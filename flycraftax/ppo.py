"""PPO for a linear readout over descending-neuron rates; the brain stays frozen."""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from flycraftax.env import ACTIONS, base_state
from flycraftax.loop import brain_feats, init_carry, make_step


def init_params(n_dn, n_actions=7):
    """Zeros, so the initial policy is uniform and the initial value is zero."""
    return dict(W=jnp.zeros((n_dn, n_actions)), b=jnp.zeros(n_actions), vw=jnp.zeros(n_dn), vb=jnp.zeros(()))


def policy(params, feats):
    return feats @ params["W"] + params["b"], feats @ params["vw"] + params["vb"]


class PPOConfig(NamedTuple):
    n_envs: int = 8
    n_steps: int = 64
    n_updates: int = 150
    lr: float = 3e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    n_epochs: int = 4
    n_minibatches: int = 4
    max_grad_norm: float = 0.5


def gae(rewards, values, dones, last_value, gamma, lam):
    """Generalised advantage estimation over (T, B); `dones` end the episode after that step."""
    def back(carry, x):
        adv_next, v_next = carry
        r, v, d = x
        nonterm = 1.0 - d.astype(jnp.float32)
        delta = r + gamma * v_next * nonterm - v
        adv = delta + gamma * lam * nonterm * adv_next
        return (adv, v), adv
    _, adv = jax.lax.scan(back, (jnp.zeros_like(last_value), last_value), (rewards, values, dones), reverse=True)
    return adv, adv + values


def loss(params, batch, clip_eps, ent_coef, vf_coef):
    """PureJaxRL's: clipped surrogate, clipped value loss, entropy bonus, normalised advantages."""
    logits, value = policy(params, batch["feats"])
    logp_all = jax.nn.log_softmax(logits)
    logp = jnp.take_along_axis(logp_all, batch["action"][:, None], axis=1)[:, 0]
    ratio = jnp.exp(logp - batch["logp_old"])
    adv = (batch["adv"] - batch["adv"].mean()) / (batch["adv"].std() + 1e-8)
    pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv).mean()
    v_clip = batch["value_old"] + jnp.clip(value - batch["value_old"], -clip_eps, clip_eps)
    vf = 0.5 * jnp.maximum((value - batch["target"]) ** 2, (v_clip - batch["target"]) ** 2).mean()
    entropy = -(jnp.exp(logp_all) * logp_all).sum(axis=1).mean()
    return pg + vf_coef * vf - ent_coef * entropy, dict(pg=pg, vf=vf, entropy=entropy)


def make_train(agent, env, cfg):
    """One jitted `train(key) -> (params, metrics)`; only the readout is in the optimiser."""
    step = make_step(agent, env, policy="linear")
    tx = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(cfg.lr))
    n_batch = cfg.n_envs * cfg.n_steps
    mb = n_batch // cfg.n_minibatches

    def update(runner, key):
        carry, opt_state = runner
        key, k_roll, k_boot, k_shuf = jax.random.split(key, 4)
        carry, logs = jax.lax.scan(step, carry, jax.random.split(k_roll, cfg.n_steps))
        obs, env_state, brain, params = carry
        # the features after the last step are not in the logs: one more brain window on the final
        # carry, env not stepped and the carried brain state left alone, just for the bootstrap
        _, _, last_feats = brain_feats(agent, obs, base_state(env_state), brain, k_boot)
        last_value = policy(params, last_feats)[1]
        adv, target = gae(logs["reward"], logs["value"], logs["done"], last_value, cfg.gamma, cfg.gae_lambda)
        flat = dict(feats=logs["feats"].reshape(n_batch, -1), action=logs["action"].reshape(n_batch),
                    logp_old=logs["logp"].reshape(n_batch), value_old=logs["value"].reshape(n_batch),
                    adv=adv.reshape(n_batch), target=target.reshape(n_batch))

        def epoch(state, k):
            params, opt_state = state
            perm = jax.random.permutation(k, n_batch)

            def minibatch(state, i):
                params, opt_state = state
                idx = jax.lax.dynamic_slice_in_dim(perm, i * mb, mb)
                batch = {k: v[idx] for k, v in flat.items()}
                (l, aux), grads = jax.value_and_grad(loss, has_aux=True)(params, batch, cfg.clip_eps, cfg.ent_coef, cfg.vf_coef)
                updates, opt_state = tx.update(grads, opt_state, params)
                return (optax.apply_updates(params, updates), opt_state), (l, aux["entropy"])

            return jax.lax.scan(minibatch, (params, opt_state), jnp.arange(cfg.n_minibatches))

        (params, opt_state), (losses, ents) = jax.lax.scan(epoch, (params, opt_state), jax.random.split(k_shuf, cfg.n_epochs))
        done = logs["ep_done"]
        n_done = jnp.maximum(done.sum(), 1)
        metrics = dict(
            ep_return=(logs["ep_return"] * done).sum() / n_done, ep_len=(logs["ep_len"] * done).sum() / n_done,
            n_episodes=done.sum(), entropy=ents.mean(), loss=losses.mean(),
            action_hist=jnp.zeros(len(ACTIONS)).at[logs["action"].reshape(-1)].add(1.0) / n_batch,
        )
        return ((obs, env_state, brain, params), opt_state), metrics

    def train(key):
        key, k0 = jax.random.split(key)
        params = init_params(len(agent.dn_idx))
        carry = init_carry(agent, env, k0, cfg.n_envs) + (params,)
        runner = (carry, tx.init(params))
        (carry, _), metrics = jax.lax.scan(update, runner, jax.random.split(key, cfg.n_updates))
        return carry[3], metrics

    return jax.jit(train)
