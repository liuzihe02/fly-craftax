"""Closed loop: obs and state -> drive -> brain window -> readout -> env step, as one scan over actions."""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from flycraftax.brain import (
    STEPS_PER_ACTION, BrainParams, build_weights, init_state, rate_hz, reset_counts, reset_envs, rfc_steps, run_window,
)
from flycraftax.drive import Drive, drive_rates, kick_prob
from flycraftax.env import ACTIONS, DO, FORWARD, base_state
from flycraftax.readout import SPIKE_HZ, Readout, act, signals

ABLATIONS = ("full", "black", "static", "disconnected", "shuffled")
WIRINGS = ("full", "shuffled")     # the only build-time ablation: the rest are applied at rollout time
POLICIES = ("readout", "random", "linear", "alternate")


@dataclass
class Agent:
    """Unhashable: close over it, never pass it as a static jit argument."""

    W: object          # BCOO (N, N)
    rfc: jax.Array     # int32 (N,) refractory steps, 0 on the driven cells
    p: BrainParams
    drive: Drive
    readout: Readout
    n: int
    dn_idx: np.ndarray  # int32 (1314,) descending neurons, the linear policy's features
    n_steps: int = STEPS_PER_ACTION
    wiring: str = "full"


def build_agent(conn, drive, readout, p=BrainParams(), wiring="full", seed=0):
    """`shuffled` permutes the edge targets, keeping the in-degree distribution.

    Wiring is the only build-time ablation: every other ablation builds the `full` agent and is
    applied in `rollout`, so a caller holding a condition name passes
    `wiring="shuffled" if ablation == "shuffled" else "full"`.
    """
    assert wiring in WIRINGS, f"unknown wiring {wiring!r}"
    post = np.random.default_rng(seed).permutation(conn.post) if wiring == "shuffled" else conn.post
    W = build_weights(conn.n, conn.pre, post, conn.signed_count(), p)
    dn_idx = np.flatnonzero(conn.superclass == "descending_neuron").astype(np.int32)
    return Agent(W, rfc_steps(conn.n, drive.idx, p), p, drive, readout, conn.n, dn_idx, wiring=wiring)


def brain_feats(agent, obs_in, st, brain, key, disconnected=False):
    """One action's worth of brain: drive -> window -> rates -> DN features (spikes per window)."""
    rates_in = drive_rates(agent.drive, obs_in, st)
    if disconnected:
        rates_in = jnp.zeros_like(rates_in)   # no kicks; the bias stays
    kp = kick_prob(agent.drive, rates_in, agent.n, agent.p)
    brain = run_window(agent.W, agent.rfc, agent.p, reset_counts(brain), kp, jnp.ones(agent.n), key,
                       agent.n_steps, bias=jnp.asarray(agent.drive.bias))
    rates = rate_hz(brain.counts, agent.n_steps, agent.p)
    return brain, rates, rates[:, agent.dn_idx] / SPIKE_HZ


def init_carry(agent, env, key, batch):
    """(obs, env_state, brain); the env must have been built with num_envs == batch."""
    assert env.num_envs == batch, f"env.num_envs {env.num_envs} != batch {batch}"
    obs, env_state = env.reset(key, env.default_params)
    return obs, env_state, init_state(agent.n, batch, agent.p)


def make_step(agent, env, ablation="full", policy="readout", keep_frames=False, greedy=False, obs0=None,
              keep_counts=False, keep_feats=None):
    """The scan body: step((obs, env_state, brain, params), key) -> (carry, log).

    `params` is None for readout and random, and a dict with W, b, vw, vb for linear; it rides in
    the carry untouched so PPO can scan the same step with fresh parameters each update. Only
    `shuffled` lives in the agent's wiring; the other ablations are applied here.

    `keep_feats` logs `feats` (B, 1314), `logp` and `value`; it defaults to the linear policy, the
    only consumer, because a 2,000-action rollout of them is ~100 MB of device memory nothing reads.

    `alternate` is the open-loop control for the trained readout: FORWARD, DO, FORWARD, DO ...,
    the mix the greedy linear policy converged on, driven by the brain's step counter and nothing else.
    """
    assert ablation in ABLATIONS and policy in POLICIES
    if keep_feats is None:
        keep_feats = policy == "linear"
    assert (ablation == "shuffled") == (agent.wiring == "shuffled"), \
        f"ablation {ablation!r} against a {agent.wiring!r} agent would mislabel the run"
    if ablation == "static" and obs0 is None:
        raise ValueError("the `static` ablation needs obs0, the frame to freeze vision on")
    if policy == "linear":
        from flycraftax.ppo import policy as linear   # deferred: ppo imports this module
    env_params = env.default_params      # static to env.step: pass it from the closure, not the carry
    l1 = agent.drive.lamina_l1

    def step(carry, k):
        obs, env_state, brain, params = carry
        k_brain, k_act, k_env = jax.random.split(k, 3)
        window = brain.t // agent.n_steps      # the action index: read before this action's window runs
        st = base_state(env_state)
        if ablation == "black":
            obs_in = jnp.zeros_like(obs)     # photoreceptors silent, lamina at its resting rate
        elif ablation == "static":
            obs_in = obs0                    # vision carries no information about the world
        else:
            obs_in = obs
        brain, rates, feats = brain_feats(agent, obs_in, st, brain, k_brain, ablation == "disconnected")
        action, z = act(agent.readout, rates)
        logp = value = jnp.zeros(env.num_envs)
        if policy == "random":
            action = jax.random.randint(k_act, (env.num_envs,), 0, len(ACTIONS))
        elif policy == "alternate":
            action = jnp.full((env.num_envs,), jnp.where(window % 2 == 0, FORWARD, DO), jnp.int32)
        elif policy == "linear":
            logits, value = linear(params, feats)
            action = jnp.argmax(logits, axis=1) if greedy else jax.random.categorical(k_act, logits)
            logp = jnp.take_along_axis(jax.nn.log_softmax(logits), action[:, None], axis=1)[:, 0]
        obs2, env_state2, reward, done, info = env.step(k_env, env_state, action, env_params)
        log = dict(
            action=action, reward=reward, done=done, sig=signals(agent.readout.groups, rates), z=z,
            active=(brain.counts > 0).mean(axis=1), lamina=rates[:, l1].mean(axis=1), achievements=st.achievements,
            health=st.player_health, food=st.player_food, drink=st.player_drink, energy=st.player_energy,
            ep_return=info["returned_episode_returns"], ep_len=info["returned_episode_lengths"],
            ep_done=info["returned_episode"],
        )
        if keep_feats:
            log.update(feats=feats, logp=logp, value=value)
        if keep_frames:
            log["frame"] = obs[0]        # the frame that drove this action, env 0 only
        if keep_counts:
            log["counts"] = brain.counts  # (B, N) spikes in this window; the viewer's only use
        brain = reset_envs(brain, done, agent.p)   # after the log: active, lamina and counts are pre-reset
        return (obs2, env_state2, brain, params), log

    return step


def rollout(agent, env, key, n_actions, batch, ablation="full", policy="readout", keep_frames=False,
            params=None, greedy=False, keep_feats=None):
    """One scan over actions; the env must have been built with num_envs == batch.

    `shuffled` is the one ablation that has to match the agent (it is built into the wiring);
    `black`, `static` and `disconnected` are applied to the full agent as the rollout runs.
    """
    key, k0 = jax.random.split(key)
    carry = init_carry(agent, env, k0, batch) + (params,)
    step = make_step(agent, env, ablation, policy, keep_frames, greedy, obs0=carry[0], keep_feats=keep_feats)
    _, logs = jax.lax.scan(step, carry, jax.random.split(key, n_actions))
    return logs


def summarise(logs, n_actions):
    """Survival, achievements held at death, action histogram, and mean reward from rollout logs."""
    done = np.asarray(logs["done"])
    first = np.where(done.any(0), done.argmax(0) + 1, n_actions)
    ach = np.asarray(logs["achievements"])
    # the log holds the pre-step state and the auto-reset zeroes achievements on the death step,
    # so read the last row before the reset; identical to ach[f] for an env that never dies
    unlocked = np.stack([ach[f - 1, b] for b, f in enumerate(first)])
    actions = np.asarray(logs["action"])
    hist = np.bincount(actions.ravel(), minlength=len(ACTIONS)) / actions.size
    return dict(survival=first.tolist(), achievements=unlocked.sum(0).tolist(),
                action_hist=hist.tolist(), reward=float(np.asarray(logs["reward"]).mean()))
