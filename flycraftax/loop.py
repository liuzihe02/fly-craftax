"""Closed loop: obs and state -> drive -> brain window -> readout -> env step, as one scan over actions."""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from flycraftax.brain import (
    STEPS_PER_ACTION, BrainParams, build_weights, init_state, rate_hz, reset_counts, reset_envs, rfc_steps, run_window,
)
from flycraftax.drive import Drive, drive_rates, kick_prob
from flycraftax.env import ACTIONS, base_state
from flycraftax.readout import Readout, act, signals

ABLATIONS = ("full", "black", "static", "disconnected", "shuffled")
POLICIES = ("readout", "random")


@dataclass
class Agent:
    """Unhashable: close over it, never pass it as a static jit argument."""

    W: object          # BCOO (N, N)
    rfc: jax.Array     # int32 (N,) refractory steps, 0 on the driven cells
    p: BrainParams
    drive: Drive
    readout: Readout
    n: int
    n_steps: int = STEPS_PER_ACTION
    ablation: str = "full"


def build_agent(conn, drive, readout, p=BrainParams(), ablation="full", seed=0):
    """`shuffled` permutes the edge targets, keeping the in-degree distribution."""
    post = np.random.default_rng(seed).permutation(conn.post) if ablation == "shuffled" else conn.post
    W = build_weights(conn.n, conn.pre, post, conn.signed_count(), p)
    return Agent(W, rfc_steps(conn.n, drive.idx, p), p, drive, readout, conn.n, ablation=ablation)


def rollout(agent, env, key, n_actions, batch, ablation=None, policy="readout", keep_frames=False):
    """One scan over actions; the env must have been built with num_envs == batch.

    `ablation` defaults to the one the agent was built with -- `shuffled` and `disconnected`
    only differ in `build_agent`, so passing a mismatched one here would silently mislabel a run.
    """
    if ablation is None:
        ablation = agent.ablation
    assert ablation == agent.ablation, f"rollout ablation {ablation!r} != agent's {agent.ablation!r}"
    assert ablation in ABLATIONS and policy in POLICIES
    assert env.num_envs == batch, f"env.num_envs {env.num_envs} != batch {batch}"
    params = env.default_params          # static to env.step: pass it from the closure, not the carry
    key, k0 = jax.random.split(key)
    obs0, env_state = env.reset(k0, params)
    silence = jnp.ones(agent.n)
    bias = jnp.asarray(agent.drive.bias)
    l1 = agent.drive.lamina_l1

    def step(carry, k):
        obs, env_state, brain = carry
        k_brain, k_act, k_env = jax.random.split(k, 3)
        st = base_state(env_state)
        if ablation == "black":
            obs_in = jnp.zeros_like(obs)     # photoreceptors silent, lamina at its resting rate
        elif ablation == "static":
            obs_in = obs0                    # vision carries no information about the world
        else:
            obs_in = obs
        rates_in = drive_rates(agent.drive, obs_in, st)
        if ablation == "disconnected":
            rates_in = jnp.zeros_like(rates_in)   # no kicks; the bias stays
        kp = kick_prob(agent.drive, rates_in, agent.n, agent.p)
        brain = run_window(agent.W, agent.rfc, agent.p, reset_counts(brain), kp, silence, k_brain, agent.n_steps,
                           bias=bias)
        rates = rate_hz(brain.counts, agent.n_steps, agent.p)
        action, z = act(agent.readout, rates)
        if policy == "random":
            action = jax.random.randint(k_act, (batch,), 0, len(ACTIONS))
        obs2, env_state2, reward, done, _ = env.step(k_env, env_state, action, params)
        log = dict(
            action=action, reward=reward, done=done, sig=signals(agent.readout.groups, rates), z=z,
            active=(brain.counts > 0).mean(axis=1), lamina=rates[:, l1].mean(axis=1), achievements=st.achievements,
            health=st.player_health, food=st.player_food, drink=st.player_drink, energy=st.player_energy,
        )
        if keep_frames:
            log["frame"] = obs[0]        # the frame that drove this action, env 0 only
        brain = reset_envs(brain, done, agent.p)   # after the log: active and lamina are pre-reset
        return (obs2, env_state2, brain), log

    keys = jax.random.split(key, n_actions)
    _, logs = jax.lax.scan(step, (obs0, env_state, init_state(agent.n, batch, agent.p)), keys)
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
