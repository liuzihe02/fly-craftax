"""Random egocentric actions through env, drive, and brain; plot frames, drive, and DN rates."""
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from flycraftax.brain import (
    STEPS_PER_ACTION, BrainParams, build_weights, init_state, rate_hz, reset_counts, reset_envs, rfc_steps, run_window,
)
from flycraftax.data import load_connectome
from flycraftax.drive import build_drive, drive_rates, kick_prob
from flycraftax.env import ACTIONS, base_state, make_env
from flycraftax.retina import build_retina

GROUPS = {"DNp09": dict(types=["DNp09"]), "MDN": dict(types=["MDN"]), "DNa02_L": dict(types=["DNa02"], side="L"),
          "DNa02_R": dict(types=["DNa02"], side="R"), "MN9": dict(types=["MN9"])}


def _groups(conn):
    g = {k: conn.index(**v) for k, v in GROUPS.items()}
    # sleep group: both dorsal fan-shaped-body layers
    g["FB6/7"] = np.union1d(conn.index(type_prefix="FB6"), conn.index(type_prefix="FB7")).astype(np.int32)
    return g


def main(n_actions=40, n_steps=STEPS_PER_ACTION, batch=4, out=Path("outputs/io_rollout.png")):
    p = BrainParams()
    conn = load_connectome()
    W = build_weights(conn.n, conn.pre, conn.post, conn.signed_count(), p)
    drive = build_drive(conn, build_retina(conn))
    rfc = rfc_steps(conn.n, drive.idx, p)
    silence = jnp.ones(conn.n)
    groups = _groups(conn)
    env = make_env(num_envs=batch, reset_ratio=4)
    key = jax.random.PRNGKey(0)
    obs, env_state = env.reset(key, env.default_params)
    brain = init_state(conn.n, batch, p)
    ever = np.zeros(conn.n, bool)  # neurons of env 0 that spiked in any window
    log, frames, actions = [], [], []
    for t in range(n_actions):
        key, k_act, k_brain, k_env = jax.random.split(key, 4)
        kp = kick_prob(drive, drive_rates(drive, obs, base_state(env_state)), conn.n, p)
        brain = run_window(W, rfc, p, reset_counts(brain), kp, silence, k_brain, n_steps)
        rates = rate_hz(brain.counts, n_steps, p)
        log.append({k: float(rates[0, g].mean()) for k, g in groups.items()})
        ever |= np.asarray(brain.counts[0]) > 0
        a = jax.random.randint(k_act, (batch,), 0, len(ACTIONS))
        actions.append(int(a[0])); frames.append(np.asarray(obs[0]))
        obs, env_state, reward, done, info = env.step(k_env, env_state, a, env.default_params)
        brain = reset_envs(brain, done, p)   # spec: brain state resets per episode
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw=dict(height_ratios=[1, 2]))
    strip = np.concatenate([frames[i] for i in range(0, n_actions, max(1, n_actions // 8))], axis=1)
    ax[0].imshow(strip); ax[0].set_axis_off(); ax[0].set_title("env 0 frames over the rollout")
    for k in groups:
        ax[1].plot([r[k] for r in log], label=k)
    ax[1].set_xlabel("action index"); ax[1].set_ylabel("mean rate (Hz)"); ax[1].legend(ncol=3)
    ax[1].set_title("actions: " + " ".join(ACTIONS[a][:2] for a in actions))
    out.parent.mkdir(exist_ok=True); fig.tight_layout(); fig.savefig(out, dpi=110)
    active = int((brain.counts[0] > 0).sum())
    print(out, f"active neurons in last window (env 0): {active}")
    print(f"{'group':8s} {'n':>4s} {'mean Hz':>9s} {'max Hz':>9s} {'silent':>8s}")
    for k, g in groups.items():
        r = np.array([row[k] for row in log])
        print(f"{k:8s} {len(g):4d} {r.mean():9.2f} {r.max():9.2f} {(r == 0).mean():8.2f}")
    print(f"distinct neurons that spiked at least once over {n_actions} actions (env 0): {int(ever.sum())}")


if __name__ == "__main__":
    main()
