# scripts/eval_ppo_controls.py
"""Ablation controls for the trained readout: does it need the brain's DN rates, or only a clock?

The M4 greedy evaluation ran the full agent on the training worlds and beat neither baseline on
survival, so nothing separated "the fly's descending-neuron rates" from "any noisy feature vector".
Here the same frozen readout is replayed on a fresh key with vision blacked out, vision frozen and
all drive removed, against an open-loop FORWARD/DO alternation and the random baseline.
"""
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from eval_zero_shot import line, plot_conditions   # both scripts live in scripts/, so this is a sibling import

from flycraftax.brain import BrainParams
from flycraftax.data import load_connectome
from flycraftax.drive import build_drive
from flycraftax.env import make_env
from flycraftax.loop import build_agent, rollout, summarise
from flycraftax.readout import load_readout
from flycraftax.retina import build_retina

# (name, ablation, policy): every condition is the full wiring, so one agent serves them all
CONDITIONS = [("ppo/greedy", "full", "linear"), ("ppo/greedy/black", "black", "linear"),
              ("ppo/greedy/static", "static", "linear"), ("ppo/greedy/disconnected", "disconnected", "linear"),
              ("alternate/full", "full", "alternate"), ("full/random", "full", "random")]
PARAM_KEYS = ("W", "b", "vw", "vb")


def load_params(path):
    with np.load(path) as z:
        missing = [k for k in PARAM_KEYS if k not in z]
        if missing:
            raise SystemExit(f"{path} is missing {missing}: it should hold the linear readout's {PARAM_KEYS}")
        return {k: jnp.asarray(z[k]) for k in PARAM_KEYS}


def main(n_actions=2000, batch=8, seed=1, out=Path("outputs")):
    out.mkdir(parents=True, exist_ok=True)
    params_path = out / "ppo_params.npz"
    if not params_path.exists():
        raise SystemExit(f"{params_path} not found: run `make train` first, it writes the trained readout")
    params = load_params(params_path)
    conn = load_connectome()
    readout, norm = load_readout(conn)
    drive = build_drive(conn, build_retina(conn), norm["max_hz"], lamina_mv=norm["lamina_mv"])
    agent = build_agent(conn, drive, readout, BrainParams(w_syn=norm["w_syn"]))
    env = make_env(batch)
    t0 = time.time()
    results = {}
    for name, ablation, policy in CONDITIONS:
        # a fresh key, not the training key: these 8 worlds are held out from the PPO run
        logs = rollout(agent, env, jax.random.PRNGKey(seed), n_actions, batch, ablation=ablation, policy=policy,
                       params=params if policy == "linear" else None, greedy=policy == "linear", keep_feats=False)
        results[name] = r = summarise(logs, n_actions)
        print(line(name, r, width=24), flush=True)
    print(f"{len(CONDITIONS)} conditions in {(time.time() - t0) / 60:.1f} min", flush=True)

    config = dict(n_actions=n_actions, batch=batch, params=str(params_path), norm=norm,
                  conditions=[list(c) for c in CONDITIONS])
    with open(out / "ppo_controls.json", "w") as f:
        json.dump(dict(config=config, key=seed, results=results), f, indent=1)
    plot_conditions(results, batch, out / "ppo_controls.png")
    print(out / "ppo_controls.json", out / "ppo_controls.png", flush=True)


if __name__ == "__main__":
    main()
