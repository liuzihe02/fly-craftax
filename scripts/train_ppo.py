# scripts/train_ppo.py
"""Train the linear DN readout with PPO, then evaluate it greedily against the M3 baselines."""
import json
import time
from pathlib import Path

import jax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from flycraftax.brain import BrainParams
from flycraftax.data import load_connectome
from flycraftax.drive import build_drive
from flycraftax.env import ACTIONS, make_env
from flycraftax.loop import build_agent, rollout, summarise
from flycraftax.ppo import PPOConfig, make_train
from flycraftax.readout import load_readout
from flycraftax.retina import build_retina


def main(cfg=PPOConfig(), out=Path("outputs")):
    conn = load_connectome()
    readout, norm = load_readout(conn)
    drive = build_drive(conn, build_retina(conn), norm["max_hz"], lamina_mv=norm["lamina_mv"])
    agent = build_agent(conn, drive, readout, BrainParams(w_syn=norm["w_syn"]))
    env = make_env(cfg.n_envs)
    t0 = time.time()
    params, metrics = make_train(agent, env, cfg)(jax.random.PRNGKey(0))
    jax.block_until_ready(params)
    metrics = {k: np.asarray(v).tolist() for k, v in metrics.items()}   # tolist: int32 counts too
    print(f"trained {cfg.n_updates} updates in {(time.time() - t0) / 60:.1f} min; "
          f"first-10 mean episode return {np.mean(metrics['ep_return'][:10]):.3f}, length {np.mean(metrics['ep_len'][:10]):.0f}; "
          f"last-10 mean episode return {np.mean(metrics['ep_return'][-10:]):.3f}, length {np.mean(metrics['ep_len'][-10:]):.0f}", flush=True)
    out.mkdir(exist_ok=True)
    np.savez(out / "ppo_params.npz", **{k: np.asarray(v) for k, v in params.items()})
    json.dump(dict(config=cfg._asdict(), metrics=metrics), open(out / "ppo_metrics.json", "w"), indent=1)

    env8 = make_env(8)
    logs = rollout(agent, env8, jax.random.PRNGKey(0), 2000, 8, policy="linear", params=params, greedy=True)
    result = summarise(logs, 2000)
    base = json.load(open(out / "zero_shot.json"))["results"]
    table = {"ppo/greedy": result, "full/readout": base["full/readout"], "full/random": base["full/random"]}
    for name, r in table.items():
        print(f"{name:14s} survival {np.mean(r['survival']):7.1f}  achievements {sum(r['achievements'])}  actions "
              + " ".join(f"{a[:2]}={h:.2f}" for a, h in zip(ACTIONS, r["action_hist"])), flush=True)
    json.dump(table, open(out / "ppo_eval.json", "w"), indent=1)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    ax[0].plot(metrics["ep_return"]); ax[0].set_title("episode return per update"); ax[0].set_xlabel("update")
    ax[1].plot(metrics["ep_len"]); ax[1].set_title("episode length per update"); ax[1].set_xlabel("update")
    hist = np.array(metrics["action_hist"])
    for i, a in enumerate(ACTIONS):
        ax[2].plot(hist[:, i], label=a)
    ax[2].set_title("action frequency per update"); ax[2].set_xlabel("update"); ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "ppo_curve.png", dpi=110); print(out / "ppo_curve.png", flush=True)


if __name__ == "__main__":
    main()
