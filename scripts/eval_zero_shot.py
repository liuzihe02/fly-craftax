# scripts/eval_zero_shot.py
"""Zero-shot fly versus ablation controls: survival, achievements, actions; JSON, figure, GIF."""
import json
from pathlib import Path

import imageio.v2 as imageio
import jax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from craftax.craftax_classic.constants import Achievement

from flycraftax.brain import BrainParams
from flycraftax.data import load_connectome
from flycraftax.drive import build_drive
from flycraftax.env import ACTIONS, make_env
from flycraftax.loop import build_agent, rollout, summarise
from flycraftax.readout import load_readout
from flycraftax.retina import build_retina

CONDITIONS = [("full", "readout"), ("black", "readout"), ("static", "readout"), ("disconnected", "readout"),
              ("shuffled", "readout"), ("full", "random")]
SURVIVAL = (Achievement.COLLECT_WOOD, Achievement.EAT_COW, Achievement.COLLECT_SAPLING, Achievement.COLLECT_DRINK,
            Achievement.DEFEAT_ZOMBIE, Achievement.DEFEAT_SKELETON, Achievement.WAKE_UP)


def line(name, r, width=22):
    """One condition's summary, the format both eval scripts print."""
    return (f"{name:{width}s} survival {np.mean(r['survival']):7.1f}  reward/action {r['reward']:+.4f}  "
            f"achievements {sum(r['achievements'])}  actions "
            + " ".join(f"{a[:2]}={h:.2f}" for a, h in zip(ACTIONS, r["action_hist"])))


def plot_conditions(results, batch, path):
    """Three panels: survival bars with per-env dots, stacked survival achievements, action frequency."""
    names = list(results)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    for i, n in enumerate(names):
        ax[0].scatter([i] * batch, results[n]["survival"], s=14)
    ax[0].bar(range(len(names)), [np.mean(results[n]["survival"]) for n in names], alpha=0.4)
    ax[0].set_xticks(range(len(names))); ax[0].set_xticklabels(names, rotation=30, ha="right"); ax[0].set_ylabel("actions survived")
    bottom = np.zeros(len(names))
    for a in SURVIVAL:
        vals = np.array([results[n]["achievements"][a.value] for n in names])
        ax[1].bar(range(len(names)), vals, bottom=bottom, label=a.name.lower()); bottom += vals
    ax[1].set_xticks(range(len(names))); ax[1].set_xticklabels(names, rotation=30, ha="right"); ax[1].set_ylabel(f"achievements ({batch} envs)"); ax[1].legend(fontsize=7)
    for n in names:
        ax[2].plot(results[n]["action_hist"], marker="o", label=n)
    ax[2].set_xticks(range(len(ACTIONS))); ax[2].set_xticklabels(ACTIONS, rotation=30, ha="right"); ax[2].set_ylabel("action frequency"); ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main(n_actions=2000, batch=8, out=Path("outputs")):
    conn = load_connectome()
    readout, cfg = load_readout(conn)
    drive = build_drive(conn, build_retina(conn), cfg["max_hz"], lamina_mv=cfg["lamina_mv"])
    p = BrainParams(w_syn=cfg["w_syn"])
    env = make_env(batch)
    results = {}
    for ablation, policy in CONDITIONS:
        name = f"{ablation}/{policy}"
        agent = build_agent(conn, drive, readout, p, wiring="shuffled" if ablation == "shuffled" else "full")
        logs = rollout(agent, env, jax.random.PRNGKey(0), n_actions, batch, ablation=ablation, policy=policy)
        results[name] = r = summarise(logs, n_actions)
        print(line(name, r), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "zero_shot.json", "w") as f:
        json.dump(dict(config=cfg, results=results), f, indent=1)

    plot_conditions(results, batch, out / "zero_shot.png")

    agent = build_agent(conn, drive, readout, p)
    logs = rollout(agent, env, jax.random.PRNGKey(1), 300, batch, keep_frames=True)
    frames = (np.asarray(logs["frame"]) * 255).astype(np.uint8).repeat(4, axis=1).repeat(4, axis=2)
    imageio.mimsave(out / "zero_shot.gif", list(frames), duration=100)   # imageio 2.37: ms, so 0.1 s
    print(out / "zero_shot.png", out / "zero_shot.gif")


if __name__ == "__main__":
    main()
