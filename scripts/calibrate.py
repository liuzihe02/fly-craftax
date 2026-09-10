# scripts/calibrate.py
"""Sweep lamina bias and synapse weight under a random policy; write the readout norm file."""
import json
from pathlib import Path

import jax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from flycraftax.brain import BrainParams
from flycraftax.data import load_connectome
from flycraftax.drive import build_drive
from flycraftax.env import make_env
from flycraftax.loop import build_agent, rollout
from flycraftax.readout import NORM_PATH, SIGNALS, Readout, readout_groups, save_readout, std_floor
from flycraftax.retina import build_retina

MAX_HZ = 100.0
GRID = [(w, mv) for w in (0.275, 0.44) for mv in (0.04, 0.06, 0.10)]


def main(n_actions=300, batch=4, out=Path("outputs/calibration.png")):
    conn = load_connectome()
    retina = build_retina(conn)
    groups = readout_groups(conn)
    blank = Readout(groups, np.zeros(6, np.float32), std_floor(groups))
    floor = std_floor(groups)
    env = make_env(batch)
    rows = []
    for w_syn, mv in GRID:
        drive = build_drive(conn, retina, MAX_HZ, lamina_mv=mv)
        agent = build_agent(conn, drive, blank, BrainParams(w_syn=w_syn))
        full = rollout(agent, env, jax.random.PRNGKey(0), n_actions, batch, policy="random")
        black = rollout(agent, env, jax.random.PRNGKey(0), 60, batch, ablation="black", policy="random")
        sig = np.asarray(full["sig"]).reshape(-1, 6)
        row = dict(w_syn=w_syn, lamina_mv=mv, mean=sig.mean(0).tolist(), std=sig.std(0).tolist(),
                   nonzero=int((sig.std(0) > 0).sum()), active=float(np.asarray(full["active"]).mean()),
                   l1_full=float(np.asarray(full["lamina"]).mean()), l1_black=float(np.asarray(black["lamina"]).mean()))
        rows.append(row)
        print(f"w_syn={w_syn:.3f} mv={mv:.2f}  nonzero={row['nonzero']}  active={row['active']:.3f}"
              f"{'  RUNAWAY' if row['active'] > 0.5 else ''}  "
              f"L1 full/black {row['l1_full']:.1f}/{row['l1_black']:.1f} Hz  "
              + "  ".join(f"{s}={m:.2f}+-{sd:.2f}{'!' if sd <= f else ''}"
                          for s, m, sd, f in zip(SIGNALS, row["mean"], row["std"], floor)), flush=True)
    ok = [r for r in rows if r["nonzero"] >= 4 and r["active"] < 0.25 and r["l1_black"] > r["l1_full"]]
    pick = ok[0] if ok else max((r for r in rows if r["active"] < 0.25), key=lambda r: r["nonzero"], default=rows[0])
    print("pick:", pick["w_syn"], pick["lamina_mv"], "qualified" if ok else "best available", flush=True)
    save_readout(NORM_PATH, MAX_HZ, pick["w_syn"], pick["lamina_mv"], pick["mean"], pick["std"])
    out.parent.mkdir(exist_ok=True)
    json.dump(rows, open(out.with_suffix(".json"), "w"), indent=1)
    labels = [f"{r['w_syn']}/{r['lamina_mv']}" for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    for i, s in enumerate(SIGNALS):
        ax[0].plot([r["mean"][i] for r in rows], marker="o", label=s)
    ax[0].set_xticks(range(len(rows))); ax[0].set_xticklabels(labels, rotation=45); ax[0].set_ylabel("mean signal (Hz)"); ax[0].legend(ncol=2)
    ax[1].bar(range(len(rows)), [r["active"] for r in rows]); ax[1].set_xticks(range(len(rows))); ax[1].set_xticklabels(labels, rotation=45); ax[1].set_ylabel("active fraction")
    ax[2].plot([r["l1_full"] for r in rows], marker="o", label="L1 real frames"); ax[2].plot([r["l1_black"] for r in rows], marker="s", label="L1 black frame")
    ax[2].set_xticks(range(len(rows))); ax[2].set_xticklabels(labels, rotation=45); ax[2].set_ylabel("Hz"); ax[2].legend()
    fig.tight_layout(); fig.savefig(out, dpi=110); print(out, flush=True)


if __name__ == "__main__":
    main()
