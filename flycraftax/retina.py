"""Photoreceptor retinotopy from the connectome, and a radial sampler over the Craftax frame."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from flycraftax.data import Connectome

CHANNEL = {"R1-R6": 0, "R8p": 1, "R8y": 2}
LAMINA = ("L1", "L2", "L3")


@dataclass
class Retina:
    idx: np.ndarray  # int32 (K,) neuron indices into the Connectome
    side: np.ndarray  # int8 (K,) 0 left, 1 right
    u: np.ndarray  # float32 (K,) 0 front .. 1 back
    v: np.ndarray  # float32 (K,) 0 near .. 1 far
    channel: np.ndarray  # int8 (K,) 0 luminance, 1 blue, 2 green


def _hex_xy(data_dir: Path, body_id: np.ndarray) -> np.ndarray:
    """Cartesian column coordinates for every neuron that has a hex assignment, else NaN."""
    ann = pd.read_feather(
        data_dir / "body-annotations.feather",
        columns=["bodyId", "assignedOlHex1", "assignedOlHex2"],
    )
    ann = ann[ann.assignedOlHex1.notna()].set_index("bodyId")
    h1 = ann.assignedOlHex1.reindex(body_id).values
    h2 = ann.assignedOlHex2.reindex(body_id).values
    return np.stack([h1 - 0.5 * h2, np.sqrt(3) / 2 * h2], axis=1)  # (N, 2), NaN where unassigned


def build_retina(conn: Connectome, data_dir: Path = Path("data")) -> Retina:
    """Each photoreceptor takes the hex column of its modal postsynaptic partner."""
    cache = data_dir / "retina.npz"
    if cache.exists():
        z = np.load(cache)
        return Retina(**{k: z[k] for k in z.files})

    xy = _hex_xy(data_dir, conn.body_id)
    has_col = ~np.isnan(xy[:, 0])
    is_lamina = np.isin(conn.type, LAMINA)
    rows = []
    for tname, ch in CHANNEL.items():
        cells = conn.index(types=[tname])
        e = np.isin(conn.pre, cells) & has_col[conn.post]
        if tname == "R1-R6":
            e &= is_lamina[conn.post]
        df = pd.DataFrame(
            {"pre": conn.pre[e], "x": xy[conn.post[e], 0], "y": xy[conn.post[e], 1], "w": conn.count[e]}
        )
        votes = df.groupby(["pre", "x", "y"], as_index=False).w.sum()
        best = votes.sort_values("w", ascending=False, kind="stable").drop_duplicates("pre")
        best["channel"] = ch
        rows.append(best)
    best = pd.concat(rows)
    idx = best.pre.values.astype(np.int32)
    side = (conn.side[idx] == "R").astype(np.int8)  # unknown / M land on the left eye
    u = np.zeros(len(idx), np.float32)
    v = np.zeros(len(idx), np.float32)
    for s in (0, 1):
        m = side == s
        for out, col in ((u, best.x.values), (v, best.y.values)):
            c = col[m]
            out[m] = (c - c.min()) / (c.max() - c.min())
    # Assumption: within an eye, increasing x runs front to back and increasing y near to far.
    r = Retina(idx=idx, side=side, u=u, v=v, channel=best.channel.values.astype(np.int8))
    np.savez(cache, **r.__dict__)
    return r
