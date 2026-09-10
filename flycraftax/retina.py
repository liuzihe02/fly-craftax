"""Photoreceptor retinotopy from the connectome, and a radial sampler over the Craftax frame."""

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax.scipy.ndimage import map_coordinates

from flycraftax.data import Connectome

_CACHE_VERSION = 1  # bump when build_retina's output changes

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
    cache = data_dir / f"retina_v{_CACHE_VERSION}.npz"
    if cache.exists():
        z = np.load(cache)
        # idx points into the Connectome, so a differently sized one invalidates the cache.
        if int(z["n"]) == conn.n:
            return Retina(**{k: z[k] for k in z.files if k != "n"})

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
    np.savez(cache, n=conn.n, **r.__dict__)
    return r


CENTER = (24.5, 31.5)
MAP_ROWS = 49
# Unit (row, col) vectors for facing values 1..4: LEFT, RIGHT, UP, DOWN. Index 0 unused.
FACING_VEC = jnp.array([[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]], jnp.float32)
LUMA = jnp.array([0.2126, 0.7152, 0.0722], jnp.float32)


def sample_points(retina: Retina, facing, r_min: float = 4.0, r_max: float = 23.0):
    """Pixel (row, col) coords, (B, K, 2), for each cell given facing (B,) Action values."""
    theta = jnp.where(retina.side == 0, 1.0, -1.0) * retina.u * jnp.pi  # 0 ahead, +pi/2 left, -pi/2 right
    radius = r_min + retina.v * (r_max - r_min)
    fwd = FACING_VEC[facing]  # (B, 2)
    left = jnp.stack([-fwd[:, 1], fwd[:, 0]], axis=1)  # 90 degrees counter-clockwise on screen
    offset = radius[None, :, None] * (
        jnp.cos(theta)[None, :, None] * fwd[:, None, :] + jnp.sin(theta)[None, :, None] * left[:, None, :]
    )
    return jnp.asarray(CENTER) + offset  # (B, K, 2)


def sample(retina: Retina, obs, facing, r_min: float = 4.0, r_max: float = 23.0):
    """Per-cell values in [0, 1], (B, K), bilinear from the frame; grey 0.5 outside the map view."""
    pts = sample_points(retina, facing, r_min, r_max)
    view = obs[:, :MAP_ROWS]  # crop the inventory bar
    planes = jnp.stack([view @ LUMA, view[..., 2], view[..., 1]], axis=1)  # (B, 3, 49, 63): luminance, blue, green

    def one(planes_b, pts_b):
        vals = jax.vmap(
            lambda pl: map_coordinates(pl, [pts_b[:, 0], pts_b[:, 1]], order=1, mode="constant", cval=0.5)
        )(planes_b)
        return vals[retina.channel, jnp.arange(len(retina.channel))]

    return jax.vmap(one)(planes, pts)
