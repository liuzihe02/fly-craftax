"""Offline viewer: fly sprite in the game render, brain activity over soma positions, readout and meters."""

from pathlib import Path

import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from craftax.craftax_classic.constants import BLOCK_PIXEL_SIZE_HUMAN, OBS_DIM, load_all_textures
from craftax.craftax_classic.renderer import make_craftax_pixel_renderer

from flycraftax.env import ACTIONS
from flycraftax.readout import SIGNALS

BODY, WING, HEAD = (40, 30, 30, 255), (200, 220, 255, 140), (230, 200, 60, 255)


def _draw(sleeping):
    """16x16 RGBA fly facing up (row 0 is the head end); callers rotate/flip.

    Left-right symmetric apart from the sleep `z`, so that the left sprite is the
    mirror of the right one.
    """
    s = np.zeros((16, 16, 4), np.uint8)
    yy, xx = np.mgrid[:16, :16]
    body = ((yy - 8.5) / 5.0) ** 2 + ((xx - 7.5) / 2.2) ** 2 <= 1.0
    if sleeping:  # wings folded over the body
        wings = ((yy - 9) / 4.0) ** 2 + ((xx - 7.5) / 3.2) ** 2 <= 1.0
    else:
        wings = (((yy - 9) / 3.5) ** 2 + ((xx - 3.5) / 3.0) ** 2 <= 1.0) | (
            ((yy - 9) / 3.5) ** 2 + ((xx - 11.5) / 3.0) ** 2 <= 1.0
        )
    s[wings] = WING
    s[body] = BODY
    head = ((yy - 3.5) / 1.6) ** 2 + ((xx - 7.5) / 1.6) ** 2 <= 1.0
    s[head] = HEAD
    if sleeping:
        s[1, 12:15] = s[3, 12:15] = (255, 255, 255, 255)
        s[2, 13] = (255, 255, 255, 255)  # a small z
    return s


def fly_sprites():
    """uint8 (5, 16, 16, 4) in the Craftax player order: left, right, up, down, sleep."""
    up = _draw(False)
    return np.stack([np.rot90(up, 1), np.rot90(up, -1), up, up[::-1], _draw(True)])


def make_fly_renderer():
    """The stock 64 px renderer with the player textures swapped for the fly sprites.

    `load_all_textures` is lru_cached and hands back a mutable dict, so overwriting the two
    player entries before building the renderer is enough (and is process-wide).
    """
    bps = BLOCK_PIXEL_SIZE_HUMAN
    tex = load_all_textures()[bps]
    sprites = fly_sprites().repeat(bps // 16, axis=1).repeat(bps // 16, axis=2)
    pad = ((OBS_DIM[0] // 2) * bps, (OBS_DIM[1] // 2) * bps)  # as constants.py pads the player
    padded = jnp.array([jnp.pad(s, ((pad[0], pad[0]), (pad[1], pad[1]), (0, 0))) for s in sprites])
    tex["full_map_player_textures"] = padded[..., :3].astype(jnp.float32)
    tex["full_map_player_textures_alpha"] = jnp.repeat(
        padded[..., 3:4].astype(jnp.float32) / 255, 3, axis=3
    )
    return make_craftax_pixel_renderer(bps)


def soma_xy(conn, data_dir=Path("data")):
    """float32 (N, 2) soma positions in connectome order, NaN where the soma is unknown."""
    ann = pd.read_feather(
        data_dir / "body-annotations.feather", columns=["bodyId", "somaLocation"]
    ).set_index("bodyId")
    loc = ann.somaLocation.reindex(conn.body_id)
    # missing somata come back as None (and reindex misses would be NaN); both have ndim 0
    xyz = np.array([p if np.ndim(p) == 1 else [np.nan] * 3 for p in loc.values], np.float32)
    # (x, y): x is the left-right axis, so the two optic lobes are the dense blobs at the far
    # left and right of a frontal view of the brain (checked by colouring the ol_* superclasses
    # in a scatter). (x, z) puts the same lobes side by side but z also carries the
    # neck/VNC-projecting somata out to 134k, squashing the brain into the bottom sixth.
    # y increases ventrally, so `compose` inverts the brain panel's y axis to put dorsal up.
    return xyz[:, :2]


def compose(frame64, counts, xy, groups, z, action, meters, z_floor, vmax=None):
    """One 16x9 panel: game frame, brain activity, the six z-scores and the meters.

    `vmax` fixes the spike-count colour scale (pass one scale for a whole run so frames compare);
    None picks this window's 95th percentile.
    """
    fig = plt.figure(figsize=(16, 9), dpi=80)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.1, 1.3, 0.6])
    ax_game, ax_brain = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[:, 1])
    ax_sig, ax_met = fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 2])
    ax_game.imshow(np.asarray(frame64) / 255.0)
    ax_game.set_axis_off()
    ax_game.set_title("Craftax")
    ok = ~np.isnan(xy[:, 0])
    ax_brain.scatter(xy[ok, 0], xy[ok, 1], s=1, c="#dddddd", linewidths=0)
    hot = ok & (counts > 0)
    if vmax is None:
        vmax = float(np.percentile(counts[hot], 95)) if hot.any() else 1.0
    vmax = max(1.0, vmax)
    ax_brain.scatter(
        xy[hot, 0], xy[hot, 1], s=3, c=counts[hot], cmap="inferno", vmin=0, vmax=vmax, linewidths=0
    )
    for g in groups.values():
        gg = g[ok[g]]
        ax_brain.scatter(
            xy[gg, 0],
            xy[gg, 1],
            s=40,
            facecolors="none",
            edgecolors="cyan" if counts[gg].sum() == 0 else "red",
            linewidths=1,
        )
    ax_brain.set_aspect("equal")
    ax_brain.invert_yaxis()   # soma y increases ventrally, so flip it to put dorsal up
    ax_brain.set_axis_off()
    ax_brain.set_title(f"{int(hot.sum())} neurons spiking this window")
    ax_sig.barh(range(6), z, color=["red" if i == np.argmax(z) and z[i] >= z_floor else "grey" for i in range(6)])
    ax_sig.axvline(z_floor, color="k", ls="--")
    ax_sig.set_yticks(range(6))
    ax_sig.set_yticklabels(SIGNALS)
    ax_sig.set_xlim(-3, 5)
    ax_sig.set_title(f"action: {ACTIONS[int(action)]}")
    names = ["health", "food", "drink", "energy"]
    ax_met.bar(names, [meters[k] for k in names], color=["crimson", "orange", "dodgerblue", "gold"])
    ax_met.set_ylim(0, 9)
    ax_met.set_title("meters")
    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img
