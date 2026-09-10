"""Overlay the retina sampling points on a real frame and show what each eye sees."""

from pathlib import Path

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from flycraftax.data import load_connectome
from flycraftax.env import EgocentricWrapper, FORWARD, TURN_LEFT
from flycraftax.retina import build_retina, sample, sample_points


def main(out=Path("outputs/retina.png")):
    from craftax.craftax_env import make_craftax_env_from_name

    env = EgocentricWrapper(make_craftax_env_from_name("Craftax-Classic-Pixels-v1", auto_reset=False))
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key, env.default_params)
    for a in (FORWARD, FORWARD, TURN_LEFT, FORWARD):
        obs, state, *_ = env.step(key, state, a, env.default_params)
    retina = build_retina(load_connectome())
    facing = state.player_direction[None]
    pts = np.asarray(sample_points(retina, facing))[0]
    val = np.asarray(sample(retina, obs[None], facing))[0]

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(np.asarray(obs))
    ax[0].set_title(f"frame, facing={int(state.player_direction)}; dots R1-R6, plus R8")
    lum = retina.channel == 0
    colour = np.where(retina.side == 0, "cyan", "orange")
    ax[0].scatter(pts[lum, 1], pts[lum, 0], s=3, c=colour[lum], alpha=0.6)
    ax[0].scatter(pts[~lum, 1], pts[~lum, 0], s=14, marker="+", c=colour[~lum], alpha=0.6, linewidths=0.5)
    for s, name in ((0, "left eye"), (1, "right eye")):
        m = lum & (retina.side == s)
        ax[1 + s].scatter(retina.u[m], retina.v[m], c=val[m], s=12, cmap="gray", vmin=0, vmax=1)
        ax[1 + s].set_title(f"{name}: R1-R6 luminance")
        ax[1 + s].set_xlabel("u (front to back)")
        ax[1 + s].set_ylabel("v (near to far)")
        ax[1 + s].set_xlim(-0.05, 1.05)
        ax[1 + s].set_ylim(-0.05, 1.05)
    out.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(out)


if __name__ == "__main__":
    main()
