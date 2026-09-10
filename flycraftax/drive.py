"""Per-step Poisson drive: retina values plus hunger, thirst, fatigue on their cell types."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from flycraftax.brain import BrainParams, rate_for_hz
from flycraftax.data import Connectome
from flycraftax.retina import Retina, sample


@dataclass
class Drive:
    """Unhashable: close over it when tracing, never pass it as a static jit argument."""

    retina: Retina
    idx: np.ndarray  # int32 (K,) retina indices, then NPFL1-I, hygrosensory, ER5
    n_retina: int
    groups: tuple  # (npf_idx, hyg_idx, er5_idx) int32 arrays, driven by food, drink, energy
    max_hz: float


def build_drive(conn: Connectome, retina: Retina, max_hz: float = 100.0) -> Drive:
    groups = (conn.index(types=["NPFL1-I"]), conn.index(cls="hygrosensory"), conn.index(types=["ER5"]))
    idx = np.concatenate([retina.idx, *groups]).astype(np.int32)
    return Drive(retina=retina, idx=idx, n_retina=len(retina.idx), groups=groups, max_hz=max_hz)


def drive_rates(drive: Drive, obs, env_state) -> jax.Array:
    """(B, K) rates in [0, 1], laid out like drive.idx."""
    vis = sample(drive.retina, obs, env_state.player_direction)
    levels = (env_state.player_food, env_state.player_drink, env_state.player_energy)
    deficits = [(9.0 - lv.astype(jnp.float32)) / 9.0 for lv in levels]
    internal = [jnp.repeat(d[:, None], len(g), axis=1) for d, g in zip(deficits, drive.groups)]
    return jnp.concatenate([vis, *internal], axis=1)


def kick_prob(drive: Drive, rates, n: int, p: BrainParams) -> jax.Array:
    """(B, n) per-step Bernoulli probabilities, zero everywhere but the driven cells."""
    prob = rates * rate_for_hz(drive.max_hz, p)
    return jnp.zeros((rates.shape[0], n), jnp.float32).at[:, drive.idx].set(prob)
