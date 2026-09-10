"""Fixed DN-to-action readout: six signals, z-scored against a random-policy baseline."""

import json
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from flycraftax.brain import STEPS_PER_ACTION, BrainParams
from flycraftax.env import BACKWARD, DO, FORWARD, NOOP, SLEEP, TURN_LEFT, TURN_RIGHT

SIGNALS = ("forward", "backward", "turn_left", "turn_right", "do", "sleep")
SIGNAL_ACTION = np.array([FORWARD, BACKWARD, TURN_LEFT, TURN_RIGHT, DO, SLEEP], np.int32)
GROUP_NAMES = ("forward", "backward", "turn_l", "turn_r", "do", "sleep")
NORM_PATH = Path(__file__).with_name("readout_norm.json")
SPIKE_HZ = 1000.0 / (STEPS_PER_ACTION * BrainParams().dt_ms)  # 50 Hz: one spike in one window


def readout_groups(conn):
    def turn(side):
        return np.concatenate(
            [conn.index(types=["DNa01"], side=side), conn.index(types=["DNa02"], side=side)]
        )

    return {
        "forward": conn.index(types=["DNp09"]),
        "backward": conn.index(types=["MDN"]),
        "turn_l": turn("L"),
        "turn_r": turn("R"),
        "do": conn.index(types=["MN9"]),
        "sleep": np.concatenate([conn.index(type_prefix="FB6"), conn.index(type_prefix="FB7")]),
    }


@dataclass
class Readout:
    """Unhashable: close over it, never pass it as a static jit argument."""

    groups: dict
    mean: np.ndarray
    std: np.ndarray
    z_floor: float = 1.0


def group_rates(groups, rates):
    return {k: rates[:, g].mean(axis=1) for k, g in groups.items()}


def signals(groups, rates):
    r = group_rates(groups, rates)
    turn = r["turn_l"] - r["turn_r"]
    return jnp.stack([r["forward"], r["backward"], turn, -turn, r["do"], r["sleep"]], axis=1)


def act(readout, rates):
    z = (signals(readout.groups, rates) - readout.mean) / readout.std
    best = jnp.argmax(z, axis=1)
    action = jnp.where(z.max(axis=1) >= readout.z_floor, jnp.asarray(SIGNAL_ACTION)[best], NOOP)
    return action, z


def std_floor(groups):
    n = {k: len(v) for k, v in groups.items()}
    turn = min(n["turn_l"], n["turn_r"])
    return SPIKE_HZ / np.array(
        [n["forward"], n["backward"], turn, turn, n["do"], n["sleep"]], np.float32
    )


def save_readout(path, max_hz, w_syn, lamina_mv, mean, std, z_floor=1.0):
    Path(path).write_text(
        json.dumps(
            {
                "max_hz": float(max_hz),
                "w_syn": float(w_syn),
                "lamina_mv": float(lamina_mv),
                "z_floor": float(z_floor),
                "mean": [float(x) for x in mean],
                "std": [float(x) for x in std],
            },
            indent=1,
        )
        + "\n"
    )


def load_readout(conn, path=NORM_PATH):
    cfg = json.loads(Path(path).read_text())
    groups = readout_groups(conn)
    std = np.maximum(np.array(cfg["std"], np.float32), std_floor(groups))
    mean = np.array(cfg["mean"], np.float32)
    # turn signals are antisymmetric; their baseline is zero by construction, and subtracting a
    # measured mean on the spike lattice inverts the bias
    mean[2] = mean[3] = 0.0
    return Readout(groups, mean, std, cfg["z_floor"]), cfg
