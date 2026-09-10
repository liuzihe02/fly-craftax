import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flycraftax.env import BACKWARD, NOOP, SLEEP, TURN_RIGHT


def _groups():
    return {
        "forward": np.array([0, 1]), "backward": np.array([2]), "turn_l": np.array([3, 4]),
        "turn_r": np.array([5, 6]), "do": np.array([7]), "sleep": np.array([8, 9, 10]),
    }


def _rates(**hz):
    r = np.zeros((1, 11), np.float32)
    for k, v in hz.items():
        r[0, _groups()[k]] = v
    return jnp.asarray(r)


def test_signals_layout():
    from flycraftax.readout import signals
    s = np.asarray(signals(_groups(), _rates(forward=10, turn_l=4, turn_r=1, sleep=2)))[0]
    assert np.allclose(s, [10, 0, 3, -3, 0, 2])


def test_act_argmax_and_floor():
    from flycraftax.readout import Readout, act
    ro = Readout(_groups(), mean=np.zeros(6, np.float32), std=np.ones(6, np.float32), z_floor=1.0)
    a, z = act(ro, _rates(backward=5))
    assert int(a[0]) == BACKWARD
    a, _ = act(ro, _rates(turn_r=3, turn_l=1))
    assert int(a[0]) == TURN_RIGHT
    a, _ = act(ro, _rates(do=0.5))          # below the floor
    assert int(a[0]) == NOOP
    a, _ = act(ro, _rates(sleep=2, forward=1))
    assert int(a[0]) == SLEEP
    a, _ = jax.jit(lambda r: act(ro, r))(_rates(backward=5))
    assert int(a[0]) == BACKWARD


def test_std_floor_is_one_spike_per_window():
    from flycraftax.readout import std_floor
    assert np.allclose(std_floor(_groups()), [25.0, 50.0, 25.0, 25.0, 50.0, 50.0 / 3])


def test_save_load_roundtrip(tmp_path):
    from flycraftax.readout import load_readout, save_readout
    p = tmp_path / "norm.json"
    save_readout(p, max_hz=100.0, w_syn=0.44, lamina_mv=0.06, mean=np.arange(6.0), std=np.full(6, 0.1))

    class Conn:
        def index(self, **kw):
            return np.array([0], np.int32)
    ro, cfg = load_readout(Conn(), p)
    assert cfg["w_syn"] == 0.44 and cfg["lamina_mv"] == 0.06 and cfg["max_hz"] == 100.0
    assert np.allclose(ro.mean, [0, 1, 0, 0, 4, 5])
    assert np.all(ro.std >= 50.0 / 2)


@pytest.mark.slow
def test_groups_on_real_connectome(conn):
    from flycraftax.readout import readout_groups
    g = readout_groups(conn)
    assert [len(g[k]) for k in ("forward", "backward", "turn_l", "turn_r", "do", "sleep")] == [2, 4, 2, 2, 2, 140]
