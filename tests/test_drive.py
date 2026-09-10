import numpy as np
import jax.numpy as jnp
import pytest

from flycraftax.brain import BrainParams


class _State:
    def __init__(self, food, drink, energy):
        self.player_food = jnp.asarray(food)
        self.player_drink = jnp.asarray(drink)
        self.player_energy = jnp.asarray(energy)
        self.player_direction = jnp.full_like(self.player_food, 3)


def _toy_drive():
    from flycraftax.drive import Drive
    from flycraftax.retina import Retina
    retina = Retina(idx=np.array([10, 11], np.int32), side=np.array([0, 1], np.int8),
                    u=np.zeros(2, np.float32), v=np.ones(2, np.float32), channel=np.zeros(2, np.int8))
    groups = (np.array([20], np.int32), np.array([30, 31], np.int32), np.array([40], np.int32))
    idx = np.concatenate([retina.idx, *groups])
    return Drive(retina=retina, idx=idx, n_retina=2, groups=groups, max_hz=100.0,
                 bias=np.zeros(50, np.float32), lamina_l1=np.array([12], np.int32))


def test_rates_layout_and_deficits():
    from flycraftax.drive import drive_rates
    d = _toy_drive()
    obs = jnp.ones((2, 63, 63, 3))
    st = _State([9, 0], [9, 3], [0, 9])
    r = np.asarray(drive_rates(d, obs, st))
    assert r.shape == (2, 6)
    assert np.allclose(r[:, :2], 1.0)                       # white frame saturates the retina
    assert np.allclose(r[0, 2:], [0.0, 0.0, 0.0, 1.0])      # env 0: full food, full drink, empty energy
    assert np.allclose(r[1, 2:], [1.0, 6 / 9, 6 / 9, 0.0])  # env 1: hungry, two hygrosensory cells share the thirst rate


def test_kick_prob_scatters_and_scales():
    from flycraftax.drive import kick_prob
    d = _toy_drive()
    p = BrainParams()
    rates = jnp.array([[1.0, 0.5, 0.0, 1.0, 1.0, 0.0]])
    kp = np.asarray(kick_prob(d, rates, 50, p))
    assert kp.shape == (1, 50)
    assert kp[0, 10] == pytest.approx(100 * p.dt_ms / 1000)
    assert kp[0, 11] == pytest.approx(50 * p.dt_ms / 1000)
    assert kp[0, 20] == 0.0 and kp[0, 30] > 0 and kp[0, 40] == 0.0
    assert kp.sum() == pytest.approx((1.0 + 0.5 + 1.0 + 1.0) * 100 * p.dt_ms / 1000)


@pytest.mark.slow
def test_build_drive_real_sizes(conn):
    from flycraftax.drive import build_drive
    from flycraftax.retina import build_retina
    d = build_drive(conn, build_retina(conn))
    assert [len(g) for g in d.groups] == [2, 66, 21]
    assert len(d.idx) == d.n_retina + 89
    assert len(np.unique(d.idx)) == len(d.idx)


def test_bias_field_on_toy():
    d = _toy_drive()
    assert d.bias.shape == (50,) and d.bias.sum() == 0.0


@pytest.mark.slow
def test_build_drive_lamina_bias(conn):
    from flycraftax.drive import LAMINA, build_drive
    from flycraftax.retina import build_retina
    d = build_drive(conn, build_retina(conn), lamina_mv=0.06)
    lam = conn.index(types=list(LAMINA))
    assert len(lam) > 8000
    assert np.allclose(d.bias[lam], 0.06) and float(np.abs(d.bias).sum()) == pytest.approx(0.06 * len(lam), rel=1e-5)
    assert len(np.intersect1d(lam, d.idx)) == 0    # lamina cells are biased, never kicked
