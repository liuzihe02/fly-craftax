import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.slow
def test_build_retina_covers_both_eyes(conn):
    from flycraftax.retina import build_retina

    r = build_retina(conn)
    assert r.idx.dtype == np.int32 and r.side.dtype == np.int8 and r.channel.dtype == np.int8
    assert r.u.dtype == np.float32 and r.v.dtype == np.float32
    # 1,983 of the 3,377 R1-R6 cells are "Out of scope", so edges-traced.feather holds no
    # synapse for them: they can be neither placed nor driven. Score against the wired ones.
    r16 = conn.index(type_prefix="R1-R6")
    n_wired = len(np.unique(conn.pre[np.isin(conn.pre, r16)]))
    assert n_wired > 1300
    assert (r.channel == 0).sum() > 0.95 * n_wired
    assert (r.channel == 1).sum() > 250 and (r.channel == 2).sum() > 350
    for side in (0, 1):
        m = r.side == side
        assert m.sum() > 800
        assert r.u[m].min() == 0.0 and r.u[m].max() == 1.0
        assert r.v[m].min() == 0.0 and r.v[m].max() == 1.0
    assert len(np.unique(np.round(np.stack([r.u, r.v, r.side], 1), 4), axis=0)) > 1000  # many distinct columns


def _toy_retina():
    from flycraftax.retina import Retina

    # 4 cells: left front, left back, right front, right back; all luminance channel
    return Retina(
        idx=np.arange(4, dtype=np.int32),
        side=np.array([0, 0, 1, 1], np.int8),
        u=np.array([0.0, 1.0, 0.0, 1.0], np.float32),
        v=np.array([1.0, 1.0, 1.0, 1.0], np.float32),
        channel=np.zeros(4, np.int8),
    )


def _frame(bright_rows, bright_cols):
    f = np.zeros((1, 63, 63, 3), np.float32)
    f[:, bright_rows, bright_cols, :] = 1.0
    return jnp.asarray(f)


def test_points_rotate_with_facing():
    from flycraftax.retina import CENTER, sample_points

    r = _toy_retina()
    up = np.asarray(sample_points(r, jnp.array([3])))[0]  # facing UP
    left = np.asarray(sample_points(r, jnp.array([1])))[0]  # facing LEFT
    # front cells (u=0) lie ahead: above centre when facing UP, left of centre when facing LEFT
    assert up[0, 0] < CENTER[0] and abs(up[0, 1] - CENTER[1]) < 1e-3
    assert left[0, 1] < CENTER[1] and abs(left[0, 0] - CENTER[0]) < 1e-3
    # a left-eye back cell (u=1) lies behind
    assert up[1, 0] > CENTER[0]


def test_bright_patch_ahead_lights_front_cells_only():
    from flycraftax.retina import sample

    r = _toy_retina()
    obs = _frame(slice(0, 10), slice(0, 63))  # bright band at the top of the map view
    val = np.asarray(sample(r, obs, jnp.array([3])))[0]  # facing UP: front cells look up
    assert val[0] > 0.9 and val[2] > 0.9
    assert val[1] < 0.1 and val[3] < 0.1


def test_inventory_bar_is_never_sampled():
    from flycraftax.retina import sample

    r = _toy_retina()
    obs = _frame(slice(49, 63), slice(0, 63))  # bright inventory bar only
    for facing in (1, 2, 3, 4):
        assert float(np.asarray(sample(r, obs, jnp.array([facing]))).max()) < 0.1
        # r_max=30 pushes the cells past row 48, so the crop is load-bearing: those points
        # sit outside the cropped plane and map_coordinates fills them with cval=0.5 --
        # "outside the view" is neutral grey, not dark, so a dark reading here would be
        # indistinguishable from genuinely dark terrain. Without the crop they would read
        # the bright bar as 1.0.
        far = np.asarray(sample(r, obs, jnp.array([facing]), r_max=30.0))[0]
        assert far.max() <= 0.5 + 1e-6
        if facing == 4:  # DOWN: the front cells point straight at the inventory bar
            assert np.isclose(far, 0.5, atol=1e-6).any()


def test_channels_pick_colour():
    from flycraftax.retina import Retina, sample

    r = Retina(
        idx=np.arange(3, dtype=np.int32),
        side=np.zeros(3, np.int8),
        u=np.zeros(3, np.float32),
        v=np.ones(3, np.float32),
        channel=np.array([0, 1, 2], np.int8),
    )
    f = np.zeros((1, 63, 63, 3), np.float32)
    f[..., 2] = 1.0  # pure blue frame
    val = np.asarray(sample(r, jnp.asarray(f), jnp.array([3])))[0]
    assert val[1] > 0.9 and val[2] < 0.1 and 0.05 < val[0] < 0.1  # blue cell on, green off, luminance 0.0722
