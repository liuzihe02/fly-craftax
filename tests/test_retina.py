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
    assert (r.channel == 0).sum() > 0.95 * n_wired
    assert (r.channel == 1).sum() > 250 and (r.channel == 2).sum() > 350
    for side in (0, 1):
        m = r.side == side
        assert m.sum() > 800
        assert r.u[m].min() == 0.0 and r.u[m].max() == 1.0
        assert r.v[m].min() == 0.0 and r.v[m].max() == 1.0
    assert len(np.unique(np.round(np.stack([r.u, r.v, r.side], 1), 4), axis=0)) > 1000  # many distinct columns
