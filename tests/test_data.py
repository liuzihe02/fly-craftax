import numpy as np
import pytest

from flycraftax.data import Connectome


@pytest.mark.slow
def test_population_and_edges(conn):
    assert conn.n == 146_271
    assert conn.pre.dtype == np.int32 and conn.post.dtype == np.int32
    assert conn.count.min() >= 5
    assert 4_500_000 < len(conn.pre) < 6_000_000
    assert np.all(conn.sign[conn.pre] != 0)


@pytest.mark.slow
def test_signs(conn):
    frac_inhib = np.mean(conn.sign == -1)
    assert 0.30 < frac_inhib < 0.40
    assert np.all(conn.sign[conn.index(type_prefix="R1-R6")] == -1)
    assert np.all(conn.sign[conn.index(types=["DNp09"])] == 1)


@pytest.mark.slow
def test_io_cell_types_exist(conn):
    assert len(conn.index(types=["DNp09"])) == 2
    assert len(conn.index(types=["DNa01"])) == 2
    assert len(conn.index(types=["DNa02"])) == 2
    assert len(conn.index(types=["MDN"])) == 4
    assert len(conn.index(types=["MN9"])) == 2
    assert len(conn.index(types=["MN9"], side="L")) == 1
    assert len(conn.index(types=["NPFL1-I"])) == 2
    assert len(conn.index(types=["ER5"])) == 21
    assert len(conn.index(cls="hygrosensory")) == 66
    assert len(conn.index(type_prefix="R1-R6")) == 3377
    assert len(conn.index(type_prefix="R8")) > 1000
    assert len(conn.index(type_prefix="LB")) == 165
    assert len(conn.index(type_prefix="FB6")) + len(conn.index(type_prefix="FB7")) == 140
    assert np.sum(conn.superclass == "descending_neuron") == 1314


def test_index_on_toy():
    c = Connectome(
        body_id=np.array([1, 2, 3]),
        type=np.array(["A", "A", "B"], dtype=object),
        cls=np.array(["", "", "x"], dtype=object),
        superclass=np.array(["s", "s", "s"], dtype=object),
        side=np.array(["L", "R", "L"], dtype=object),
        sign=np.array([1, -1, 1], dtype=np.float32),
        pre=np.array([0, 1], dtype=np.int32),
        post=np.array([2, 2], dtype=np.int32),
        count=np.array([3, 5], dtype=np.int32),
    )
    assert c.index(types=["A"]).tolist() == [0, 1]
    assert c.index(types=["A"], side="R").tolist() == [1]
    assert c.index(cls="x").tolist() == [2]
    assert c.signed_count().tolist() == [3.0, -5.0]
