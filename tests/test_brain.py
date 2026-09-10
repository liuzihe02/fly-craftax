import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flycraftax.brain import (
    BrainParams, build_weights, init_state, rate_for_hz, rate_hz, rfc_steps, run_window, step,
)
from flycraftax.oracle import run_brian2


def run_jax_with_kicks(n, pre, post, signed_count, kicks, p):
    """Drive the JAX kernel with a fixed kick schedule (prob 1 where kicks is True)."""
    W = build_weights(n, pre, post, signed_count, p)
    rfc = rfc_steps(n, np.flatnonzero(kicks.any(axis=0)), p)
    state = init_state(n, 1, p)
    silence = jnp.ones(n)
    out = np.zeros_like(kicks)
    key = jax.random.PRNGKey(0)
    for t in range(kicks.shape[0]):
        kick_prob = jnp.asarray(kicks[t][None].astype(np.float32))
        state, spikes = step(W, rfc, p, state, kick_prob, silence, key)
        out[t] = np.asarray(spikes[0])
    return out


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_oracle_on_random_toy(seed):
    """20 neurons, random signed edges, random kicks: spike rasters identical."""
    rng = np.random.default_rng(seed)
    n, e, T = 20, 60, 400
    pre = rng.integers(0, n, e)
    post = rng.integers(0, n, e)
    # ~162 synapses fire a resting neuron from one spike, so small edges never propagate.
    signed_count = rng.integers(100, 600, e) * rng.choice([-1.0, 1.0], e, p=[0.3, 0.7])
    kicks = rng.random((T, n)) < 0.01
    kicks[:, 10:] = False  # only neurons 0-9 are driven
    p = BrainParams()
    ref = run_brian2(n, pre, post, signed_count * p.w_syn, kicks, T * p.dt_ms, p)
    got = run_jax_with_kicks(n, pre, post, signed_count, kicks, p)
    assert ref.sum() > 20, "toy network too quiet to be a useful test"
    assert ref[:, 10:].sum() > 5, "no propagation to undriven neurons"
    np.testing.assert_array_equal(got, ref)


def test_refractory_caps_rate():
    """A neuron kicked every step fires at most once per refractory period."""
    p = BrainParams()
    n, T = 1, 500
    W = build_weights(n, np.array([], int), np.array([], int), np.array([], np.float32), p)
    rfc = jnp.array([p.n_rfc], dtype=jnp.int32)  # not treated as driven, so refractory applies
    state = init_state(n, 1, p)
    key = jax.random.PRNGKey(0)
    total = 0
    for t in range(T):
        state, s = step(W, rfc, p, state, jnp.ones((1, n)), jnp.ones(n), key)
        total += int(s.sum())
    assert abs(total - T / (p.n_rfc + 1)) <= 1


def test_silence_mask_blocks_output():
    # 6000 synapses is the measured edge size that fires a resting neuron from one
    # spike (see tests/test_oracle.py); a small edge would make this test vacuous.
    p = BrainParams()
    kicks = np.zeros((100, 2), dtype=bool)
    kicks[10, 0] = True
    W = build_weights(2, np.array([0]), np.array([1]), np.array([6000.0], np.float32), p)
    rfc = rfc_steps(2, np.array([0]), p)
    key = jax.random.PRNGKey(0)

    def fired_1(silence):
        state = init_state(2, 1, p)
        fired = 0
        for t in range(100):
            kick_prob = jnp.asarray(kicks[t][None].astype(np.float32))
            state, s = step(W, rfc, p, state, kick_prob, silence, key)
            fired += int(s[0, 1])
        return fired

    assert fired_1(jnp.ones(2)) == 1, "control: neuron 1 fires when nothing is silenced"
    assert fired_1(jnp.array([0.0, 1.0])) == 0


def test_run_window_counts_and_poisson_rate():
    """A driven neuron under Poisson kicks fires at roughly the drive rate, as in Shiu."""
    p = BrainParams()
    n, hz, n_steps = 1, 100.0, 10_000  # 1 s
    W = build_weights(n, np.array([], int), np.array([], int), np.array([], np.float32), p)
    rfc = rfc_steps(n, np.array([0]), p)
    state = init_state(n, 4, p)
    kick_prob = jnp.full((4, n), hz * p.dt_ms / 1000.0)
    state = run_window(W, rfc, p, state, kick_prob, jnp.ones(n), jax.random.PRNGKey(0), n_steps)
    r = np.asarray(rate_hz(state.counts, n_steps, p))
    assert r.shape == (4, n)
    assert 85 < r.mean() < 115


@pytest.mark.slow
def test_matches_oracle_on_malecns_subgraph():
    """~2000 real neurons downstream of the labellar GRNs, driven at 100 Hz for 200 ms."""
    from flycraftax.data import load_connectome, subgraph
    conn = load_connectome()
    seeds = conn.index(type_prefix="LB")
    nodes, pre, post, sc = subgraph(conn, seeds, hops=2, max_n=2000)
    p = BrainParams()
    T = 2000
    rng = np.random.default_rng(0)
    kicks = np.zeros((T, len(nodes)), dtype=bool)
    kicks[:, : len(seeds)] = rng.random((T, len(seeds))) < rate_for_hz(100.0, p)
    ref = run_brian2(len(nodes), pre, post, sc * p.w_syn, kicks, T * p.dt_ms, p)
    got = run_jax_with_kicks(len(nodes), pre, post, sc, kicks, p)
    assert ref.sum() > 1000
    # 2 of 4M raster cells differ: neuron 110 fires at t=1131 instead of 1130, after
    # 3542 identical spikes. At t=1130 the kernel's float32 v is -45.000004, exactly one
    # float32 ULP under v_th = -45.0, while Brian2's float64 lands just over. Spike totals
    # match exactly (6283 each) and no other neuron differs: accumulation precision.
    assert np.mean(got == ref) > 0.999
    assert np.array_equal(got.sum(0), ref.sum(0))  # a one-step shift moves no spike between neurons
    assert ref[:, len(seeds):].sum() > 100  # propagation to undriven neurons, not just seed echo


@pytest.mark.slow
def test_mn9_responds_to_labellar_drive():
    """Labellar GRNs at 100 Hz for 500 ms make MN9 fire; silent input makes it silent."""
    from flycraftax.data import load_connectome
    conn = load_connectome()
    p = BrainParams()
    W = build_weights(conn.n, conn.pre, conn.post, conn.signed_count(), p)
    lb, mn9 = conn.index(type_prefix="LB"), conn.index(types=["MN9"])
    rfc = rfc_steps(conn.n, lb, p)
    n_steps = 5000
    on = jnp.zeros((1, conn.n)).at[:, lb].set(rate_for_hz(100.0, p))
    off = jnp.zeros((1, conn.n))
    key = jax.random.PRNGKey(0)
    r_on = rate_hz(run_window(W, rfc, p, init_state(conn.n, 1, p), on, jnp.ones(conn.n), key, n_steps).counts, n_steps, p)
    r_off = rate_hz(run_window(W, rfc, p, init_state(conn.n, 1, p), off, jnp.ones(conn.n), key, n_steps).counts, n_steps, p)
    assert float(r_off.sum()) == 0.0
    assert float(r_on[0, mn9].max()) > 5.0
