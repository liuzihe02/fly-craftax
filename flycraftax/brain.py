"""Shiu et al. 2024 LIF on a sparse connectome, in JAX."""
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import sparse

STEPS_PER_ACTION = 200  # 20 ms of brain time per env action, fixed after the M1 benchmark


class BrainParams(NamedTuple):
    dt_ms: float = 0.1
    v_rest: float = -52.0
    v_reset: float = -52.0
    v_th: float = -45.0
    t_mbr: float = 20.0
    tau: float = 5.0
    t_rfc: float = 2.2
    t_dly: float = 1.8
    w_syn: float = 0.275
    kick: float = 0.275 * 250

    @property
    def n_dly(self) -> int:
        return round(self.t_dly / self.dt_ms)

    @property
    def n_rfc(self) -> int:
        return round(self.t_rfc / self.dt_ms)

    def decay(self) -> tuple[float, float, float]:
        a = math.exp(-self.dt_ms / self.tau)
        b = math.exp(-self.dt_ms / self.t_mbr)
        c = (b - a) * self.tau / (self.t_mbr - self.tau)
        return a, b, c


class BrainState(NamedTuple):
    v: jax.Array        # (B, N) membrane potential, mV
    g: jax.Array        # (B, N) synaptic drive, mV
    refrac: jax.Array   # (B, N) int32, steps of refractoriness remaining
    buf: jax.Array      # (n_dly, B, N) float32 ring of spike vectors
    counts: jax.Array   # (B, N) float32, spikes since the last reset_counts
    t: jax.Array        # int32 scalar, steps run; slot t % n_dly is the oldest, then reused


def build_weights(n, pre, post, signed_count, p):
    # JAX drops out-of-range scatter indices silently, so an off-by-one in the loader
    # would build a wrong graph instead of failing: check on numpy, before the tracer.
    assert len(pre) == 0 or (pre.min() >= 0 and pre.max() < n), "pre index out of range"
    assert len(post) == 0 or (post.min() >= 0 and post.max() < n), "post index out of range"
    idx = jnp.stack([jnp.asarray(post, jnp.int32), jnp.asarray(pre, jnp.int32)], axis=1)
    vals = jnp.asarray(signed_count, jnp.float32) * p.w_syn
    return sparse.BCOO((vals, idx), shape=(n, n)).sort_indices()


def rfc_steps(n, driven_idx, p):
    # Out-of-range indices are dropped silently by .at[].set, which would leave the
    # "driven" neurons refractory and the whole brain silent.
    assert len(driven_idx) == 0 or (
        driven_idx.min() >= 0 and driven_idx.max() < n
    ), "driven_idx out of range"
    return jnp.full(n, p.n_rfc, jnp.int32).at[jnp.asarray(driven_idx, jnp.int32)].set(0)


def init_state(n, batch, p):
    z = jnp.zeros((batch, n), jnp.float32)
    return BrainState(
        v=jnp.full((batch, n), p.v_rest, jnp.float32),
        g=z, refrac=jnp.zeros((batch, n), jnp.int32),
        buf=jnp.zeros((p.n_dly, batch, n), jnp.float32), counts=z, t=jnp.int32(0),
    )


def rate_for_hz(hz, p):
    return hz * p.dt_ms / 1000.0


def step(W, rfc, p, state, kick_prob, silence, key, bias=None):
    """One dt of the Shiu LIF, in Brian2's slot order.

    Measured Brian2 semantics (probe, 2026-09-09), reproduced here:
      * `v` and `g` carry `(unless refractory)`, so Brian2 skips *every* write to
        them while a neuron is refractory -- not just the ODE. A synaptic
        `g += w` (and a Poisson `v += kick`) landing on a refractory target is
        dropped outright: it does not accumulate and it does not fire the neuron
        when refractoriness ends. StateMonitor: a 6000-synapse input delivered at
        step 39 to a neuron that spiked at step 30 left g at exactly 0.0.
      * A neuron that spikes at step t is frozen for t+1..t+21 and accepts input
        again at t+22, so the counter is decremented *before* `active` is read.
        Scanned arrival step X: X<=51 gives no second spike, X=52 fires at 53.
        Kicked every step with rfc=2.2 ms, Brian2 spikes at 1, 24, 47, ... (ISI 23).
      * Reset runs after the synapse slot, so a kick landing on a spiking step is
        wiped by `v = v_reset`.
    """
    a, b, c = p.decay()
    refrac = jnp.maximum(state.refrac - 1, 0)
    active = refrac == 0
    v = jnp.where(active, p.v_rest + b * (state.v - p.v_rest) + c * state.g, state.v)
    g = jnp.where(active, a * state.g, state.g)
    if bias is not None:                                  # tonic input, dropped while refractory
        v = jnp.where(active, v + bias, v)
    spikes = active & (v > p.v_th)                        # Brian2: (v > v_th) and not_refractory
    slot = state.t % p.n_dly                              # ring: read the oldest slot, then reuse it
    delayed = jax.lax.dynamic_index_in_dim(state.buf, slot, 0, False) * silence   # (B, N)
    syn_in = (W @ delayed.T).T                            # (N, N) @ (N, B) -> (B, N)
    kicked = p.kick * jax.random.bernoulli(key, kick_prob)
    g = g + jnp.where(active, syn_in, 0.0)                # dropped while refractory
    v = v + jnp.where(active, kicked, 0.0)                # dropped while refractory
    v = jnp.where(spikes, p.v_reset, v)
    g = jnp.where(spikes, 0.0, g)
    refrac = jnp.where(spikes, rfc[None, :], refrac)
    sp = spikes.astype(jnp.float32)
    buf = jax.lax.dynamic_update_index_in_dim(state.buf, sp, slot, 0)
    return BrainState(v, g, refrac, buf, state.counts + sp, state.t + 1), spikes


def run_window(W, rfc, p, state, kick_prob, silence, key, n_steps, bias=None):
    def body(carry, k):
        st, _ = step(W, rfc, p, carry, kick_prob, silence, k, bias)
        return st, None
    keys = jax.random.split(key, n_steps)
    state, _ = jax.lax.scan(body, state, keys)
    return state


def rate_hz(counts, n_steps, p):
    return counts / (n_steps * p.dt_ms / 1000.0)


def reset_counts(state):
    return state._replace(counts=jnp.zeros_like(state.counts))


def reset_envs(state, done, p):
    """Reset the brain of every env where done is True; other envs and the step counter are untouched."""
    d = done[:, None]                       # (B, 1) against the (B, N) fields
    return state._replace(
        v=jnp.where(d, p.v_rest, state.v), g=jnp.where(d, 0.0, state.g),
        refrac=jnp.where(d, 0, state.refrac), buf=jnp.where(d[None], 0.0, state.buf),
        counts=jnp.where(d, 0.0, state.counts),
    )


step = jax.jit(step, static_argnames=("p",))
run_window = jax.jit(run_window, static_argnames=("p", "n_steps"))
