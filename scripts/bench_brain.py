"""Steps per second of the full MaleCNS brain in JAX, for several batch sizes."""
import time

import jax
import jax.numpy as jnp

from flycraftax.brain import (
    BrainParams, build_weights, init_state, rate_for_hz, reset_counts, rfc_steps, run_window,
)
from flycraftax.data import load_connectome


def main():
    p = BrainParams()
    conn = load_connectome()
    W = build_weights(conn.n, conn.pre, conn.post, conn.signed_count(), p)
    driven = conn.index(type_prefix="LB")
    rfc = rfc_steps(conn.n, driven, p)
    silence = jnp.ones(conn.n)
    n_steps = 500  # 50 ms window
    print(f"N={conn.n} E={len(conn.pre)} driven={len(driven)} device={jax.devices()[0]}")
    for batch in (1, 8, 32):
        state = init_state(conn.n, batch, p)
        kick_prob = jnp.zeros((batch, conn.n)).at[:, driven].set(rate_for_hz(100.0, p))
        key = jax.random.PRNGKey(0)
        state = run_window(W, rfc, p, state, kick_prob, silence, key, n_steps)  # compile
        state.v.block_until_ready()
        state = reset_counts(state)  # so "active neurons/env" describes the timed windows only
        t0 = time.perf_counter()
        for _ in range(5):
            state = run_window(W, rfc, p, state, kick_prob, silence, key, n_steps)
        state.v.block_until_ready()
        dt = (time.perf_counter() - t0) / 5
        active = int((state.counts > 0).sum(axis=1).mean())
        print(f"batch={batch:3d}  {n_steps/dt:8.0f} steps/s  {n_steps*p.dt_ms/1000/dt:6.3f}x realtime per env  "
              f"{batch*n_steps/dt:8.0f} env-steps/s  active neurons/env={active}")
    mem = jax.devices()[0].memory_stats()  # None on backends that do not report it
    if isinstance(mem, dict):
        print(f"GPU bytes_in_use={mem['bytes_in_use']/2**30:.2f} GiB  "
              f"peak_bytes_in_use={mem['peak_bytes_in_use']/2**30:.2f} GiB")


if __name__ == "__main__":
    main()
