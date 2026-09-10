"""MN9 firing rate versus labellar GRN drive rate on the full brain."""
import jax
import jax.numpy as jnp
import numpy as np

from flycraftax.brain import BrainParams, build_weights, init_state, rate_for_hz, rate_hz, rfc_steps, run_window
from flycraftax.data import load_connectome


def main(threshold=5, hz_list=(25, 50, 100, 150, 200), t_ms=500, seeds=3):
    p = BrainParams()
    conn = load_connectome(threshold)
    W = build_weights(conn.n, conn.pre, conn.post, conn.signed_count(), p)
    lb = conn.index(type_prefix="LB")
    mn9_l, mn9_r = conn.index(types=["MN9"], side="L"), conn.index(types=["MN9"], side="R")
    rfc = rfc_steps(conn.n, lb, p)
    n_steps = round(t_ms / p.dt_ms)
    keys = jax.random.split(jax.random.PRNGKey(0), len(hz_list))  # one key per drive rate
    print(f"threshold={threshold} N={conn.n} E={len(conn.pre)} LB={len(lb)}")
    for hz, key in zip(hz_list, keys):
        kick = jnp.zeros((seeds, conn.n)).at[:, lb].set(rate_for_hz(hz, p))
        st = run_window(W, rfc, p, init_state(conn.n, seeds, p), kick, jnp.ones(conn.n), key, n_steps)
        r = np.asarray(rate_hz(st.counts, n_steps, p))
        print(f"drive {hz:4d} Hz  MN9_L {r[:, mn9_l].mean():6.1f} Hz  MN9_R {r[:, mn9_r].mean():6.1f} Hz  "
              f"active {int((r > 0).sum(axis=1).mean())}")


if __name__ == "__main__":
    import sys
    main(threshold=int(sys.argv[1]) if len(sys.argv) > 1 else 5,
         seeds=int(sys.argv[2]) if len(sys.argv) > 2 else 3)
