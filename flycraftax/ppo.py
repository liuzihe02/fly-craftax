"""PPO for a linear readout over descending-neuron rates; the brain stays frozen."""
import jax.numpy as jnp


def init_params(n_dn, n_actions=7):
    """Zeros, so the initial policy is uniform and the initial value is zero."""
    return dict(W=jnp.zeros((n_dn, n_actions)), b=jnp.zeros(n_actions), vw=jnp.zeros(n_dn), vb=jnp.zeros(()))


def policy(params, feats):
    return feats @ params["W"] + params["b"], feats @ params["vw"] + params["vb"]
