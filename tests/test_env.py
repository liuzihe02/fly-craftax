import jax
import jax.numpy as jnp
import pytest


@pytest.mark.slow
def test_base_env_one_step():
    from craftax.craftax_env import make_craftax_env_from_name

    env = make_craftax_env_from_name("Craftax-Classic-Pixels-v1", auto_reset=False)
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key, env.default_params)
    assert obs.shape == (63, 63, 3) and obs.dtype == jnp.float32
    assert float(obs.max()) <= 1.0
    assert int(state.player_direction) == 3  # Action.UP at spawn
    obs, state, reward, done, info = env.step(key, state, 0, env.default_params)
    assert int(state.timestep) == 1
