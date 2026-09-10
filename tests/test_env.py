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


def _base_env():
    from craftax.craftax_env import make_craftax_env_from_name

    return make_craftax_env_from_name("Craftax-Classic-Pixels-v1", auto_reset=False)


@pytest.fixture(scope="module")
def ego():
    from flycraftax.env import EgocentricWrapper

    env = EgocentricWrapper(_base_env())
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key, env.default_params)
    return env, key, state


@pytest.mark.slow
def test_turn_changes_facing_not_position(ego):
    from flycraftax.env import TURN_LEFT, TURN_RIGHT

    env, key, state = ego
    pos0 = state.player_position
    obs, s1, _, _, _ = env.step(key, state, TURN_LEFT, env.default_params)
    assert int(s1.player_direction) == 1  # UP -> LEFT
    assert (s1.player_position == pos0).all()
    assert int(s1.timestep) == int(state.timestep) + 1
    obs, s2, _, _, _ = env.step(key, s1, TURN_RIGHT, env.default_params)
    assert int(s2.player_direction) == 3  # LEFT -> UP


@pytest.mark.slow
def test_forward_moves_along_facing_and_backward_keeps_facing(ego):
    from flycraftax.env import BACKWARD, FORWARD, TURN_LEFT

    env, key, state = ego
    obs, s, _, _, _ = env.step(key, state, TURN_LEFT, env.default_params)  # face LEFT
    obs, s1, _, _, _ = env.step(key, s, FORWARD, env.default_params)
    moved = s1.player_position - s.player_position
    assert int(s1.player_direction) == 1
    assert int(moved[0]) == 0 and int(moved[1]) in (0, -1)  # -1 unless blocked
    obs, s2, _, _, _ = env.step(key, s1, BACKWARD, env.default_params)
    moved = s2.player_position - s1.player_position
    assert int(s2.player_direction) == 1  # facing restored
    assert int(moved[0]) == 0 and int(moved[1]) in (0, 1)


@pytest.mark.slow
def test_obs_rerendered_after_turn(ego):
    from flycraftax.env import TURN_LEFT

    env, key, state = ego
    obs, s1, _, _, _ = env.step(key, state, TURN_LEFT, env.default_params)
    # get_obs outside the jitted step rounds 1 ULP differently; a render of the
    # pre-override facing would differ by ~0.8.
    assert jnp.allclose(obs, env.get_obs(s1), atol=1e-6)


@pytest.mark.slow
def test_turn_ignored_while_sleeping(ego):
    from flycraftax.env import TURN_LEFT

    env, key, state = ego
    asleep = state.replace(is_sleeping=True, player_energy=3)
    obs, s1, _, _, _ = env.step(key, asleep, TURN_LEFT, env.default_params)
    assert int(s1.player_direction) == int(asleep.player_direction)


def test_rotation_tables():
    from flycraftax.env import LEFT_OF, OPPOSITE, RIGHT_OF

    for f in (1, 2, 3, 4):
        assert int(RIGHT_OF[LEFT_OF[f]]) == f
        assert int(OPPOSITE[OPPOSITE[f]]) == f
        assert int(LEFT_OF[LEFT_OF[f]]) == int(OPPOSITE[f])
