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
    assert env.num_actions == 7  # not the 17 grid actions of the inner env
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


@pytest.mark.slow
def test_reward_counts_survival_achievements_once():
    from flycraftax.env import EgocentricWrapper, NOOP, SurvivalRewardWrapper

    env = SurvivalRewardWrapper(EgocentricWrapper(_base_env()))
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key, env.default_params)
    ach = state.env_state.achievements.at[0].set(True).at[1].set(True)
    # COLLECT_WOOD (counts), PLACE_TABLE (does not)
    poked = state.replace(env_state=state.env_state.replace(achievements=ach))
    obs, s1, r1, _, _ = env.step(key, poked, NOOP, env.default_params)
    assert float(r1) == pytest.approx(1.0, abs=0.11)  # +1 wood, health may move by 0.1
    obs, s2, r2, _, _ = env.step(key, s1, NOOP, env.default_params)
    assert abs(float(r2)) <= 0.11  # no repeat


@pytest.mark.slow
def test_reward_health_term():
    from flycraftax.env import EgocentricWrapper, NOOP, SurvivalRewardWrapper

    env = SurvivalRewardWrapper(EgocentricWrapper(_base_env()))
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key, env.default_params)
    hurt = state.replace(
        env_state=state.env_state.replace(player_health=5, player_recover=25.5)
    )
    obs, s1, r, _, _ = env.step(key, hurt, NOOP, env.default_params)
    assert float(r) == pytest.approx(0.1)


@pytest.mark.slow
def test_make_env_batched_step():
    from flycraftax.env import base_state, make_env

    env = make_env(num_envs=4, reset_ratio=4)  # reset_ratio must divide num_envs
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key, env.default_params)
    assert obs.shape == (4, 63, 63, 3)
    actions = jnp.zeros(4, jnp.int32)
    obs, state, reward, done, info = env.step(key, state, actions, env.default_params)
    assert reward.shape == (4,)
    assert base_state(state).player_food.shape == (4,)
