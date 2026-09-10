"""Craftax-Classic with fly-frame actions and a survival reward."""

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from craftax.craftax_classic.constants import Achievement, Action
from craftax.craftax_env import make_craftax_env_from_name
from craftax.environment_base import spaces
from flax import struct

from flycraftax.wrappers import (
    GymnaxWrapper,
    LogWrapper,
    OptimisticResetVecEnvWrapper,
)

ACTIONS = ("noop", "forward", "backward", "turn_left", "turn_right", "do", "sleep")
NOOP, FORWARD, BACKWARD, TURN_LEFT, TURN_RIGHT, DO, SLEEP = range(7)
# Facing is an Action value 1..4 (LEFT, RIGHT, UP, DOWN); index 0 is unused.
LEFT_OF = jnp.array([0, 4, 3, 1, 2], jnp.int32)
RIGHT_OF = jnp.array([0, 3, 4, 2, 1], jnp.int32)
OPPOSITE = jnp.array([0, 2, 1, 4, 3], jnp.int32)


class EgocentricWrapper(GymnaxWrapper):
    """Seven fly-frame actions over the grid actions. A turn costs one NOOP step."""

    @property
    def num_actions(self):
        return len(ACTIONS)

    def action_space(self, params=None):
        return spaces.Discrete(len(ACTIONS))

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        facing = state.player_direction
        grid = jnp.array(
            [
                Action.NOOP.value,
                0,
                0,
                Action.NOOP.value,
                Action.NOOP.value,
                Action.DO.value,
                Action.SLEEP.value,
            ],
            jnp.int32,
        )
        grid = grid.at[FORWARD].set(facing).at[BACKWARD].set(OPPOSITE[facing])
        obs, new, reward, done, info = self._env.step(key, state, grid[action], params)
        wanted = jnp.select(
            [action == TURN_LEFT, action == TURN_RIGHT, action == BACKWARD],
            [LEFT_OF[facing], RIGHT_OF[facing], facing],
            new.player_direction,
        )
        # The engine ignores actions while asleep, so keep whatever it produced.
        new = new.replace(
            player_direction=jnp.where(state.is_sleeping, new.player_direction, wanted)
        )
        return self._env.get_obs(new), new, reward, done, info


def base_state(state):
    """Unwrap LogWrapper / reward wrapper states down to the Craftax EnvState."""
    while not hasattr(state, "player_food"):
        state = state.env_state
    return state


_SURVIVAL_ACHIEVEMENTS = (
    Achievement.COLLECT_WOOD,
    Achievement.EAT_COW,
    Achievement.COLLECT_SAPLING,
    Achievement.COLLECT_DRINK,
    Achievement.DEFEAT_ZOMBIE,
    Achievement.DEFEAT_SKELETON,
    Achievement.WAKE_UP,
)
SURVIVAL = (
    jnp.zeros(len(Achievement), jnp.float32)
    .at[jnp.array([a.value for a in _SURVIVAL_ACHIEVEMENTS])]
    .set(1.0)
)


@struct.dataclass
class RewardState:
    env_state: Any
    prev_achievements: jnp.ndarray


class SurvivalRewardWrapper(GymnaxWrapper):
    """+1 per new survival achievement, +0.1 per health point; others are worth 0."""

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        obs, s = self._env.reset(key, params)
        return obs, RewardState(s, s.achievements)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        health_before = state.env_state.player_health
        obs, s, _, done, info = self._env.step(key, state.env_state, action, params)
        newly = jnp.logical_and(s.achievements, jnp.logical_not(state.prev_achievements))
        reward = (newly.astype(jnp.float32) * SURVIVAL).sum() + 0.1 * (
            s.player_health - health_before
        )
        return obs, RewardState(s, s.achievements), reward, done, info


def make_env(num_envs, reset_ratio=16):
    """The full stack: reward inside the auto-reset, logging outside it."""
    env = make_craftax_env_from_name("Craftax-Classic-Pixels-v1", auto_reset=False)
    env = LogWrapper(SurvivalRewardWrapper(EgocentricWrapper(env)))
    return OptimisticResetVecEnvWrapper(env, num_envs=num_envs, reset_ratio=reset_ratio)
