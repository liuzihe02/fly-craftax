"""Craftax-Classic with fly-frame actions and a survival reward."""

from functools import partial

import jax
import jax.numpy as jnp
from craftax.craftax_classic.constants import Action
from craftax.environment_base import spaces

from flycraftax.wrappers import GymnaxWrapper

ACTIONS = ("noop", "forward", "backward", "turn_left", "turn_right", "do", "sleep")
NOOP, FORWARD, BACKWARD, TURN_LEFT, TURN_RIGHT, DO, SLEEP = range(7)
# Facing is an Action value 1..4 (LEFT, RIGHT, UP, DOWN); index 0 is unused.
LEFT_OF = jnp.array([0, 4, 3, 1, 2], jnp.int32)
RIGHT_OF = jnp.array([0, 3, 4, 2, 1], jnp.int32)
OPPOSITE = jnp.array([0, 2, 1, 4, 3], jnp.int32)


class EgocentricWrapper(GymnaxWrapper):
    """Seven fly-frame actions over the grid actions. A turn costs one NOOP step."""

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
