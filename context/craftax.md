# Craftax-Classic (Pixels) reference

Reference notes for driving `Craftax-Classic-Pixels-v1` from an external controller.
All paths relative to `/home/flowingpurplecrane/personal/fly-craftax/external/Craftax/` unless absolute.
Pinned to the local clone at tag **v1.6.1** (commit `c3c2e0d`, "lazily load textures").

## Install and env creation

- Package
  - PyPI name is `craftax`, current version `1.6.1` (`pyproject.toml:6-7`).
  - Python `>=3.9`; CI badge lists 3.9 / 3.10 / 3.11 / 3.12.
  - Runtime deps are unpinned: `jax`, `flax`, `numpy`, `pygame`, `matplotlib`, `imageio` (`pyproject.toml:19-26`).
    - No JAX version pin at all. Whatever `pip install jax` resolves is what you get.
    - Code uses `jax.tree.map` (not `jax.tree_util.tree_map`) in `environment_bases.py:40`, which needs **jax >= 0.4.25**. That is the real floor.
    - Also uses `tuple[int, int, int, int]` annotation at `craftax/craftax_classic/envs/craftax_state.py:74`, which needs Python >= 3.9 with `from __future__` not required at runtime only because it is inside a dataclass annotation evaluated lazily by flax. Practically: use 3.10+.
  - `PIL` (`Pillow`) is imported by `constants.py:9` but is **not** in `dependencies`. Install it explicitly or texture loading fails on a cold cache.
  - Optional extra: `dev` gives `ruff` + `pre-commit` only. There is no `gpu` extra.
- Install recipe
  - `pip install craftax` gives CPU JAX.
  - GPU: `pip install -U "jax[cuda12]"` afterwards (README lines 77-82).
  - Editable local clone: `pip install -e ".[dev]"` from the repo root, needs `pip>=23.0`.
- Making the env
  - Two factory functions, both in `craftax/craftax_env.py`.

```python
# craftax/craftax_env.py:1, :20-28
def make_craftax_env_from_name(name: str, auto_reset: bool): ...
# name in {"Craftax-Classic-Pixels-v1", "Craftax-Classic-Pixels-AutoReset-v1", ...}

# craftax/craftax_env.py:58
def make_craftax_env_from_params(classic: bool, symbolic: bool, auto_reset: bool): ...
```

  - `auto_reset=True` returns `CraftaxClassicPixelsEnv` (`craftax/craftax_classic/envs/craftax_pixels_env.py:97`).
  - `auto_reset=False` returns `CraftaxClassicPixelsEnvNoAutoReset` (same file, line 23).
  - Neither factory takes `static_env_params`. To change map size or mob caps you must construct the class directly: `CraftaxClassicPixelsEnv(StaticEnvParams(...))`.
- Params objects
  - `EnvParams` is a flax `struct.dataclass`, passed through as a traced pytree by the base env (`craftax/craftax_classic/envs/craftax_state.py:77-95`).
  - `StaticEnvParams` is baked into the env object at construction and controls array shapes (`:98-107`).
- gymnax-style API (`craftax/environment_base/environment_bases.py:12-54`)

```python
# environment_bases.py:19-45  (EnvironmentAutoReset)
@partial(jax.jit, static_argnums=(0,))
def step(self, key, state, action, params=None):
    ...
    return obs, state, reward, done, info

@partial(jax.jit, static_argnums=(0,))
def reset(self, key, params=None):
    ...
    return obs, state
```

  - `reset(key, params) -> (obs, state)`.
  - `step(key, state, action, params) -> (obs, state, reward, done, info)`, a 5-tuple, not a gymnasium 6-tuple. There is no separate truncation flag.
  - `action` is a scalar int (int32).
  - `key` is a single `PRNGKey`; the env splits it internally.
  - `params=None` falls back to `self.default_params`.
  - Only `self` is a static argnum, so `params` is traced here. Note the Craftax_Baselines wrappers do the opposite (see Wrapping / Gotchas).
- Minimal usage

```python
import jax
from craftax.craftax_env import make_craftax_env_from_name

env = make_craftax_env_from_name("Craftax-Classic-Pixels-v1", auto_reset=True)
params = env.default_params                     # EnvParams()
rng = jax.random.PRNGKey(0)
rng, r1, r2 = jax.random.split(rng, 3)
obs, state = env.reset(r1, params)              # obs (63, 63, 3) float32
obs, state, reward, done, info = env.step(r2, state, 5, params)   # 5 = Action.DO
```

## Observation

### Pixels (what we use)

- Shape and dtype
  - Declared by `observation_space` (`craftax_pixels_env.py:158-168`) as
    `((OBS_DIM[0] + INVENTORY_OBS_HEIGHT) * BLOCK_PIXEL_SIZE_AGENT, OBS_DIM[1] * BLOCK_PIXEL_SIZE_AGENT, 3)`.
  - Constants (`craftax/craftax_classic/constants.py:14-20`): `OBS_DIM = (7, 9)`, `INVENTORY_OBS_HEIGHT = 2`, `BLOCK_PIXEL_SIZE_AGENT = 7`.
  - So the agent observation is **(63, 63, 3) float32**, values in **[0.0, 1.0]**.
    - Height 63 = (7 map rows + 2 inventory rows) * 7 px.
    - Width 63 = 9 map cols * 7 px.
    - Square by coincidence, not by design.
  - `get_obs` divides the raw uint-range render by 255 (`craftax_pixels_env.py:140-142`):

```python
# craftax/craftax_classic/envs/craftax_pixels_env.py:140
def get_obs(self, state: EnvState) -> jax.Array:
    pixels = self._render_fn(state) / 255.0
    return pixels
```

- Layout
  - Rows 0..48 are the **map view**: 7 tiles tall by 9 tiles wide at 7 px per tile.
  - Rows 49..62 are the **inventory / status bar**, 2 tiles tall. It **is** part of the observation image.
  - Concatenation happens at `craftax/craftax_classic/renderer.py:705`: `pixels = jnp.concatenate([map_pixels, inv_pixels], axis=0)`.
- Where the player sits
  - The view is centred on the player: top-left corner is `player_position - OBS_DIM // 2` in the padded map (`renderer.py:121-129`).
  - `OBS_DIM // 2 = (3, 4)`, so the player is at **tile (row 3, col 4)**, i.e. pixels rows 21..27, cols 28..34.
  - Out-of-map area is padded with `BlockType.OUT_OF_BOUNDS` (value 1), rendered as flat grey 128 (`constants.py:204-206`).
- Axis convention
  - `state.map[i, j]`: axis 0 is the vertical/screen-row axis, axis 1 is the horizontal/screen-column axis.
  - The renderer maps `local_position[0] * block_pixel_size` to image axis 0. So the obs array is standard `[y, x, channel]`.
  - `play_craftax_classic.py:81` transposes `(1, 0, 2)` only because pygame's `surfarray` wants `[x, y, c]`.
- Inventory bar contents (row, col in tile units, `renderer.py:497-702`)
  - Row 0: health(0), food(1), drink(2), energy(3), sapling(4), wood(5), stone(6), coal(7), iron(8).
  - Row 1: diamond(0), wood pickaxe(1), stone pickaxe(2), iron pickaxe(3), wood sword(4), stone sword(5), iron sword(6).
  - Each cell is an icon plus a small digit glyph 0..9. At 7 px per tile the digit is 4 px; it is barely legible, which is one reason to read intrinsics off the state struct instead.
- Day/night and sleep post-processing (`renderer.py:408-452`)
  - When `light_level < 0.5` a noisy dark overlay is applied, blended by `daylight * map_pixels + (1 - daylight) * night_pixels`.
  - The night static uses `state.state_rng`, so the render is a deterministic function of the state. Same state gives the same image.
  - While `is_sleeping` the whole map view is desaturated to luminance and tinted blue.
  - The inventory bar is **not** darkened at night or during sleep. It is appended after the tinting.
- Textures
  - Source PNGs: `craftax/craftax_classic/assets/*.png`, 59 files, all 16x16 RGBA (`constants.py:139` asserts this).
  - Built atlas is pickled to `craftax/craftax_classic/assets/texture_cache_classic.pbz2` (`constants.py:21-23`).
  - `load_all_textures()` is `functools.lru_cache`d and builds three sizes: 7 (agent), 16 (img), 64 (human) (`constants.py:437-484`).
  - Force a rebuild with `export CRAFTAX_RELOAD_TEXTURES=true`.

### Symbolic (for comparison)

- Classic symbolic obs is a flat **(1345,) float32** vector in [0, 1]. Derived from `craftax/craftax_classic/envs/craftax_symbolic_env.py:23-42`:
  - map part: `OBS_DIM[0] * OBS_DIM[1] * (len(BlockType) + 4)` = `7 * 9 * (17 + 4)` = 1323.
  - inventory part: `12 inv + 4 intrinsics + 1 light + 1 sleeping + 4 direction` = 22.
- Layout (`craftax/craftax_classic/renderer.py:7-110`)
  - `[0:1323]` reshapes to `(7, 9, 21)`. Channels `[0:17]` one-hot block id, `[17:21]` mob presence in order zombie, cow, skeleton, arrow.
  - `[1323:1335]` inventory counts / 10.0, order: wood, stone, coal, iron, diamond, sapling, wood_pickaxe, stone_pickaxe, iron_pickaxe, wood_sword, stone_sword, iron_sword.
  - `[1335:1339]` intrinsics / 10.0: health, food, drink, energy.
  - `[1339:1343]` one-hot direction, `one_hot(player_direction - 1, 4)`, order left, right, up, down.
  - `[1343]` light_level, `[1344]` is_sleeping.
- Note: the repo's `obs_description.md` describes **full Craftax** (8268-dim), not Classic. Do not use it for Classic.
- Symbolic runs roughly 10x faster than pixels (paper).

## State struct

File: `craftax/craftax_classic/envs/craftax_state.py`.

```python
# craftax/craftax_classic/envs/craftax_state.py:33-74
@struct.dataclass
class EnvState:
    map: jnp.ndarray
    mob_map: jnp.ndarray
    player_position: jnp.ndarray
    player_direction: int
    player_health: int
    player_food: int
    player_drink: int
    player_energy: int
    is_sleeping: bool
    player_recover: float
    player_hunger: float
    player_thirst: float
    player_fatigue: float
    inventory: Inventory
    zombies: Mobs
    cows: Mobs
    skeletons: Mobs
    arrows: Mobs
    arrow_directions: jnp.ndarray
    growing_plants_positions: jnp.ndarray
    growing_plants_age: jnp.ndarray
    growing_plants_mask: jnp.ndarray
    light_level: float
    achievements: jnp.ndarray
    state_rng: Any
    timestep: int
    fractal_noise_angles: tuple[int, int, int, int] = (None, None, None, None)
```

- Field-by-field, with the shape/dtype produced by `generate_world` (`craftax/craftax_classic/world_gen.py:10-256`)

| field | shape | dtype | notes |
|---|---|---|---|
| `map` | (64, 64) | int32 | `BlockType` values 0..16. Size from `StaticEnvParams.map_size`. |
| `mob_map` | (64, 64) | bool | true where any mob occupies a tile. Used for collision. |
| `player_position` | (2,) | int32 | `[row, col]`. Starts at `(32, 32)` (`world_gen.py:14-16`). |
| `player_direction` | scalar | int32 | Holds an **Action value** in {1,2,3,4}. Starts at `Action.UP.value = 3` (`world_gen.py:232`). |
| `player_health` | scalar | int32 | 0..9, starts 9. |
| `player_food` | scalar | int32 | 0..9, starts 9. |
| `player_drink` | scalar | int32 | 0..9, starts 9. |
| `player_energy` | scalar | int32 | 0..9, starts 9. |
| `is_sleeping` | scalar | bool | starts False. |
| `player_recover` | scalar | float32 | health accumulator, starts 0.0. |
| `player_hunger` | scalar | float32 | hunger accumulator, starts 0.0. |
| `player_thirst` | scalar | float32 | thirst accumulator, starts 0.0. |
| `player_fatigue` | scalar | float32 | fatigue accumulator, starts 0.0. Goes negative while sleeping. |
| `inventory` | struct of 12 scalars | int32 | see `craftax_state.py:9-22`, all capped at 9 each step. |
| `zombies` | `Mobs`, cap 3 | see below | |
| `cows` | `Mobs`, cap 3 | | |
| `skeletons` | `Mobs`, cap 2 | | |
| `arrows` | `Mobs`, cap 3 | | `health` unused for arrows. |
| `arrow_directions` | (3, 2) | int32 | per-arrow unit step. |
| `growing_plants_positions` | (10, 2) | int32 | |
| `growing_plants_age` | (10,) | int32 | ripens at >= 600. |
| `growing_plants_mask` | (10,) | bool | |
| `light_level` | scalar | float32 | in [0, 1], see day/night below. |
| `achievements` | (22,) | **bool** | monotone within an episode. Index by `Achievement`. |
| `state_rng` | (2,) | uint32 | PRNGKey. Drives the night static in the render. |
| `timestep` | scalar | int32 | starts 0. |
| `fractal_noise_angles` | tuple of 4 | None | world-gen determinism hook, normally all `None`. |

- `Mobs` (`craftax_state.py:25-30`)
  - `position` (N, 2) int32, `health` (N,) int32, `mask` (N,) bool (alive/active), `attack_cooldown` (N,) int32.
- Everything the fly needs for interoception is directly readable
  - hunger: `state.player_food` (0..9 integer) and/or `state.player_hunger` (0..25 float ramp).
  - thirst: `state.player_drink` and `state.player_thirst` (0..20 ramp).
  - fatigue: `state.player_energy` and `state.player_fatigue` (-10..30 ramp, negative while asleep).
  - Using the float ramps gives a smooth signal instead of a 10-level staircase.

## Actions

```python
# craftax/craftax_classic/constants.py:47-64
class Action(Enum):
    NOOP = 0               # q in the pygame script
    LEFT = 1               # a
    RIGHT = 2              # d
    UP = 3                 # w
    DOWN = 4               # s
    DO = 5                 # space
    SLEEP = 6              # tab
    PLACE_STONE = 7        # r
    PLACE_TABLE = 8        # t
    PLACE_FURNACE = 9      # f
    PLACE_PLANT = 10       # p
    MAKE_WOOD_PICKAXE = 11 # 1
    MAKE_STONE_PICKAXE = 12
    MAKE_IRON_PICKAXE = 13
    MAKE_WOOD_SWORD = 14
    MAKE_STONE_SWORD = 15
    MAKE_IRON_SWORD = 16
```

- `num_actions` and `action_space` both report **17** (`craftax_pixels_env.py:152-156`).
- **The seven actions we want are exactly values 0..6, contiguous and in order.** The action mask is therefore an identity map into `Discrete(7)`.

### Movement and facing

```python
# craftax/craftax_classic/constants.py:68-74
DIRECTIONS = jnp.concatenate((
    jnp.array([[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]], dtype=jnp.int32),
    jnp.zeros((11, 2), dtype=jnp.int32),
), axis=0)
```

- Offsets, indexed by action value
  - `NOOP(0)` = `[0, 0]`
  - `LEFT(1)` = `[0, -1]` (column decreases, screen-left)
  - `RIGHT(2)` = `[0, +1]`
  - `UP(3)` = `[-1, 0]` (row decreases, screen-up)
  - `DOWN(4)` = `[+1, 0]`
  - everything else `[0, 0]`.
- `DIRECTIONS` has only **16** rows while there are 17 actions. `DIRECTIONS[16]` is out of bounds and JAX clamps it to index 15, which is `[0, 0]`. Harmless, but worth knowing it is an off-by-one in the source.

```python
# craftax/craftax_classic/game_logic.py:1374-1397
def move_player(state, action):
    proposed_position = state.player_position + DIRECTIONS[action]
    valid_move = is_position_in_bounds_not_in_wall_not_in_mob_not_in_lava(state, proposed_position)
    valid_move = jnp.logical_or(
        valid_move,
        state.map[proposed_position[0], proposed_position[1]] == BlockType.LAVA.value,
    )
    position = state.player_position + valid_move.astype(jnp.int32) * DIRECTIONS[action]
    is_new_direction = jnp.sum(jnp.abs(DIRECTIONS[action])) != 0
    new_direction = state.player_direction * (1 - is_new_direction) + action * is_new_direction
    state = state.replace(player_position=position, player_direction=new_direction)
    return state
```

- **Does moving change facing?** Yes. Facing is set to the action itself whenever the action is one of LEFT/RIGHT/UP/DOWN.
- **Is there a turn without moving?** No dedicated action. But:
  - **Blocked movement still updates facing.** `is_new_direction` depends only on the action, not on `valid_move`. Walking into a wall, a mob, or the map edge is a pure turn in place.
  - So "turn to face X" is expressible only as "attempt to move X", which will also move you if the tile is free.
- Blocking rules
  - `SOLID_BLOCKS` (`constants.py:91-105`): water, stone, tree, coal, iron, diamond, crafting table, furnace, plant, ripe plant. Grass, sand, path, wood, lava are walkable.
  - Mobs block movement (via `mob_map`).
  - **Lava is deliberately walkable.** `move_player` ORs lava back into `valid_move`, then `update_health` (`game_logic.py:1640-1652`) sets health to 0 and `is_game_over` terminates. Stepping into lava is instant suicide, not a bumped move.
- Facing values are Action values, so `DIRECTIONS[state.player_direction]` gives the offset to the tile in front.

### DO

- Target is always **the single tile in front**: `block_position = state.player_position + DIRECTIONS[state.player_direction]` (`game_logic.py:72`).
- Resolution order inside `do_action` (`game_logic.py:69-382`), all computed then masked:
  1. Attack zombie / cow / skeleton if one occupies the target tile.
  2. If **no** mob was attacked and the tile is in bounds, resolve the block interaction. `action_block_in_bounds = in_bounds AND NOT did_attack_mob` (`:339-342`). Mobs shadow blocks completely.
  3. The whole state delta is then selected on `action == Action.DO` (`:375-380`), so DO is the only action that triggers any of this.
- Block interactions
  - TREE: always minable, gives +1 wood, becomes GRASS, sets `COLLECT_WOOD`.
  - STONE / COAL: needs `wood_pickaxe`; becomes PATH.
  - IRON: needs `stone_pickaxe`. DIAMOND: needs `iron_pickaxe`.
  - GRASS: **10% chance per DO** of yielding a sapling (`:288-292`), sets `COLLECT_SAPLING`. The block is unchanged.
  - WATER: `player_drink = min(9, drink + 1)`, `player_thirst = 0`, sets `COLLECT_DRINK`. One drink point per action.
  - RIPE_PLANT: becomes PLANT, `player_food = min(9, food + 4)`, `player_hunger = 0`, sets `EAT_PLANT`, and resets that plant's age to 0.
- Attack damage: `max(1, 2*wood_sword, 3*stone_sword, 5*iron_sword)` (`game_logic.py:43-53`). Bare-handed is 1.
  - Cow health 3, zombie health 5, skeleton health 3 (`craftax_state.py:84-86`). So bare-handed: 3 DOs for a cow, 5 for a zombie, 3 for a skeleton.
  - Killing a cow gives `player_food = min(9, food + 6)` and `player_hunger = 0`, and sets `EAT_COW`.

### SLEEP

```python
# craftax/craftax_classic/game_logic.py:1239-1253
is_starting_sleep = jnp.logical_and(action == Action.SLEEP.value, state.player_energy < 9)
new_is_sleeping = jnp.logical_or(state.is_sleeping, is_starting_sleep)
...
is_waking_up = jnp.logical_and(state.player_energy >= 9, state.is_sleeping)
```

- **When can you sleep?** Only when `player_energy < 9`. SLEEP at full energy is a no-op, so the WAKE_UP achievement is unavailable until you have been awake ~31 steps.
- **While asleep, all actions are ignored.** `craftax_step` overrides the action first thing:

```python
# craftax/craftax_classic/game_logic.py:1660
action = jax.lax.select(state.is_sleeping, Action.NOOP.value, action)
```

  - The agent cannot choose to wake up. Sleeping is a commitment.
- **Waking, three routes**
  1. Energy reaches 9 (`update_player_intrinsics`). Sets `WAKE_UP`.
  2. A zombie attacks you (`game_logic.py:840-852`). Sets `is_sleeping = False` **and** sets `WAKE_UP`.
  3. An arrow hits you (`game_logic.py:1210`). Sets `is_sleeping = False` but **does not** set `WAKE_UP`. This asymmetry looks unintentional but it is what the code does.
- Sleeping side effects
  - Zombie damage is 7 instead of 2 (`game_logic.py:831-835`). Sleeping in the open is very dangerous.
  - Hunger and thirst accumulate at 0.5/step instead of 1.0.
  - Fatigue is set to `min(fatigue - 1, 0)`, so it snaps to 0 on the first sleep step then decrements. Every time it drops below -10, energy +1 and fatigue resets. About **11 steps per energy point**, roughly 100 steps for a full 0 to 9 recharge.
  - Health recovery doubles (recover accumulator +2.0 instead of +1.0), and energy counts as satisfied for the necessities check even at 0.

### NOOP

- `Action.NOOP = 0`, `DIRECTIONS[0] = [0, 0]`. Nothing is triggered.
- Time still advances: intrinsics decay, mobs move and attack, mobs spawn, plants grow, `timestep += 1`, light level updates.
- NOOP is the resting/waiting action, and is also what SLEEP degenerates into once asleep.

## Achievements and reward

```python
# craftax/craftax_classic/constants.py:109-131
class Achievement(Enum):
    COLLECT_WOOD = 0
    PLACE_TABLE = 1
    EAT_COW = 2
    COLLECT_SAPLING = 3
    COLLECT_DRINK = 4
    MAKE_WOOD_PICKAXE = 5
    MAKE_WOOD_SWORD = 6
    PLACE_PLANT = 7
    DEFEAT_ZOMBIE = 8
    COLLECT_STONE = 9
    PLACE_STONE = 10
    EAT_PLANT = 11
    DEFEAT_SKELETON = 12
    MAKE_STONE_PICKAXE = 13
    MAKE_STONE_SWORD = 14
    WAKE_UP = 15
    PLACE_FURNACE = 16
    COLLECT_COAL = 17
    COLLECT_IRON = 18
    COLLECT_DIAMOND = 19
    MAKE_IRON_PICKAXE = 20
    MAKE_IRON_SWORD = 21
```

- 22 achievements. `state.achievements` is a `(22,)` bool array, set with `logical_or`, so **each fires at most once per episode**.

### The eight survival achievements

| name | idx | condition | reachable with mask {noop,left,right,up,down,do,sleep}? |
|---|---|---|---|
| `COLLECT_WOOD` | 0 | DO facing a TREE tile (`game_logic.py:189-205`) | yes |
| `EAT_COW` | 2 | DO kills a cow, i.e. its health crosses to <= 0 (`:126-131`) | yes |
| `COLLECT_SAPLING` | 3 | DO facing GRASS, 10% roll (`:288-301`) | yes |
| `COLLECT_DRINK` | 4 | DO facing WATER (`:303-315`) | yes |
| `DEFEAT_ZOMBIE` | 8 | DO kills a zombie (`:93-100`) | yes |
| `EAT_PLANT` | 11 | DO facing a `RIPE_PLANT` tile (`:317-333`) | **no, see below** |
| `DEFEAT_SKELETON` | 12 | DO kills a skeleton (`:159-166`) | yes |
| `WAKE_UP` | 15 | energy reaches 9 while sleeping, or a zombie attacks you while sleeping (`:847-851`, `:1250-1252`) | yes |

- **`EAT_PLANT` is unreachable without `PLACE_PLANT`. Confirmed.**
  - `RIPE_PLANT` blocks only appear from `update_plants` (`game_logic.py:1335-1371`), which promotes tiles listed in `growing_plants_positions` once `growing_plants_age >= 600`.
  - `growing_plants_*` entries are only ever created by `add_new_growing_plant`, called only from `place_block` under `is_placing_sapling`, which requires `action == Action.PLACE_PLANT.value` (action 10).
  - World generation never emits `PLANT` or `RIPE_PLANT`: `world_gen.py` writes only WATER, GRASS, SAND, STONE, PATH, COAL, IRON, DIAMOND, TREE, LAVA.
  - So with our action mask, only **7 of the 8** survival achievements are attainable, and the theoretical max achievement reward per episode is 7.0.
  - It also needs `COLLECT_SAPLING` first (sapling inventory), plus 600 steps of growth, plus surviving to return to the tile.

### Default reward

```python
# craftax/craftax_classic/game_logic.py:1655-1710 (craftax_step)
init_achievements = state.achievements
init_health = state.player_health
...
achievement_reward = (
    state.achievements.astype(jnp.float32).sum()
    - init_achievements.astype(jnp.float32).sum()
)
health_reward = (state.player_health - init_health) * 0.1
reward = achievement_reward + health_reward
```

- Two components:
  - `+1.0` for each achievement bit that flips 0 to 1 on this step. Never repeats.
  - `+/- 0.1` per point of health gained or lost this step. This is the only dense signal.
- Max episode reward for Craftax-Classic with the full action set is 22 (all achievements) plus whatever the health term nets out to. The "226" figure in the README scoreboard is for **full Craftax**, not Classic.
- `update_health` clamps health at 0 before the diff is taken (`game_logic.py:1650`), so death does not produce an unbounded negative. Fixed in v1.6.0 per the README errata.
- To replace the reward you must recompute the achievement delta yourself, because the env returns only the scalar. Store the previous `achievements` bool vector in wrapper state (see below).
- `info` from `compute_score` (`craftax/craftax_classic/envs/common.py:5-12`)
  - `info["Achievements/<lowercase_name>"] = achievements[i] * done * 100.0` for all 22. Non-zero only on the terminal step.
  - `info["score"] = exp(mean(log(1 + achievements*done*100))) - 1`, the Crafter-style geometric mean.
  - `info["discount"]` added in `step_env`, 0.0 if terminal else 1.0.

## Survival dynamics

All in `update_player_intrinsics` (`craftax/craftax_classic/game_logic.py:1237-1332`), run once per step after mobs and plants.

- Hunger and food
  - `player_hunger += 1.0` per step, or `+0.5` while sleeping.
  - When `hunger > 25`: `food = max(food - 1, 0)`, `hunger = 0`.
  - So one food point every **26 steps** awake. Full food (9) lasts about **234 steps**.
  - Refills: cow kill `+6`, ripe plant `+4`, both capped at 9 and both zero the hunger accumulator.
- Thirst and drink
  - `player_thirst += 1.0` per step (`+0.5` sleeping). When `thirst > 20`: `drink -= 1`, reset.
  - One drink point every **21 steps**. Full drink lasts about **189 steps**. Thirst is the tightest clock.
  - Refill: DO on water, `+1` per action. Topping up from 0 to 9 costs 9 actions.
- Fatigue and energy
  - Awake: `fatigue += 1`. When `fatigue > 30`: `energy -= 1`, reset. One energy point every **31 steps**, full energy lasts about **279 steps**.
  - Asleep: `fatigue = min(fatigue - 1, 0)`. When `fatigue < -10`: `energy = min(energy + 1, 9)`, reset. About 11 steps per point.
- Health
  - `necessities = [food > 0, drink > 0, (energy > 0 or is_sleeping)]`.
  - All satisfied: `recover += 1.0` awake, `+2.0` asleep. When `recover > 25`: `health = min(health + 1, 9)`, reset. About **26 steps per health point** awake, 13 asleep.
  - Any necessity at 0: `recover -= 1.0` awake, `-0.5` asleep. When `recover < -15`: `health -= 1`, reset. About **16 steps per health point lost**.
  - There is no partial credit: one empty bar is as bad as three.
- Damage
  - Zombie melee: 2 awake, 7 asleep, with a 5-step cooldown, only from an adjacent tile (`game_logic.py:816-838`).
  - Skeleton arrow: 2 on hit (`game_logic.py:1209`).
  - Lava: health set to 0 immediately.
- Day/night

```python
# craftax/craftax_classic/game_logic.py:581-583
def calculate_light_level(timestep, params):
    progress = (timestep / params.day_length) % 1 + 0.3
    return 1 - jnp.abs(jnp.cos(jnp.pi * progress)) ** 3
```

  - `day_length = 300` steps (`craftax_state.py:80`).
  - `light_level` starts at about 0.797 at t=0, is darkest (0.0) at `t mod 300 == 210`.
  - The renderer treats `light_level < 0.5` as night. Solving gives night for roughly **`t mod 300` in [148, 272]**, about 125 of every 300 steps.
- Mob spawning (`game_logic.py:1400-1629`), attempted every step
  - Cows: p = 0.1, only on GRASS, player distance (manhattan) in (3, 14), tile free of mobs. Cap 3.
  - Zombies: p = `0.02 + 0.1 * (1 - light_level)^2`, so 0.02 by day up to **0.12 at midnight**. On GRASS or PATH, distance in (9, 14). Cap 3.
  - Skeletons: p = 0.05, only on PATH tiles, distance in (9, 14). Cap 2.
  - Despawn: any mob with manhattan distance >= `mob_despawn_distance = 14` has its mask cleared.
  - Zombies home in on the player when within manhattan distance 10 (75% of the time); skeletons keep to a 4-5 tile band and shoot.
- Termination

```python
# craftax/craftax_classic/game_logic.py:5-13
def is_game_over(state, params):
    done_steps = state.timestep >= params.max_timesteps
    in_lava = state.map[state.player_position[0], state.player_position[1]] == BlockType.LAVA.value
    is_dead = state.player_health <= 0
    return done_steps | in_lava | is_dead
```

  - `max_timesteps = 10000` (`craftax_state.py:79`).
  - Timeout and death are indistinguishable in the returned tuple; both come back as `done=True`. There is no truncation flag. If you need one, compare `state.timestep` in your wrapper.

## Wrapping

### What exists

- **Nothing ships in the `craftax` package.** `grep -rn "class.*Wrapper" craftax/` returns nothing at v1.6.1, and `craftax/environment_base/wrappers.py` 404s on GitHub for both `main` and older tags.
  - Consequence: `Craftax_Baselines/analysis/view_ppo_agent.py:10` does `from craftax.environment_base.wrappers import AutoResetEnvWrapper`, which is a **broken import** against any released craftax. Copy the wrapper instead.
- The wrappers live in the separate repo `MichaelTMatthews/Craftax_Baselines`, file `wrappers.py` (209 lines). Copy it into the project; it has no craftax-specific code, only `jax`, `chex`, `flax`.
  - `GymnaxWrapper` (`:10`): base class, proxies unknown attributes via `__getattr__`.
  - `BatchEnvWrapper(env, num_envs)` (`:21`): `jax.vmap(reset, in_axes=(0, None))` and `jax.vmap(step, in_axes=(0, 0, 0, None))`. Splits the key for you.
  - `AutoResetEnvWrapper(env)` (`:48`): recovers gymnax auto-reset for a `NoAutoReset` env. Steps and resets every call, then `tree.map(lax.select(done, ...))`.
  - `OptimisticResetVecEnvWrapper(env, num_envs, reset_ratio)` (`:83`): batches and resets, but generates only `num_envs // reset_ratio` fresh worlds per step and scatters them to whichever envs are done. Much cheaper because world gen is the expensive part. Baselines default `reset_ratio=16` with `num_envs=1024`.
    - If more than `num_resets` envs finish on the same step, some do not get a fresh world. That is the "optimistic" part.
  - `LogWrapper(env)` (`:170`): wraps state in `LogEnvState(env_state, episode_returns, episode_lengths, returned_episode_returns, returned_episode_lengths, timestep)` and adds `returned_episode_returns`, `returned_episode_lengths`, `timestep`, `returned_episode` to `info`.
- The README's own gotcha (lines 98-104): an `auto_reset=False` env that is not wrapped will happily keep stepping past termination into garbage states.

### Recommended stack for this project

```python
env = make_craftax_env_from_name("Craftax-Classic-Pixels-v1", auto_reset=False)
env = ActionMaskWrapper(env)         # stateless
env = SurvivalRewardWrapper(env)     # adds prev_achievements to state
env = LogWrapper(env)
env = OptimisticResetVecEnvWrapper(env, num_envs=NUM_ENVS, reset_ratio=16)
```

- Order matters. The reward wrapper must sit **inside** the auto-reset so that the post-step `state.achievements` it diffs against is the real one, not a freshly reset world.
  - If you instead wrap an `auto_reset=True` env, the terminal step returns the **reset** state, whose achievements are all zero, and your diff goes negative and loses the last achievement. Do not do this.
  - When the outer auto-reset fires, it `tree.map`s over the whole wrapper state including `prev_achievements`, pulling the zeros from your wrapper's own `reset`. That is exactly the behaviour you want.
- Put the action mask innermost so `LogWrapper` and the PPO code see `Discrete(7)`.

### Action mask wrapper

Since noop/left/right/up/down/do/sleep are exactly action values 0..6, this is an identity gather. Kept as an explicit table so an egocentric remap can drop in later.

```python
from functools import partial
import jax, jax.numpy as jnp
from craftax.environment_base import spaces
from craftax.craftax_classic.constants import Action

ALLOWED = jnp.array([
    Action.NOOP.value, Action.LEFT.value, Action.RIGHT.value,
    Action.UP.value, Action.DOWN.value, Action.DO.value, Action.SLEEP.value,
], dtype=jnp.int32)   # [0, 1, 2, 3, 4, 5, 6]

class ActionMaskWrapper(GymnaxWrapper):
    """Restrict the 17-action space to the 7 survival actions."""

    def action_space(self, params=None):
        return spaces.Discrete(ALLOWED.shape[0])

    @property
    def num_actions(self):
        return int(ALLOWED.shape[0])

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        return self._env.step(key, state, ALLOWED[action], params)
```

- jit-safe because the remap is a traced gather on a constant array, not Python control flow.
- `reset` is inherited through `__getattr__`.
- Do **not** clip or validate `action`; an out-of-range index clamps under JAX rather than erroring, which silently maps to `SLEEP`. Sample from `Discrete(7)` and you are fine.

### Survival-only reward wrapper

```python
from typing import Any
from flax import struct

SURVIVAL_IDX = jnp.array([0, 2, 3, 4, 8, 11, 12, 15], dtype=jnp.int32)
# collect_wood, eat_cow, collect_sapling, collect_drink,
# defeat_zombie, eat_plant, defeat_skeleton, wake_up

@struct.dataclass
class SurvivalRewardState:
    env_state: Any
    prev_achievements: jnp.ndarray      # (22,) bool

class SurvivalRewardWrapper(GymnaxWrapper):
    def __init__(self, env, keep_health_reward=False, health_coef=0.1):
        super().__init__(env)
        self.mask = jnp.zeros(22, dtype=jnp.float32).at[SURVIVAL_IDX].set(1.0)
        self.keep_health_reward = keep_health_reward
        self.health_coef = health_coef

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        obs, env_state = self._env.reset(key, params)
        return obs, SurvivalRewardState(env_state, env_state.achievements)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        prev_health = state.env_state.player_health
        obs, env_state, _, done, info = self._env.step(
            key, state.env_state, action, params
        )
        newly = jnp.logical_and(
            env_state.achievements, jnp.logical_not(state.prev_achievements)
        )
        reward = (newly.astype(jnp.float32) * self.mask).sum()
        if self.keep_health_reward:
            reward = reward + (env_state.player_health - prev_health) * self.health_coef
        new_state = SurvivalRewardState(env_state, env_state.achievements)
        return obs, new_state, reward, done, info
```

- `keep_health_reward` is a Python bool read at trace time, so the `if` is fine under jit.
- With it off, reward is 0 on almost every step and at most 7.0 per episode. That is extremely sparse for a linear readout. Consider keeping the health term, or adding a small shaped survival bonus, and note that in the spec.
- `state.env_state.player_health` assumes the reward wrapper sits directly on the base env or on a stateless wrapper. If you insert another stateful wrapper below it, adjust the path.

### vmap over envs

- `jax.vmap(env.reset, in_axes=(0, None))` and `jax.vmap(env.step, in_axes=(0, 0, 0, None))`.
  - Axis 0 over keys, state pytree, and actions. `params` is broadcast.
- The whole `EnvState` is a flax `struct.dataclass`, so it vmaps and `lax.scan`s without any extra registration.
- Obs batches as `(num_envs, 63, 63, 3)`.
- Do not vmap over `params` unless you want per-env difficulty; every `EnvParams` field is a traced leaf.

## Baselines

`Craftax_Baselines/ppo.py`, 738 lines. PureJaxRL-derived (its own header says so at line 36-37).

- Structure
  - `make_train(config)` returns a `train(rng)` closure. Everything is inside jit; `run_ppo` does `jax.vmap(jax.jit(make_train(config)))` over seeds for `NUM_REPEATS` parallel runs.
  - Env setup at `ppo.py:61-75`: `make_craftax_env_from_name(env_name, not use_optimistic_resets)`, then `LogWrapper`, then either `OptimisticResetVecEnvWrapper` or `AutoResetEnvWrapper` + `BatchEnvWrapper`.
  - Network choice at `:87-92`: `ActorCritic` (MLP) for `"Symbolic" in env_name`, else `ActorCriticConv`.
  - `_update_step` = `lax.scan(_env_step, ..., NUM_STEPS)` collecting a `Transition` NamedTuple, then GAE, then `UPDATE_EPOCHS` passes of `NUM_MINIBATCHES` minibatches.
  - Loss (`ppo.py:351-387`): clipped value loss with `value_pred_clipped`, clipped surrogate policy loss, advantage normalisation per minibatch, entropy bonus. Textbook PPO.
- Defaults (`ppo.py:669-723`)
  - `num_envs=1024`, `num_steps=64`, `total_timesteps=1e9`, `lr=2e-4` with linear anneal, `update_epochs=4`, `num_minibatches=8`, `gamma=0.99`, `gae_lambda=0.8`, `clip_eps=0.2`, `ent_coef=0.01`, `vf_coef=0.5`, `max_grad_norm=1.0`, `layer_size=512`, `optimistic_reset_ratio=16`.
  - `gae_lambda=0.8` is notably lower than the usual 0.95.
- `ActorCriticConv` (`Craftax_Baselines/models/actor_critic.py:83`) is the pixels network: three `Conv(32, 5x5) + relu + max_pool(3,3, stride 3)` blocks, flatten, then heads.
  - On a 63x63 input this reduces to roughly 2x2x32 = 128 features. It is small.
- What to reuse for a linear readout on an external feature vector
  - **Reuse wholesale**: the `Transition` NamedTuple, the `_env_step` scan, `_calculate_gae`, the minibatch shuffle/reshape, the clipped `_loss_fn`, `optax.chain(clip_by_global_norm, adam)`, `TrainState`, and the `LogWrapper` metric aggregation at `ppo.py:348-352`.
  - **Replace**: `network` becomes a single `nn.Dense(7)` actor head (plus a value head) applied to the fly's DN feature vector rather than to `obs`.
  - **The hard part**: the brain simulation is stateful across steps and is not part of `network.apply`. Two options:
    - Carry the brain's neuron state in the `runner_state` tuple alongside `env_state`, advance it inside `_env_step`, and store the resulting DN vector in `Transition.obs`. Then the PPO update is unchanged: it re-applies a stateless linear layer to stored features. This is the clean route and keeps everything jittable if the brain sim is JAX.
    - If the brain is not JAX, you cannot keep the scan; fall back to a host-side loop and lose most of the throughput.
  - `Transition.obs` becomes `(num_steps, num_envs, n_dn)` floats. With ~1300 DNs and 1024 envs x 64 steps that is ~340 MB in float32. Budget for it or use bf16 / fewer envs.
- `purejaxrl/ppo.py` comparison
  - Same skeleton, simpler: no ICM/E3B/RND branches, no wandb/orbax, no `next_obs` in the transition.
  - Network is a fixed 64-64 tanh MLP (`purejaxrl/ppo.py:14-49`).
  - Uses `gymnax.make` + `FlattenObservationWrapper` + `LogWrapper` instead of the craftax factories.
  - If you want the smallest readable starting point, copy `purejaxrl/ppo.py` and swap in the craftax env plus the Craftax_Baselines wrappers. Everything else transfers line for line.

## Rendering for a viewer

- Renderer factory

```python
# craftax/craftax_classic/renderer.py:113
def make_craftax_pixel_renderer(block_pixel_size):
    textures = load_all_textures()[block_pixel_size]
    def render_craftax_pixels(state): ...
    return render_craftax_pixels
```

  - `block_pixel_size` must be one of the three cached sizes: 7 (`BLOCK_PIXEL_SIZE_AGENT`), 16 (`BLOCK_PIXEL_SIZE_IMG`), 64 (`BLOCK_PIXEL_SIZE_HUMAN`).
  - For a viewer, build a **second** renderer at size 64. Output is `(9*64, 9*64, 3) = (576, 576, 3)`, values 0..255 float32. Do not divide by 255 unless your display expects floats.
  - It is a pure function of `state`, so you can render any stored state without stepping.
- There is **no built-in full-map renderer** in Classic. `full_map_block_textures` is a misnomer; it is a single block tiled to fill the 7x9 view, used for the masked-add trick at `renderer.py:145-155`.
- Full 64x64 map render, using the raw block atlas:

```python
from craftax.craftax_classic.constants import load_all_textures, BLOCK_PIXEL_SIZE_IMG
B = BLOCK_PIXEL_SIZE_IMG                                    # 16
tex = load_all_textures()[B]["block_textures"]              # (17, 16, 16, 3) int32
tiles = tex[state.map]                                      # (64, 64, 16, 16, 3)
img = tiles.transpose(0, 2, 1, 3, 4).reshape(64 * B, 64 * B, 3)   # (1024, 1024, 3)
```

  - Overlay the player and mobs yourself with `dynamic_update_slice` at `position * B`, copying the pattern from `renderer.py:171-220`.
  - jit-safe as written.
- Swapping the player sprite
  - Sprites are `assets/player-{left,right,up,down,sleep}.png`, 16x16 RGBA, loaded at `constants.py:228-234` into `player_textures` in exactly that order.
  - The renderer picks the index at `renderer.py:158-160`: `select(is_sleeping, 4, player_direction - 1)`. Since `player_direction` is the Action value 1..4, index 0 is left, 1 right, 2 up, 3 down, 4 sleeping.
  - Two ways to swap in a fly sprite:
    1. Overwrite the five PNGs in `assets/` (keep 16x16 RGBA) and run once with `CRAFTAX_RELOAD_TEXTURES=true` to rebuild `texture_cache_classic.pbz2`. Cleanest, affects every renderer.
    2. Mutate the cached dict before constructing the renderer. `load_all_textures()` is `lru_cache`d and returns the same mutable dict, and `make_craftax_pixel_renderer` captures `textures` by reference at construction. Replace `full_map_player_textures` and `full_map_player_textures_alpha` (both `(5, 7*bps, 9*bps, 3)` padded arrays built at `constants.py:236-257`) before calling the factory.
  - The padding at `constants.py:223-226` centres the sprite in a full-view-sized canvas so it can be alpha-composited in one shot. If you build replacements, reuse that padding code.
- pygame play script
  - `craftax/craftax_classic/play_craftax_classic.py`, entry point `play_craftax_classic` (`pyproject.toml:36`).
  - `CraftaxRenderer` class at `:43-98` is directly reusable: it holds a jitted 64px renderer, blits via `pygame.surfarray.make_surface(np.array(pixels).transpose((1, 0, 2)))`, and does nearest-neighbour upscaling with `jnp.repeat`.
  - Key map at `:22-40`. Note it uses `q` for NOOP, not space.
  - `get_action_from_keypress` returns `Action.NOOP.value` unconditionally while `state.is_sleeping` (`:91-92`), mirroring the engine's own override.
  - It builds the env as `Craftax-Classic-Symbolic-v1` and renders separately, which is the right pattern for us too: run the agent on pixels at size 7, render the viewer at size 64 from the same state.
  - First frame takes ~30s to compile, first step another ~20s (README line 93).

## Gotchas

- jit and static args
  - Craftax's own `EnvironmentAutoReset.step` / `.reset` use `static_argnums=(0,)`, so `params` is **traced**.
  - The Craftax_Baselines wrappers use `static_argnums=(0, 2)` on reset and `(0, 4)` on step, so `params` is **static** there. It works because `EnvParams` is a frozen dataclass with hashable fields, but any change to `params` retriggers compilation, and an unhashable field would blow up. Keep the convention consistent in your own wrappers; I used `(0,)` in the snippets above.
  - `self` as a static arg means the wrapper object must be hashable. Plain classes are (identity hash), but do not add `__eq__` without `__hash__`.
- Integer dtypes
  - `player_direction` stores an **Action value** (1..4), not a 0..3 index. Every consumer does `player_direction - 1`. Easy to get wrong when writing an egocentric wrapper.
  - `achievements` is `bool`, not int. `sum()` on it gives a bool-summed int; the engine casts to float32 before differencing. Do the same.
  - `state.map` is int32 holding `BlockType` values; `BlockType.INVALID = 0` and `OUT_OF_BOUNDS = 1` are padding-only.
  - Out-of-bounds gathers clamp silently in JAX (`DIRECTIONS[16]`, an over-range action index). No error, wrong behaviour.
- Seeds and determinism
  - `reset(key)` fully determines the world. Same key gives the same map and same player start `(32, 32)`.
  - `state.state_rng` is carried in the state and refreshed each step (`game_logic.py:1702-1708`); it is what makes the night static reproducible from a state alone.
  - `EnvParams.fractal_noise_angles` lets you pin world-gen noise phases if you want a fixed world across seeds.
- Auto-reset
  - `auto_reset=True` calls `reset_env` on **every** step and `lax.select`s the result. World generation therefore runs every step, in every env. This is the single biggest cost and the entire reason `OptimisticResetVecEnvWrapper` exists.
  - For any batched training run, use `auto_reset=False` plus `OptimisticResetVecEnvWrapper`.
  - For a single-env interactive viewer, `auto_reset=True` is fine and simpler.
- Reward and episode boundaries
  - With auto-reset, the `obs` returned on a terminal step is the **new episode's first obs**, while `reward` and `info` belong to the old episode. Standard gymnax semantics; the PPO code relies on it.
  - `done` conflates death, lava, and timeout.
- Textures
  - Cold start rebuilds the atlas (~1 minute) and writes into the installed package directory. In a read-only install this fails. Pre-warm the cache, or ship the `.pbz2`.
  - `Pillow` is an undeclared dependency of that path.
- Performance (paper, section on speed; single RTX 4090 + i9-13900K, 4096 parallel envs)
  - Craftax-Classic: **405,618 steps/sec**.
  - Craftax (full): 266,961 steps/sec.
  - Original Crafter: 1,580 steps/sec. So 257x and 169x respectively.
  - Unverified which observation variant those headline numbers used. The paper separately states symbolic runs about 10x faster than pixels, which would put Classic-Pixels near 40k steps/sec at that batch size. Treat the pixels figure as an estimate, not a measured number.
  - A 1e9-step PPO run finishes in under an hour on one GPU.
  - On an 8 GB 4060 you will be memory-bound long before you are compute-bound, mostly on the stored pixel observations and the brain state. Start at `num_envs` in the low hundreds.

## Open questions for spec

- **Egocentric actions versus a grid with absolute facing.** This is the main design conflict.
  - Craftax has no turn action. Facing is set by the move action, and the only way to turn without moving is to attempt a move into a blocked tile, which you cannot rely on.
  - Options:
    1. **Absolute mapping (identity).** Map the fly's left/right/forward DNs onto `LEFT/RIGHT/UP/DOWN` directly. Trivial to implement, but the fly's turn circuits then mean "go west", which throws away the whole point of using DNa02-style left-minus-right steering.
    2. **Egocentric wrapper with an in-place turn.** Action set `{noop, forward, backward, turn_left, turn_right, do, sleep}`. `forward` issues the grid action equal to the current facing. `turn_left` / `turn_right` issue a `NOOP` step (so time still advances, mobs move, intrinsics decay) and then overwrite `player_direction` in the returned state and **re-render the obs** via `self._env.get_obs(new_state)`, because the player sprite encodes facing. Rotation tables, indexed by facing value 1..4:
       - `LEFT_OF  = [0, 4, 3, 1, 2]` (counter-clockwise: up to left, left to down, down to right, right to up)
       - `RIGHT_OF = [0, 3, 4, 2, 1]`
       - `OPPOSITE = [0, 2, 1, 4, 3]`
    3. **Backward walking** is the awkward case. Issuing `OPPOSITE[facing]` moves you back but also flips your facing, which is not what MDN-driven backward walking does. To keep facing you would step with the opposite action and then restore `player_direction`, same trick as the turn.
  - Question for the spec: is the turn allowed to consume a timestep (option 2, recommended), or must turning be free? Free turns would require stepping the env zero times and mutating state, which desynchronises the brain's clock from the world's.
  - Second question: does modifying `player_direction` post-step count as "two wrappers only"? It is one wrapper, but it writes to the state struct rather than only filtering actions.
- **`eat_plant` is unreachable under the proposed action mask.** Confirmed by source. It needs `PLACE_PLANT` (action 10), which needs a sapling, plus 600 steps of growth. Either drop it from the reward set (leaving 7 achievements) or add `place_plant` as an eighth action. Which?
- **Reward sparsity.** With achievements only and no health term, the max episode return is 7.0 over up to 10,000 steps, and most of those 7 are one-shot. A linear readout on DN activity is unlikely to find that signal. Do we keep the `0.1 * delta_health` dense term, add a per-step alive bonus, or reward the intrinsic bars directly (food/drink/energy above threshold)?
- **Repeatable achievements.** `EAT_COW` fires once per episode but eating cows is the actual survival behaviour we want. If the goal is survival rather than achievement-scoring, consider rewarding the *event* (cow killed, water drunk) every time it happens, computed from state deltas, rather than the achievement bit. That is a bigger departure from "achievement reward" and should be an explicit decision.
- **Inventory bar in the observation.** It occupies rows 49..62 of the 63x63 image, 22% of the pixels, and is mostly irrelevant under our action mask (no crafting, no placing). Do we feed the full 63x63 to the retina, crop to the 49x63 map view, or crop to a square 49x49 centred on the player? The retina is a ring of ~800 photoreceptors per eye, so the crop determines the sampling geometry.
- **View window shape.** The map view is 7 tall by 9 wide, not square, and the player is off-centre vertically (row 3 of 7) but centred horizontally (col 4 of 9). A radially symmetric ring retina over a 7x9 window sees more sideways than forwards/backwards. Worth deciding whether to letterbox to 9x9.
- **Which JAX version to pin.** The package pins nothing. Pick one (e.g. `jax==0.4.35`) in our own requirements so the brain sim and the env agree, and so `jax.tree.map` exists.
