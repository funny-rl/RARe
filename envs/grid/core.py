"""Functional JAX grid navigation with explicit episode state."""

from dataclasses import dataclass
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class GridSpec:
    """Host-side layout. Coordinates and action offsets use (row, column)."""

    name: str
    shape: tuple[int, int]
    starts: tuple[tuple[int, int], ...]
    goals: tuple[tuple[int, int], ...]
    pits: tuple[tuple[int, int], ...] = ()
    actions: tuple[tuple[int, int], ...] = ((0, -1), (-1, 0), (0, 1), (1, 0))
    max_steps: int = 100
    step_reward: float = -1.0
    goal_reward: float = -1.0
    pit_reward: float = -100.0
    walls: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        if len(self.shape) != 2 or any(type(n) is not int or n <= 0 for n in self.shape):
            raise ValueError("shape must contain two positive integers")
        if type(self.max_steps) is not int or self.max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        if not self.starts or not self.goals or not self.actions:
            raise ValueError("starts, goals, and actions must be nonempty")
        for positions in (self.starts, self.goals, self.pits, self.walls):
            for position in positions:
                if len(position) != 2 or any(type(n) is not int for n in position):
                    raise ValueError("positions must contain integer coordinate pairs")
                if not all(0 <= n < size for n, size in zip(position, self.shape)):
                    raise ValueError(f"position {position} is outside shape {self.shape}")
        for move in self.actions:
            if len(move) != 2 or any(type(n) is not int for n in move):
                raise ValueError("actions must contain integer coordinate offsets")
        if set(self.pits).intersection((*self.starts, *self.goals)):
            raise ValueError("starts and goals must not overlap pits")
        if set(self.walls).intersection((*self.starts, *self.goals, *self.pits)):
            raise ValueError("walls must not overlap starts, goals, or pits")
        if not all(math.isfinite(r) for r in (self.step_reward, self.goal_reward, self.pit_reward)):
            raise ValueError("rewards must be finite")


class GridParams(NamedTuple):
    """Array-only pytree, shared across seeds or batched across equal-size grids."""

    transitions: jax.Array
    pit_mask: jax.Array
    starts: jax.Array
    goals: jax.Array
    max_steps: jax.Array
    step_reward: jax.Array
    goal_reward: jax.Array
    pit_reward: jax.Array


class GridState(NamedTuple):
    """Per-episode pytree. Position and goal are flattened cell indices."""

    position: jax.Array
    goal: jax.Array
    steps: jax.Array


def build_params(spec: GridSpec) -> GridParams:
    """Build deterministic transitions once, outside JIT and rollout loops."""
    height, width = spec.shape

    def index(position: tuple[int, int]) -> int:
        return position[0] * width + position[1]

    transitions = []
    for row in range(height):
        for col in range(width):
            successors = []
            for dr, dc in spec.actions:
                nr, nc = row + dr, col + dc
                valid = 0 <= nr < height and 0 <= nc < width
                successors.append(index((nr, nc)) if valid else index((row, col)))
            transitions.append(successors)
    pits = set(map(index, spec.pits))
    walls = set(map(index, spec.walls))
    if walls:
        transitions = [
            [cell if successor in walls else successor for successor in successors]
            for cell, successors in enumerate(transitions)
        ]
    return GridParams(
        transitions=jnp.asarray(transitions, dtype=jnp.int32),
        pit_mask=jnp.asarray([i in pits for i in range(height * width)]),
        starts=jnp.asarray(list(map(index, spec.starts)), dtype=jnp.int32),
        goals=jnp.asarray(list(map(index, spec.goals)), dtype=jnp.int32),
        max_steps=jnp.asarray(spec.max_steps, dtype=jnp.int32),
        step_reward=jnp.asarray(spec.step_reward, dtype=jnp.float32),
        goal_reward=jnp.asarray(spec.goal_reward, dtype=jnp.float32),
        pit_reward=jnp.asarray(spec.pit_reward, dtype=jnp.float32),
    )


def observe(state: GridState, params: GridParams) -> jax.Array:
    """Preserve the original encoding: agent=1, goal=2, overlap=3, pits hidden."""
    obs = jnp.zeros(params.pit_mask.shape, dtype=jnp.float32)
    return obs.at[state.goal].set(2.0).at[state.position].add(1.0)


def reset(key: jax.Array, params: GridParams) -> tuple[jax.Array, GridState]:
    """Sample start and goal independently; the caller owns key advancement."""
    start_key, goal_key = jax.random.split(key)
    state = GridState(
        position=params.starts[jax.random.randint(start_key, (), 0, params.starts.shape[0])],
        goal=params.goals[jax.random.randint(goal_key, (), 0, params.goals.shape[0])],
        steps=jnp.asarray(0, dtype=jnp.int32),
    )
    return observe(state, params), state


def step(
    key: jax.Array, state: GridState, action: jax.Array, params: GridParams
) -> tuple[jax.Array, GridState, jax.Array, jax.Array, jax.Array, dict]:
    """Take one primitive step. Pass a valid action and reset after either done flag.

    The key is reserved for stochastic environments; these transitions are
    deterministic. Termination takes precedence over the time limit, matching
    the original implementation. No automatic reset occurs.
    """
    del key
    position = params.transitions[state.position, action]
    next_state = GridState(position, state.goal, state.steps + 1)
    pit = params.pit_mask[position]
    goal = position == state.goal
    terminated = pit | goal
    truncated = (next_state.steps >= params.max_steps) & ~terminated
    reward = jnp.where(pit, params.pit_reward, jnp.where(goal, params.goal_reward, params.step_reward))
    return observe(next_state, params), next_state, reward, terminated, truncated, {}


class GridEnv:
    """Bind a layout to pure functions compatible with jit, vmap, and scan.

    Layout construction is host-side. All rollout state is explicit, so one
    instance can serve independent seeds without sharing episode state.
    """

    backend = "rare_jax"
    supports_exact_risk = True
    requires_global_step = False
    observation_storage_dtype = jnp.uint8
    observation_codec = "grid_indices"

    def __init__(self, spec: GridSpec):
        self.spec = spec
        self.params = build_params(spec)

    @property
    def observation_shape(self) -> tuple[int]:
        return (self.spec.shape[0] * self.spec.shape[1],)

    @property
    def num_actions(self) -> int:
        return len(self.spec.actions)

    def reset(self, key: jax.Array, *, global_step: int | jax.Array = 0) -> tuple[jax.Array, GridState]:
        del global_step
        return reset(key, self.params)

    def evaluation_resets(self, *, global_step: int | jax.Array = 0) -> tuple[jax.Array, GridState]:
        """Enumerate start-major, goal-minor reset pairs without sampling RNG."""
        del global_step
        starts, goals = self.params.starts, self.params.goals
        state = GridState(
            position=jnp.repeat(starts, goals.shape[0]),
            goal=jnp.tile(goals, starts.shape[0]),
            steps=jnp.zeros((starts.shape[0] * goals.shape[0],), dtype=jnp.int32),
        )
        return jax.vmap(observe, in_axes=(0, None))(state, self.params), state

    def with_global_step(self, observation: jax.Array, state: GridState,
                         global_step: int | jax.Array) -> tuple[jax.Array, GridState]:
        """Static grids do not change their evaluation state with training time."""
        del global_step
        return observation, state

    def episode_outcome(self, state: GridState) -> tuple[jax.Array, jax.Array]:
        """Classify the real final state for performance evaluation."""
        pit = self.params.pit_mask[state.position]
        return (state.position == state.goal) & ~pit, pit

    def step(
        self, key: jax.Array, state: GridState, action: jax.Array
    ) -> tuple[jax.Array, GridState, jax.Array, jax.Array, jax.Array, dict]:
        return step(key, state, action, self.params)
