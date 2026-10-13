"""A compact grid whose wind direction changes on the training-step clock."""

from dataclasses import dataclass
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .core import GridEnv, GridSpec, observe as observe_grid, reset as reset_grid


@dataclass(frozen=True)
class WindShiftSpec(GridSpec):
    wind_probability: float = 0.5
    switch_interval: int = 0
    initial_wind: str = "north"
    observe_wind: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if not math.isfinite(self.wind_probability) or not 0 <= self.wind_probability <= 1:
            raise ValueError("wind_probability must be in [0, 1]")
        if type(self.switch_interval) is not int or self.switch_interval < 0:
            raise ValueError("switch_interval must be a nonnegative integer; zero disables shifts")
        if self.initial_wind not in ("north", "south"):
            raise ValueError("initial_wind must be 'north' or 'south'")
        if type(self.observe_wind) is not bool:
            raise TypeError("observe_wind must be bool")
        if self.actions != ((0, -1), (-1, 0), (0, 1), (1, 0)):
            raise ValueError("WindShift requires left, up, right, down action order")


class WindShiftState(NamedTuple):
    position: jax.Array
    goal: jax.Array
    steps: jax.Array
    global_steps: jax.Array


class WindShiftEnv(GridEnv):
    """Use the existing hidden-terrain observation and a persistent global clock."""

    supports_exact_risk = False
    requires_global_step = True

    def __init__(self, spec: WindShiftSpec):
        super().__init__(spec)
        self.observation_codec = "raw" if spec.observe_wind else "grid_indices"
        self.observation_storage_dtype = jnp.float32 if spec.observe_wind else jnp.uint8

    @property
    def observation_shape(self) -> tuple[int]:
        cells = self.spec.shape[0] * self.spec.shape[1]
        return (cells + int(self.spec.observe_wind),)

    def wind_south(self, global_steps: jax.Array) -> jax.Array:
        phase = (global_steps // self.spec.switch_interval if self.spec.switch_interval
                 else jnp.zeros_like(global_steps))
        return jnp.logical_xor(self.spec.initial_wind == "south", (phase & 1) == 1)

    def observe(self, state: WindShiftState) -> jax.Array:
        grid = observe_grid(state, self.params)
        if not self.spec.observe_wind:
            return grid
        direction = self.wind_south(state.global_steps).astype(jnp.float32)
        return jnp.concatenate((grid, direction[None]), axis=0)

    def reset(self, key: jax.Array, *, global_step: int | jax.Array = 0) -> tuple[jax.Array, WindShiftState]:
        _, base = reset_grid(key, self.params)
        state = WindShiftState(base.position, base.goal, base.steps, jnp.asarray(global_step, jnp.int32))
        return self.observe(state), state

    def evaluation_resets(self, *, global_step: int | jax.Array = 0) -> tuple[jax.Array, WindShiftState]:
        _, base = super().evaluation_resets()
        state = WindShiftState(base.position, base.goal, base.steps,
                               jnp.broadcast_to(jnp.asarray(global_step, jnp.int32), base.steps.shape))
        return jax.vmap(self.observe)(state), state

    def with_global_step(self, observation: jax.Array, state: WindShiftState,
                         global_step: int | jax.Array) -> tuple[jax.Array, WindShiftState]:
        del observation
        updated = state._replace(global_steps=jnp.asarray(global_step, jnp.int32))
        return self.observe(updated), updated

    def step(self, key: jax.Array, state: WindShiftState, action: jax.Array):
        intended = self.params.transitions[state.position, action]
        intended_terminal = self.params.pit_mask[intended] | (intended == state.goal)
        south = self.wind_south(state.global_steps)
        wind_action = jnp.where(south, 3, 1)
        blown = self.params.transitions[intended, wind_action]
        gust_probability = jnp.asarray(self.spec.wind_probability, dtype=jnp.float32)
        gust = jax.random.bernoulli(key, gust_probability) & ~intended_terminal
        position = jnp.where(gust, blown, intended)
        next_state = WindShiftState(position, state.goal, state.steps + 1, state.global_steps + 1)
        pit = self.params.pit_mask[position]
        goal = position == state.goal
        terminated = pit | goal
        truncated = (next_state.steps >= self.params.max_steps) & ~terminated
        reward = jnp.where(pit, self.params.pit_reward,
                           jnp.where(goal, self.params.goal_reward, self.params.step_reward))
        info = {"wind_south": south, "gust_applied": gust}
        return self.observe(next_state), next_state, reward, terminated, truncated, info


def make_wind_shift_env(*, switch_interval: int, wind_probability: float = 0.5,
                        layout: str = "descending", initial_wind: str = "north",
                        observe_wind: bool = False) -> WindShiftEnv:
    """Create one of two vertically mirrored corner-to-corner layouts."""
    if layout not in ("descending", "ascending"):
        raise ValueError("layout must be 'descending' or 'ascending'")
    descending = layout == "descending"
    start = (0, 0) if descending else (4, 0)
    goal = (4, 8) if descending else (0, 8)
    walls = ((0, 7), (0, 8), (4, 0), (4, 1)) if descending else (
        (4, 7), (4, 8), (0, 0), (0, 1))
    spec = WindShiftSpec(
        name="WindShift", shape=(5, 9), starts=(start,), goals=(goal,),
        pits=tuple((row, col) for row in (0, 4) for col in range(2, 7)),
        walls=walls, max_steps=100, wind_probability=wind_probability,
        switch_interval=switch_interval, initial_wind=initial_wind,
        observe_wind=observe_wind,
    )
    return WindShiftEnv(spec)
