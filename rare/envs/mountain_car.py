"""JAX port of Gymnasium's MountainCarContinuous-v0 transition rules."""

from dataclasses import dataclass
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp


class MountainCarParams(NamedTuple):
    min_position: jax.Array
    max_position: jax.Array
    max_speed: jax.Array
    action_limit: jax.Array
    goal_position: jax.Array
    goal_velocity: jax.Array
    power: jax.Array
    gravity: jax.Array
    reset_low: jax.Array
    reset_high: jax.Array
    max_steps: jax.Array


class MountainCarState(NamedTuple):
    position: jax.Array
    velocity: jax.Array
    steps: jax.Array


@dataclass(frozen=True)
class MountainCarSpec:
    min_position: float = -1.2
    max_position: float = 0.6
    max_speed: float = 0.07
    action_limit: float = 1.0
    goal_position: float = 0.45
    goal_velocity: float = 0.0
    power: float = 0.0015
    gravity: float = 0.0025
    reset_low: float = -0.6
    reset_high: float = -0.4
    max_steps: int = 999

    def __post_init__(self):
        if type(self.max_steps) is not int or self.max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        for name in ("min_position", "max_position", "max_speed", "action_limit",
                     "goal_position", "goal_velocity", "power", "gravity",
                     "reset_low", "reset_high"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not self.min_position < self.reset_low < self.reset_high < self.max_position:
            raise ValueError("reset bounds must lie strictly within position bounds")
        if not self.min_position < self.goal_position <= self.max_position:
            raise ValueError("goal_position must lie within position bounds")
        if self.max_speed <= 0 or self.action_limit <= 0 or self.power <= 0 or self.gravity < 0:
            raise ValueError("speed, action limit and power must be positive; gravity nonnegative")
        if not -self.max_speed <= self.goal_velocity <= self.max_speed:
            raise ValueError("goal_velocity must lie within velocity bounds")


def build_params(spec: MountainCarSpec) -> MountainCarParams:
    return MountainCarParams(*(
        jnp.asarray(getattr(spec, name), dtype=jnp.int32 if name == "max_steps" else jnp.float32)
        for name in MountainCarParams._fields
    ))


def observe(state: MountainCarState) -> jax.Array:
    return jnp.stack((state.position, state.velocity))


def reset(key: jax.Array, params: MountainCarParams) -> tuple[jax.Array, MountainCarState]:
    position = jax.random.uniform(key, (), minval=params.reset_low,
                                  maxval=params.reset_high, dtype=jnp.float32)
    state = MountainCarState(position, jnp.float32(0), jnp.int32(0))
    return observe(state), state


def step(key: jax.Array, state: MountainCarState, action: jax.Array,
         params: MountainCarParams):
    """Use clipped force for motion, but the original action for its cost."""
    del key
    raw_action = jnp.asarray(action, dtype=jnp.float32).reshape(())
    force = jnp.clip(raw_action, -params.action_limit, params.action_limit)
    velocity = jnp.clip(
        state.velocity + force * params.power - params.gravity * jnp.cos(3. * state.position),
        -params.max_speed, params.max_speed,
    )
    position = jnp.clip(state.position + velocity, params.min_position, params.max_position)
    velocity = jnp.where((position == params.min_position) & (velocity < 0), 0., velocity)
    next_state = MountainCarState(position, velocity, state.steps + 1)
    terminated = (position >= params.goal_position) & (velocity >= params.goal_velocity)
    truncated = next_state.steps >= params.max_steps
    reward = jnp.where(terminated, 100., 0.) - 0.1 * raw_action**2
    return observe(next_state), next_state, reward, terminated, truncated, {}


class MountainCarEnv:
    """Explicit-key adapter compatible with the existing continuous runner."""

    backend = "rare_jax"
    supports_exact_risk = False
    requires_global_step = False
    observation_storage_dtype = jnp.float32
    observation_codec = "raw"

    def __init__(self, spec: MountainCarSpec = MountainCarSpec()):
        self.spec = spec
        self.params = build_params(spec)

    @property
    def name(self) -> str:
        return "MountainCarContinuous-v0"

    @property
    def observation_shape(self) -> tuple[int, ...]:
        return (2,)

    @property
    def action_shape(self) -> tuple[int, ...]:
        return (1,)

    @property
    def action_low(self) -> jax.Array:
        return jnp.reshape(-self.params.action_limit, (1,))

    @property
    def action_high(self) -> jax.Array:
        return jnp.reshape(self.params.action_limit, (1,))

    def reset(self, key: jax.Array, *, global_step: int | jax.Array = 0
              ) -> tuple[jax.Array, MountainCarState]:
        del global_step
        return reset(key, self.params)

    def with_global_step(self, observation: jax.Array, state: MountainCarState,
                         global_step: int | jax.Array) -> tuple[jax.Array, MountainCarState]:
        del global_step
        return observation, state

    def step(self, key: jax.Array, state: MountainCarState, action: jax.Array):
        return step(key, state, action, self.params)


def make_mountain_car_env(**settings) -> MountainCarEnv:
    return MountainCarEnv(MountainCarSpec(**settings))
