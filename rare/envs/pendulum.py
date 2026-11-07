"""JAX Pendulum-v1 with Gymnasium's dense reward."""

from dataclasses import dataclass
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp


class PendulumParams(NamedTuple):
    """Array-only parameters so equal-shape experiments can be vmapped."""

    max_speed: jax.Array
    max_torque: jax.Array
    dt: jax.Array
    gravity: jax.Array
    mass: jax.Array
    length: jax.Array
    max_steps: jax.Array


class PendulumState(NamedTuple):
    theta: jax.Array
    theta_dot: jax.Array
    steps: jax.Array


@dataclass(frozen=True)
class PendulumSpec:
    """Host configuration."""

    max_steps: int = 200
    max_speed: float = 8.0
    max_torque: float = 2.0
    dt: float = 0.05
    gravity: float = 10.0
    mass: float = 1.0
    length: float = 1.0

    def __post_init__(self) -> None:
        if type(self.max_steps) is not int or self.max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        for name in ("max_speed", "max_torque", "dt", "gravity", "mass", "length"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")


def build_params(spec: PendulumSpec) -> PendulumParams:
    return PendulumParams(
        max_speed=jnp.asarray(spec.max_speed, dtype=jnp.float32),
        max_torque=jnp.asarray(spec.max_torque, dtype=jnp.float32),
        dt=jnp.asarray(spec.dt, dtype=jnp.float32),
        gravity=jnp.asarray(spec.gravity, dtype=jnp.float32),
        mass=jnp.asarray(spec.mass, dtype=jnp.float32),
        length=jnp.asarray(spec.length, dtype=jnp.float32),
        max_steps=jnp.asarray(spec.max_steps, dtype=jnp.int32),
    )


def angle_normalize(theta: jax.Array) -> jax.Array:
    """Match Gymnasium's [-pi, pi) normalization at the reward boundary."""
    return (theta + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def observe(state: PendulumState) -> jax.Array:
    return jnp.stack((jnp.cos(state.theta), jnp.sin(state.theta), state.theta_dot))


def reset(key: jax.Array, params: PendulumParams) -> tuple[jax.Array, PendulumState]:
    """Sample independent state coordinates; no hidden or global RNG state."""
    sample = jax.random.uniform(key, shape=(2,), dtype=jnp.float32)
    state = PendulumState(
        theta=(2.0 * sample[0] - 1.0) * jnp.pi,
        theta_dot=2.0 * sample[1] - 1.0,
        steps=jnp.asarray(0, dtype=jnp.int32),
    )
    return observe(state), state


def step(
    key: jax.Array, state: PendulumState, action: jax.Array, params: PendulumParams
) -> tuple[jax.Array, PendulumState, jax.Array, jax.Array, jax.Array, dict]:
    """Return the real successor before reset, with a separate time-limit flag."""
    del key
    torque = jnp.clip(jnp.asarray(action, dtype=jnp.float32).reshape(()),
                      -params.max_torque, params.max_torque)
    reward = -(
        angle_normalize(state.theta) ** 2
        + 0.1 * state.theta_dot**2
        + 0.001 * torque**2
    )
    acceleration = (
        3.0 * params.gravity / (2.0 * params.length) * jnp.sin(state.theta)
        + 3.0 / (params.mass * params.length**2) * torque
    )
    next_dot = jnp.clip(state.theta_dot + acceleration * params.dt,
                        -params.max_speed, params.max_speed)
    next_state = PendulumState(
        theta=state.theta + next_dot * params.dt,
        theta_dot=next_dot,
        steps=state.steps + 1,
    )
    terminated = jnp.asarray(False)
    truncated = next_state.steps >= params.max_steps
    return observe(next_state), next_state, reward, terminated, truncated, {}


class PendulumEnv:
    """Bind immutable settings to pure reset/step functions for jit/vmap/scan."""

    backend = "rare_jax"
    supports_exact_risk = False
    requires_global_step = False
    observation_storage_dtype = jnp.float32
    observation_codec = "raw"

    def __init__(self, spec: PendulumSpec = PendulumSpec()):
        self.spec = spec
        self.params = build_params(spec)

    @property
    def name(self) -> str:
        return "Pendulum-v1"

    @property
    def observation_shape(self) -> tuple[int, ...]:
        return (3,)

    @property
    def action_shape(self) -> tuple[int, ...]:
        return (1,)

    @property
    def action_low(self) -> jax.Array:
        return jnp.reshape(-self.params.max_torque, (1,))

    @property
    def action_high(self) -> jax.Array:
        return jnp.reshape(self.params.max_torque, (1,))

    def reset(self, key: jax.Array, *, global_step: int | jax.Array = 0) -> tuple[jax.Array, PendulumState]:
        del global_step
        return reset(key, self.params)

    def with_global_step(self, observation: jax.Array, state: PendulumState,
                         global_step: int | jax.Array) -> tuple[jax.Array, PendulumState]:
        del global_step
        return observation, state

    def step(self, key: jax.Array, state: PendulumState, action: jax.Array
             ) -> tuple[jax.Array, PendulumState, jax.Array, jax.Array, jax.Array, dict]:
        return step(key, state, action, self.params)


def make_pendulum_env(**settings) -> PendulumEnv:
    return PendulumEnv(PendulumSpec(**settings))
