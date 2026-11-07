"""Unmodified Brax HalfCheetah on the spring physics backend."""

from dataclasses import dataclass
from importlib.metadata import version
from typing import NamedTuple

import jax
import jax.numpy as jnp


BACKEND = "spring"


@dataclass(frozen=True)
class BraxSpec:
    max_steps: int = 1000

    def __post_init__(self):
        if type(self.max_steps) is not int or self.max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")


class BraxState(NamedTuple):
    brax: object
    steps: jax.Array


class HalfCheetahEnv:
    backend = f"brax_{BACKEND}"
    supports_exact_risk = False
    requires_global_step = False
    observation_storage_dtype = jnp.float32
    observation_codec = "raw"

    def __init__(self, spec: BraxSpec = BraxSpec()):
        from brax import envs

        self.spec = spec
        self.brax = envs.get_environment("halfcheetah", backend=BACKEND)
        self.software_versions = {"brax": version("brax"), "backend": BACKEND}

    @property
    def name(self) -> str:
        return f"Brax/halfcheetah-{BACKEND}-v0"

    @property
    def observation_shape(self) -> tuple[int, ...]:
        return (self.brax.observation_size,)

    @property
    def action_shape(self) -> tuple[int, ...]:
        return (self.brax.action_size,)

    @property
    def action_low(self) -> jax.Array:
        return jnp.asarray(self.brax.sys.actuator.ctrl_range[:, 0], jnp.float32)

    @property
    def action_high(self) -> jax.Array:
        return jnp.asarray(self.brax.sys.actuator.ctrl_range[:, 1], jnp.float32)

    def reset(self, key: jax.Array, *, global_step: int | jax.Array = 0):
        del global_step
        state = self.brax.reset(key)
        return state.obs, BraxState(state, jnp.int32(0))

    def with_global_step(self, observation, state, global_step):
        del global_step
        return observation, state

    def step(self, key: jax.Array, state: BraxState, action: jax.Array):
        del key
        action = jnp.clip(jnp.asarray(action, jnp.float32), self.action_low, self.action_high)
        next_brax = self.brax.step(state.brax, action)
        steps = state.steps + 1
        return (next_brax.obs, BraxState(next_brax, steps), next_brax.reward,
                next_brax.done.astype(bool), steps >= self.spec.max_steps, {})


def make_halfcheetah_env(**settings) -> HalfCheetahEnv:
    return HalfCheetahEnv(BraxSpec(**settings))
