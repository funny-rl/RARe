"""Common contract for functional JAX environment adapters."""

from typing import Any, Protocol

import jax


class FunctionalEnv(Protocol):
    """Explicit key and state interface; step returns the real successor before any reset."""

    @property
    def observation_shape(self) -> tuple[int, ...]: ...

    @property
    def num_actions(self) -> int: ...

    def reset(self, key: jax.Array, *, global_step: int | jax.Array = 0) -> tuple[jax.Array, Any]: ...

    def with_global_step(self, observation: jax.Array, state: Any,
                         global_step: int | jax.Array) -> tuple[jax.Array, Any]: ...

    def step(self, key: jax.Array, state: Any, action: jax.Array) -> tuple[
        jax.Array, Any, jax.Array, jax.Array, jax.Array, dict
    ]: ...


class ContinuousFunctionalEnv(Protocol):
    """Same explicit transition semantics, with bounded vector actions."""

    @property
    def observation_shape(self) -> tuple[int, ...]: ...

    @property
    def action_shape(self) -> tuple[int, ...]: ...

    @property
    def action_low(self) -> jax.Array: ...

    @property
    def action_high(self) -> jax.Array: ...

    def reset(self, key: jax.Array, *, global_step: int | jax.Array = 0) -> tuple[jax.Array, Any]: ...

    def with_global_step(self, observation: jax.Array, state: Any,
                         global_step: int | jax.Array) -> tuple[jax.Array, Any]: ...

    def step(self, key: jax.Array, state: Any, action: jax.Array) -> tuple[
        jax.Array, Any, jax.Array, jax.Array, jax.Array, dict
    ]: ...
