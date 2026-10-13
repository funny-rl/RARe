"""Common contract for functional JAX environment adapters."""

from typing import Any, Protocol

import jax


class FunctionalEnv(Protocol):
    """Explicit-key/state adapter; step returns the real successor before reset.

    States and info must be JAX pytrees. Adapters for auto-reset backends
    must expose the final observation/state here, leaving reset to the runner.
    Plain Gymnasium requires a separate host collector.
    """

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
