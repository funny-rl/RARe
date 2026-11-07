"""Optax Adam with explicit state."""

from typing import Any

import jax
import jax.numpy as jnp
import optax


AdamState = optax.ScaleByAdamState
_ADAM = optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8, eps_root=0.0)


def init_adam(params: Any) -> AdamState:
    """Initialize a zero step count and zero moments. A UTE ensemble shares one count."""
    return _ADAM.init(params)


def adam_step(
    params: Any, gradients: Any, state: AdamState, learning_rate: jax.Array
) -> tuple[Any, AdamState]:
    """Take one Adam step with beta=(0.9, 0.999) and epsilon=1e-8 outside the square root."""
    learning_rate = jnp.asarray(learning_rate, dtype=jnp.float32)
    if learning_rate.shape != ():
        raise ValueError("learning_rate must be scalar")
    updates, state = _ADAM.update(gradients, state, params)
    updates = jax.tree.map(lambda update: -learning_rate * update, updates)
    return optax.apply_updates(params, updates), state
