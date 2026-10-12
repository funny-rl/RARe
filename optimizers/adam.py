"""Optax Adam with the legacy Torch defaults and explicit resumable state."""

from typing import Any

import jax
import jax.numpy as jnp
import optax


AdamState = optax.ScaleByAdamState
_ADAM = optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8, eps_root=0.0)


def init_adam(params: Any) -> AdamState:
    """Initialize a scalar int32 count and zero first/second-moment pytrees.

    Each optimizer owns one state, including an independent state per seed.
    Use vmap(init_adam) on seed-batched parameters, rather than creating one
    shared counter. A UTE ensemble can have one optimizer with member-batched
    moments, matching its legacy joint optimizer. Static pytree metadata is
    preserved and is not trainable.
    """
    return _ADAM.init(params)


def adam_step(
    params: Any, gradients: Any, state: AdamState, learning_rate: jax.Array
) -> tuple[Any, AdamState]:
    """Return updated parameters/state without modifying the caller's inputs.

    Match default torch.optim.Adam: beta=(0.9, 0.999), epsilon=1e-8 outside
    the square root, bias correction, no weight decay or AMSGrad. Provide a
    finite nonnegative scalar learning rate each step to support schedules.
    Parameters/gradients/moments must have the same pytree structure and
    shapes. Save params and the entire AdamState to resume the next update;
    record the learning-rate schedule separately. No RNG is consumed.
    """
    learning_rate = jnp.asarray(learning_rate, dtype=jnp.float32)
    if learning_rate.shape != ():
        raise ValueError("learning_rate must be scalar")
    updates, state = _ADAM.update(gradients, state, params)
    updates = jax.tree.map(lambda update: -learning_rate * update, updates)
    return optax.apply_updates(params, updates), state
