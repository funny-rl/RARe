"""Shared TempoRL/RARe skip targets, loss and explicit Adam learning state."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from rare.networks import SkipParams, apply_skip, init_skip_params
from rare.replay.skip import SkipBatch

from .adam import AdamState, adam_step, init_adam


class SkipState(NamedTuple):
    """Skip parameters and Adam state. UTE adds a leading ensemble axis and shares one Adam count."""

    params: SkipParams
    optimizer_state: AdamState


def init_skip_state(
    key: jax.Array, obs_dim: int, num_actions: int, max_skip: int, hidden_dim: int = 64
) -> SkipState:
    """Initialize one skip network and its Adam state."""
    params = init_skip_params(key, obs_dim, num_actions, max_skip, hidden_dim)
    return SkipState(params, init_adam(params))


def _skip_labels(batch: SkipBatch) -> jax.Array:
    observations = jnp.asarray(batch.observations)
    actions = jnp.asarray(batch.actions)
    skips = jnp.asarray(batch.skips)
    if observations.ndim != 2 or observations.shape[0] < 1:
        raise ValueError("observations must have nonempty shape (B, obs_dim)")
    shape = (observations.shape[0],)
    if any(x.shape != shape or not jnp.issubdtype(x.dtype, jnp.integer) for x in (actions, skips)):
        raise ValueError("actions and skips must be integers with shape (B,)")
    return skips - 1


def _smooth_l1(predictions: jax.Array, targets: jax.Array) -> jax.Array:
    error = jnp.abs(predictions - jax.lax.stop_gradient(targets))
    quadratic = jnp.minimum(error, 1.0)
    return 0.5 * quadratic ** 2 + (error - quadratic)


def skip_targets(
    returns: jax.Array,
    discounts: jax.Array,
    bootstrap_values: jax.Array,
) -> jax.Array:
    """Return detached (B,) targets G + d * V.

    d is gamma**executed_steps, or zero at an episode end, so a repeat ending at step 2 gives
    the same target for labels 2 to 6. A zero discount ignores even a NaN endpoint value."""
    returns = jnp.asarray(returns, dtype=jnp.float32)
    discounts = jnp.asarray(discounts, dtype=jnp.float32)
    bootstrap_values = jnp.asarray(bootstrap_values, dtype=jnp.float32)
    if (returns.ndim != 1 or returns.shape[0] < 1
            or discounts.shape != returns.shape or bootstrap_values.shape != returns.shape):
        raise ValueError("returns, discounts and bootstrap_values must share nonempty shape (B,)")
    bootstrap = jnp.where(discounts == 0, 0.0, discounts * bootstrap_values)
    return jax.lax.stop_gradient(returns + bootstrap)


def skip_loss(params: SkipParams, batch: SkipBatch, targets: jax.Array) -> jax.Array:
    """Mean Smooth L1 (beta=1) between the output of each observed skip label and its target."""
    indices = _skip_labels(batch)
    targets = jnp.asarray(targets, dtype=jnp.float32)
    if targets.shape != indices.shape:
        raise ValueError("targets must have shape (B,)")
    values = apply_skip(params, batch.observations, batch.actions)
    predictions = jnp.take_along_axis(values, indices[:, None], axis=-1)[:, 0]
    return jnp.mean(_smooth_l1(predictions, targets))


def skip_train_step(
    state: SkipState, batch: SkipBatch, bootstrap_values: jax.Array, learning_rate: jax.Array
) -> tuple[SkipState, jax.Array]:
    """Take one Adam step on the skip network; return the new state and the pre-update loss."""
    targets = skip_targets(batch.returns, batch.discounts, bootstrap_values)
    loss, gradients = jax.value_and_grad(skip_loss)(state.params, batch, targets)
    params, optimizer_state = adam_step(state.params, gradients, state.optimizer_state, learning_rate)
    return SkipState(params, optimizer_state), loss
