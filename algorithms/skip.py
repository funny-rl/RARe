"""Shared TempoRL/RARe skip targets, loss and explicit Adam learning state."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from buffers import SkipBatch
from networks import SkipParams, apply_skip, init_skip_params
from optimizers import AdamState, adam_step, init_adam


class SkipState(NamedTuple):
    """Skip parameters and complete Adam state; one instance per experiment.

    UTE uses the same container with a leading ensemble axis in parameters
    and moments, and one joint Adam counter. There is no skip target network.
    """

    params: SkipParams
    optimizer_state: AdamState


def init_skip_state(
    key: jax.Array, obs_dim: int, num_actions: int, max_skip: int, hidden_dim: int = 64
) -> SkipState:
    """Initialize the common single skip network and zero Adam for TempoRL/RARe.

    Split/advance the caller's model_init stream before calling. Dimensions
    are static under jit. Use an outer vmap for independent experiment states.
    """
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
    """Return detached (B,) targets G + d*V for TempoRL, UTE and RARe.

    All inputs must share nonempty shape (B,). Pass SkipBatch.returns and
    SkipBatch.discounts directly: d is gamma**executed_steps for a continuing
    record, zero for termination or the agreed absorbing episode time limit.
    Discounts must be finite and lie in [0,1]. Do not recompute d from the
    requested skip label, which may extend beyond observed episode completion.
    A zero discount selects zero bootstrap even for a NaN/Inf endpoint value.
    Thus a repeat ending at step 2 gives identical targets for labels 2..6.
    Bootstrap construction and gathering skip predictions remain separate.
    """
    returns = jnp.asarray(returns, dtype=jnp.float32)
    discounts = jnp.asarray(discounts, dtype=jnp.float32)
    bootstrap_values = jnp.asarray(bootstrap_values, dtype=jnp.float32)
    if (returns.ndim != 1 or returns.shape[0] < 1
            or discounts.shape != returns.shape or bootstrap_values.shape != returns.shape):
        raise ValueError("returns, discounts and bootstrap_values must share nonempty shape (B,)")
    bootstrap = jnp.where(discounts == 0, 0.0, discounts * bootstrap_values)
    return jax.lax.stop_gradient(returns + bootstrap)


def skip_loss(params: SkipParams, batch: SkipBatch, targets: jax.Array) -> jax.Array:
    """Mean Smooth L1 (beta=1) of observed skip labels, matching TempoRL/RARe.

    Gather output j-1 for batch.skips, then compare to detached targets (B,).
    Actions must lie in [0,num_actions), skips in [1,max_skip]. Targets are
    inputs so greedy and separate evaluation bootstrap can share this loss.
    Only the skip parameters receive gradients, never the evaluation critic.
    """
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
    """Compose replay targets, compute skip loss/gradients and take one Adam step.

    TempoRL supplies greedy_bootstrap from the updated base critic. Legacy
    separate_RARe supplies expected_bootstrap after its evaluation critic
    update/target synchronization. The caller owns that update order and
    replay sampling. Return the new state and pre-update loss. No RNG or
    target synchronization occurs here; save both state fields to resume.
    """
    targets = skip_targets(batch.returns, batch.discounts, bootstrap_values)
    loss, gradients = jax.value_and_grad(skip_loss)(state.params, batch, targets)
    params, optimizer_state = adam_step(state.params, gradients, state.optimizer_state, learning_rate)
    return SkipState(params, optimizer_state), loss
