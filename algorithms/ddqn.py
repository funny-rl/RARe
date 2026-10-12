"""Single-estimator DDQN objective and functional learning state/update."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from buffers import ReplayBatch
from networks import QParams, apply_q, init_q_params
from optimizers import AdamState, adam_step, init_adam

from .targets import update_target_params


class DDQNState(NamedTuple):
    """One experiment's online/target parameters and complete Adam state."""

    online_params: QParams
    target_params: QParams
    optimizer_state: AdamState


def init_ddqn_state(
    key: jax.Array, obs_dim: int, num_actions: int, hidden_dim: int = 64
) -> DDQNState:
    """Initialize online once, copy its values to the target, and zero Adam.

    Arrays are immutable, so initial target storage can share online values.
    Dimensions are static under jit. The caller splits its model_init key
    before calling. No independent random target network is initialized.
    Use an outer vmap to initialize each seed's parameters and Adam counter.
    """
    params = init_q_params(key, obs_dim, num_actions, hidden_dim)
    return DDQNState(params, params, init_adam(params))


def ddqn_targets(
    online_next_q: jax.Array,
    target_next_q: jax.Array,
    rewards: jax.Array,
    terminated: jax.Array,
    truncated: jax.Array,
    gamma: jax.Array,
) -> jax.Array:
    """Select the next action online and evaluate it with the target network.

    Q arrays have shape (B, A); rewards/end flags and returned targets have
    shape (B,). Either episode-end flag suppresses bootstrap, including the
    agreed max-step boundary. Gamma must be scalar and lie in [0, 1]. The
    returned targets are detached from differentiation.
    """
    online_next_q = jnp.asarray(online_next_q, dtype=jnp.float32)
    target_next_q = jnp.asarray(target_next_q, dtype=jnp.float32)
    rewards = jnp.asarray(rewards, dtype=jnp.float32)
    terminated = jnp.asarray(terminated, dtype=jnp.bool_)
    truncated = jnp.asarray(truncated, dtype=jnp.bool_)
    gamma = jnp.asarray(gamma, dtype=jnp.float32)
    if (online_next_q.ndim != 2 or min(online_next_q.shape) < 1
            or target_next_q.shape != online_next_q.shape):
        raise ValueError("next Q arrays must share nonempty shape (B, A)")
    batch_shape = (online_next_q.shape[0],)
    if any(x.shape != batch_shape for x in (rewards, terminated, truncated)):
        raise ValueError("rewards and episode-end flags must have shape (B,)")
    if gamma.shape != ():
        raise ValueError("gamma must be scalar")
    next_actions = jnp.argmax(online_next_q, axis=-1)
    next_values = jnp.take_along_axis(target_next_q, next_actions[:, None], axis=-1)[:, 0]
    # Mask by selection rather than multiplication so terminal NaNs cannot
    # contaminate a target that depends only on the observed reward.
    bootstrap = jnp.where(terminated | truncated, 0.0, gamma * next_values)
    return jax.lax.stop_gradient(rewards + bootstrap)


def ddqn_loss(
    online_params: QParams,
    target_params: QParams,
    batch: ReplayBatch,
    gamma: jax.Array,
) -> jax.Array:
    """Return scalar mean Smooth L1 loss (beta=1), matching legacy DDQN.

    This is the n_target=1 baseline objective. Inputs are one experiment's
    replay batch: observations (B, obs_dim), integer actions and scalar
    transition fields (B,). Action indices must lie in [0, num_actions).
    Differentiate with respect to online_params only. Use an outer vmap
    for independent experiments. Optimizer and target updates are separate.
    """
    observations = jnp.asarray(batch.observations, dtype=jnp.float32)
    next_observations = jnp.asarray(batch.next_observations, dtype=jnp.float32)
    actions = jnp.asarray(batch.actions)
    if observations.ndim != 2 or next_observations.shape != observations.shape:
        raise ValueError("observations must share shape (B, obs_dim)")
    if actions.shape != (observations.shape[0],) or not jnp.issubdtype(actions.dtype, jnp.integer):
        raise ValueError("actions must be integers with shape (B,)")
    targets = ddqn_targets(
        apply_q(online_params, next_observations),
        apply_q(target_params, next_observations),
        batch.rewards, batch.terminated, batch.truncated, gamma,
    )
    current_q = apply_q(online_params, observations)
    predictions = jnp.take_along_axis(current_q, actions[:, None], axis=-1)[:, 0]
    absolute_error = jnp.abs(predictions - targets)
    quadratic = jnp.minimum(absolute_error, 1.0)
    return jnp.mean(0.5 * quadratic ** 2 + (absolute_error - quadratic))


def ddqn_train_step(
    state: DDQNState,
    batch: ReplayBatch,
    gamma: jax.Array,
    learning_rate: jax.Array,
    training_steps: jax.Array,
    *,
    tau: jax.Array = 0.005,
    use_hard_update: bool = False,
    update_interval: int = 1000,
) -> tuple[DDQNState, jax.Array]:
    """Compute loss, update online with Adam, then synchronize target values.

    Return (new_state, pre-update_loss). The old target defines the TD target;
    the newly updated online parameters define synchronization. This function
    performs exactly one optimizer update and consumes no RNG. Sampling,
    warmup/budget guards and advancing the external training_steps counter
    belong to the collector. Save all DDQNState fields to resume learning;
    the collector also owns the interaction count, schedules, replay and RNG.
    Target scheduling matches update_target_params; its mode and interval
    are static under jit. Use an outer vmap over independent states/batches.
    """
    loss, gradients = jax.value_and_grad(ddqn_loss)(
        state.online_params, state.target_params, batch, gamma,
    )
    online_params, optimizer_state = adam_step(
        state.online_params, gradients, state.optimizer_state, learning_rate,
    )
    target_params = update_target_params(
        state.target_params, online_params, training_steps,
        tau=tau, use_hard_update=use_hard_update, update_interval=update_interval,
    )
    return DDQNState(online_params, target_params, optimizer_state), loss
