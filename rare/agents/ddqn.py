"""Single-estimator DDQN objective and functional learning state/update."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from rare.networks import QParams, apply_q, init_q_params
from rare.replay.replay import ReplayBatch

from .adam import AdamState, adam_step, init_adam
from .bootstrap import expected_bootstrap
from .policies import epsilon_greedy_probs
from .targets import update_target_params


class DDQNState(NamedTuple):
    """One experiment's online/target parameters and complete Adam state."""

    online_params: QParams
    target_params: QParams
    optimizer_state: AdamState


def init_ddqn_state(
    key: jax.Array, obs_dim: int, num_actions: int, hidden_dim: int = 64
) -> DDQNState:
    """Initialize the online network, copy it to the target, and zero Adam."""
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
    """Double Q targets: online selects, target evaluates, and an episode end stops the bootstrap."""
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
    target_mode: str = "greedy",
    epsilon: jax.Array = 0.,
) -> jax.Array:
    """Mean Huber loss (delta=1) of the action critic.

    target_mode="expected" gives the Expected SARSA target of the ablation in Table 5."""
    observations = jnp.asarray(batch.observations, dtype=jnp.float32)
    next_observations = jnp.asarray(batch.next_observations, dtype=jnp.float32)
    actions = jnp.asarray(batch.actions)
    if observations.ndim != 2 or next_observations.shape != observations.shape:
        raise ValueError("observations must share shape (B, obs_dim)")
    if actions.shape != (observations.shape[0],) or not jnp.issubdtype(actions.dtype, jnp.integer):
        raise ValueError("actions must be integers with shape (B,)")
    if target_mode not in ("greedy", "expected"):
        raise ValueError("target_mode must be greedy or expected")
    online_next = apply_q(online_params, next_observations)
    target_next = apply_q(target_params, next_observations)
    if target_mode == "greedy":
        targets = ddqn_targets(online_next, target_next, batch.rewards,
                               batch.terminated, batch.truncated, gamma)
    else:
        values = expected_bootstrap(epsilon_greedy_probs(online_next, epsilon), target_next)
        targets = jax.lax.stop_gradient(batch.rewards + jnp.where(
            batch.terminated | batch.truncated, 0., gamma * values))
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
    *,
    tau: jax.Array = 0.005,
    target_mode: str = "greedy",
    epsilon: jax.Array = 0.,
) -> tuple[DDQNState, jax.Array]:
    """Take one Adam step and update the target; return the new state and the pre-update loss."""
    loss, gradients = jax.value_and_grad(lambda params: ddqn_loss(
        params, state.target_params, batch, gamma, target_mode, epsilon,
    ))(state.online_params)
    online_params, optimizer_state = adam_step(
        state.online_params, gradients, state.optimizer_state, learning_rate,
    )
    target_params = update_target_params(state.target_params, online_params, tau)
    return DDQNState(online_params, target_params, optimizer_state), loss
