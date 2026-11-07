"""Separate RARe evaluation critic using its own epsilon-greedy policy."""

import jax
import jax.numpy as jnp

from rare.networks import apply_q

from .adam import adam_step
from .bootstrap import expected_bootstrap, greedy_bootstrap
from .ddqn import DDQNState
from .policies import epsilon_greedy_probs
from .targets import update_target_params


def evaluation_loss(online_params, target_params, batch, gamma, epsilon, target_mode="expected"):
    # The auxiliary critic retains its own greedy action identities. Expected
    # backup uses the configured target-policy epsilon; greedy ignores it.
    if target_mode not in ("greedy", "expected"):
        raise ValueError("target_mode must be greedy or expected")
    online_next = apply_q(online_params, batch.next_observations)
    target_next = apply_q(target_params, batch.next_observations)
    if target_mode == "expected":
        probabilities = epsilon_greedy_probs(online_next, epsilon)
        values = expected_bootstrap(probabilities, target_next)
    else:
        values = greedy_bootstrap(online_next, target_next)
    targets = jax.lax.stop_gradient(batch.rewards + jnp.where(
        batch.terminated | batch.truncated, 0., gamma * values,
    ))
    predictions = jnp.take_along_axis(apply_q(online_params, batch.observations), batch.actions[:, None], axis=-1)[:, 0]
    error = jnp.abs(predictions - targets)
    quadratic = jnp.minimum(error, 1.)
    return jnp.mean(.5 * quadratic ** 2 + error - quadratic)


def evaluation_train_step(state, batch, gamma, epsilon, learning_rate, *, tau=0.005,
                          target_mode="expected"):
    """Update the separate endpoint critic before RARe skip targets."""
    loss, gradients = jax.value_and_grad(lambda params: evaluation_loss(
        params, state.target_params, batch, gamma, epsilon, target_mode,
    ))(state.online_params)
    online, optimizer = adam_step(state.online_params, gradients, state.optimizer_state, learning_rate)
    target = update_target_params(state.target_params, online, tau)
    return DDQNState(online, target, optimizer), loss
