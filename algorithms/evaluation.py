"""Separate RARe evaluation critic using its own epsilon-greedy policy."""

import jax
import jax.numpy as jnp

from networks import apply_q
from optimizers import adam_step
from policies import epsilon_greedy_probs
from .bootstrap import expected_bootstrap
from .ddqn import DDQNState
from .targets import update_target_params


def evaluation_loss(online_params, target_params, batch, gamma, epsilon):
    # Separate RARe retains its own greedy policy; only its epsilon schedule
    # is tied to the behavior schedule.
    probabilities = epsilon_greedy_probs(apply_q(online_params, batch.next_observations), epsilon)
    values = expected_bootstrap(probabilities, apply_q(target_params, batch.next_observations))
    targets = jax.lax.stop_gradient(batch.rewards + jnp.where(
        batch.terminated | batch.truncated, 0., gamma * values,
    ))
    predictions = jnp.take_along_axis(apply_q(online_params, batch.observations), batch.actions[:, None], axis=-1)[:, 0]
    error = jnp.abs(predictions - targets)
    quadratic = jnp.minimum(error, 1.)
    return jnp.mean(.5 * quadratic ** 2 + error - quadratic)


def evaluation_train_step(state, batch, gamma, epsilon, learning_rate, training_steps, **target_settings):
    """Update the separate evaluation critic before RARe skip targets."""
    loss, gradients = jax.value_and_grad(evaluation_loss)(
        state.online_params, state.target_params, batch, gamma, epsilon,
    )
    online, optimizer = adam_step(state.online_params, gradients, state.optimizer_state, learning_rate)
    target = update_target_params(state.target_params, online, training_steps, **target_settings)
    return DDQNState(online, target, optimizer), loss
