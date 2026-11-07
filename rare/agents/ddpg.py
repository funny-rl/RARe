"""DDPG-backed continuous TempoRL, UTE and RARe learning functions."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from rare.networks.continuous import (apply_actor, apply_continuous_skip, apply_critic,
                                      init_actor_params, init_continuous_skip_params,
                                      init_critic_params)

from .adam import adam_step, init_adam
from .targets import update_target_params
from .ute import sample_ute_masks


class DDPGLearner(NamedTuple):
    actor: object
    target_actor: object
    actor_optimizer: object
    critic: object
    target_critic: object
    critic_optimizer: object


class CriticLearner(NamedTuple):
    online: object
    target: object
    optimizer: object


class ContinuousSkipLearner(NamedTuple):
    params: object
    optimizer: object


def init_ddpg(key, obs_dim, action_dim, hidden_dim):
    actor_key, critic_key = jax.random.split(key)
    actor = init_actor_params(actor_key, obs_dim, action_dim, hidden_dim)
    critic = init_critic_params(critic_key, obs_dim, action_dim, hidden_dim)
    return DDPGLearner(actor, actor, init_adam(actor), critic, critic, init_adam(critic))


def init_evaluation_critic(key, obs_dim, action_dim, hidden_dim):
    critic = init_critic_params(key, obs_dim, action_dim, hidden_dim)
    return CriticLearner(critic, critic, init_adam(critic))


def init_continuous_skip(key, obs_dim, action_dim, max_skip, hidden_dim, ensemble_size=1):
    if ensemble_size == 1:
        params = init_continuous_skip_params(key, obs_dim, action_dim, max_skip, hidden_dim)
    else:
        keys = jax.random.split(key, ensemble_size)
        params = jax.vmap(lambda k: init_continuous_skip_params(
            k, obs_dim, action_dim, max_skip, hidden_dim))(keys)
    return ContinuousSkipLearner(params, init_adam(params))


def ddpg_train_step(learner, batch, gamma, learning_rate, low, high, *, tau=.005):
    next_actions = apply_actor(learner.target_actor, batch.next_observations, low, high)
    next_values = apply_critic(learner.target_critic, batch.next_observations, next_actions)
    targets = jax.lax.stop_gradient(batch.rewards + jnp.where(
        batch.terminated | batch.truncated, 0., gamma * next_values))

    def critic_loss(params):
        errors = jnp.abs(apply_critic(params, batch.observations, batch.actions) - targets)
        quadratic = jnp.minimum(errors, 1.)
        return jnp.mean(.5 * quadratic**2 + errors - quadratic)

    loss, gradients = jax.value_and_grad(critic_loss)(learner.critic)
    critic, critic_optimizer = adam_step(learner.critic, gradients, learner.critic_optimizer, learning_rate)

    def actor_loss(params):
        actions = apply_actor(params, batch.observations, low, high)
        return -jnp.mean(apply_critic(critic, batch.observations, actions))

    policy_loss, gradients = jax.value_and_grad(actor_loss)(learner.actor)
    actor, actor_optimizer = adam_step(learner.actor, gradients, learner.actor_optimizer, learning_rate)
    target_actor = update_target_params(learner.target_actor, actor, tau)
    target_critic = update_target_params(learner.target_critic, critic, tau)
    return DDPGLearner(actor, target_actor, actor_optimizer, critic, target_critic,
                       critic_optimizer), loss, policy_loss


def gaussian_expected_q(key, actor, critic, observations, low, high, sigma, samples):
    """Monte Carlo E[Q(s, clip(mu_target(s)+N(0,sigma), bounds))]."""
    mean_actions = apply_actor(actor, observations, low, high)
    noise = jax.random.normal(key, (samples, *mean_actions.shape), jnp.float32) * sigma
    actions = jnp.clip(mean_actions[None] + noise, low, high)
    expanded_states = jnp.broadcast_to(observations[None],
                                       (samples, *observations.shape))
    return jnp.mean(apply_critic(critic, expanded_states, actions), axis=0)


def evaluation_critic_train_step(learner, actor, batch, key, gamma, learning_rate, low, high,
                                 sigma, samples, *, tau=.005):
    expected = gaussian_expected_q(key, actor, learner.target, batch.next_observations,
                                   low, high, sigma, samples)
    targets = jax.lax.stop_gradient(batch.rewards + jnp.where(
        batch.terminated | batch.truncated, 0., gamma * expected))

    def loss_fn(params):
        errors = jnp.abs(apply_critic(params, batch.observations, batch.actions) - targets)
        quadratic = jnp.minimum(errors, 1.)
        return jnp.mean(.5 * quadratic**2 + errors - quadratic)

    loss, gradients = jax.value_and_grad(loss_fn)(learner.online)
    online, optimizer = adam_step(learner.online, gradients, learner.optimizer, learning_rate)
    target = update_target_params(learner.target, online, tau)
    return CriticLearner(online, target, optimizer), loss


def continuous_skip_train_step(learner, batch, bootstrap, learning_rate,
                               *, mask_key=None, ensemble_size=1, bootstrap_probability=.5):
    targets = jax.lax.stop_gradient(batch.returns + jnp.where(
        batch.discounts == 0, 0., batch.discounts * bootstrap))
    if ensemble_size == 1:
        masks = None
    else:
        masks = sample_ute_masks(mask_key, ensemble_size=ensemble_size,
                                 batch_size=batch.returns.shape[0], probability=bootstrap_probability)

    def loss_fn(params):
        values = (apply_continuous_skip(params, batch.observations, batch.actions)
                  if ensemble_size == 1 else
                  jax.vmap(apply_continuous_skip, in_axes=(0, None, None))(
                      params, batch.observations, batch.actions))
        predictions = jnp.take_along_axis(values, (batch.skips - 1)[..., None], axis=-1)[..., 0] if masks is None else jnp.take_along_axis(
            values, (batch.skips - 1)[None, :, None], axis=-1)[..., 0]
        errors = jnp.abs(predictions - targets)
        quadratic = jnp.minimum(errors, 1.)
        huber = .5 * quadratic**2 + errors - quadratic
        if masks is None:
            return jnp.mean(huber)
        counts = jnp.sum(masks, axis=1)
        return jnp.mean(jnp.sum(jnp.where(masks, huber, 0.), axis=1) / jnp.maximum(counts, 1))

    loss, gradients = jax.value_and_grad(loss_fn)(learner.params)
    params, optimizer = adam_step(learner.params, gradients, learner.optimizer, learning_rate)
    return ContinuousSkipLearner(params, optimizer), loss
