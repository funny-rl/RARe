"""DDPG-based TAAC: action proposal, repeat/switch gate and compare-through targets."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from rare.networks.continuous import apply_actor, apply_critic, init_actor_params, init_critic_params

from .adam import adam_step, init_adam
from .targets import update_target_params


class TAACLearner(NamedTuple):
    actor: object
    target_actor: object
    actor_optimizer: object
    critic: object
    target_critic: object
    critic_optimizer: object
    log_temperature: jax.Array
    temperature_optimizer: object


def tempered_logits(q_values, log_temperature):
    """Keep TAAC's two-action softmax finite as learned temperature approaches zero."""
    centered = q_values - jnp.max(q_values, axis=-1, keepdims=True)
    temperature = jnp.maximum(jnp.exp(log_temperature), jnp.finfo(q_values.dtype).tiny)
    return centered / temperature


def init_taac(key, obs_dim, action_dim, hidden_dim, temperature):
    actor_key, critic_key = jax.random.split(key)
    actor = init_actor_params(actor_key, obs_dim, action_dim, hidden_dim,
                              use_previous_action=True)
    critic = init_critic_params(critic_key, obs_dim, action_dim, hidden_dim)
    log_temperature = jnp.log(jnp.asarray(temperature, jnp.float32))
    return TAACLearner(actor, actor, init_adam(actor), critic, critic,
                       init_adam(critic), log_temperature, init_adam(log_temperature))


def select_taac_action(learner, observation, previous_action, has_previous,
                       key, low, high, sigma, *, deterministic=False):
    proposal_key, gate_key = jax.random.split(key)
    candidate = apply_actor(learner.actor, observation, low, high, previous_action)
    if not deterministic:
        candidate = jnp.clip(candidate + sigma * jax.random.normal(
            proposal_key, candidate.shape), low, high)
    stay_q = apply_critic(learner.critic, observation, previous_action)
    switch_q = apply_critic(learner.critic, observation, candidate)
    if deterministic:
        switched = switch_q > stay_q
    else:
        logits = tempered_logits(jnp.stack((stay_q, switch_q)), learner.log_temperature)
        switched = jax.random.categorical(gate_key, logits) == 1
    switched = ~has_previous | switched
    return jnp.where(switched, candidate, previous_action), switched


def compare_through_targets(learner, batch, gamma, low, high):
    """Stay-through recursion on chronological replay windows.

    A time-limit truncation ends the return like a terminal state, as in the
    other continuous learners.
    """
    next_actions = apply_actor(learner.target_actor, batch.next_observations,
                               low, high, batch.actions)
    stay_q = apply_critic(learner.target_critic, batch.next_observations, batch.actions)
    switch_q = apply_critic(learner.target_critic, batch.next_observations, next_actions)
    target_q = jnp.maximum(stay_q, switch_q)
    cut = batch.switched | (switch_q > stay_q)
    ended = batch.terminated | batch.truncated
    future = target_q[:, -1]
    reverse_targets = []
    for t in reversed(range(batch.rewards.shape[1])):
        future = batch.rewards[:, t] + gamma * jnp.where(
            ended[:, t], 0., jnp.where(cut[:, t], target_q[:, t], future))
        reverse_targets.append(future)
    return jax.lax.stop_gradient(jnp.stack(reverse_targets[::-1], axis=1))


def taac_train_step(learner, batch, gamma, learning_rate, low, high, target_entropy_delta,
                    *, tau=.005):
    targets = compare_through_targets(learner, batch, gamma, low, high)
    observations = batch.observations.reshape((-1, batch.observations.shape[-1]))
    actions = batch.actions.reshape((-1, batch.actions.shape[-1]))
    previous = batch.previous_actions.reshape((-1, batch.previous_actions.shape[-1]))
    targets = targets.reshape((-1,))

    def critic_loss(params):
        errors = jnp.abs(apply_critic(params, observations, actions) - targets)
        quadratic = jnp.minimum(errors, 1.)
        return jnp.mean(.5 * quadratic**2 + errors - quadratic)

    loss, gradients = jax.value_and_grad(critic_loss)(learner.critic)
    critic, critic_optimizer = adam_step(learner.critic, gradients,
                                         learner.critic_optimizer, learning_rate)
    stay_q = jax.lax.stop_gradient(apply_critic(critic, observations, previous))

    def actor_loss(params):
        proposals = apply_actor(params, observations, low, high, previous)
        switch_q = apply_critic(critic, observations, proposals)
        gate = jax.nn.softmax(tempered_logits(
            jnp.stack((stay_q, switch_q), -1), learner.log_temperature), -1)[..., 1]
        return -jnp.mean(jax.lax.stop_gradient(gate) * switch_q)

    policy_loss, gradients = jax.value_and_grad(actor_loss)(learner.actor)
    actor, actor_optimizer = adam_step(learner.actor, gradients,
                                       learner.actor_optimizer, learning_rate)
    proposals = apply_actor(actor, observations, low, high, previous)
    q_values = jnp.stack((stay_q, jax.lax.stop_gradient(apply_critic(
        critic, observations, proposals))), -1)
    probabilities = jax.nn.softmax(tempered_logits(q_values, learner.log_temperature), -1)
    entropy = -jnp.mean(jnp.sum(probabilities * jnp.log(jnp.maximum(probabilities, 1e-10)), -1))
    delta = jnp.asarray(target_entropy_delta, jnp.float32)
    target_entropy = -delta * jnp.log(delta) - (1. - delta) * jnp.log1p(-delta)
    log_temperature, temperature_optimizer = adam_step(
        learner.log_temperature, jax.lax.stop_gradient(entropy - target_entropy),
        learner.temperature_optimizer, learning_rate)
    target_actor = update_target_params(learner.target_actor, actor, tau)
    target_critic = update_target_params(learner.target_critic, critic, tau)
    return TAACLearner(actor, target_actor, actor_optimizer, critic, target_critic,
                       critic_optimizer, log_temperature, temperature_optimizer), loss, policy_loss, entropy
