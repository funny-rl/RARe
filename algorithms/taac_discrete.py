"""Discrete TAAC with categorical proposals and multi-step compare-through."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from algorithms.targets import update_target_params
from networks import apply_q, init_q_params
from optimizers import adam_step, init_adam


class TAACDiscreteLearner(NamedTuple):
    online_params: object
    target_params: object
    optimizer_state: object
    actor_params: object
    target_actor_params: object
    actor_optimizer_state: object
    log_temperature: jax.Array
    temperature_optimizer_state: object
    log_actor_temperature: jax.Array
    actor_temperature_optimizer_state: object


def previous_features(previous_actions, num_actions):
    """One-hot previous action plus a validity bit; -1 marks episode start."""
    valid = previous_actions >= 0
    one_hot = jax.nn.one_hot(jnp.maximum(previous_actions, 0), num_actions)
    return jnp.concatenate((one_hot * valid[..., None], valid[..., None].astype(jnp.float32)), axis=-1)


def actor_logits(params, observations, previous_actions, num_actions):
    features = previous_features(previous_actions, num_actions)
    return apply_q(params, jnp.concatenate((observations, features), axis=-1))


def init_taac_discrete(key, obs_dim, num_actions, hidden_dim, temperature, actor_temperature):
    if num_actions < 2:
        raise ValueError("TAACDiscrete requires at least two actions")
    critic_key, actor_key = jax.random.split(key)
    critic = init_q_params(critic_key, obs_dim, num_actions, hidden_dim)
    actor = init_q_params(actor_key, obs_dim + num_actions + 1, num_actions, hidden_dim)
    log_temperature = jnp.log(jnp.asarray(temperature, jnp.float32))
    log_actor_temperature = jnp.log(jnp.asarray(actor_temperature, jnp.float32))
    return TAACDiscreteLearner(critic, critic, init_adam(critic), actor, actor,
                               init_adam(actor), log_temperature, init_adam(log_temperature),
                               log_actor_temperature, init_adam(log_actor_temperature))


def select_taac_action(learner, observation, previous_action, key, *, deterministic=False):
    """Sample an actual candidate first, then compare it with the last action."""
    num_actions = learner.online_params[-1].bias.shape[0]
    proposal_key, gate_key = jax.random.split(key)
    logits = actor_logits(learner.actor_params, observation, previous_action, num_actions)
    proposal = (jnp.argmax(logits) if deterministic else jax.random.categorical(proposal_key, logits)).astype(jnp.int32)
    q = apply_q(learner.online_params, observation)
    q_stay = q[jnp.maximum(previous_action, 0)]
    q_switch = q[proposal]
    if deterministic:
        switch = q_switch > q_stay
    else:
        gate_logits = jnp.stack((q_stay, q_switch)) / jnp.exp(learner.log_temperature)
        switch = jax.random.categorical(gate_key, gate_logits) == 1
    switch = (previous_action < 0) | switch
    return jnp.where(switch, proposal, previous_action).astype(jnp.int32), switch


def compare_through_targets(learner, batch, key, gamma):
    """Continue a replay segment only while the current policy matches it.

    Batch fields have (batch, horizon, ...). Each bootstrap action is sampled
    from the two-stage target policy at the actual successor state. Terminal
    transitions stop the recursion; a replay action mismatch cuts it off.
    """
    num_actions = learner.online_params[-1].bias.shape[0]
    horizon = batch.actions.shape[1]
    if horizon < 1:
        raise ValueError("compare-through horizon must be positive")
    keys = jax.random.split(key, horizon)
    bootstraps = []
    chosen = []
    for index in range(horizon):
        next_obs = batch.next_observations[:, index]
        previous = batch.actions[:, index]
        logits = actor_logits(learner.target_actor_params, next_obs, previous, num_actions)
        proposal_key, gate_key = jax.random.split(keys[index])
        proposal = jax.random.categorical(proposal_key, logits, axis=-1).astype(jnp.int32)
        policy_q = apply_q(learner.online_params, next_obs)
        policy_stay = jnp.take_along_axis(policy_q, previous[:, None], axis=-1)[:, 0]
        policy_switch = jnp.take_along_axis(policy_q, proposal[:, None], axis=-1)[:, 0]
        gate = jax.random.categorical(
            gate_key, jnp.stack((policy_stay, policy_switch), axis=-1) / jnp.exp(learner.log_temperature), axis=-1)
        target_action = jnp.where(gate == 1, proposal, previous)
        target_q = apply_q(learner.target_params, next_obs)
        chosen.append(target_action)
        bootstraps.append(jnp.take_along_axis(target_q, target_action[:, None], axis=-1)[:, 0])
    gamma = jnp.asarray(gamma, jnp.float32)
    returns = []
    continuation = None
    for index in reversed(range(horizon)):
        bootstrap = bootstraps[index]
        if continuation is not None:
            matches_replay = chosen[index] == batch.actions[:, index + 1]
            bootstrap = jnp.where(matches_replay, continuation, bootstrap)
        ended = batch.terminated[:, index] | batch.truncated[:, index]
        continuation = batch.rewards[:, index] + gamma * jnp.where(ended, 0., bootstrap)
        returns.append(continuation)
    return jax.lax.stop_gradient(jnp.stack(returns[::-1], axis=1))


def taac_discrete_train_step(learner, batch, key, gamma, learning_rate, training_steps,
                             target_entropy_delta, target_actor_entropy_ratio,
                             *, tau=.005, use_hard_update=False, update_interval=1000):
    """Critic compare-through, exact categorical actor expectation, two temperatures."""
    num_actions = learner.online_params[-1].bias.shape[0]
    targets = compare_through_targets(learner, batch, key, gamma)
    observations = batch.observations.reshape((-1, batch.observations.shape[-1]))
    actions = batch.actions.reshape((-1,))
    previous = batch.previous_actions.reshape((-1,))
    targets = targets.reshape((-1,))

    def critic_loss(params):
        predicted = jnp.take_along_axis(apply_q(params, observations), actions[:, None], axis=-1)[:, 0]
        error = jnp.abs(predicted - targets)
        quadratic = jnp.minimum(error, 1.)
        return jnp.mean(.5 * quadratic ** 2 + error - quadratic)

    loss, critic_grad = jax.value_and_grad(critic_loss)(learner.online_params)
    critic, critic_optimizer = adam_step(learner.online_params, critic_grad,
                                         learner.optimizer_state, learning_rate)
    q = jax.lax.stop_gradient(apply_q(critic, observations))
    gate_temperature = jnp.exp(learner.log_temperature)
    actor_temperature = jnp.exp(learner.log_actor_temperature)
    valid_previous = previous >= 0
    q_stay = jnp.take_along_axis(q, jnp.maximum(previous, 0)[:, None], axis=-1)[:, 0]
    gate_logits = jnp.stack((jnp.broadcast_to(q_stay[:, None], q.shape), q), axis=-1) / gate_temperature
    gate_probs = jax.nn.softmax(gate_logits, axis=-1)
    gate_entropy = -jnp.sum(gate_probs * jnp.log(jnp.maximum(gate_probs, 1e-10)), axis=-1)
    candidate_value = gate_temperature * jax.nn.logsumexp(gate_logits, axis=-1)
    candidate_value = jnp.where(valid_previous[:, None], candidate_value, q)
    gate_entropy = jnp.where(valid_previous[:, None], gate_entropy, 0.)

    def actor_loss(params):
        logits = actor_logits(params, observations, previous, num_actions)
        probabilities = jax.nn.softmax(logits, axis=-1)
        entropy = -jnp.sum(probabilities * jnp.log(jnp.maximum(probabilities, 1e-10)), axis=-1)
        objective = jnp.sum(probabilities * jax.lax.stop_gradient(candidate_value), axis=-1)
        return -jnp.mean(objective + jax.lax.stop_gradient(actor_temperature) * entropy)

    policy_loss, actor_grad = jax.value_and_grad(actor_loss)(learner.actor_params)
    actor, actor_optimizer = adam_step(learner.actor_params, actor_grad,
                                       learner.actor_optimizer_state, learning_rate)
    logits = actor_logits(learner.actor_params, observations, previous, num_actions)
    proposal_probs = jax.lax.stop_gradient(jax.nn.softmax(logits, axis=-1))
    actor_entropy = -jnp.mean(jnp.sum(
        proposal_probs * jnp.log(jnp.maximum(proposal_probs, 1e-10)), axis=-1))
    valid_count = jnp.maximum(jnp.sum(valid_previous), 1)
    mean_gate_entropy = jnp.sum(jnp.sum(proposal_probs * gate_entropy, axis=-1)) / valid_count
    delta = jnp.asarray(target_entropy_delta, jnp.float32)
    target_gate_entropy = jnp.where(jnp.any(valid_previous),
                                    -delta * jnp.log(delta) - (1. - delta) * jnp.log1p(-delta), 0.)
    gate_grad = jax.lax.stop_gradient(mean_gate_entropy - target_gate_entropy)
    target_actor_entropy = target_actor_entropy_ratio * jnp.log(jnp.float32(num_actions))
    actor_temp_grad = jax.lax.stop_gradient(actor_entropy - target_actor_entropy)
    log_temperature, temp_optimizer = adam_step(learner.log_temperature, gate_grad,
                                                 learner.temperature_optimizer_state, learning_rate)
    log_actor_temperature, actor_temp_optimizer = adam_step(
        learner.log_actor_temperature, actor_temp_grad, learner.actor_temperature_optimizer_state,
        learning_rate)
    critic_target = update_target_params(learner.target_params, critic, training_steps,
                                          tau=tau, use_hard_update=use_hard_update,
                                          update_interval=update_interval)
    actor_target = update_target_params(learner.target_actor_params, actor, training_steps,
                                        tau=tau, use_hard_update=use_hard_update,
                                        update_interval=update_interval)
    return TAACDiscreteLearner(critic, critic_target, critic_optimizer, actor, actor_target,
                               actor_optimizer, log_temperature, temp_optimizer,
                               log_actor_temperature, actor_temp_optimizer), loss, policy_loss, mean_gate_entropy
