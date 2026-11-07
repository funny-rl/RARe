"""Shared two-hidden-layer continuous actor, critic and skip estimators."""

import jax
import jax.numpy as jnp

from .q import QParams, apply_q, init_q_params


def init_actor_params(key: jax.Array, obs_dim: int, action_dim: int,
                      hidden_dim: int, *, use_previous_action: bool = False) -> QParams:
    return init_q_params(key, obs_dim + (action_dim if use_previous_action else 0),
                         action_dim, hidden_dim)


def apply_actor(params: QParams, observations: jax.Array, low: jax.Array, high: jax.Array,
                previous_actions: jax.Array | None = None) -> jax.Array:
    observations = jnp.asarray(observations, jnp.float32)
    if previous_actions is not None:
        observations = jnp.concatenate((observations, jnp.asarray(previous_actions, jnp.float32)), -1)
    raw = jnp.tanh(apply_q(params, observations))
    low, high = jnp.asarray(low, jnp.float32), jnp.asarray(high, jnp.float32)
    return (high + low) * .5 + (high - low) * .5 * raw


def init_critic_params(key: jax.Array, obs_dim: int, action_dim: int,
                       hidden_dim: int) -> QParams:
    return init_q_params(key, obs_dim + action_dim, 1, hidden_dim)


def apply_critic(params: QParams, observations: jax.Array, actions: jax.Array) -> jax.Array:
    inputs = jnp.concatenate((jnp.asarray(observations, jnp.float32),
                              jnp.asarray(actions, jnp.float32)), -1)
    return apply_q(params, inputs)[..., 0]


def init_continuous_skip_params(key: jax.Array, obs_dim: int, action_dim: int,
                                max_skip: int, hidden_dim: int) -> QParams:
    return init_q_params(key, obs_dim + action_dim, max_skip, hidden_dim)


def apply_continuous_skip(params: QParams, observations: jax.Array,
                          actions: jax.Array) -> jax.Array:
    inputs = jnp.concatenate((jnp.asarray(observations, jnp.float32),
                              jnp.asarray(actions, jnp.float32)), -1)
    return apply_q(params, inputs)
