"""Primitive replay with vector actions; TAAC also reads previous actions and gate decisions."""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class ContinuousBatch(NamedTuple):
    observations: jax.Array
    actions: jax.Array
    rewards: jax.Array
    next_observations: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    previous_actions: jax.Array
    switched: jax.Array


class ContinuousReplay(NamedTuple):
    data: ContinuousBatch
    write_index: jax.Array
    size: jax.Array


def init_continuous_replay(capacity: int, obs_dim: int, action_dim: int) -> ContinuousReplay:
    if any(type(n) is not int or n <= 0 for n in (capacity, obs_dim, action_dim)):
        raise ValueError("replay dimensions must be positive integers")
    return ContinuousReplay(
        ContinuousBatch(
            jnp.zeros((capacity, obs_dim), jnp.float32),
            jnp.zeros((capacity, action_dim), jnp.float32),
            jnp.zeros(capacity, jnp.float32),
            jnp.zeros((capacity, obs_dim), jnp.float32),
            jnp.zeros(capacity, bool), jnp.zeros(capacity, bool),
            jnp.zeros((capacity, action_dim), jnp.float32),
            jnp.zeros(capacity, bool),
        ), jnp.int32(0), jnp.int32(0),
    )


def add_continuous_transition(replay: ContinuousReplay, observation, action, reward,
                              next_observation, terminated, truncated, previous_action,
                              switched=False) -> ContinuousReplay:
    row = ContinuousBatch(
        jnp.asarray(observation, jnp.float32), jnp.asarray(action, jnp.float32),
        jnp.asarray(reward, jnp.float32), jnp.asarray(next_observation, jnp.float32),
        jnp.asarray(terminated, bool), jnp.asarray(truncated, bool),
        jnp.asarray(previous_action, jnp.float32), jnp.asarray(switched, bool),
    )
    for name, value, storage in zip(ContinuousBatch._fields, row, replay.data):
        if value.shape != storage.shape[1:]:
            raise ValueError(f"{name} must have shape {storage.shape[1:]}")
    data = jax.tree.map(lambda storage, value: storage.at[replay.write_index].set(value), replay.data, row)
    capacity = replay.data.rewards.shape[0]
    return ContinuousReplay(data, (replay.write_index + 1) % capacity,
                            jnp.minimum(replay.size + 1, capacity))


def sample_continuous_batch(key, replay: ContinuousReplay, batch_size: int) -> ContinuousBatch:
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be positive")
    indices = jax.random.randint(key, (batch_size,), 0, replay.size)
    return jax.tree.map(lambda storage: storage[indices], replay.data)
