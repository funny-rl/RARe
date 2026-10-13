"""Chronological replay windows for TAAC compare-through targets."""

import jax
import jax.numpy as jnp

from .observation_codec import decode_observation


def sample_taac_sequence(key, state, batch_size, horizon, *, observation_codec="raw", observation_dim=None):
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if type(horizon) is not int or horizon <= 0 or horizon > state.data.actions.shape[0]:
        raise ValueError("horizon must be positive and no larger than replay capacity")
    if not isinstance(state.size, jax.core.Tracer) and int(state.size) < horizon:
        raise ValueError("replay has fewer transitions than the requested horizon")
    capacity = state.data.actions.shape[0]
    oldest = jnp.where(state.size == capacity, state.write_index, 0)
    starts = jax.random.randint(key, (batch_size,), 0, state.size - horizon + 1)
    indices = (oldest + starts[:, None] + jnp.arange(horizon)) % capacity
    batch = jax.tree.map(lambda storage: storage[indices], state.data)
    return batch._replace(
        observations=decode_observation(batch.observations, observation_codec, observation_dim),
        next_observations=decode_observation(batch.next_observations, observation_codec, observation_dim),
    )
