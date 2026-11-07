"""Chronological replay windows for TAAC compare-through targets."""

import jax
import jax.numpy as jnp


def sample_taac_sequence(key, state, batch_size, horizon):
    capacity = state.data.actions.shape[0]
    oldest = jnp.where(state.size == capacity, state.write_index, 0)
    starts = jax.random.randint(key, (batch_size,), 0, state.size - horizon + 1)
    indices = (oldest + starts[:, None] + jnp.arange(horizon)) % capacity
    return jax.tree.map(lambda storage: storage[indices], state.data)
