"""Dynamic collection with fixed-size metric buffers for one compiled executable."""

import jax
import jax.numpy as jnp


def collect_steps(step, state, inactive_metrics, capacity, count):
    """Stop at count; unused metric slots retain their inactive sentinel values."""
    metrics = jax.tree.map(
        lambda value: jnp.broadcast_to(value, (capacity, *value.shape)), inactive_metrics,
    )

    def collect(carry):
        index, current, records = carry
        updated, record = step(current)
        records = jax.tree.map(
            lambda buffer, value: jax.lax.dynamic_update_index_in_dim(buffer, value, index, axis=0),
            records, record,
        )
        return index + 1, updated, records

    _, updated, metrics = jax.lax.while_loop(
        lambda carry: carry[0] < count, collect, (jnp.asarray(0, jnp.int32), state, metrics),
    )
    return updated, metrics
