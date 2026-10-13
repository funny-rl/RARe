"""Skip replay with equal-return labels beyond episode completion."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import checkify


class SkipSegment(NamedTuple):
    """One fixed-shape repeat: observations (J+1, ...), rewards/end flags (J,).

    Only the first executed_steps transitions are real. Keep the actual final
    observation at observations[executed_steps]; later padding is ignored.
    selected_skip is the original requested duration, at most J.
    """

    observations: jax.Array
    action: jax.Array
    rewards: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    selected_skip: jax.Array
    executed_steps: jax.Array


class SkipBatch(NamedTuple):
    """Replay arrays; scalar fields have one leading storage/sample axis.

    skips identifies the output head to train. executed_steps counts actual
    interactions for this record; selected_skips/segment_steps describe the
    original collected repeat, including for augmented subsegments.
    """

    observations: jax.Array
    actions: jax.Array
    skips: jax.Array
    returns: jax.Array
    discounts: jax.Array
    next_observations: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    selected_skips: jax.Array
    segment_steps: jax.Array
    executed_steps: jax.Array


class SkipReplayState(NamedTuple):
    """One experiment's skip storage and scalar int32 cursor/count."""

    data: SkipBatch
    write_index: jax.Array
    size: jax.Array


def init_skip_replay(capacity: int, obs_shape: tuple[int, ...], *, observation_dtype=jnp.float32) -> SkipReplayState:
    """Allocate skip replay; uint8 observations require exact integral encodings."""
    if type(capacity) is not int or capacity <= 0:
        raise ValueError("capacity must be a positive Python integer")
    if (not isinstance(obs_shape, tuple) or not obs_shape
            or any(type(size) is not int or size <= 0 for size in obs_shape)):
        raise ValueError("obs_shape must be a nonempty tuple of positive Python integers")
    observation_dtype = jnp.dtype(observation_dtype)
    if observation_dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.uint8)):
        raise ValueError("observation storage dtype must be float32 or uint8")
    return SkipReplayState(
        data=SkipBatch(
            observations=jnp.zeros((capacity, *obs_shape), dtype=observation_dtype),
            actions=jnp.zeros(capacity, dtype=jnp.int32),
            skips=jnp.zeros(capacity, dtype=jnp.int32),
            returns=jnp.zeros(capacity, dtype=jnp.float32),
            discounts=jnp.zeros(capacity, dtype=jnp.float32),
            next_observations=jnp.zeros((capacity, *obs_shape), dtype=observation_dtype),
            terminated=jnp.zeros(capacity, dtype=jnp.bool_),
            truncated=jnp.zeros(capacity, dtype=jnp.bool_),
            selected_skips=jnp.zeros(capacity, dtype=jnp.int32),
            segment_steps=jnp.zeros(capacity, dtype=jnp.int32),
            executed_steps=jnp.zeros(capacity, dtype=jnp.int32),
        ),
        write_index=jnp.asarray(0, dtype=jnp.int32),
        size=jnp.asarray(0, dtype=jnp.int32),
    )


def _require(condition: jax.Array, message: str) -> None:
    """Check concrete inputs eagerly and expose compiled checks to checkify."""
    if not isinstance(condition, jax.core.Tracer) and not bool(condition):
        raise ValueError(message)
    checkify.debug_check(condition, message)


def add_skip_segment(
    state: SkipReplayState,
    segment: SkipSegment,
    gamma: jax.Array,
    *,
    augment_subsegments: bool = True,
) -> SkipReplayState:
    """Insert a repeat and its supported skip labels, preserving terminal credit.

    Termination or episode max-step truncation extends labels through the
    originally selected duration, even with augment_subsegments=False.
    Extra duration adds no reward and has zero bootstrap discount. This
    treats the episode time limit as an absorbing boundary, not a cut in a
    continuing task. Partial collection without an end flag does not extend
    labels. augment_subsegments is static under jit.
    """
    if type(augment_subsegments) is not bool:
        raise TypeError("augment_subsegments must be a bool")
    observations = jnp.asarray(segment.observations, dtype=jnp.float32)
    rewards = jnp.asarray(segment.rewards, dtype=jnp.float32)
    terminated = jnp.asarray(segment.terminated, dtype=jnp.bool_)
    truncated = jnp.asarray(segment.truncated, dtype=jnp.bool_)
    action = jnp.asarray(segment.action)
    selected = jnp.asarray(segment.selected_skip)
    executed = jnp.asarray(segment.executed_steps)
    gamma = jnp.asarray(gamma, dtype=jnp.float32)
    if rewards.ndim != 1 or rewards.shape[0] < 1:
        raise ValueError("segment rewards must have shape (J,) with J >= 1")
    max_skip = rewards.shape[0]
    if observations.shape != (max_skip + 1, *state.data.observations.shape[1:]):
        raise ValueError("segment observations must have shape (J+1, *obs_shape)")
    if terminated.shape != rewards.shape or truncated.shape != rewards.shape:
        raise ValueError("segment end flags must have shape (J,)")
    for name, value in (("action", action), ("selected_skip", selected), ("executed_steps", executed)):
        if value.shape != () or not jnp.issubdtype(value.dtype, jnp.integer):
            raise ValueError(f"{name} must be a scalar integer")
    if gamma.shape != ():
        raise ValueError("gamma must be scalar")
    action, selected, executed = (x.astype(jnp.int32) for x in (action, selected, executed))
    _require((selected >= 1) & (selected <= max_skip), "selected_skip must lie in [1, J]")
    _require((executed >= 1) & (executed <= selected), "executed_steps must lie in [1, selected_skip]")
    _require(jnp.isfinite(gamma) & (gamma >= 0) & (gamma <= 1), "gamma must lie in [0, 1]")
    times = jnp.arange(max_skip)
    _require(
        ~jnp.any((terminated | truncated) & (times < executed - 1)),
        "a segment cannot contain real transitions after an episode end",
    )
    _require(
        jnp.all(jnp.isfinite(jnp.where(times < executed, rewards, 0.0))),
        "real rewards must be finite",
    )

    # End-major order preserves the original subsegment insertion order.
    end_rows, starts = jnp.tril_indices(max_skip)
    ends = end_rows + 1
    actual_ends = jnp.minimum(ends, executed)
    actual_steps = jnp.maximum(actual_ends - starts, 0)
    final_done = terminated[executed - 1] | truncated[executed - 1]
    valid = (starts < executed) & (ends <= selected)
    valid &= (ends <= executed) | final_done
    if not augment_subsegments:
        # Episode-end equivalence remains mandatory; ordinary prefix/suffix
        # augmentation can be disabled independently.
        valid &= (starts == 0) & jnp.where(final_done, ends >= executed, ends == executed)

    # Gather the same accumulated return for every equivalent episode-end label.
    # No division by gamma is needed, including when gamma is zero.
    offsets = times[None, :] - times[:, None]
    weights = jnp.where(offsets >= 0, gamma ** jnp.maximum(offsets, 0), 0.0)
    real_rewards = jnp.where(times < executed, rewards, 0.0)
    return_table = jnp.cumsum(weights * real_rewards[None, :], axis=1)
    record_terminated = terminated[actual_ends - 1]
    record_truncated = truncated[actual_ends - 1]
    records = SkipBatch(
        observations=observations[starts],
        actions=jnp.full(ends.shape, action, dtype=jnp.int32),
        skips=(ends - starts).astype(jnp.int32),
        returns=return_table[starts, actual_ends - 1],
        discounts=jnp.where(record_terminated | record_truncated, 0.0, gamma ** actual_steps),
        next_observations=observations[actual_ends],
        terminated=record_terminated,
        truncated=record_truncated,
        selected_skips=jnp.full(ends.shape, selected, dtype=jnp.int32),
        segment_steps=jnp.full(ends.shape, executed, dtype=jnp.int32),
        executed_steps=actual_steps.astype(jnp.int32),
    )

    ranks = jnp.cumsum(valid.astype(jnp.int32)) - 1
    count = jnp.sum(valid, dtype=jnp.int32)
    capacity = state.data.actions.shape[0]
    # Keep only the final capacity accepted records, so scatter writes have
    # no duplicate in-bounds indices even when one segment exceeds capacity.
    keep = valid & (ranks >= count - capacity)
    indices = jnp.where(keep, (state.write_index + ranks) % capacity, capacity)
    data = jax.tree.map(
        lambda storage, values: storage.at[indices].set(values.astype(storage.dtype), mode="drop"),
        state.data, records,
    )
    return SkipReplayState(
        data=data,
        write_index=(state.write_index + count) % capacity,
        size=jnp.minimum(state.size + count, capacity),
    )


def sample_skip_batch(key: jax.Array, state: SkipReplayState, batch_size: int) -> SkipBatch:
    """Sample nonempty skip replay uniformly with replacement using an explicit key.

    batch_size is static under jit. Compiled callers must guard size > 0;
    checkify diagnostics can enforce that precondition. The key is unchanged.
    """
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive Python integer")
    _require(state.size > 0, "cannot sample an empty skip replay")
    indices = jax.random.randint(key, (batch_size,), 0, state.size)
    sampled = jax.tree.map(lambda storage: storage[indices], state.data)
    return sampled._replace(observations=sampled.observations.astype(jnp.float32),
                            next_observations=sampled.next_observations.astype(jnp.float32))
