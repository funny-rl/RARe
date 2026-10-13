"""Uniform replay for primitive transitions with discrete actions."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import checkify

from .observation_codec import decode_observation, encode_observation, storage_spec


class ReplayBatch(NamedTuple):
    """Arrays with a leading storage/sample axis; scalar fields have shape (B,)."""

    observations: jax.Array
    actions: jax.Array
    rewards: jax.Array
    next_observations: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    previous_actions: jax.Array


class ReplayState(NamedTuple):
    """One experiment's storage and scalar int32 cursor/count, without RNG state."""

    data: ReplayBatch
    write_index: jax.Array
    size: jax.Array


def init_replay(capacity: int, obs_shape: tuple[int, ...], *, observation_dtype=jnp.float32,
                observation_codec="raw") -> ReplayState:
    """Allocate float32 rewards and configurable observation storage.

    Actions are scalar int32 indices. Previous action -1 means absent, as at
    episode start. Capacity and observation shape must be static under jit.
    Use uint8 only when environment observations are exactly integral 0–255.
    """
    if type(capacity) is not int or capacity <= 0:
        raise ValueError("capacity must be a positive Python integer")
    if (not isinstance(obs_shape, tuple) or not obs_shape
            or any(type(size) is not int or size <= 0 for size in obs_shape)):
        raise ValueError("obs_shape must be a nonempty tuple of positive Python integers")
    storage_shape, observation_dtype = storage_spec(obs_shape, observation_dtype, observation_codec)
    return ReplayState(
        data=ReplayBatch(
            observations=jnp.zeros((capacity, *storage_shape), dtype=observation_dtype),
            actions=jnp.zeros((capacity,), dtype=jnp.int32),
            rewards=jnp.zeros((capacity,), dtype=jnp.float32),
            next_observations=jnp.zeros((capacity, *storage_shape), dtype=observation_dtype),
            terminated=jnp.zeros((capacity,), dtype=jnp.bool_),
            truncated=jnp.zeros((capacity,), dtype=jnp.bool_),
            previous_actions=jnp.full((capacity,), -1, dtype=jnp.int32),
        ),
        write_index=jnp.asarray(0, dtype=jnp.int32),
        size=jnp.asarray(0, dtype=jnp.int32),
    )


def add_transition(
    state: ReplayState,
    observation: jax.Array,
    action: jax.Array,
    reward: jax.Array,
    next_observation: jax.Array,
    terminated: jax.Array,
    truncated: jax.Array,
    *,
    previous_action: jax.Array | None = None,
    observation_codec="raw",
    observation_dim=None,
) -> ReplayState:
    """Store one primitive transition, overwriting the oldest when full.

    Pass the actual successor observation before any reset. End flags are
    stored separately; this function does not decide whether to bootstrap.
    Each action/reward/flag is scalar; observations must match obs_shape.
    The input state is unchanged. Batch independent states with vmap.
    """
    observation = jnp.asarray(observation, dtype=jnp.float32)
    next_observation = jnp.asarray(next_observation, dtype=jnp.float32)
    if observation_codec == "grid_indices" and (
        type(observation_dim) is not int or observation.shape != (observation_dim,)
        or next_observation.shape != observation.shape
    ):
        raise ValueError("grid_indices requires matching observations of observation_dim")
    transition = ReplayBatch(
        observations=encode_observation(observation, observation_codec),
        actions=jnp.asarray(action),
        rewards=jnp.asarray(reward, dtype=jnp.float32),
        next_observations=encode_observation(next_observation, observation_codec),
        terminated=jnp.asarray(terminated, dtype=jnp.bool_),
        truncated=jnp.asarray(truncated, dtype=jnp.bool_),
        previous_actions=jnp.asarray(-1 if previous_action is None else previous_action),
    )
    for name, value, storage in zip(ReplayBatch._fields, transition, state.data):
        if value.shape != storage.shape[1:]:
            raise ValueError(f"{name} must have shape {storage.shape[1:]}, got {value.shape}")
    for name, value in (("action", transition.actions), ("previous_action", transition.previous_actions)):
        if not jnp.issubdtype(value.dtype, jnp.integer):
            raise ValueError(f"{name} must be an integer action index")
    transition = transition._replace(
        actions=transition.actions.astype(jnp.int32),
        previous_actions=transition.previous_actions.astype(jnp.int32),
    )
    data = jax.tree.map(
        lambda storage, value: storage.at[state.write_index].set(value.astype(storage.dtype)),
        state.data, transition,
    )
    capacity = state.data.actions.shape[0]
    return ReplayState(
        data=data,
        write_index=(state.write_index + 1) % capacity,
        size=jnp.minimum(state.size + 1, capacity),
    )


def sample_batch(key: jax.Array, state: ReplayState, batch_size: int, *,
                 observation_codec="raw", observation_dim=None) -> ReplayBatch:
    """Sample stored transitions uniformly with replacement, without mutation.

    The caller owns key advancement. batch_size must be static under jit.
    Never sample an empty replay: eager calls raise ValueError; compiled
    callers must guard with lax.cond(state.size > 0, ...). checkify.checkify
    can enforce the nonempty precondition during compiled diagnostics.
    """
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive Python integer")
    if not isinstance(state.size, jax.core.Tracer) and int(state.size) == 0:
        raise ValueError("cannot sample an empty replay")
    checkify.debug_check(state.size > 0, "cannot sample an empty replay")
    indices = jax.random.randint(key, (batch_size,), 0, state.size)
    sampled = jax.tree.map(lambda storage: storage[indices], state.data)
    return sampled._replace(
        observations=decode_observation(sampled.observations, observation_codec, observation_dim),
        next_observations=decode_observation(sampled.next_observations, observation_codec, observation_dim),
    )
