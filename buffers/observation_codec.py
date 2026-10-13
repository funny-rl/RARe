"""Lossless compact storage for the Grid observation used by RARe."""

import jax
import jax.numpy as jnp


def storage_spec(obs_shape, observation_dtype, codec):
    if codec == "raw":
        dtype = jnp.dtype(observation_dtype)
        if dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.uint8)):
            raise ValueError("raw observation storage dtype must be float32 or uint8")
        return obs_shape, dtype
    if codec != "grid_indices":
        raise ValueError(f"unknown observation codec: {codec}")
    if len(obs_shape) != 1:
        raise ValueError("grid_indices requires one-dimensional observations")
    cells = obs_shape[0]
    dtype = jnp.uint8 if cells <= 256 else jnp.uint16 if cells <= 65536 else jnp.int32
    return (2,), jnp.dtype(dtype)


def encode_observation(observation, codec):
    if codec == "raw":
        return observation
    if codec != "grid_indices":
        raise ValueError(f"unknown observation codec: {codec}")
    # A cell with value 3 contains both the agent and the goal.
    agent = jnp.argmax((observation == 1) | (observation == 3), axis=-1)
    goal = jnp.argmax((observation == 2) | (observation == 3), axis=-1)
    return jnp.stack((agent, goal), axis=-1)


def decode_observation(stored, codec, observation_dim):
    if codec == "raw":
        return stored.astype(jnp.float32)
    if codec != "grid_indices":
        raise ValueError(f"unknown observation codec: {codec}")
    if type(observation_dim) is not int or observation_dim <= 0:
        raise ValueError("grid_indices sampling requires a positive observation_dim")
    agent = jax.nn.one_hot(stored[..., 0].astype(jnp.int32), observation_dim, dtype=jnp.float32)
    goal = jax.nn.one_hot(stored[..., 1].astype(jnp.int32), observation_dim, dtype=jnp.float32)
    return agent + 2.0 * goal
