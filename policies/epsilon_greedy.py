"""Discrete epsilon-greedy policy matching legacy first-argmax exploitation."""

import jax
import jax.numpy as jnp


def _inputs(values: jax.Array, epsilon: jax.Array) -> tuple[jax.Array, jax.Array]:
    values = jnp.asarray(values, dtype=jnp.float32)
    epsilon = jnp.asarray(epsilon, dtype=jnp.float32)
    if values.ndim < 1 or values.shape[-1] < 1:
        raise ValueError("values must have nonempty final action axis")
    if epsilon.shape != ():
        raise ValueError("epsilon must be scalar")
    return values, epsilon


def epsilon_greedy_probs(values: jax.Array, epsilon: jax.Array) -> jax.Array:
    """Return (..., A) probabilities, epsilon/A plus mass on first argmax.

    Values must be finite and epsilon lie in [0,1]. Exploration includes the
    greedy action; tied maxima do not share exploitation mass in legacy DDQN.
    """
    values, epsilon = _inputs(values, epsilon)
    count = values.shape[-1]
    greedy = jax.nn.one_hot(jnp.argmax(values, axis=-1), count, dtype=jnp.float32)
    return epsilon / count + (1.0 - epsilon) * greedy


def sample_epsilon_greedy(key: jax.Array, values: jax.Array, epsilon: jax.Array) -> jax.Array:
    """Return int32 actions (...) using an explicit, caller-advanced key.

    Values must be finite and epsilon lie in [0,1]. Draw an exploration
    decision and a uniform action independently; no categorical/log transform
    is needed. Use an outer vmap with one key per experiment.
    """
    values, epsilon = _inputs(values, epsilon)
    explore_key, action_key = jax.random.split(key)
    shape = values.shape[:-1]
    explore = jax.random.bernoulli(explore_key, epsilon, shape=shape)
    random_actions = jax.random.randint(action_key, shape, 0, values.shape[-1])
    return jnp.where(explore, random_actions, jnp.argmax(values, axis=-1)).astype(jnp.int32)
