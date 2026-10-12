"""Common discrete skip architecture for TempoRL, UTE, and RARe."""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp

from .q import QParams, apply_q, init_q_params


@partial(jax.tree_util.register_dataclass, data_fields=("layers",), meta_fields=("num_actions",))
@dataclass(frozen=True)
class SkipParams:
    """Trainable layers and a static action count, preserved under jit/vmap."""

    layers: QParams
    num_actions: int


def init_skip_params(
    key: jax.Array,
    obs_dim: int,
    num_actions: int,
    max_skip: int,
    hidden_dim: int = 64,
) -> SkipParams:
    """Initialize (obs_dim + num_actions) -> hidden -> hidden -> max_skip.

    Use the legacy Linear initialization distribution, with independent
    weight/bias keys. Dimensions are positive Python integers, static under
    jit. Split the caller's model_init stream before each initialization;
    this function leaves the supplied key unchanged. Torch seeds do not
    produce identical JAX samples. UTE ensembles independent copies of
    this same architecture; the legacy UTE extra hidden layer is omitted
    intentionally to keep architecture constant across algorithms.
    """
    for name, size in (("obs_dim", obs_dim), ("num_actions", num_actions)):
        if type(size) is not int or size <= 0:
            raise ValueError(f"{name} must be a positive Python integer")
    layers = init_q_params(key, obs_dim + num_actions, max_skip, hidden_dim)
    return SkipParams(layers=layers, num_actions=num_actions)


def apply_skip(
    params: SkipParams, observations: jax.Array, actions: jax.Array
) -> jax.Array:
    """Return Q_J(s, a, j) with output (..., max_skip), index j-1 for length j.

    Observations have shape (..., obs_dim); integer actions have shape (...),
    matching replay's (B,) convention. Callers must supply action indices
    in [0, num_actions). Concatenate float32 action one-hots with observations,
    then apply two ReLU hidden layers and a linear output. Use vmap over
    parameters and inputs for independent experiment seeds.
    """
    observations = jnp.asarray(observations, dtype=jnp.float32)
    actions = jnp.asarray(actions)
    obs_dim = params.layers[0].weight.shape[0] - params.num_actions
    if observations.ndim < 1 or observations.shape[-1] != obs_dim:
        raise ValueError("observations must have shape (..., obs_dim)")
    if actions.shape != observations.shape[:-1] or not jnp.issubdtype(actions.dtype, jnp.integer):
        raise ValueError("actions must be integers with shape observations.shape[:-1]")
    encoded_actions = jax.nn.one_hot(actions, params.num_actions, dtype=jnp.float32)
    inputs = jnp.concatenate((observations, encoded_actions), axis=-1)
    return apply_q(params.layers, inputs)


def init_skip_ensemble_params(
    key: jax.Array,
    obs_dim: int,
    num_actions: int,
    max_skip: int,
    *,
    ensemble_size: int,
    hidden_dim: int = 64,
) -> SkipParams:
    """Initialize independent copies of the common two-hidden-layer network.

    Each trainable array gains a leading member axis E. Members share no
    weights or encoder; only their architecture and static action count
    match. ensemble_size and dimensions are static positive Python integers.
    Split the caller's model_init stream before calling; this key is unchanged.
    """
    if type(ensemble_size) is not int or ensemble_size <= 0:
        raise ValueError("ensemble_size must be a positive Python integer")
    member_keys = jax.random.split(key, ensemble_size)
    return jax.vmap(lambda member_key: init_skip_params(
        member_key, obs_dim, num_actions, max_skip, hidden_dim
    ))(member_keys)


def apply_skip_ensemble(
    params: SkipParams, observations: jax.Array, actions: jax.Array
) -> jax.Array:
    """Evaluate all members on the same inputs, returning (E, ..., max_skip).

    Preserve the legacy UTE member-first output axis. Inputs follow apply_skip:
    observations (..., obs_dim), integer actions (...). Values remain separate;
    reduction and skip selection belong to the algorithm. For independent
    seeds, use an outer vmap to obtain (S, E, B, max_skip) from (S, B, obs_dim).
    """
    if params.layers[0].weight.ndim != 3:
        raise ValueError("ensemble parameters must have one leading member axis")
    return jax.vmap(apply_skip, in_axes=(0, None, None))(params, observations, actions)
