"""Two-hidden-layer Q network matching the original RARe architecture."""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class LinearParams(NamedTuple):
    """Immutable pytree; weight axes are (input, output)."""

    weight: jax.Array
    bias: jax.Array


QParams = tuple[LinearParams, LinearParams, LinearParams]


def init_q_params(
    key: jax.Array, obs_dim: int, num_actions: int, hidden_dim: int = 64
) -> QParams:
    """Initialize float32 layers, uniform on (-1/sqrt(fan_in), 1/sqrt(fan_in))."""
    for name, size in (
        ("obs_dim", obs_dim), ("num_actions", num_actions), ("hidden_dim", hidden_dim)
    ):
        if type(size) is not int or size <= 0:
            raise ValueError(f"{name} must be a positive Python integer")

    keys = jax.random.split(key, 6)
    dimensions = (obs_dim, hidden_dim, hidden_dim, num_actions)
    layers = []
    for index, (fan_in, fan_out) in enumerate(zip(dimensions[:-1], dimensions[1:])):
        bound = fan_in ** -0.5
        layers.append(LinearParams(
            weight=jax.random.uniform(
                keys[2 * index], (fan_in, fan_out), dtype=jnp.float32,
                minval=-bound, maxval=bound,
            ),
            bias=jax.random.uniform(
                keys[2 * index + 1], (fan_out,), dtype=jnp.float32,
                minval=-bound, maxval=bound,
            ),
        ))
    return tuple(layers)


def apply_q(params: QParams, observations: jax.Array) -> jax.Array:
    """Map (..., obs_dim) to Q values (..., num_actions) with ReLU layers and a linear output."""
    x = jnp.asarray(observations, dtype=jnp.float32)
    if x.ndim < 1 or x.shape[-1] != params[0].weight.shape[0]:
        raise ValueError("observations must have shape (..., obs_dim)")
    for layer in params[:-1]:
        x = jax.nn.relu(
            jnp.matmul(x, layer.weight, precision=jax.lax.Precision.HIGHEST) + layer.bias
        )
    output = params[-1]
    return jnp.matmul(x, output.weight, precision=jax.lax.Precision.HIGHEST) + output.bias
