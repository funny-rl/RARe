"""Independent random streams for one experiment seed."""

from typing import NamedTuple

import jax


class RNGState(NamedTuple):
    """Immutable JAX pytree of the current key for each random stream."""

    model_init: jax.Array
    environment: jax.Array
    action: jax.Array
    skip: jax.Array
    replay: jax.Array
    target: jax.Array
    evaluation: jax.Array


def init_rngs(seed: int) -> RNGState:
    """Derive independent keys without using global RNG or device identity.

    Each consumer must split and advance its own key before reusing its stream.
    """
    root = jax.random.key(seed, impl="threefry2x32")

    # Keep these IDs stable when adding streams so existing keys do not change.
    return RNGState(
        model_init=jax.random.fold_in(root, 0),
        environment=jax.random.fold_in(root, 1),
        action=jax.random.fold_in(root, 2),
        skip=jax.random.fold_in(root, 3),
        replay=jax.random.fold_in(root, 4),
        target=jax.random.fold_in(root, 5),
        evaluation=jax.random.fold_in(root, 6),
    )
