"""Advance a compiled step function a dynamic number of times."""

import jax
import jax.numpy as jnp


def run_steps(step, state, count):
    """Apply `step` `count` times; one executable serves every chunk length."""
    return jax.lax.while_loop(lambda carry: carry[0] < count,
                              lambda carry: (carry[0] + 1, step(carry[1])),
                              (jnp.asarray(0, jnp.int32), state))[1]
