"""Polyak averaging of target networks."""

import jax
import jax.numpy as jnp


def update_target_params(target_params, online_params, tau=0.005):
    """Return detached tau * online + (1 - tau) * target, applied at every learning update."""
    tau = jnp.asarray(tau, dtype=jnp.float32)
    updated = jax.tree.map(lambda target, online: tau * online + (1.0 - tau) * target,
                           target_params, online_params)
    return jax.tree.map(jax.lax.stop_gradient, updated)
