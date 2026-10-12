"""Target synchronization shared by matching network parameter pytrees."""

from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import checkify


def update_target_params(
    target_params: Any,
    online_params: Any,
    training_steps: jax.Array,
    *,
    tau: jax.Array = 0.005,
    use_hard_update: bool = False,
    update_interval: int = 1000,
) -> Any:
    """Return detached target parameters after the online optimizer update.

    Soft mode: tau*online + (1-tau)*target on every learning update.
    Hard mode: copy online when training_steps % update_interval == 0.
    training_steps is the external primitive interaction count INCLUDING
    warmup, as in legacy main.py; never substitute the Adam update count.
    use_hard_update and update_interval are static under jit. Tau must be
    finite and lie in [0,1]; matching structures and leaf shapes are required.
    No environment/optimizer counter is advanced by this function.
    """
    if type(use_hard_update) is not bool:
        raise TypeError("use_hard_update must be a bool")
    if type(update_interval) is not int or update_interval <= 0:
        raise ValueError("update_interval must be a positive Python integer")
    if jax.tree.structure(target_params) != jax.tree.structure(online_params):
        raise ValueError("target and online parameter structures must match")
    if any(a.shape != b.shape for a, b in zip(jax.tree.leaves(target_params), jax.tree.leaves(online_params))):
        raise ValueError("target and online parameter shapes must match")
    training_steps = jnp.asarray(training_steps)
    if training_steps.shape != () or not jnp.issubdtype(training_steps.dtype, jnp.integer):
        raise ValueError("training_steps must be a scalar integer")
    if use_hard_update:
        updated = jax.lax.cond(
            training_steps % update_interval == 0,
            lambda _: online_params, lambda _: target_params, operand=None,
        )
    else:
        tau = jnp.asarray(tau, dtype=jnp.float32)
        if tau.shape != ():
            raise ValueError("tau must be scalar")
        valid = jnp.isfinite(tau) & (tau >= 0) & (tau <= 1)
        if not isinstance(valid, jax.core.Tracer) and not bool(valid):
            raise ValueError("tau must lie in [0, 1]")
        checkify.debug_check(valid, "tau must lie in [0, 1]")
        updated = jax.tree.map(
            lambda target, online: tau * online + (1.0 - tau) * target,
            target_params, online_params,
        )
    return jax.tree.map(jax.lax.stop_gradient, updated)
