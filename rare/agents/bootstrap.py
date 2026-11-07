"""Detached endpoint values for greedy and expected skip targets."""

import jax
import jax.numpy as jnp


def _matching_action_arrays(left, right):
    left = jnp.asarray(left, dtype=jnp.float32)
    right = jnp.asarray(right, dtype=jnp.float32)
    if left.ndim != 2 or min(left.shape) < 1 or right.shape != left.shape:
        raise ValueError("action arrays must share nonempty shape (B, A)")
    return left, right


def greedy_bootstrap(online_q: jax.Array, target_q: jax.Array) -> jax.Array:
    """Endpoint values of TempoRL and UTE: the online argmax evaluated by the target Q."""
    online_q, target_q = _matching_action_arrays(online_q, target_q)
    actions = jnp.argmax(online_q, axis=-1)
    values = jnp.take_along_axis(target_q, actions[:, None], axis=-1)[:, 0]
    return jax.lax.stop_gradient(values)


def expected_bootstrap(
    policy_probs: jax.Array, target_eval_q: jax.Array
) -> jax.Array:
    """Endpoint values of RARe: the exact expectation of target Q under the given probabilities.

    RARe-se builds the probabilities from the evaluation critic, RARe-sh from the base critic."""
    policy_probs, target_eval_q = _matching_action_arrays(policy_probs, target_eval_q)
    return jax.lax.stop_gradient(jnp.sum(policy_probs * target_eval_q, axis=-1))
