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
    """Return detached (B,) endpoint values for TempoRL and UTE.

    Select the first argmax from online Q, then gather from target Q, as in
    legacy DDQN-backed skip updates. Inputs have shape (B, A). This does
    not maximize target Q or bootstrap from the skip network. Use outer
    vmap for independent experiments; no RNG is consumed.
    """
    online_q, target_q = _matching_action_arrays(online_q, target_q)
    actions = jnp.argmax(online_q, axis=-1)
    values = jnp.take_along_axis(target_q, actions[:, None], axis=-1)[:, 0]
    return jax.lax.stop_gradient(values)


def expected_bootstrap(
    policy_probs: jax.Array, target_eval_q: jax.Array
) -> jax.Array:
    """Return detached (B,) exact action-wise expected endpoint values.

    Inputs have shape (B, A); probabilities must be finite, nonnegative
    and sum to one per row. Nonterminal Q values must be finite. In the
    separate_RARe reference, construct probabilities with epsilon_greedy_probs
    from the ONLINE EVALUATION critic, then evaluate TARGET EVALUATION Q.
    Shared RARe uses the corresponding online and target BASE critics.
    Evaluation epsilon is independent of behavior/skip epsilon. Expectation
    requires no sampling or RNG. Use outer vmap for independent experiments.
    """
    policy_probs, target_eval_q = _matching_action_arrays(policy_probs, target_eval_q)
    return jax.lax.stop_gradient(jnp.sum(policy_probs * target_eval_q, axis=-1))
