"""Temperature-scaled Softmax policy, separate from epsilon-greedy targets."""

import jax
import jax.numpy as jnp
from jax.experimental import checkify


def _softmax_logits(values: jax.Array, temperature: jax.Array) -> jax.Array:
    values = jnp.asarray(values, dtype=jnp.float32)
    temperature = jnp.asarray(temperature, dtype=jnp.float32)
    if values.ndim < 1 or values.shape[-1] < 1:
        raise ValueError("values must have shape (..., num_actions) with num_actions >= 1")
    if temperature.shape != ():
        raise ValueError("temperature must be scalar")
    valid = jnp.isfinite(temperature) & (temperature > 0) & jnp.all(jnp.isfinite(values))
    message = "Softmax requires finite values and a finite positive temperature"
    if not isinstance(valid, jax.core.Tracer) and not bool(valid):
        raise ValueError(message)
    checkify.debug_check(valid, message)
    # Center before division: large finite Q/T can overflow before a normal
    # softmax has the opportunity to subtract its maximum.
    return (values - jnp.max(values, axis=-1, keepdims=True)) / temperature


def softmax_probs(values: jax.Array, temperature: jax.Array = 1.0) -> jax.Array:
    """Return float32 probabilities (..., A) proportional to exp(Q/T).

    Normalize the last axis. Inputs must be finite and T finite/positive;
    smaller T concentrates on large values, larger T approaches uniform.
    Compiled callers must meet those preconditions; checkify can diagnose
    violations. This utility does not change any algorithm's behavior or
    expected-value policy until it is explicitly used there.
    """
    return jax.nn.softmax(_softmax_logits(values, temperature), axis=-1)


def sample_softmax(
    key: jax.Array, values: jax.Array, temperature: jax.Array = 1.0
) -> jax.Array:
    """Sample int32 action indices (...) from the Softmax policy.

    Same key/inputs reproduce the same draw. The caller splits and advances
    its action or skip stream; this function does not advance the key.
    Use outer vmap with each experiment's key for independent seeds. Sample
    from centered logits directly, avoiding a probability-to-log round trip
    that can underflow.
    """
    logits = _softmax_logits(values, temperature)
    return jax.random.categorical(key, logits, axis=-1).astype(jnp.int32)
