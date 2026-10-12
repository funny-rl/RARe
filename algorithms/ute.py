"""UTE skip learning with corrected per-member bootstrap losses."""

import jax
import jax.numpy as jnp
from jax.experimental import checkify

from buffers import SkipBatch
from networks import SkipParams, apply_skip_ensemble, init_skip_ensemble_params
from optimizers import adam_step, init_adam

from .skip import SkipState, _skip_labels, _smooth_l1, skip_targets


def init_ute_skip_state(
    key: jax.Array, obs_dim: int, num_actions: int, max_skip: int,
    *, ensemble_size: int, hidden_dim: int = 64,
) -> SkipState:
    """Initialize independent common skip networks and one joint Adam state.

    Parameters/moments have leading member axis E, with one optimizer count
    per experiment, matching legacy UTE. The agreed common architecture has
    two hidden layers; the old UTE extra hidden layer is intentionally omitted.
    Split/advance the model_init key outside; dimensions are static under jit.
    """
    params = init_skip_ensemble_params(
        key, obs_dim, num_actions, max_skip, ensemble_size=ensemble_size, hidden_dim=hidden_dim,
    )
    return SkipState(params, init_adam(params))


def sample_ute_masks(
    key: jax.Array, *, ensemble_size: int, batch_size: int, probability: float = 0.5
) -> jax.Array:
    """Return bool masks (E,B), Bernoulli(0.5) with empty members repaired.

    Give each empty member one uniformly sampled row, matching the corrected
    rebuttal collector's mask rule. Draw repair rows for every member to keep
    shapes static. The caller splits/advances the mask RNG stream; this key
    is not mutated. Use outer vmap with separate keys for independent seeds.
    """
    for name, value in (("ensemble_size", ensemble_size), ("batch_size", batch_size)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive Python integer")
    draw_key, repair_key = jax.random.split(key)
    masks = jax.random.bernoulli(draw_key, probability, (ensemble_size, batch_size))
    empty = ~jnp.any(masks, axis=-1)
    rows = jax.random.randint(repair_key, (ensemble_size,), 0, batch_size)
    repairs = empty[:, None] & (jnp.arange(batch_size)[None, :] == rows[:, None])
    return masks | repairs


def ute_skip_loss(
    params: SkipParams, batch: SkipBatch, targets: jax.Array, masks: jax.Array
) -> jax.Array:
    """Average Smooth L1 over each member's active rows, then active members.

    Targets (B,) are detached; boolean masks (E,B) align with member-first
    predictions. No (B,1)*(B,) broadcasting or extra batch-size division.
    Supplied empty members are excluded, as in rebuttal masked_bootstrap_loss;
    at least one active member is required (eager/checkify diagnostic).
    sample_ute_masks guarantees every member is active during normal training.
    Action/skip ranges follow skip_loss. Outer vmap keeps seeds independent.
    """
    indices = _skip_labels(batch)
    targets = jnp.asarray(targets, dtype=jnp.float32)
    masks = jnp.asarray(masks)
    if targets.shape != indices.shape:
        raise ValueError("targets must have shape (B,)")
    values = apply_skip_ensemble(params, batch.observations, batch.actions)
    if masks.shape != values.shape[:2] or masks.dtype != jnp.bool_:
        raise ValueError("masks must be boolean with shape (E, B)")
    predictions = jnp.take_along_axis(values, indices[None, :, None], axis=-1)[..., 0]
    counts = jnp.sum(masks, axis=-1)
    active = counts > 0
    valid = jnp.any(active)
    if not isinstance(valid, jax.core.Tracer) and not bool(valid):
        raise ValueError("at least one UTE member must have an active sample")
    checkify.debug_check(valid, "at least one UTE member must have an active sample")
    errors = _smooth_l1(predictions, targets)
    member_losses = jnp.sum(jnp.where(masks, errors, 0.0), axis=-1) / jnp.maximum(counts, 1)
    return jnp.sum(member_losses) / jnp.maximum(jnp.sum(active), 1)


def ute_skip_train_step(
    state: SkipState, batch: SkipBatch, bootstrap_values: jax.Array,
    learning_rate: jax.Array, mask_key: jax.Array, *, bootstrap_probability: float = 0.5,
) -> tuple[SkipState, jax.Array]:
    """Compose greedy endpoint targets, draw masks and take one joint Adam step.

    Return new state and pre-update loss. The caller supplies greedy_bootstrap
    from the updated base DDQN, samples replay and advances mask_key before
    calling again. No skip target network or environment counter is updated.
    Checkpoint parameters, Adam state and the caller's mask RNG to resume.
    """
    if state.params.layers[0].weight.ndim != 3:
        raise ValueError("UTE state must have one leading ensemble axis")
    targets = skip_targets(batch.returns, batch.discounts, bootstrap_values)
    masks = sample_ute_masks(
        mask_key, ensemble_size=state.params.layers[0].weight.shape[0], batch_size=targets.shape[0],
        probability=bootstrap_probability,
    )
    loss, gradients = jax.value_and_grad(ute_skip_loss)(state.params, batch, targets, masks)
    params, optimizer_state = adam_step(state.params, gradients, state.optimizer_state, learning_rate)
    return SkipState(params, optimizer_state), loss
