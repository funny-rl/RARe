"""Portable model weights only; replay, optimizer and RNG state are not saved."""

import json
from pathlib import Path


def default_model_save_steps(total_steps):
    """Save at each fifth of the budget, rounding up to an observed step.

    Budgets below five steps have fewer than five distinct save positions.
    """
    return tuple(sorted({(total_steps * index + 4) // 5 for index in range(1, 6)}))


def save_models(path, models, *, seeds, env_step, source_sha256):
    """Save a named parameter pytree with its ordered leading seed axis."""
    import jax
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    leaves, structure = jax.tree_util.tree_flatten(models)
    arrays = {f"leaf_{index}": np.asarray(value) for index, value in enumerate(jax.device_get(leaves))}
    if any(value.ndim == 0 or value.shape[0] != len(seeds) for value in arrays.values()):
        raise ValueError("every parameter must have a leading seed axis")
    metadata = {
        "schema_version": 1, "kind": "model_weights", "seeds": list(seeds),
        "env_step": int(env_step), "source_sha256": source_sha256,
        "tree_structure": str(structure),
        "leaves": [{"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}
                   for name, value in arrays.items()],
    }
    temporary = path.with_suffix(".tmp")
    try:
        with temporary.open("xb") as stream:
            np.savez(stream, metadata=json.dumps(metadata), **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return metadata


def load_models(path, template):
    """Restore weights into a matching batched pytree; return models, metadata.

    The template supplies Python structure without executable pickle data.
    Seed order comes from metadata and must be respected by the caller.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    leaves, structure = jax.tree_util.tree_flatten(template)
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        if (metadata["schema_version"] != 1 or metadata["kind"] != "model_weights"
                or metadata["tree_structure"] != str(structure) or len(metadata["leaves"]) != len(leaves)):
            raise ValueError("model archive does not match the template structure")
        restored = []
        for expected, entry in zip(leaves, metadata["leaves"]):
            value = archive[entry["name"]]
            if (list(value.shape) != entry["shape"] or str(value.dtype) != entry["dtype"]
                    or value.shape != expected.shape or value.dtype != expected.dtype):
                raise ValueError("model archive does not match the template shape/dtype")
            restored.append(jnp.asarray(value))
    return jax.tree_util.tree_unflatten(structure, restored), metadata
