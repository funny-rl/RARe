# JAX grid environments

Preserve the RARe_prev environment semantics while using explicit JAX state
and keys. File layout and class inheritance do not need to mirror the old code.

```python
import jax
from envs.grid import make_grid_env

env = make_grid_env("Bridge")  # Also CliffWalking and ZigZag.
env = make_grid_env("Bridge", random=True)  # Resample endpoints on reset.
key = jax.random.key(0, impl="threefry2x32")
key, reset_key = jax.random.split(key)
obs, state = jax.jit(env.reset)(reset_key)
key, step_key = jax.random.split(key)
obs, state, reward, terminated, truncated, info = jax.jit(env.step)(
    step_key, state, 3
)
```

Each step advances one primitive interaction. Action repetition belongs to
the rollout code. Reset explicitly after either done flag; returned observations
belong to the transition just taken. Termination takes precedence over the
time limit, as in the original code.

Observations are float32 vectors: agent=1, goal=2, overlap=3, with pits hidden.
The old code returned float64 despite declaring float32; this port follows the
declared dtype. NumPy and JAX RNG streams do not match from the same numeric
seed. Pass independent keys/states to vmap, and use scan for compiled rollouts.

`random=False` is the default and keeps the original fixed endpoints. Set
`random=True` when constructing any of the three environments. Randomized
variants use the base name and this option instead of separate R-prefixed names.

`make_grid_env("Bridge", random=True)` and
`make_grid_env("ZigZag", random=True)` preserve their base
terrain, actions, rewards, and time limits. At each reset, the start is sampled
uniformly from rows 0-7, columns 0-1, and the goal independently from rows 0-7,
columns 10-11. Each region has 16 cells, giving 256 possible pairs. Both
positions remain fixed for that episode; ZigZag's middle region is used for
transit. These region-based RZigZag candidates expand the old corner-only
RZigZag definition.

`make_grid_env("CliffWalking", random=True)` preserves the original cliff and samples the
start from rows 6-7, columns 0-1, and the goal independently from rows 6-7,
columns 10-11. These lower side regions have four cells each, giving 16 pairs.
Every pair requires navigating around the cliff. Terrain, actions, rewards,
and time limits match CliffWalking, and endpoints remain fixed per episode.

Reset has no hidden RNG state: the same layout and key reproduce the same
sample. Split and advance the environment stream before each reset to generate
new samples; a new key can still select the same pair by chance. Replaying the
same initial key and split sequence reproduces the episode sequence, including
when independent experiment keys are batched or reordered. Record the layout,
`random` option, initial seed, and RNG implementation with the experiment;
save current stream keys in checkpoints for resuming.

## Adding a grid

```python
from envs.grid import GridEnv, GridSpec

env = GridEnv(GridSpec(
    name="CustomGrid",
    shape=(4, 6),
    starts=((3, 0),),
    goals=((0, 5),),
    pits=((2, 2),),
    max_steps=50,
))
```

GridSpec stores an immutable tuple layout using (row, column) coordinates.
Default action order is left, up, right, down. Multiple starts and goals are
sampled independently and uniformly. Dimensions, action offsets, pits, episode
limit, and rewards are configurable. New transition or observation rules need
additional implementation using the same explicit-state reset/step convention.

The pure core.reset(key, params) and core.step(key, state, action, params) also
accept parameters explicitly. Equal-shape parameter pytrees can be stacked for
batching different layouts; different shapes compile separately.
