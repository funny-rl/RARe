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

## WindShift

`WindShift` uses a 5×9 corner-to-corner layout. `descending` starts at `(0, 0)`
and ends at `(4, 8)`; `ascending` is its vertical mirror. The outer rows have
five central pits each. Four unused corner cells are walls, which block movement
without ending the episode. Pits and walls both remain hidden in the grid
observation. Rewards retain the other grids' values: step and goal `-1`, pit
`-100`. Wind is applied after the intended move with probability 0.5 by default;
reaching the goal or a pit on the intended move ends the step before wind.

```python
from envs.grid import make_wind_shift_env

env = make_wind_shift_env(switch_interval=250, layout="descending",
                          initial_wind="north", observe_wind=False)
```

Wind direction reverses after every `switch_interval` primitive training steps,
independent of episode resets. `switch_interval=0` disables reversals for a
stationary control. The direction used at global step `t` is the initial
direction for `0 <= t < K`, its opposite for `K <= t < 2K`, and so on.
Training resets preserve the current global step. Evaluations start their own
counterfactual rollout at the checkpoint's training step; they never advance
the training clock. The deterministic-grid exact risk diagnostic is disabled
for this stochastic environment.

The hidden condition observes only agent and goal positions and uses compact
grid-index replay storage. `observe_wind=True` appends one float, `0` for north
and `1` for south, and uses raw replay storage. Balance the `descending` and
`ascending` layouts and the two initial directions across experimental runs;
they are explicit run settings rather than additional episode randomness.

The training command accepts the same settings, with an explicit interval:

```bash
python -m experiments.train --env WindShift --algorithm RARe \
  --wind-switch-interval 250 --wind-layout descending --wind-initial north \
  --wind-observation hidden --platform cpu --wandb-mode disabled
```

## Discrete TAAC baseline

`TAACDiscrete` implements a categorical adaptation of
[Yu et al. (2021)](https://proceedings.neurips.cc/paper_files/paper/2021/file/f337d999d9ad116a7b4f3d409fcc6480-Paper.pdf), not
the earlier rebuttal's one-step approximation. At every primitive step its
actor samples a candidate conditioned on the observation and previous action;
a learned-temperature softmax of the critic's Q values decides whether to
repeat or switch. The critic uses chronological replay windows and compare-through
targets, cutting off when the current policy's sampled action differs from
the recorded next action or the episode ends. The actor optimizes the exact
categorical expectation of the two-stage objective, with separate learned
temperatures for actor and gate entropy. Evaluation uses the deterministic
mode of each stage. A categorical candidate can equal the previous action;
both latent gate choices then execute the same action, as implied by the
paper's two-stage policy.

```bash
python -m experiments.train --env WindShift --algorithm TAACDiscrete \
  --wind-switch-interval 250 --compare-horizon 10 \
  --wind-observation hidden --platform cpu --wandb-mode disabled
```

The original paper and released implementation are continuous-action methods;
the categorical actor, exact discrete actor expectation, and action-equality
comparison are explicit adaptations. `TAACDiscrete` here is therefore a
paper-informed discrete baseline, not an exact reproduction of their
continuous-control results. The earlier rebuttal `TAACDiscrete` used one-step
targets and `argmax` plus epsilon exploration, so its results are not directly
reproducible by this implementation. The paper's task-dependent reward
normalization is not applied to Grid; all compared Grid algorithms learn from
the same raw rewards. `replay_capacity` sets the TAAC transition buffer size,
and `compare_horizon` sets its multi-step backup window.
