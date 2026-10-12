# Replay buffers

`replay.py` stores discrete-action primitive transitions as JAX pytrees.
`skip.py` separately stores action-repeat segments and their derived labels.

## Primitive replay

```python
import jax
from buffers import add_transition, init_replay, sample_batch
from envs.grid import make_grid_env
from rng import init_rngs

env = make_grid_env("Bridge", random=True)
rngs = init_rngs(42)
replay = init_replay(capacity=23_000, obs_shape=env.observation_shape)

environment_key, reset_key, step_key = jax.random.split(rngs.environment, 3)
rngs = rngs._replace(environment=environment_key)
obs, env_state = env.reset(reset_key)
next_obs, env_state, reward, terminated, truncated, _ = env.step(
    step_key, env_state, 2
)
replay = jax.jit(add_transition)(
    replay, obs, 2, reward, next_obs, terminated, truncated,
    previous_action=-1,
)

replay_key, sample_key = jax.random.split(rngs.replay)
rngs = rngs._replace(replay=replay_key)
batch = jax.jit(sample_batch, static_argnames=("batch_size",))(
    sample_key, replay, batch_size=64
)
```

`init_replay(capacity, obs_shape)` allocates one experiment's replay and sets
`write_index` and `size` to zero. The storage is on the selected JAX device;
initialize within a device context when assigning experiments to GPUs.

`add_transition(...)` returns a new replay state. It stores one observation,
scalar integer action, scalar reward, actual successor observation, separate
termination/truncation flags, and optional scalar previous action. Missing
previous action is encoded as -1, matching the old TAAC-Discrete convention.
An episode's first transition must use -1; following TAAC transitions must
supply the action executed on the preceding primitive step. Other methods
may leave the optional field absent. At capacity, insertion overwrites the
oldest transition. The buffer does not reset environments, pad transitions,
or derive learning targets.

`sample_batch(key, state, batch_size)` samples uniformly with replacement
from the `size` written slots. A batch may be larger than the current replay.
It returns a `ReplayBatch` with observations shaped `(B, *obs_shape)` and
all action/reward/flag fields shaped `(B,)`, rather than `(B, 1)`. Keep this
convention in loss code to avoid unintended `(B, B)` broadcasting.

Observations/rewards are float32, actions/previous actions are int32, and
end flags are bool. `terminated` and `truncated` remain separate; the learner
will decide the time-limit bootstrap rule. Store a terminal successor before
resetting so that a subsequent episode's initial observation cannot replace it.

Sampling requires `size > 0`. Eager validation reads the count scalar and
raises for an empty buffer. Under JIT there is no host read; the compiled
caller must gate sampling with `jax.lax.cond`, or diagnostics can use
`checkify.checkify` to detect an empty buffer. Warmup/update eligibility stays
with the training loop. Capacity, observation shape, and sample batch size
are static compilation inputs; changing them can require compilation again.

Every experiment owns its replay state and `rngs.replay` stream. Initialize
independent states and use `vmap` over the leading experiment axis; do not
share one replay across independent learners. The same replay and key produce
the same sampled batch. Save the entire `ReplayState` and current replay key
in checkpoints to resume the same stream. Sampling has no NumPy/Torch RNG,
and the compiled path neither transfers a sampled batch to the host nor
changes the caller's state/key.

## Skip replay

`init_skip_replay(capacity, obs_shape)` allocates one independent skip replay.
`add_skip_segment(state, segment, gamma, augment_subsegments=True)` inserts
the supported labels from a fixed-shape `SkipSegment`. Its observations have
shape `(J+1, *obs_shape)` and rewards/end flags have shape `(J,)`, with scalar
action, originally selected duration, and actual executed count. Only the
executed prefix is real; its final observation must be preserved before reset.
Padding after that prefix is ignored. J is a static array dimension, while
the selected and executed lengths can vary between seeds and episodes.

For a selected duration of six that terminates or hits the episode's maximum
step count at step two, the
original start/action receives labels 2,3,4,5,6 with exactly the same stored
discounted return `r1 + gamma*r2`, the actual final observation, and zero
bootstrap discount. Later duration adds no reward. The length-one prefix
has its own return and nonterminal discount. Ordinary augmentation also
creates subsegments from later real start states, preserving the old
end-major insertion order. This example creates 11 records when enabled.
Equivalent labels from a single start have identical training targets; their
learned network predictions are not forcibly tied.

Episode-end equivalence is mandatory even with `augment_subsegments=False`.
That option disables ordinary prefix/suffix augmentation, leaving labels
2,3,4,5,6 from the original start in this example (five records). This
deliberately differs from legacy `use_data_aug=false`, which inserted only
the selected length. No label extends beyond the originally selected duration.

The episode max-step limit is treated as an absorbing boundary, so truncation
receives the same extension and zero bootstrap discount as termination.
This is an explicit finite-episode objective; it does not estimate continuing
task value beyond a time-limit wrapper. The original `terminated` and
`truncated` flags remain separate. The last observed reward is included once,
and all later rewards are zero; repeating the last reward would change the
target. Actual final observations are preserved, so zero-vector observations
are unnecessary.

Partial segments interrupted by a training budget without either end flag
only produce labels for observed steps. With ordinary augmentation disabled,
store the full observed prefix. Preserve the original selected length
separately. Collectors must not mark an administrative training-budget stop
as episode truncation.

`SkipBatch.skips` is the candidate duration whose network output is trained.
`executed_steps` counts real interactions in that particular derived record.
`selected_skips` and `segment_steps` retain the original repeat's selected and
executed counts; they may differ from an augmented record's lengths.
`returns` contains the already discounted reward sum. `discounts` equals
`gamma**executed_steps`, set to zero for either episode-end flag. The future
learner can construct `returns + discounts*next_value`; primitive-value
targets must use the same episode-end rule. All scalar fields have shape `(B,)` when
sampled. Observations/returns/discounts are float32, integer fields int32,
and flags bool.

`sample_skip_batch(key, state, batch_size)` uses uniform sampling with
replacement, with the same nonempty-buffer and explicit-key contracts as
primitive replay. Insertion uses a fixed-shape masked scatter on device;
even a segment larger than capacity retains the last accepted records
deterministically. Use jit/vmap for independent seed states. Shape/type checks
run during tracing; value preconditions raise eagerly and are available to
`checkify.checkify` in compiled diagnostics. Normal compiled callers must
provide valid collected segments. Save the entire skip replay state and
the sampling stream key to restore the same sampling sequence.
