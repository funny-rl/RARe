# Grid experiment launchers

Each environment directory contains `ddqn.sh`, `temporl.sh`, `ute.sh` and
`rare.sh`. The six directories are `cliffwalking`, `bridge`, `zigzag`,
`rcliffwalking`, `rbridge` and `rzigzag`. The `r` variants enable
reproducible start/goal sampling on every reset.

Launch one experiment:

```bash
bash /data/RARe/scripts/grid/cliffwalking/ddqn.sh --gpu 0 --seeds 0:20
bash /data/RARe/scripts/grid/bridge/temporl.sh --gpu 1 --seeds 0:20
bash /data/RARe/scripts/grid/zigzag/ute.sh --gpu 2 --seeds 0:20
bash /data/RARe/scripts/grid/rbridge/rare.sh --gpu 3 --seeds 0:20
```

Each command runs in the foreground. `0:20` means seeds 0 through 19,
batched on the selected GPU. Only physical GPUs 0,1,2,3 are allowed.
If omitted, GPU and seeds follow the unified CLI defaults: GPU 0 and seed 0.

Fixed Grid launchers use the YAML budgets: CliffWalking 15,000, Bridge 23,000,
and ZigZag 35,000 primitive steps per seed. Random Grid launchers set
CliffWalking to 50,000, Bridge to 150,000 and ZigZag to 200,000 steps per seed. These
launcher defaults can be changed with an explicit `--total-steps N` argument.
For Random Bridge and ZigZag, launcher defaults keep primitive replay at
100,000 and skip replay at 300,000 records per seed even as training continues
past 100,000 steps. DDQN has no skip replay. All Grid observations are stored
as exact uint8 values 0–3 and converted back to float32 when sampled.
The current 20-seed campaign passed `--model-save-interval 0` to every run.

All remaining options go to `python -m experiments.train` unchanged.
The script fixes the environment, algorithm, random mode and W&B project;
use another script to change them. Hyperparameters come from `configs/grid.yaml`,
then an optional `--config` YAML file and explicit CLI overrides.

RARe supports `--rare-q-mode separate|shared` on all six environment launchers.
The default is `separate`, also set by `algorithms.RARe.rare_q_mode` in YAML.
Separate learns an auxiliary Expected SARSA critic for its own evaluation policy.
Separate uses that critic's own greedy actions to define its expected-target
policy. Shared uses the base DDQN Q for both greedy actions and target values.
The base critic keeps its DDQN objective in both modes. Shared has no auxiliary
network, optimizer, auxiliary replay sample or auxiliary-loss log; its base Q
learning rate is `learning_rate`, and `eval_learning_rate` applies only to
Separate. Both use the same epsilon schedule as the training behavior policy:
1 during warmup, then 1 to 0 over the configured decay steps. Equal epsilon
does not make Separate's distinct greedy policy identical to behavior.

```bash
bash /data/RARe/scripts/grid/cliffwalking/rare.sh --gpu 0 --seeds 0:20 --rare-q-mode shared
bash /data/RARe/scripts/grid/cliffwalking/rare.sh --gpu 1 --seeds 0:20 --rare-q-mode separate
```

RARe run names and W&B groups use `RARe-sh` for Shared and `RARe-se` for
Separate; default output directories use the corresponding lowercase label.
W&B projects stay unchanged. The mode is also recorded
in local run metadata, resolved YAML and W&B config. Both modes save base and
skip weights at the usual five boundaries; Separate additionally saves the
auxiliary online/target weights. A custom `--wandb-name` still replaces the
automatic name prefix.

```bash
bash /data/RARe/scripts/grid/rzigzag/rare.sh \
    --gpu 0 --seeds 7,19 \
    --max-skip 12 --model-save-interval 5000 \
    --wandb-mode online
```

W&B projects are fixed to `RARe-CliffWalking`, `RARe-Bridge`, `RARe-ZigZag`,
`RARe-RCliffWalking`, `RARe-RBridge`, `RARe-RZigZag`. All five model variants
for an environment share that project. A launcher rejects `--wandb-project`
overrides; direct unified CLI calls also reject a mismatching project name.
Each seed has its own W&B run, grouped by algorithm (DDQN, TempoRL, UTE,
RARe-sh or RARe-se). Thus `--seeds 0:20` creates 20 runs for seeds 0 through 19. Default
run names include algorithm, environment, timestamp, a launch identifier and
`-seed<N>`, for example `DDQN-CliffWalking-<timestamp>-<id>-seed0`.
`--wandb-name` changes the shared name prefix; the seed suffix is retained.
Each run uses common `train/*` and `eval/*` metric names for comparison and
group aggregation. Training episode records are flat `train/return`,
`train/length`, `train/terminated`, `train/truncated` and `train/episode_index`,
using the episode's ending `env_step` on the same axis as training summaries.
`env_step` is sent only as axis data; its automatic chart and summary are disabled.
Episode and summary rows are written in step order and every completed episode
is retained. Local `episodes.jsonl` keeps the individual raw records.
`train/epsilon` records the epsilon-greedy action-selection rate for the next
decision at each logged step (1 during warmup). RARe also records its separate
`train/expected_policy_epsilon`. Greedy evaluation logs `eval/policy_epsilon=0`;
Grid risk diagnostics log their counterfactual rate as `eval/risk_epsilon`.
Batched GPU training stays simultaneous: one command
finishes once all selected seeds reach the per-seed interaction budget.
Other W&B settings follow the unified CLI, including its current online
default. Use `--wandb-mode disabled` to disable W&B or
`--wandb-mode offline` to record locally.

The launcher changes to the project root before training, so relative
`--config` and `--output` paths are relative to the project root.
Default outputs are unique directories under `runs/`; default model saves
occur at 20%,40%,60%,80%,100% of the per-seed interaction budget.
`--model-save-interval 0` disables every model save, including the final one.
Use `--help` to inspect training options without starting training.

## Performance evaluation

`configs/grid.yaml` separates performance evaluation from learning settings:

```yaml
evaluation:
  interval_fraction: 0.01
  interval_steps: null
  episodes: 1
  random_episodes: all
  risk_diagnostics: true
```

Default evaluation starts after warmup. The interval remains 1% of the total
primitive-step budget, anchored at warmup end, and the final step is included.
For CliffWalking this gives 2150,2300,...,14900,15000 with total_steps=15000
and warmup_steps=2000. Bridge starts at 3230 with interval 230; ZigZag starts
at 3350 with interval 350. Fractional offsets round up and duplicate points
are removed. All-warmup budgets have no evaluation. Previously completed
runs keep their original checkpoint records until explicitly replaced.
With the Random Grid launcher budgets, RCliffWalking starts at 2500 and ends
at 50000; RBridge starts at 4500 and ends at 150000; RZigZag starts at 5000
and ends at 200000.
Fixed environments evaluate one episode per checkpoint. Random environments
enumerate every start-goal pair exactly once: RCliffWalking evaluates 16 pairs,
RBridge and RZigZag each evaluate 256. The model, transitions and greedy policy
are deterministic, so repeating the same pair adds no information. Aggregates
weight each pair equally, matching the independent uniform training reset
distribution. Training still samples random endpoints at every reset.
An environment's `evaluation` section can override the global YAML settings.
CLI overrides are also available:

```bash
bash /data/RARe/scripts/grid/rbridge/rare.sh --gpu 0 --seeds 0:20 \
    --eval-interval-fraction 0.01 --eval-episodes all
```

`--eval-interval-steps N` overrides the fraction with every N primitive steps
after warmup plus the final step; `--eval-interval-steps 0` disables evaluation. A positive
YAML `interval_steps` likewise takes precedence over `interval_fraction`.
`--eval-episodes all` enumerates all grid start-goal pairs; positive N explicitly
switches to N sampled episodes for either fixed or random mode. YAML `episodes`
and `random_episodes` accept a positive integer or `all`. `run.json`,
`resolved.yaml` and W&B config record the requested option, actual episode
count, reset mode and start-major/goal-minor case order. Exhaustive mode requires
the GridEnv adapter and is not applied automatically to other backends.

Evaluation freezes the current online action/skip networks and uses greedy
actions and skip lengths, without epsilon exploration. UTE retains its
mean + uncertainty_factor * sample standard deviation selector. Each episode
ends on termination or time limit, even during a repeat. Evaluation has its
own environment state and a checkpoint key derived from the existing
evaluation RNG root plus training step. It never changes training RNG keys,
replay, optimizer, pending repeats or the training interaction budget.

W&B logs `eval/reward_sum`, `eval/reward_sum_std`, `eval/success_rate`,
`eval/pit_rate`, `eval/timeout_rate`, `eval/length_mean`, `eval/decisions_mean`
and `eval/episodes` against `env_step` in each seed's run. Local
`evaluations.jsonl` stores aggregates and checkpoint RNG keys;
`eval/reward_sum` is the mean of undiscounted episode reward sums, and
`eval/reward_sum_std` is their population standard deviation. The local
aggregate records use the same names without the `eval/` prefix.
`evaluation_episodes.jsonl` stores each episode's `reward_sum` and outcome; exhaustive cases also
include case_index (zero-based), start and goal coordinates. The resolved protocol
is recorded in `run.json`, `resolved.yaml` and W&B config. Evaluation time and
compilation time are separate summary fields. Evaluation runs even with W&B
disabled. Separate RARe's `train/evaluation_loss_mean` remains the auxiliary critic's
training loss, separate from the exact diagnostics below. RODR is not implemented.

## Exact Grid skip-value and True Risk errors

`evaluation.risk_diagnostics: true` enables the diagnostic by default for
TempoRL, UTE and RARe, on GridEnv only. DDQN keeps ordinary performance
evaluation without risk fields. `--no-risk-diagnostics` disables these
measurements while retaining performance evaluation; `--risk-diagnostics`
enables them. No additional model checkpoints are needed. Existing saved
weights can be passed to the same pure diagnostic functions; there is no
separate offline-analysis launcher in this change.

At each evaluation checkpoint, freeze the behavior epsilon-greedy policy for
the common True Risk comparison. All algorithms use their online base Q to
choose greedy action identities and the same training-epsilon schedule;
during uniform warmup epsilon is 1. Separate RARe additionally freezes its
own evaluation-critic policy at the same epsilon for a separately named
target-policy diagnostic. The network's predicted Q magnitudes never supply
exact reference values. Ties follow the existing first-argmax rule.
Each frozen policy is defined on every cell for every possible goal, including
cells not visited by greedy evaluation. Continuation uses primitive actions
under this policy, as in the skip bootstrap definition, rather than further
skip decisions. The reference epsilon is fixed throughout the calculation.

`diagnostics/grid_risk.py` separates five operations:

1. `grid_policy_probabilities` constructs pi[goal, cell, action].
2. `solve_grid_policy_values` computes exact finite-horizon Q^pi for the
   checkpoint's frozen epsilon policy. `solve_grid_optimal_values` computes
   Q* with a max over actions at **every** later step. Both use the known
   transition/reward model. A JAX scan builds horizons 0 through the episode limit. Goals and
   pits are absorbing, Q_0 is zero, and timeout ends continuation after the
   last observed reward. There is no neural function approximation, Monte
   Carlo estimation or iterative convergence tolerance. Float32 rounding
   still applies; separate Bellman residuals check both horizon equations.
3. `score_decision_risks` repeats the actual decision's action separately for
   every candidate length 1..J. It computes discounted prefix return G_j,
   endpoint state s_j, exact expected skip value Y_pi, optimal-control baseline
   Y_max, and the full-continuation risk used in this experiment:

   ```text
   Y_pi  = G_j + gamma^j * sum_b pi(b|s_j) Q^pi(s_j,b)
   Y_max = G_j + gamma^j * max_b Q*(s_j,b)
   R_true = max_b Q*(s_j,b) - sum_b pi(b|s_j) Q^pi(s_j,b)
   R_hat  = (Y_max - predicted_skip_value_j) / gamma^j
   ```

   The first term assumes optimal control through the entire remaining
   episode; the second follows the checkpoint policy through continuation.
   This operational definition extends the current manuscript equation,
   which has not yet been edited. The implied risk is not clipped,
   and UTE uses its ensemble mean without its skip-selection uncertainty
   penalty. Goal, pit or timeout freezes prefix return for every later
   candidate length: Y_pi=Y_max=G and R_true=0. Prediction errors at these
   lengths still count in skip-value MAE. Risk MAE excludes the ending length
   and every later length, because there is no continuing control to assess.
   Reconstruction uses requested gamma^j, not the replay bootstrap discount.
4. `evaluate_grid_risk` attaches a pure observer to the existing greedy
   evaluator. It scores each actual decision point once; intermediate
   repeated-action states are not extra probes. All unselected lengths are
   inspected; only nonterminal candidates enter risk sums and counts.
   Repeated visits count as repeated decisions. The
   observer accumulates sums on-device instead of storing dense trajectories
   or recalculating skip outputs. It does not change actions, RNG or state.
5. `aggregate_risk_errors` first computes MAE_j over visited decisions across
   evaluation episodes. Value MAE uses all candidates; risk MAE uses only
   candidates still continuing after length j. It then records the equal-length
   mean and population variance of MAE_j over lengths with valid samples.
   Empty lengths are null and excluded, never treated as zero. Episodes with more decisions contribute
   more probes; initial reset pairs still have equal performance-evaluation
   weight. The variance describes differences between lengths, not uncertainty
   across seeds or a confidence interval. Host aggregation uses float64;
   empty data produces null, not zero.

W&B and local `evaluations.jsonl` receive:

```text
eval/risk_mae_mean             eval/risk_mae_variance
eval/skip_value_mae_mean       eval/skip_value_mae_variance
```

Local JSONL field names omit `eval/`. `risk_diagnostics.jsonl` additionally
stores each length's two MAEs, total count, valid risk count, ended-prefix
count and risk-excluded fraction. It also retains frozen epsilon, policy and
optimal Bellman residuals, risk-valid fraction over all candidate pairs and number of lengths
with valid risk samples for each seed/checkpoint. Per-length
curves and raw decision traces are not sent to W&B. `run.json`, resolved
YAML and W&B config record the full measurement protocol. The local source
snapshot includes the diagnostics module.

The common fields measure accuracy under the same behavior-policy definition
on each algorithm's own visited decision distribution. Separate RARe's
`target_*` fields use the same formula but its distinct evaluation policy,
so they diagnose calibration to the actual skip-learning target rather than
the cross-algorithm behavior reference. Both policies share the same numeric
epsilon at each checkpoint; their greedy actions can differ.
TempoRL/UTE are compared against the expected-value reference even though
their training targets use max/DDQN bootstrap, so this is not their native
TD loss. Each nonterminal candidate's inferred risk error is its skip-value
error divided by gamma^j, up to rounding. The logged value MAE additionally
includes terminal candidates, so its aggregate is not a direct rescaling of
logged risk MAE. This measurement does not independently
validate a separate risk head or imply a fall probability. It does not include
an error in a learned max reference critic, because Y_max is exact.
Reconstruction requires gamma > 0 and representable gamma^J; invalid settings
raise an explanatory error and can still use performance-only evaluation.
Grid observations omit elapsed time while the exact oracle includes remaining
time, so aliasing near timeout can contribute to the measured value error.
Terminal exclusion changes the probe distribution: dangerous long repeats can
drop out after hitting a pit. Read valid counts/excluded fractions alongside
eval/pit_rate rather than treating lower risk MAE alone as evidence of safety.

`run.sh` is the shared launcher. No script schedules additional experiments,
selects another GPU automatically or starts background jobs. Shell launchers
are included in each run's local source archive.

## Offline normalized AUC analysis

Analysis runs only when explicitly invoked after the entire environment cohort
finishes. It never runs inside training. The default cohort contains DDQN,
TempoRL, UTE, RARe-sh and RARe-se, with the same seed set. Every declared run
must have completed its full training budget and post-warmup evaluation grid.
Missing algorithms, unfinished seeds, duplicate seeds, mismatched environments,
evaluation schedules or reset protocols stop analysis before creating outputs.
Use `--algorithms` only to explicitly declare a different complete comparison
cohort. Multiple input directories may supply disjoint seed shards.

For each start-goal case c and seed s, the primary score is:

```text
z_s,c(t) = clip((episode_reward_s,c(t) - L) / (U_c - L), 0, 1)
AUC_s,c  = trapezoidal_integral(z_s,c(t), observed_env_steps) / observed_duration
AUC_s    = mean_c(AUC_s,c)
```

The common floor L defaults to -100 for all six grids. Values below it clip
before integration. U_c is the exact maximum undiscounted reward for that
case, computed on CPU from the known grid model by finite-horizon optimal
dynamic programming, including the episode time limit. Each algorithm uses
the same oracle and floor. Rewards above the theoretical ceiling raise an
error. Every U_c must exceed L. The resulting AUC is between 0 and 1,
with higher values better. This score combines learning speed and reward
level over the specified window; it is not a time-to-convergence threshold.

Fixed-grid ceilings are -13 for CliffWalking, -17 for Bridge, and -19 for
ZigZag. Random grids retain separate ceilings and curves for all 16
RCliffWalking cases or 256 RBridge/RZigZag cases. Case-normalized AUC requires
exhaustive evaluation and reads `evaluation_episodes.jsonl`, validating every
case index, start, goal, seed and checkpoint. Aggregate rewards alone cannot
recover this metric. Do not normalize the average reward by an averaged
ceiling: clipping and different case ranges make that a different statistic.
`--reward-max` can override a single-case ceiling, but is rejected for multiple
cases. Legacy `return_mean`/`return` and current `reward_sum` logs are supported.

Case AUCs are averaged within each seed before computing seed statistics.
The default paper/report metric is mean AUC ± sample standard deviation
(ddof=1) across all independently trained seeds. Every declared completed seed,
including failed seeds, contributes without trimming. For 20 seeds, all 20
seed means enter the statistics. IQM, confidence intervals and bootstrap
calculations are excluded from this pipeline.

Only observations strictly after warmup enter the integral, on the original
primitive-step axis. Analysis uses the first actual post-warmup evaluation
through the final evaluation and never extrapolates into missing intervals.
Existing CliffWalking logs cover 2100 through 15000; newly scheduled runs
cover 2150 through 15000. The actual bounds and duration are saved. Cohorts
with different retained evaluation grids are rejected.

Analyze the five completed CliffWalking experiments without training or W&B:

```bash
cd /data/RARe
python -m experiments.analyze \
    runs/campaigns/grid-all-20261012T145908Z/experiments/CliffWalking/DDQN \
    runs/campaigns/grid-all-20261012T145908Z/experiments/CliffWalking/TempoRL \
    runs/campaigns/grid-all-20261012T145908Z/experiments/CliffWalking/UTE \
    runs/campaigns/grid-all-20261012T145908Z/experiments/CliffWalking/RARe-sh \
    runs/campaigns/grid-all-20261012T145908Z/experiments/CliffWalking/RARe-se \
    --reward-min -100 --output runs/analyses/cliffwalking-auc
```

The output directory must be new. `auc_summary.csv` contains the requested
statistics; `seed_auc.csv` stores each seed mean and raw reward area for audit;
`case_auc.csv` preserves every normalized case AUC with its endpoints and bounds.
`auc_summary.json` retains the protocol, per-case oracle rewards, clipping
counts, input/source hashes, the sample-standard-deviation convention and timing.
The default exports only CSV and JSON. Explicit `--plot` also creates `auc.png`
and `auc.pdf` showing mean ± sample standard deviation. `--no-plot` is retained
as an explicit data-only option. Input logs and W&B are never changed.

The implementation has three stages: `load_completed_cohort` in
`experiments/analyze.py` validates completion and reads each input log once;
`grid_reward_ceiling`, `normalize_rewards` and `seed_auc` in `diagnostics/auc.py`
compute the exact bounds and integrate the case curves; `auc_statistics`
computes the mean and sample standard deviation of the small seed-mean vector. `write_report` exports results.
The analysis modules import no JAX or W&B and allocate no GPU memory. Case
curves are integrated once and statistical aggregation is linear in the seed
count, without resampling. Matplotlib is imported only when `--plot` is requested.

## Local profiling without W&B

Run a short RARe/RBridge case on GPU 0 and collect a host profile:

```bash
cd /data/RARe
XLA_PYTHON_CLIENT_PREALLOCATE=false python -m cProfile \
    -o /tmp/rare-profile.pstats -m experiments.train \
    --algorithm RARe --env Bridge --random --seeds 0:4 --gpu 0 \
    --wandb-mode disabled --total-steps 1001 --warmup-steps 100 \
    --chunk-steps 256
python -c "import pstats; pstats.Stats('/tmp/rare-profile.pstats').strip_dirs().sort_stats('cumtime').print_stats(20)"
```

Disabled mode imports no W&B SDK and uploads no records. Local logging and
the five default model saves and performance evaluation remain enabled. Add
`--eval-interval-steps 0` to profile training without performance evaluation.
The run.json summary separates initialization, training-loop compilation,
training/logging, evaluation execution and evaluation compilation time. It also
records training-loop compilation_count, compiled_chunk_steps and synchronized
execution_and_transfer_seconds. The compilation count excludes initialization
and small helper operations. cProfile measures host time and blocking waits,
not a breakdown of GPU kernels.

The training loop compiles one fixed metric-buffer capacity and supplies the actual
chunk length dynamically. Its capacity is capped by the largest gap between evaluation
and model-save boundaries, and by a positive model-save interval. With the
default requested chunk of 256, CliffWalking allocates 150 slots, Bridge 230
and ZigZag 256. A dynamic while loop stops at the actual collection count,
including short remainder chunks and the final training budget. ZigZag's
280-step evaluation intervals still collect 256 then 24 steps, but the second
chunk executes only 24 iterations rather than 256. Unused metric slots retain
inactive sentinel values without extra collection-loop iterations or RNG draws.
Collection/logging/evaluation/save boundaries stay unchanged, and changing
the actual count does not trigger another compilation. Training metrics, counters,
schedules and raw RNG keys are returned together by the compiled chunk and
transferred once for logging. Risk and episode results likewise share one
evaluation transfer. Learning objectives, update order and RNG draws are
unchanged; these changes apply to newly launched processes.
Short cold-start timings primarily measure initialization/compilation;
steady-state speed needs longer runs and repeated measurements.
