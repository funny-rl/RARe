"""Human-readable meaning of every experiment chart and local risk field.

W&B stores these descriptions in each run's config, and run.json keeps the
same snapshot. Step is the number of primitive environment interactions per
seed; vectorized seeds never share a training trajectory or replay buffer.
"""

TRAIN = {
    "env_step": "Hidden x-axis: primitive environment interactions completed by this seed, including warmup.",
    "interactions": "Primitive interactions collected in this logging chunk; smaller at a checkpoint boundary.",
    "base_q_updates": "Base Q critic optimizer updates in this logging chunk; warmup has none.",
    "loss_mean": "Mean base Q critic Huber loss over updates in this chunk; absent when no update ran.",
    "reward_sum": "Sum of immediate rewards collected in this chunk; may span parts of several episodes.",
    "episodes_finished": "Number of training episodes ending in this chunk.",
    "episodes_total": "Cumulative completed training episodes for this seed.",
    "episode_return_mean": "Mean complete-episode undiscounted return among episodes ending in this chunk; absent if none end.",
    "partial_episode_return": "Undiscounted reward accumulated in the still-open training episode at chunk end.",
    "partial_episode_length": "Primitive steps already taken in that still-open episode.",
    "replay_size": "Primitive transitions currently retained in this seed's bounded replay buffer.",
    "epsilon": "Epsilon for the next training action decision: 1 during warmup; DDQN/skip methods then follow the configured 1-to-0 schedule, while TAACDiscrete samples its actor with epsilon 0.",
    "next_epsilon": "Legacy alias of train/epsilon, retained for existing log readers.",
    "next_learning_rate": "Base critic optimizer learning rate after this chunk, including any configured decay.",
    "action_change_rate": "TAACDiscrete: fraction of post-warmup actions differing from the previous primitive action.",
    "actor_loss_mean": "TAACDiscrete: mean negative categorical actor objective over updates in this chunk.",
    "gate_entropy_mean": "TAACDiscrete: mean repeat/switch gate entropy over updates in this chunk.",
    "gate_temperature": "TAACDiscrete: learned repeat/switch entropy temperature after this chunk.",
    "actor_temperature": "TAACDiscrete: learned categorical proposal entropy temperature after this chunk.",
    "chunk_seconds": "Wall-clock seconds for this seed-batched training chunk; the same timing appears in every seed row.",
    "skip_loss_mean": "Mean skip-network Huber loss across skip updates in this chunk; absent if none ran.",
    "skip_q_updates": "Skip Q-network optimizer updates in this logging chunk.",
    "skip_replay_size": "Skip transitions currently retained in this seed's bounded skip replay buffer.",
    "decisions": "New action-and-skip decisions in this chunk; repeated primitive steps are not extra decisions.",
    "decisions_total": "Cumulative new action-and-skip decisions for this seed.",
    "selected_skip_mean": "Mean selected repeat length at decisions in this chunk, before an episode may end early.",
    "executed_skip_mean": "Mean primitive steps actually executed per completed repeat segment in this chunk.",
    "next_skip_epsilon": "Epsilon used to explore skip lengths at the next decision; equal to train/epsilon.",
    "next_skip_learning_rate": "Skip-network optimizer learning rate after this chunk.",
    "expected_policy_epsilon": "RARe skip-target policy epsilon: RARe-se normally uses its independent 0.1/0.2-to-0.01 schedule; RARe-se-behavior and RARe-sh use behavior epsilon. RARe-sh-eval uses the independent schedule.",
    "next_evaluation_epsilon": "Legacy alias of train/expected_policy_epsilon.",
    "evaluation_loss_mean": "RARe-se auxiliary Expected SARSA critic loss over this chunk, not an evaluation-episode score.",
    "next_evaluation_learning_rate": "RARe-se auxiliary critic optimizer learning rate after this chunk.",
    "return": "Undiscounted reward sum of one completed exploratory training episode.",
    "length": "Primitive environment steps in that completed training episode, not the number of skip decisions.",
    "terminated": "True if that training episode ended at a goal or pit.",
    "truncated": "True if that training episode ended at the environment time limit.",
    "episode_index": "One-based completed training episode number within this seed.",
}

EVAL = {
    "episodes": "Number of frozen greedy evaluation episodes at this checkpoint; random grids enumerate every start-goal pair.",
    "reward_sum": "Mean undiscounted episode reward across those evaluation episodes.",
    "reward_sum_std": "Population standard deviation of those episode rewards; zero for one fixed-grid episode.",
    "length_mean": "Mean primitive environment steps per evaluation episode.",
    "decisions_mean": "Mean action-and-skip decisions per evaluation episode; intermediate repeat steps are excluded.",
    "success_rate": "Fraction of evaluation episodes reaching the goal.",
    "pit_rate": "Fraction of evaluation episodes ending in a pit or cliff.",
    "timeout_rate": "Fraction of evaluation episodes reaching the environment time limit.",
    "policy_epsilon": "Exploration rate used to choose evaluation actions and skips; always zero for greedy evaluation.",
    "risk_epsilon": "Training action epsilon at this checkpoint, frozen for exact behavior-policy True Risk calculations.",
    "risk_mae_mean": "Mean across valid lengths of absolute error versus V_star minus V_behavior; per valid decision this equals skip-value absolute error divided by gamma^length.",
    "risk_mae_variance": "Population variance across those per-length risk MAEs; measures length sensitivity, not seed uncertainty.",
    "skip_value_mae_mean": "Mean across skip lengths of each length's mean absolute skip-value error under the behavior policy.",
    "skip_value_mae_variance": "Population variance across those per-length skip-value MAEs.",
    "target_risk_epsilon": "RARe-se or RARe-sh-eval only: epsilon defining the skip-target policy pi_e; RARe-se-behavior matches behavior epsilon but uses a distinct critic.",
    "target_risk_mae_mean": "RARe-se or RARe-sh-eval only: absolute error versus V_star minus V_pi_e under its skip-target policy.",
    "target_risk_mae_variance": "RARe-se or RARe-sh-eval only: population variance across per-length risk MAEs under pi_e.",
    "target_skip_value_mae_mean": "RARe-se or RARe-sh-eval only: mean per-length skip-value MAE against pi_e.",
    "target_skip_value_mae_variance": "RARe-se or RARe-sh-eval only: population variance across per-length skip-value MAEs under pi_e.",
}

LOCAL_RISK_DETAILS = {
    "epsilon": "Frozen behavior-policy epsilon at this checkpoint; target_policy.epsilon records a distinct pi_e epsilon when present.",
    "bellman_residual": "Maximum absolute finite-horizon Bellman equation residual for the exact behavior-policy Grid solver.",
    "optimal_bellman_residual": "Maximum absolute finite-horizon Bellman residual for Q_star, whose continuation maximizes at every step.",
    "risk_valid_lengths": "Number of skip lengths with at least one nonterminal visited decision.",
    "risk_valid_fraction": "Valid nonterminal decision-length pairs divided by all observed decision-length pairs.",
    "lengths[].count": "Visited decision count for this skip length, including terminal endpoints.",
    "lengths[].risk_count": "Visited decisions with a continuing endpoint at this skip length; denominator of risk MAE.",
    "lengths[].ended_count": "Visited decisions whose endpoint ended the episode at or before this length.",
    "lengths[].risk_excluded_fraction": "Ended count divided by all visits for this length; null when count is zero.",
    "lengths[].value_mae": "Mean absolute skip-value error at this length; includes terminal endpoints.",
    "lengths[].risk_mae": "Mean absolute implied-risk error at this length; excludes terminal endpoints.",
    "target_policy": "RARe-se or RARe-sh-eval: identical detail schema recomputed for skip-target policy pi_e.",
}

LOCAL_TRAIN = {
    "base_q_updates_total": "Cumulative base Q-network optimizer updates, retained in local metrics.jsonl only.",
    "skip_q_updates_total": "Cumulative skip Q-network optimizer updates, retained in local metrics.jsonl only.",
}

METRIC_DEFINITIONS = {"train": TRAIN, "eval": EVAL,
                      "local_train": LOCAL_TRAIN, "local_risk_details": LOCAL_RISK_DETAILS}
