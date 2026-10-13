"""Human-readable meaning of every experiment chart and local risk field.

W&B stores these descriptions in each run's config, and run.json keeps the
same snapshot. Step is the number of primitive environment interactions per
seed; vectorized seeds never share a training trajectory or replay buffer.
"""

TRAIN = {
    "env_step": "Hidden x-axis: primitive environment interactions completed by this seed, including warmup.",
    "interactions": "Primitive interactions collected in this logging chunk; smaller at a checkpoint boundary.",
    "updates": "Base DDQN optimizer updates in this chunk, excluding warmup.",
    "updates_total": "Cumulative base DDQN optimizer updates for this seed.",
    "loss_mean": "Mean base DDQN Huber loss over updates in this chunk; absent when no update ran.",
    "reward_sum": "Sum of immediate rewards collected in this chunk; may span parts of several episodes.",
    "episodes_finished": "Number of training episodes ending in this chunk.",
    "episodes_total": "Cumulative completed training episodes for this seed.",
    "episode_return_mean": "Mean complete-episode undiscounted return among episodes ending in this chunk; absent if none end.",
    "partial_episode_return": "Undiscounted reward accumulated in the still-open training episode at chunk end.",
    "partial_episode_length": "Primitive steps already taken in that still-open episode.",
    "replay_size": "Primitive transitions currently retained in this seed's bounded replay buffer.",
    "epsilon": "Epsilon for the next training action decision: 1 during warmup, then the configured 1-to-0 schedule.",
    "next_epsilon": "Legacy alias of train/epsilon, retained for existing log readers.",
    "next_learning_rate": "Base DDQN optimizer learning rate after this chunk, including any configured decay.",
    "chunk_seconds": "Wall-clock seconds for this seed-batched training chunk; the same timing appears in every seed row.",
    "skip_loss_mean": "Mean skip-network Huber loss across skip updates in this chunk; absent if none ran.",
    "skip_updates": "Skip-network optimizer updates in this chunk.",
    "skip_updates_total": "Cumulative skip-network optimizer updates for this seed.",
    "skip_replay_size": "Skip transitions currently retained in this seed's bounded skip replay buffer.",
    "decisions": "New action-and-skip decisions in this chunk; repeated primitive steps are not extra decisions.",
    "decisions_total": "Cumulative new action-and-skip decisions for this seed.",
    "selected_skip_mean": "Mean selected repeat length at decisions in this chunk, before an episode may end early.",
    "executed_skip_mean": "Mean primitive steps actually executed per completed repeat segment in this chunk.",
    "next_skip_epsilon": "Epsilon used to explore skip lengths at the next decision; equal to train/epsilon.",
    "next_skip_learning_rate": "Skip-network optimizer learning rate after this chunk.",
    "expected_policy_epsilon": "RARe expected-target policy epsilon, tied to train/epsilon; Separate still selects greedy actions with its own critic.",
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
    "target_risk_mae_mean": "RARe-se only: absolute error versus V_star minus V_pi_e under its separate critic's policy.",
    "target_risk_mae_variance": "RARe-se only: population variance across per-length risk MAEs under pi_e.",
    "target_skip_value_mae_mean": "RARe-se only: mean per-length skip-value MAE against its own expected-policy target.",
    "target_skip_value_mae_variance": "RARe-se only: population variance across per-length skip-value MAEs under pi_e.",
}

LOCAL_RISK_DETAILS = {
    "epsilon": "Frozen policy epsilon at this checkpoint; the behavior and RARe-se target policies use the same numeric epsilon.",
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
    "target_policy": "RARe-se only: identical detail schema recomputed for its separate evaluation policy pi_e.",
}

METRIC_DEFINITIONS = {"train": TRAIN, "eval": EVAL, "local_risk_details": LOCAL_RISK_DETAILS}
