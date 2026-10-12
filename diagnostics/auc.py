"""CPU-only, seed-level AUC statistics for completed experiments."""

import numpy as np


def normalize_rewards(scores, reward_min, reward_max):
    """Normalize before integration; case ceilings broadcast along the last axis."""
    reward_max = np.asarray(reward_max, dtype=np.float64)
    if (not np.isfinite(reward_min) or not np.isfinite(reward_max).all() or np.any(reward_min >= reward_max)):
        raise ValueError("reward_min and reward_max must be finite with reward_min < reward_max")
    scores = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("reward scores must be finite")
    if np.any(scores > reward_max + 1e-10):
        raise ValueError("an observed reward exceeds reward_max; check the theoretical ceiling and evaluation cases")
    return np.clip((scores - reward_min) / (reward_max - reward_min), 0., 1.)


def grid_reward_ceiling(spec):
    """Solve optimal finite-horizon rewards for all grid reset pairs on CPU.

    Unlike shortest-path-only logic, this also honors timeout, changed rewards,
    and unreachable goals. Keep only the previous value table, not trajectories.
    """
    height, width = spec["shape"]
    horizon = spec["max_steps"]
    if any(type(value) is not int or value <= 0 for value in (height, width, horizon)):
        raise ValueError("grid shape and max_steps must be positive integers")
    starts, goals = spec["starts"], spec["goals"]
    if not starts or not goals or not spec["actions"]:
        raise ValueError("grid starts, goals and actions must be nonempty")
    positions = np.arange(height * width)
    rows, cols = np.divmod(positions, width)
    actions = np.asarray(spec["actions"], dtype=np.int64)
    next_rows = rows[:, None] + actions[None, :, 0]
    next_cols = cols[:, None] + actions[None, :, 1]
    valid = (next_rows >= 0) & (next_rows < height) & (next_cols >= 0) & (next_cols < width)
    successors = np.where(valid, next_rows * width + next_cols, positions[:, None])
    goal_ids = np.asarray([row * width + col for row, col in goals])
    pit_ids = np.asarray([row * width + col for row, col in spec["pits"]], dtype=np.int64)
    pit = np.isin(successors, pit_ids)[None]
    ended = pit | (successors[None] == goal_ids[:, None, None])
    rewards = np.where(pit, spec["pit_reward"], np.where(
        successors[None] == goal_ids[:, None, None], spec["goal_reward"], spec["step_reward"]))
    if not np.isfinite(rewards).all():
        raise ValueError("grid rewards must be finite")
    values = np.zeros((len(goals), len(positions)), dtype=np.float64)
    for _ in range(horizon):
        values = np.max(rewards + np.where(ended, 0., values[:, successors]), axis=-1)
    cases = [{"start": list(start), "goal": list(goal),
              "optimal_reward": float(values[goal_index, start[0] * width + start[1]])}
             for start in starts for goal_index, goal in enumerate(goals)]
    return {"mean": float(np.mean([case["optimal_reward"] for case in cases])),
            "method": "exact finite-horizon optimal grid dynamic programming; undiscounted rewards",
            "case_order": "starts_major_goals_minor", "cases": cases}


def seed_auc(steps, scores):
    """Integrate each seed's curve on observed steps, without extrapolation."""
    steps = np.asarray(steps, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if (steps.ndim != 1 or steps.size < 2 or not np.isfinite(steps).all()
            or not np.all(np.diff(steps) > 0)):
        raise ValueError("AUC requires at least two finite, strictly increasing evaluation steps")
    if scores.ndim != 2 or scores.shape[1] != steps.size or scores.shape[0] < 1 or not np.isfinite(scores).all():
        raise ValueError("scores must be finite with shape (seeds, evaluation_steps)")
    values = ((scores[:, :-1] + scores[:, 1:]) * .5) @ np.diff(steps)
    if not np.isfinite(values).all():
        raise ValueError("AUC overflowed; check evaluation scores and step units")
    return values


def auc_statistics(values):
    """Summarize all seed means, including failed seeds, using sample std."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("AUC mean/std require at least two finite seed AUCs")
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1))}
