"""Unmodified Safety-Gymnasium transitions with the paper's reward scalarization."""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SafeTask:
    name: str
    env_id: str
    cost_weight: float
    reward_weight: float = 10.0


TASKS = {
    "Goal": SafeTask("Goal", "SafetyRacecarGoal1-v0", 0.1),
    "Button": SafeTask("Button", "SafetyRacecarButton1-v0", 0.05),
}


@dataclass(frozen=True)
class SafeStep:
    observation: np.ndarray
    reward: float
    raw_reward: float
    cost: float
    goal_met: bool
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class ReferenceSafeEnv:
    """Keep the CPU reference separate from JAX collection and learner code."""

    def __init__(self, task: SafeTask, environment: Any):
        self.task = task
        self.environment = environment
        self.observation_shape = tuple(environment.observation_space.shape)
        self.action_shape = tuple(environment.action_space.shape)
        self.action_low = np.asarray(environment.action_space.low).copy()
        self.action_high = np.asarray(environment.action_space.high).copy()
        if len(self.observation_shape) != 1 or len(self.action_shape) != 1:
            raise ValueError("the continuous learner needs flat observations and actions")
        self._ready = False

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = self.environment.reset(seed=seed)
        self._ready = True
        return np.array(observation, copy=True), dict(info)

    def step(self, action: np.ndarray) -> SafeStep:
        if not self._ready:
            raise RuntimeError("reset before stepping, including after termination or truncation")
        action = np.asarray(action)
        if action.shape != self.action_shape or not np.all(np.isfinite(action)):
            raise ValueError(f"action must be a finite array of shape {self.action_shape}")
        observation, raw_reward, cost, terminated, truncated, info = self.environment.step(action)
        raw_reward = float(raw_reward)
        cost = float(cost)
        terminated = bool(terminated)
        truncated = bool(truncated)
        if terminated or truncated:
            self._ready = False
        return SafeStep(
            observation=np.array(observation, copy=True),
            reward=self.task.reward_weight * raw_reward - self.task.cost_weight * cost,
            raw_reward=raw_reward,
            cost=cost,
            goal_met=bool(info.get("goal_met", False)),
            terminated=terminated,
            truncated=truncated,
            info=dict(info),
        )

    def close(self) -> None:
        self.environment.close()

    def __enter__(self) -> "ReferenceSafeEnv":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def make_reference_env(task_name: str) -> ReferenceSafeEnv:
    """Load the pinned external benchmark only when this adapter is selected."""
    if task_name not in TASKS:
        raise ValueError(f"unknown SafeRL task {task_name!r}; choose Goal or Button")
    try:
        import safety_gymnasium
    except ImportError as error:
        raise ImportError(
            "activate the rare-safety conda environment (environment-safety.yml)"
        ) from error
    task = TASKS[task_name]
    return ReferenceSafeEnv(task, safety_gymnasium.make(task.env_id))
