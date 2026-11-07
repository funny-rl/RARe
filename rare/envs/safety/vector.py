"""CPU vector collector that exposes terminal successors before auto-reset."""

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .reference import TASKS, SafeTask


@dataclass(frozen=True)
class SafeVectorStep:
    observations: np.ndarray
    reset_observations: np.ndarray
    rewards: np.ndarray
    raw_rewards: np.ndarray
    costs: np.ndarray
    goal_met: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    info: dict[str, Any]


class SafeVectorReference:
    """Expose both replay successors and current observations after auto-reset."""

    def __init__(self, task: SafeTask, environment: Any, seeds: Sequence[int]):
        self.task = task
        self.environment = environment
        self.seeds = tuple(int(seed) for seed in seeds)
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be nonempty and distinct")
        self.num_envs = len(self.seeds)
        if environment.num_envs != self.num_envs:
            raise ValueError("number of environments must equal number of seeds")
        self.observation_shape = tuple(environment.single_observation_space.shape)
        self.action_shape = tuple(environment.single_action_space.shape)
        self.action_low = np.asarray(environment.single_action_space.low).copy()
        self.action_high = np.asarray(environment.single_action_space.high).copy()
        if len(self.observation_shape) != 1 or len(self.action_shape) != 1:
            raise ValueError("the continuous learner needs flat observations and actions")
        self._ready = False

    def reset(self, seeds: Sequence[int] | None = None) -> np.ndarray:
        selected = self.seeds if seeds is None else tuple(int(seed) for seed in seeds)
        if len(selected) != self.num_envs:
            raise ValueError("reset seed count must equal number of environments")
        observations, _ = self.environment.reset(seed=list(selected))
        self._ready = True
        return np.array(observations, copy=True)

    def step(self, actions: np.ndarray) -> SafeVectorStep:
        if not self._ready:
            raise RuntimeError("reset before stepping")
        actions = np.asarray(actions)
        shape = (self.num_envs, *self.action_shape)
        if actions.shape != shape or not np.all(np.isfinite(actions)):
            raise ValueError(f"actions must be a finite array of shape {shape}")
        reset_obs, raw_reward, cost, terminated, truncated, info = self.environment.step(actions)
        reset_obs = np.array(reset_obs, copy=True)
        raw_reward = np.asarray(raw_reward).copy()
        cost = np.asarray(cost).copy()
        goal_met = np.zeros(self.num_envs, dtype=np.bool_)
        if "goal_met" in info:
            mask = np.asarray(info.get("_goal_met", np.ones(self.num_envs)), dtype=np.bool_)
            goal_met[mask] = np.asarray(info["goal_met"])[mask]
        terminated = np.asarray(terminated, dtype=np.bool_).copy()
        truncated = np.asarray(truncated, dtype=np.bool_).copy()
        observations = reset_obs.copy()
        done = terminated | truncated
        if np.any(done):
            final = info.get("final_observation")
            mask = info.get("_final_observation")
            if final is None or mask is None or not np.all(np.asarray(mask)[done]):
                raise RuntimeError("vector environment lost a terminal observation")
            for index in np.flatnonzero(done):
                observations[index] = np.asarray(final[index])
        return SafeVectorStep(
            observations=observations,
            reset_observations=reset_obs,
            rewards=self.task.reward_weight * raw_reward - self.task.cost_weight * cost,
            raw_rewards=raw_reward,
            costs=cost,
            goal_met=goal_met,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def close(self) -> None:
        self.environment.close()

    def __enter__(self) -> "SafeVectorReference":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def make_vector_env(task_name: str, seeds: Sequence[int], *,
                    asynchronous: bool = True) -> SafeVectorReference:
    """Create the Safety-Gymnasium vector backend without importing it into JAX workers."""
    if task_name not in TASKS:
        raise ValueError(f"unknown SafeRL task {task_name!r}; choose Goal or Button")
    try:
        import safety_gymnasium
    except ImportError as error:
        raise ImportError(
            "activate the rare-safety conda environment (environment-safety.yml)"
        ) from error
    task = TASKS[task_name]
    environment = safety_gymnasium.vector.make(
        task.env_id, num_envs=len(seeds), asynchronous=asynchronous, wrappers=None
    )
    return SafeVectorReference(task, environment, seeds)
