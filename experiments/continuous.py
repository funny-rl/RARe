"""Train and evaluate one continuous cell whose simulator runs in JAX."""

from dataclasses import asdict
import time

import jax
import jax.numpy as jnp
import numpy as np

from experiments.config import checkpoints
from experiments.logging import RunLog
from rare.evaluation.continuous import evaluate_continuous
from rare.training.continuous import ContinuousRunConfig, ContinuousRunner, _UTE_LAMBDAS


def make_environment(name):
    if name == "MountainCar":
        from rare.envs.mountain_car import make_mountain_car_env
        return make_mountain_car_env()
    if name == "PointMaze":
        from rare.envs.point_maze import make_point_maze_env
        return make_point_maze_env()
    if name == "Pendulum":
        from rare.envs.pendulum import make_pendulum_env
        return make_pendulum_env()
    if name == "HalfCheetah":
        from rare.envs.halfcheetah import make_halfcheetah_env
        return make_halfcheetah_env()
    raise ValueError(f"unknown continuous environment: {name}")


def train(cell, seeds, output, stop_step=None):
    env = make_environment(cell.environment)
    config = ContinuousRunConfig(**cell.config)
    runner = ContinuousRunner(env, config)
    episodes = cell.evaluation["episodes"]
    steps = checkpoints(config.total_steps, config.warmup_steps, cell.evaluation)
    last_step = stop_step or config.total_steps
    metadata = {"arm": cell.arm, "algorithm": config.algorithm, "seeds": seeds, "config": asdict(config),
                "environment": {"name": cell.environment, "random": False, "spec": asdict(env.spec)},
                "evaluation": {**cell.evaluation, "steps": steps}}
    state = jax.jit(jax.vmap(runner.initialize))(jnp.asarray(seeds, dtype=jnp.uint32))
    execute = jax.jit(jax.vmap(runner.run, in_axes=(0, None)), donate_argnums=(0,))

    def evaluate_one(learner, skip, bandit, key):
        factor = None if bandit is None else _UTE_LAMBDAS[bandit.current]
        return evaluate_continuous(env, config, learner, skip, key, episodes=episodes,
                                   uncertainty_factor=factor)

    evaluate = jax.jit(jax.vmap(evaluate_one))
    position, started = 0, time.perf_counter()
    with RunLog(output, metadata) as log:
        for step in (point for point in steps if point <= last_step):
            state = execute(state, jnp.asarray(step - position, jnp.int32))
            position = step
            keys = jax.vmap(lambda key: jax.random.fold_in(key, step))(state.rngs.evaluation)
            result = jax.device_get(evaluate(state.learner, state.skip_learner, state.ute_bandit, keys))
            rows = []
            for index, seed in enumerate(seeds):
                returns = np.asarray(result.returns[index])
                rows.append({"seed": seed, "env_step": step, "episodes": episodes,
                             "reward_sum": float(returns.mean()), "reward_sum_std": float(returns.std()),
                             "length_mean": float(np.asarray(result.lengths[index]).mean()),
                             "decisions_mean": float(np.asarray(result.decisions[index]).mean()),
                             "timeout_rate": float(np.asarray(result.truncated[index]).mean())})
            log.append("evaluations", rows)
            print(f"{cell.environment} {cell.arm}: {step}/{config.total_steps} "
                  f"({time.perf_counter() - started:.0f}s)", flush=True)
    return output
