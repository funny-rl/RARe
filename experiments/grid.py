"""Train and evaluate one Grid or non-stationary Grid cell."""

from dataclasses import asdict
import time

import jax
import jax.numpy as jnp

from experiments.config import checkpoints
from experiments.logging import RunLog
from rare.envs.grid import GridEnv, make_grid_env, make_windy_goal_env
from rare.evaluation.grid import evaluate_policy
from rare.metrics.risk import aggregate_risk, evaluate_grid_decisions, evaluate_grid_risk
from rare.metrics.rodr import serialize_grid_episodes
from rare.training.ddqn import DDQNRunConfig, DDQNRunner
from rare.training.skip import SkipRunConfig, SkipRunner


def make_environment(settings):
    if "preset" in settings:
        return make_windy_goal_env(settings["preset"], shift_steps=tuple(settings["shift_steps"]))
    return make_grid_env(settings["layout"], random=settings["random"])


def make_runner(env, values):
    if values["algorithm"] == "DDQN":
        settings = {key: value for key, value in values.items() if key != "algorithm"}
        return DDQNRunner(env, DDQNRunConfig(**settings))
    return SkipRunner(env, SkipRunConfig(**values))


def make_evaluation(env, config, algorithm, episodes, evaluation):
    """Return a batched evaluator over (online, skip, key, global_step, risk epsilon)."""
    initial_states = env.evaluation_resets() if evaluation["episodes"] == "all" else None
    factor = getattr(config, "uncertainty_factor", -1.5)
    common = dict(algorithm=algorithm, episodes=episodes, uncertainty_factor=factor,
                  initial_states=initial_states)

    def evaluate(online, skip, key, step, epsilon):
        if evaluation["risk"]:
            return evaluate_grid_risk(env, online, skip, epsilon, key, gamma=config.gamma, **common)
        if evaluation["decision_traces"]:
            return evaluate_grid_decisions(env, online, key, skip_params=skip, global_step=step, **common)
        return evaluate_policy(env, online, key, outcome_fn=env.episode_outcome, skip_params=skip,
                               global_step=step, **common)

    return jax.jit(jax.vmap(evaluate, in_axes=(0, 0 if algorithm != "DDQN" else None, 0, None, 0)))


def evaluation_rows(results, seeds, step):
    rows = []
    for index, seed in enumerate(seeds):
        values = {name: field[index] for name, field in results._asdict().items()}
        rows.append({"seed": seed, "env_step": step, "episodes": len(values["returns"]),
                     "reward_sum": float(values["returns"].mean()),
                     "reward_sum_std": float(values["returns"].std()),
                     "length_mean": float(values["lengths"].mean()),
                     "decisions_mean": float(values["decisions"].mean()),
                     "success_rate": float(values["success"].mean()),
                     "pit_rate": float(values["pit"].mean()),
                     "timeout_rate": float(values["truncated"].mean())})
    return rows


def train(cell, seeds, output, stop_step=None):
    env = make_environment(cell.settings)
    runner = make_runner(env, cell.config)
    config, algorithm = runner.config, cell.config["algorithm"]
    evaluation = cell.evaluation
    if evaluation["episodes"] == "all":
        cases = [{"start": list(start), "goal": list(goal)}
                 for start in env.spec.starts for goal in env.spec.goals]
        episodes = len(cases)
    else:
        cases, episodes = None, evaluation["episodes"]
    steps = checkpoints(config.total_steps, config.warmup_steps, evaluation)
    last_step = stop_step or config.total_steps
    metadata = {"arm": cell.arm, "algorithm": algorithm, "seeds": seeds, "config": asdict(config),
                "environment": {"name": cell.environment, "random": cell.settings.get("random", False),
                                "spec": asdict(env.spec)},
                "evaluation": {**evaluation, "episodes": episodes, "steps": steps}}
    state = jax.jit(jax.vmap(runner.initialize))(jnp.asarray(seeds, dtype=jnp.uint32))
    execute = jax.jit(jax.vmap(runner.run, in_axes=(0, None)), donate_argnums=(0,))
    evaluate = make_evaluation(env, config, algorithm, episodes, evaluation)
    epsilon = jax.jit(jax.vmap(lambda current: jnp.where(
        current.training_steps < config.warmup_steps, 1., runner._epsilon(current.training_steps))))
    position, started = 0, time.perf_counter()
    with RunLog(output, metadata) as log:
        for step in (*(point for point in steps if point <= last_step),):
            state = execute(state, jnp.asarray(step - position, jnp.int32))
            position = step
            # Checkpoint keys derive from the evaluation stream without advancing training streams.
            keys = jax.vmap(lambda key: jax.random.fold_in(key, step))(state.rngs.evaluation)
            skip = state.skip_learner.params if algorithm != "DDQN" else None
            result = jax.device_get(evaluate(state.learner.online_params, skip, keys,
                                             jnp.asarray(step, jnp.int32), epsilon(state)))
            episodes_result = result if not (evaluation["risk"] or evaluation["decision_traces"]) else result.episodes
            rows = evaluation_rows(episodes_result, seeds, step)
            if evaluation["risk"]:
                for index, row in enumerate(rows):
                    row.update(aggregate_risk(jax.tree.map(lambda value: value[index], result.risk)))
            if evaluation["decision_traces"]:
                traces = result.decision_traces if evaluation["risk"] else result.diagnostics
                log.append("evaluation_decisions", [
                    row for index, seed in enumerate(seeds)
                    for row in serialize_grid_episodes(jax.tree.map(lambda value: value[index], traces),
                                                       seed=seed, env_step=step, cases=cases)])
            log.append("evaluations", rows)
            print(f"{cell.environment} {cell.arm}: {step}/{config.total_steps} "
                  f"({time.perf_counter() - started:.0f}s)", flush=True)
    return output
