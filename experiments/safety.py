"""Train JAX learners on the Safety-Gymnasium Racecar tasks, simulated in a separate CPU process."""

from dataclasses import asdict
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from experiments.config import checkpoints
from experiments.logging import RunLog
from rare.envs.safety.client import ReferenceBridge
from rare.training.continuous import ContinuousRunConfig, ContinuousRunner
from rare.training.safety import evaluation_action
from rare.training.safety_compiled import make_evaluation, make_train_chunk


def evaluation_seeds(seeds, step):
    return [(int(seed) * 1000003 + step * 31 + 20261021) % (2**32) for seed in seeds]


def rollout_compiled(bridge, state, keys, observation, compiled):
    """Evaluate in one compiled loop that calls the simulator once per step."""
    run, records = compiled
    records.clear()
    final = run(state, jnp.asarray(observation), keys)
    if len(records) != int(np.asarray(final[-1])) or bool(np.asarray(final[-2]).any()):
        raise RuntimeError("Racecar evaluation exceeded its 1000-step time limit")
    return records


def rollout_host(bridge, state, keys, observation, policy):
    """Evaluate with one jitted policy call per step, as in the Goal runs of the paper."""
    count = observation.shape[0]
    previous = np.zeros((count, 2), np.float32)
    has_previous = np.zeros(count, bool)
    repeated = previous.copy()
    remaining = np.zeros(count, np.int32)
    active = np.ones(count, bool)
    records = []
    for _ in range(1000):
        action, repeated_next, remaining_next, decision = policy(
            state, jnp.asarray(observation), jnp.asarray(previous), jnp.asarray(has_previous),
            jnp.asarray(repeated), jnp.asarray(remaining), keys)
        action = np.array(action, copy=True)
        action[~active] = 0.
        result = bridge.call("step_eval", action)
        records.append((result, np.asarray(decision), active.copy()))
        newly_done = active & (result.terminated | result.truncated)
        active &= ~newly_done
        if not active.any():
            return records
        observation = np.asarray(result.reset_observations, dtype=np.float32)
        previous = action
        has_previous |= ~newly_done
        repeated = np.asarray(repeated_next)
        remaining = np.asarray(remaining_next)
    raise RuntimeError("Racecar evaluation exceeded its 1000-step time limit")


def evaluate(bridge, state, seeds, step, rollout):
    count = len(seeds)
    per_episode = evaluation_seeds(seeds, step)
    observation = np.asarray(bridge.call("reset_eval", per_episode), dtype=np.float32)
    keys = jax.vmap(jax.random.PRNGKey)(jnp.asarray(per_episode, jnp.uint32))
    returns, raw, costs = (np.zeros(count, np.float64) for _ in range(3))
    goals, decisions, lengths = (np.zeros(count, np.int32) for _ in range(3))
    terminated, truncated = (np.zeros(count, bool) for _ in range(2))
    for result, decision, active in rollout(bridge, state, keys, observation):
        returns += np.where(active, result.rewards, 0.)
        raw += np.where(active, result.raw_rewards, 0.)
        costs += np.where(active, result.costs, 0.)
        goals += np.where(active, result.goal_met, 0)
        decisions += np.where(active, decision, 0)
        lengths += active.astype(np.int32)
        terminated |= active & result.terminated
        truncated |= active & result.truncated
    return [{"seed": seed, "env_step": step, "episodes": 1, "reward_sum": float(returns[i]),
             "raw_reward_sum": float(raw[i]), "cost_sum": float(costs[i]),
             "goal_events": int(goals[i]), "length_mean": int(lengths[i]),
             "decisions_mean": int(decisions[i]), "terminated": bool(terminated[i]),
             "truncated": bool(truncated[i])} for i, seed in enumerate(seeds)]


def train(cell, seeds, output, stop_step=None):
    config = ContinuousRunConfig(**cell.config)
    steps = checkpoints(config.total_steps, config.warmup_steps, cell.evaluation)
    last_step = stop_step or config.total_steps
    metadata = {"arm": cell.arm, "algorithm": config.algorithm, "seeds": seeds, "config": asdict(config),
                "environment": {"name": cell.environment, "random": False,
                                "registry_id": f"SafetyRacecar{cell.environment}1-v0"},
                "evaluation": {**cell.evaluation, "steps": steps}}
    started = time.perf_counter()
    output = Path(output)
    with RunLog(output, metadata) as log, \
            ReferenceBridge(cell.environment, seeds, output / "bridge.log") as bridge:
        class ShapeEnv:
            """Shapes and bounds of the external simulator; only `reset` is called, for shapes."""

            name = f"SafetyRacecar{cell.environment}1-v0"
            observation_shape = bridge.spec["observation_shape"]
            action_shape = bridge.spec["action_shape"]
            action_low = jnp.asarray(bridge.spec["action_low"], jnp.float32)
            action_high = jnp.asarray(bridge.spec["action_high"], jnp.float32)

            def reset(self, key):
                del key
                return jnp.zeros(self.observation_shape, jnp.float32), jnp.int32(0)

        runner = ContinuousRunner(ShapeEnv(), config)
        observation = np.asarray(bridge.call("reset_train"), dtype=np.float32)
        state = jax.jit(jax.vmap(runner.initialize))(jnp.asarray(seeds, jnp.uint32))
        if state.pending is not None:
            state = state._replace(pending=state.pending._replace(
                observations=state.pending.observations.at[:, 0].set(observation)))
        state = state._replace(observation=jnp.asarray(observation))
        if cell.evaluation.get("host_loop", False):
            policy = jax.jit(jax.vmap(lambda current, obs, previous, has_previous, repeated, remaining, key:
                                      evaluation_action(runner, current, obs, previous, has_previous,
                                                        repeated, remaining, key)))

            def rollout(bridge, state, keys, observation):
                return rollout_host(bridge, state, keys, observation, policy)
        else:
            records = []
            compiled = (make_evaluation(runner, bridge, len(seeds), runner.env.observation_shape,
                                        records), records)

            def rollout(bridge, state, keys, observation):
                return rollout_compiled(bridge, state, keys, observation, compiled)
        chunks, position = {}, 0
        for step in (point for point in steps if point <= last_step):
            length = step - position
            if length not in chunks:
                chunks[length] = make_train_chunk(runner, bridge, len(seeds),
                                                  runner.env.observation_shape, length)
            state = chunks[length](state)
            jax.block_until_ready(state.observation)
            position = step
            rows = evaluate(bridge, state, seeds, step, rollout)
            log.append("evaluations", rows)
            print(f"{cell.environment} {cell.arm}: {step}/{config.total_steps} "
                  f"reward={np.mean([row['reward_sum'] for row in rows]):.3f} "
                  f"({time.perf_counter() - started:.0f}s)", flush=True)
    return output
