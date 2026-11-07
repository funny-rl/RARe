"""Exact finite-horizon Grid values, Risk per decision (Eq. 4), and decision traces."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from rare.envs.grid.core import GridEnv, GridState, observe
from rare.networks.q import apply_q
from rare.agents.policies import epsilon_greedy_probs
from rare.evaluation.grid import evaluate_policy


class RiskTotals(NamedTuple):
    """Per-episode sums over the selected non-terminal skip endpoints."""

    selected_sum: jax.Array
    selected_count: jax.Array


class GridDecisionTrace(NamedTuple):
    count: jax.Array
    positions: jax.Array
    goals: jax.Array
    steps: jax.Array
    actions: jax.Array
    skips: jax.Array
    q_values: jax.Array
    skip_values: jax.Array
    skip_selection_scores: jax.Array


class GridRiskEvaluation(NamedTuple):
    episodes: object
    risk: RiskTotals
    decision_traces: GridDecisionTrace


def empty_grid_decision_trace(env, max_skip):
    return GridDecisionTrace(
        jnp.int32(0), *(jnp.full(env.spec.max_steps, -1, jnp.int32) for _ in range(5)),
        jnp.zeros((env.spec.max_steps, env.num_actions), jnp.float32),
        jnp.zeros((env.spec.max_steps, max_skip), jnp.float32),
        jnp.zeros((env.spec.max_steps, max_skip), jnp.float32))


def append_grid_decision_trace(trace, observation, state, action, duration, predictions, scores,
                               online_params):
    index = trace.count
    return GridDecisionTrace(index + 1,
                             trace.positions.at[index].set(state.position),
                             trace.goals.at[index].set(state.goal),
                             trace.steps.at[index].set(state.steps),
                             trace.actions.at[index].set(action),
                             trace.skips.at[index].set(duration),
                             trace.q_values.at[index].set(apply_q(online_params, observation)),
                             trace.skip_values.at[index].set(predictions),
                             trace.skip_selection_scores.at[index].set(scores))


def evaluate_grid_decisions(env, online_params, key, *, algorithm, episodes, skip_params=None,
                            uncertainty_factor=-1.5, initial_states=None, global_step=0):
    """Record every deterministic Grid decision, for RODR, without computing Risk."""
    max_skip = 0 if algorithm == "DDQN" else skip_params.layers[-1].bias.shape[-1]
    return evaluate_policy(
        env, online_params, key, episodes=episodes, algorithm=algorithm,
        outcome_fn=env.episode_outcome, skip_params=skip_params,
        uncertainty_factor=uncertainty_factor, initial_states=initial_states,
        global_step=global_step,
        decision_observer=lambda trace, observation, state, action, duration, predictions, scores:
            append_grid_decision_trace(trace, observation, state, action, duration,
                                       predictions, scores, online_params),
        initial_diagnostics=empty_grid_decision_trace(env, max_skip),
    )


def _grid_model(env):
    if not isinstance(env, GridEnv):
        raise TypeError("exact grid risk requires a GridEnv")
    params = env.params
    successors = params.transitions
    goals = params.goals[:, None, None]
    pit = params.pit_mask[successors][None, :, :]
    goal = successors[None, :, :] == goals
    ended = pit | goal
    rewards = jnp.where(pit, params.pit_reward,
                        jnp.where(goal, params.goal_reward, params.step_reward))
    cells = jnp.arange(successors.shape[0])
    absorbing = params.pit_mask[None, :] | (cells[None, :] == params.goals[:, None])
    return successors, rewards, ended, absorbing


def grid_policy_probabilities(env, policy_params, epsilon):
    """Freeze pi[goal, position, action]; the neural Q only selects greedy actions."""
    positions = jnp.arange(env.params.transitions.shape[0], dtype=jnp.int32)
    states = GridState(
        jnp.broadcast_to(positions, (env.params.goals.shape[0], positions.shape[0])),
        jnp.broadcast_to(env.params.goals[:, None], (env.params.goals.shape[0], positions.shape[0])),
        jnp.zeros((env.params.goals.shape[0], positions.shape[0]), jnp.int32),
    )
    observations = jax.vmap(jax.vmap(observe, in_axes=(0, None)), in_axes=(0, None))(states, env.params)
    return epsilon_greedy_probs(apply_q(policy_params, observations), epsilon)


def solve_grid_policy_values(env, policy_probs, gamma):
    """Return Q[goal, remaining_steps, position, action] of a frozen policy, with Q_0 = 0."""
    successors, rewards, ended, absorbing = _grid_model(env)
    shape = (env.params.goals.shape[0], *successors.shape)
    zero = jnp.zeros(shape, dtype=policy_probs.dtype)

    def backup(previous, _):
        expected = jnp.sum(policy_probs * previous, axis=-1)
        current = rewards + gamma * jnp.where(ended, 0., expected[:, successors])
        current = jnp.where(absorbing[:, :, None], 0., current)
        return current, current

    _, values = jax.lax.scan(backup, zero, None, length=env.spec.max_steps)
    return jnp.concatenate((zero[:, None], jnp.swapaxes(values, 0, 1)), axis=1)


def _repeat_endpoints(env, state, action, max_skip):
    """Follow one repeated action; return endpoint position, step and end flag per length."""
    def repeat(carry, _):
        current, ended = carry

        def advance(_):
            _, following, _, terminated, truncated, _ = env.step(None, current, action)
            return following, terminated | truncated

        following, ended = jax.lax.cond(ended, lambda _: carry, advance, operand=None)
        return (following, ended), (following.position, following.steps, ended)

    _, trace = jax.lax.scan(repeat, (state, jnp.bool_(False)), None, length=max_skip)
    return trace


def decision_risks(env, q_values, policy_probs, state, action, max_skip):
    """Risk (Eq. 4) at the endpoint of every skip length, and the episode-end flags."""
    positions, steps, ended = _repeat_endpoints(env, state, action, max_skip)
    goal_index = jnp.argmax(env.params.goals == state.goal)
    remaining = jnp.maximum(env.params.max_steps - steps, 0)
    endpoint_q = jnp.where(ended[:, None], 0., q_values[goal_index, remaining, positions])
    endpoint_expected = jnp.sum(policy_probs[goal_index, positions] * endpoint_q, axis=-1)
    return jnp.max(endpoint_q, axis=-1) - endpoint_expected, ended


def evaluate_grid_risk(env, online_params, skip_params, epsilon, key, *, gamma, algorithm, episodes,
                       uncertainty_factor=-1.5, initial_states=None):
    """Roll out the greedy evaluation policy; score Risk under the frozen epsilon-greedy pi."""
    max_skip = skip_params.layers[-1].bias.shape[-1]
    probabilities = grid_policy_probabilities(env, online_params, epsilon)
    q_values = solve_grid_policy_values(env, probabilities, gamma)

    def observe_decision(totals, observation, state, action, duration, predictions, scores):
        risks, ended = decision_risks(env, q_values, probabilities, state, action, max_skip)
        chosen = duration - 1
        valid = ~ended[chosen]
        risk = RiskTotals(totals[0].selected_sum + jnp.where(valid, risks[chosen], 0.),
                          totals[0].selected_count + valid.astype(jnp.int32))
        return risk, append_grid_decision_trace(totals[1], observation, state, action, duration,
                                                predictions, scores, online_params)

    result = evaluate_policy(
        env, online_params, key, episodes=episodes, algorithm=algorithm,
        outcome_fn=env.episode_outcome, skip_params=skip_params,
        uncertainty_factor=uncertainty_factor, initial_states=initial_states,
        decision_observer=observe_decision,
        initial_diagnostics=(RiskTotals(jnp.float32(0), jnp.int32(0)),
                             empty_grid_decision_trace(env, max_skip)),
    )
    return GridRiskEvaluation(result.episodes, *result.diagnostics)


def aggregate_risk(risk):
    """Average episode-level Risk sums; called on one seed's host arrays."""
    sums = np.asarray(risk.selected_sum, dtype=np.float64)
    if not np.isfinite(sums).all():
        raise ValueError("Risk produced nonfinite values")
    return {"risk_direct_sum_mean": float(sums.mean()), "risk_direct_sum_std": float(sums.std()),
            "risk_direct_valid_decisions_mean": float(np.asarray(risk.selected_count).mean())}
