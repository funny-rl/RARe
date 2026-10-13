"""Exact finite-horizon Grid values and Eq. 3 skip-risk diagnostics."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from envs.grid.core import GridEnv, GridState, observe
from networks.q import apply_q
from policies import epsilon_greedy_probs
from runners.evaluation import evaluate_policy


class DecisionRisks(NamedTuple):
    expected_values: jax.Array
    max_values: jax.Array
    true_risks: jax.Array
    estimated_risks: jax.Array
    value_errors: jax.Array
    risk_errors: jax.Array
    ended: jax.Array


class RiskTotals(NamedTuple):
    value_error_sum: jax.Array
    risk_error_sum: jax.Array
    counts: jax.Array
    ended_counts: jax.Array


class GridRiskEvaluation(NamedTuple):
    episodes: object
    totals: RiskTotals
    epsilon: jax.Array
    bellman_residual: jax.Array
    target_totals: RiskTotals | None = None
    target_bellman_residual: jax.Array | None = None
    optimal_bellman_residual: jax.Array | None = None


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
    """Freeze pi[goal, position, action]; neural Q only selects greedy actions."""
    positions = jnp.arange(env.params.transitions.shape[0], dtype=jnp.int32)
    states = GridState(
        jnp.broadcast_to(positions, (env.params.goals.shape[0], positions.shape[0])),
        jnp.broadcast_to(env.params.goals[:, None], (env.params.goals.shape[0], positions.shape[0])),
        jnp.zeros((env.params.goals.shape[0], positions.shape[0]), jnp.int32),
    )
    observations = jax.vmap(jax.vmap(observe, in_axes=(0, None)), in_axes=(0, None))(states, env.params)
    return epsilon_greedy_probs(apply_q(policy_params, observations), epsilon)


def solve_grid_policy_values(env, policy_probs, gamma):
    """Return Q[goal, remaining_steps, position, action], including Q_0=0.

    Recompute this table for each checkpoint's frozen epsilon policy. Q^pi
    takes the indexed first action, then follows that same policy. Optimal
    control has a separate Bellman backup below.
    Policy probabilities are stationary because current Grid observations do
    not encode elapsed time. Values retain time to honor the absorbing timeout.
    No learned critic, sampled trajectory, tolerance or iterative convergence
    criterion is used. Arithmetic remains subject to floating-point rounding.
    """
    successors, rewards, ended, absorbing = _grid_model(env)
    shape = (env.params.goals.shape[0], *successors.shape)
    if policy_probs.shape != shape:
        raise ValueError("policy_probs must have shape (goals, cells, actions)")
    zero = jnp.zeros(shape, dtype=policy_probs.dtype)

    def backup(previous, _):
        expected = jnp.sum(policy_probs * previous, axis=-1)
        current = rewards + gamma * jnp.where(ended, 0., expected[:, successors])
        current = jnp.where(absorbing[:, :, None], 0., current)
        return current, current

    _, values = jax.lax.scan(backup, zero, None, length=env.spec.max_steps)
    return jnp.concatenate((zero[:, None], jnp.swapaxes(values, 0, 1)), axis=1)


def solve_grid_optimal_values(env, gamma):
    """Compute finite-horizon Q* with optimal control at every future step."""
    successors, rewards, ended, absorbing = _grid_model(env)
    zero = jnp.zeros((env.params.goals.shape[0], *successors.shape), jnp.float32)

    def backup(previous, _):
        optimal = jnp.max(previous, axis=-1)
        current = rewards + gamma * jnp.where(ended, 0., optimal[:, successors])
        current = jnp.where(absorbing[:, :, None], 0., current)
        return current, current

    _, values = jax.lax.scan(backup, zero, None, length=env.spec.max_steps)
    return jnp.concatenate((zero[:, None], jnp.swapaxes(values, 0, 1)), axis=1)


def policy_bellman_residual(env, policy_probs, gamma, q_values):
    """Check all finite-horizon Bellman equations and absorbing boundaries."""
    successors, rewards, ended, absorbing = _grid_model(env)
    expected = jnp.sum(policy_probs[:, None] * q_values[:, :-1], axis=-1)
    targets = rewards[:, None] + gamma * jnp.where(ended[:, None], 0., expected[:, :, successors])
    targets = jnp.where(absorbing[:, None, :, None], 0., targets)
    return jnp.maximum(jnp.max(jnp.abs(q_values[:, 0])), jnp.max(jnp.abs(q_values[:, 1:] - targets)))


def optimal_bellman_residual(env, gamma, q_values):
    """Check that Q* uses a maximum at every continuation step."""
    successors, rewards, ended, absorbing = _grid_model(env)
    optimal = jnp.max(q_values[:, :-1], axis=-1)
    targets = rewards[:, None] + gamma * jnp.where(ended[:, None], 0., optimal[:, :, successors])
    targets = jnp.where(absorbing[:, None, :, None], 0., targets)
    return jnp.maximum(jnp.max(jnp.abs(q_values[:, 0])), jnp.max(jnp.abs(q_values[:, 1:] - targets)))


def score_decision_risks(env, q_values, optimal_q_values, policy_probs, state, action, predictions, gamma):
    """Score every length, including unselected and post-termination labels.

    The exact optimal-control baseline minus predicted skip value, divided by
    gamma**j, is the skip network's implied risk. It is not a learned risk head.
    The baseline cancels in implied-minus-true risk, so its absolute error is
    exactly skip-value error / gamma**j at every nonterminal endpoint.
    Terminal prefixes stop accumulating reward; all later labels share the
    same exact skip value and zero true risk. Their value errors still count;
    the evaluation observer excludes them from risk-error aggregation.
    """
    goal_index = jnp.argmax(env.params.goals == state.goal)
    discounts = jnp.asarray(gamma, predictions.dtype) ** jnp.arange(1, predictions.shape[0] + 1)

    def repeat(carry, index):
        current, prefix, ended = carry

        def advance(_):
            _, following, reward, terminated, truncated, _ = env.step(None, current, action)
            return following, prefix + gamma ** index * reward, terminated | truncated

        following, prefix, ended = jax.lax.cond(ended, lambda _: carry, advance, operand=None)
        remaining = jnp.maximum(env.params.max_steps - following.steps, 0)
        endpoint_q = q_values[goal_index, remaining, following.position]
        endpoint_optimal_q = optimal_q_values[goal_index, remaining, following.position]
        endpoint_q = jnp.where(ended, 0., endpoint_q)
        endpoint_optimal_q = jnp.where(ended, 0., endpoint_optimal_q)
        maximum = jnp.max(endpoint_optimal_q)
        expected = jnp.sum(policy_probs[goal_index, following.position] * endpoint_q)
        discount = discounts[index]
        values = (prefix + discount * expected, prefix + discount * maximum, maximum - expected, ended)
        return (following, prefix, ended), values

    _, (expected, maximum, true_risks, ended) = jax.lax.scan(
        repeat, (state, jnp.asarray(0., predictions.dtype), jnp.bool_(False)),
        jnp.arange(predictions.shape[0]),
    )
    estimated = (maximum - predictions) / discounts
    return DecisionRisks(expected, maximum, true_risks, estimated,
                         jnp.abs(predictions - expected), jnp.abs(estimated - true_risks), ended)


def evaluate_grid_risk(env, online_params, skip_params, policy_params, epsilon, key, *,
                       gamma, algorithm, episodes, uncertainty_factor=-1.5, initial_states=None,
                       target_policy_params=None):
    """Score behavior risk, and optionally RARe-se's distinct target-policy risk."""
    if algorithm not in ("TempoRL", "UTE", "RARe"):
        raise ValueError("grid risk is only defined here for skip algorithms")
    if not 0 < gamma <= 1:
        raise ValueError("grid risk reconstruction requires gamma > 0; disable risk diagnostics for gamma=0")
    max_skip = skip_params.layers[-1].bias.shape[-1]
    if gamma ** max_skip < float(jnp.finfo(jnp.float32).tiny):
        raise ValueError("gamma**max_skip underflows; reduce max_skip or disable risk diagnostics")
    probabilities = grid_policy_probabilities(env, policy_params, epsilon)
    q_values = solve_grid_policy_values(env, probabilities, gamma)
    optimal_q_values = solve_grid_optimal_values(env, gamma)
    if target_policy_params is not None:
        target_probabilities = grid_policy_probabilities(env, target_policy_params, epsilon)
        target_q_values = solve_grid_policy_values(env, target_probabilities, gamma)
    zero = jnp.zeros(max_skip, dtype=q_values.dtype)
    empty = RiskTotals(zero, zero, jnp.zeros(max_skip, jnp.int32), jnp.zeros(max_skip, jnp.int32))

    def accumulate(totals, errors):
        return RiskTotals(totals.value_error_sum + errors.value_errors,
                          totals.risk_error_sum + jnp.where(errors.ended, 0., errors.risk_errors),
                          totals.counts + 1, totals.ended_counts + errors.ended.astype(jnp.int32))

    def observe_decision(totals, observation, state, action, predictions):
        del observation
        errors = score_decision_risks(env, q_values, optimal_q_values, probabilities,
                                      state, action, predictions, gamma)
        if target_policy_params is None:
            return accumulate(totals, errors)
        target_errors = score_decision_risks(
            env, target_q_values, optimal_q_values, target_probabilities,
            state, action, predictions, gamma,
        )
        return (accumulate(totals[0], errors), accumulate(totals[1], target_errors))

    result = evaluate_policy(
        env, online_params, key, episodes=episodes, algorithm=algorithm,
        outcome_fn=env.episode_outcome, skip_params=skip_params,
        uncertainty_factor=uncertainty_factor, initial_states=initial_states,
        decision_observer=observe_decision,
        initial_diagnostics=empty if target_policy_params is None else (empty, empty),
    )
    totals = jax.tree.map(lambda values: values.sum(axis=0), result.diagnostics)
    behavior_totals, target_totals = (totals, None) if target_policy_params is None else totals
    target_residual = (None if target_policy_params is None else
                       policy_bellman_residual(env, target_probabilities, gamma, target_q_values))
    return GridRiskEvaluation(result.episodes, behavior_totals, epsilon,
                              policy_bellman_residual(env, probabilities, gamma, q_values),
                              target_totals, target_residual,
                              optimal_bellman_residual(env, gamma, optimal_q_values))


def aggregate_risk_errors(result):
    """Host-side equal-length macro mean/variance; raw per-length MAEs stay local."""
    import numpy as np

    totals = result.totals
    counts = np.asarray(totals.counts)
    risk_counts = counts - np.asarray(totals.ended_counts)

    def aggregate(sums, sample_counts):
        valid = sample_counts > 0
        sums = np.asarray(sums, dtype=np.float64)
        maes = np.divide(sums, sample_counts, out=np.zeros_like(sums), where=valid)
        if not np.isfinite(maes[valid]).all():
            raise ValueError("grid risk produced nonfinite errors")
        return (float(maes[valid].mean()) if valid.any() else None,
                float(maes[valid].var()) if valid.any() else None,
                [float(value) if keep else None for value, keep in zip(maes, valid)])

    value_mean, value_variance, value_maes = aggregate(np.asarray(totals.value_error_sum), counts)
    risk_mean, risk_variance, risk_maes = aggregate(np.asarray(totals.risk_error_sum), risk_counts)
    metrics = {"risk_mae_mean": risk_mean, "risk_mae_variance": risk_variance,
               "skip_value_mae_mean": value_mean, "skip_value_mae_variance": value_variance}
    details = {"epsilon": float(result.epsilon), "bellman_residual": float(result.bellman_residual),
               "risk_valid_lengths": int((risk_counts > 0).sum()),
               "risk_valid_fraction": float(risk_counts.sum() / counts.sum()) if counts.sum() else None,
               "lengths": [{"skip": index + 1, "count": int(count),
                            "risk_count": int(risk_counts[index]),
                            "ended_count": int(totals.ended_counts[index]),
                            "risk_excluded_fraction": float(totals.ended_counts[index] / count) if count else None,
                            "value_mae": value_maes[index], "risk_mae": risk_maes[index]}
                           for index, count in enumerate(counts)]}
    if result.optimal_bellman_residual is not None:
        details["optimal_bellman_residual"] = float(result.optimal_bellman_residual)
    return metrics, details


def aggregate_target_risk_errors(result):
    """Expose RARe-se calibration to its own pi_e without mixing reference policies."""
    if result.target_totals is None:
        raise ValueError("target-policy risk was not measured")
    metrics, details = aggregate_risk_errors(result._replace(
        totals=result.target_totals, bellman_residual=result.target_bellman_residual,
    ))
    return {f"target_{name}": value for name, value in metrics.items()}, details
