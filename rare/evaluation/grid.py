"""Frozen greedy-policy evaluation with independent environment state and keys."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from rare.networks.q import apply_q
from rare.networks.skip import apply_skip, apply_skip_ensemble


class EvaluationEpisodes(NamedTuple):
    returns: jax.Array
    lengths: jax.Array
    decisions: jax.Array
    success: jax.Array
    pit: jax.Array
    terminated: jax.Array
    truncated: jax.Array


class ObservedEvaluation(NamedTuple):
    episodes: EvaluationEpisodes
    diagnostics: object


def greedy_decision(algorithm, online_params, skip_params, observation, uncertainty_factor=-1.5):
    """One frozen decision: greedy action, then skip length (1 without a skip network)."""
    action = jnp.argmax(apply_q(online_params, observation)).astype(jnp.int32)
    if algorithm == "DDQN":
        predictions = jnp.empty((0,), dtype=jnp.float32)
        return action, jnp.int32(1), predictions, predictions
    if algorithm == "UTE":
        values = apply_skip_ensemble(skip_params, observation, action)
        predictions = values.mean(axis=0)
        scores = predictions + uncertainty_factor * values.std(axis=0, ddof=1)
    else:
        scores = apply_skip(skip_params, observation, action)
        predictions = scores
    return action, jnp.argmax(scores).astype(jnp.int32) + 1, predictions, scores


def evaluate_policy(env, online_params, key, *, episodes, algorithm, outcome_fn,
                    skip_params=None, uncertainty_factor=-1.5, initial_states=None,
                    decision_observer=None, initial_diagnostics=None, global_step=0):
    """Evaluate complete episodes with greedy actions and skips; UTE keeps mean + lambda * std.

    Episodes use fold_in(checkpoint_key, episode index). Optional (observations, states) replace
    the random reset for exhaustive Grid cases, and an optional observer measures each decision."""
    if algorithm not in ("DDQN", "TempoRL", "UTE", "RARe"):
        raise ValueError("unsupported evaluation algorithm")
    if type(episodes) is not int or episodes <= 0:
        raise ValueError("episodes must be a positive integer")
    if algorithm != "DDQN" and skip_params is None:
        raise ValueError("skip parameters are required for skip algorithms")
    if (decision_observer is None) != (initial_diagnostics is None):
        raise ValueError("decision_observer and initial_diagnostics must be supplied together")
    if initial_states is not None and any(
        value.ndim == 0 or value.shape[0] != episodes for value in jax.tree.leaves(initial_states)
    ):
        raise ValueError("initial_states must have an episode axis matching episodes")

    def episode(episode_key, index):
        reset_key, rollout_key = jax.random.split(episode_key)
        if initial_states is None:
            reset = getattr(env, "evaluation_reset", env.reset)
            observation, env_state = reset(reset_key, global_step=global_step)
        else:
            observation, env_state = jax.tree.map(lambda value: value[index], initial_states)
            observation, env_state = env.with_global_step(observation, env_state, global_step)
        initial = (rollout_key, observation, env_state, jnp.int32(0), jnp.int32(0),
                   jnp.float32(0), jnp.int32(0), jnp.int32(0), jnp.bool_(False), jnp.bool_(False),
                   () if initial_diagnostics is None else initial_diagnostics)

        def step(carry):
            rng, observation, state, action, remaining, reward_sum, length, decisions, _, _, diagnostics = carry

            def choose(_):
                action, duration, predictions, scores = greedy_decision(
                    algorithm, online_params, skip_params, observation, uncertainty_factor)
                measured = diagnostics if decision_observer is None else decision_observer(
                    diagnostics, observation, state, action, duration, predictions, scores)
                return action, duration, measured

            new_decision = remaining == 0
            action, remaining, diagnostics = jax.lax.cond(
                new_decision, choose, lambda _: (action, remaining, diagnostics), operand=None)
            rng, step_key = jax.random.split(rng)
            observation, state, reward, terminated, truncated, _ = env.step(step_key, state, action)
            return (rng, observation, state, action, remaining - 1, reward_sum + reward,
                    length + 1, decisions + new_decision.astype(jnp.int32), terminated, truncated, diagnostics)

        final = jax.lax.while_loop(lambda carry: ~(carry[8] | carry[9]), step, initial)
        success, pit = outcome_fn(final[2])
        results = EvaluationEpisodes(final[5], final[6], final[7], success, pit, final[8], final[9])
        return results if decision_observer is None else ObservedEvaluation(results, final[10])

    indices = jnp.arange(episodes, dtype=jnp.uint32)
    keys = jax.vmap(lambda index: jax.random.fold_in(key, index))(indices)
    return jax.vmap(episode)(keys, indices)
