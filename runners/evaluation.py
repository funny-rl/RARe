"""Frozen greedy-policy evaluation with independent environment state and keys."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from networks.q import apply_q
from networks.skip import apply_skip, apply_skip_ensemble
from algorithms.taac_discrete import actor_logits


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


def evaluate_policy(env, online_params, key, *, episodes, algorithm, outcome_fn,
                    skip_params=None, uncertainty_factor=-1.5, initial_states=None,
                    decision_observer=None, initial_diagnostics=None, global_step=0):
    """Evaluate complete episodes without accessing training state or replay.

    Episodes use fold_in(checkpoint_key, zero-based episode index). Base actions
    and skip lengths are greedy; UTE retains its mean + lambda * sample-std
    selector. The outcome adapter keeps environment-specific success semantics
    outside this rollout. Optional batched (observations, states) bypass random
    reset, permitting exhaustive Grid cases without coupling rollout to Grid.
    An optional pure observer measures each decision, including every raw skip
    prediction. UTE supplies its ensemble mean, while selection retains its
    uncertainty penalty. Algorithm, episode count and adapters are static.
    """
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
            observation, env_state = env.reset(reset_key, global_step=global_step)
        else:
            observation, env_state = jax.tree.map(lambda value: value[index], initial_states)
            observation, env_state = env.with_global_step(observation, env_state, global_step)
        initial = (rollout_key, observation, env_state, jnp.int32(0), jnp.int32(0),
                   jnp.float32(0), jnp.int32(0), jnp.int32(0), jnp.bool_(False), jnp.bool_(False),
                   () if initial_diagnostics is None else initial_diagnostics)

        def step(carry):
            rng, observation, state, action, remaining, reward_sum, length, decisions, _, _, diagnostics = carry

            def choose(_):
                action = jnp.argmax(apply_q(online_params, observation)).astype(jnp.int32)
                if algorithm == "DDQN":
                    duration = jnp.int32(1)
                    predictions = jnp.empty((0,), dtype=jnp.float32)
                else:
                    if algorithm == "UTE":
                        values = apply_skip_ensemble(skip_params, observation, action)
                        predictions = values.mean(axis=0)
                        scores = predictions + uncertainty_factor * values.std(axis=0, ddof=1)
                    else:
                        scores = apply_skip(skip_params, observation, action)
                        predictions = scores
                    duration = jnp.argmax(scores).astype(jnp.int32) + 1
                measured = diagnostics if decision_observer is None else decision_observer(
                    diagnostics, observation, state, action, predictions)
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


def evaluate_taac_discrete(env, critic_params, actor_params, key, *, episodes, outcome_fn,
                           initial_states=None, global_step=0):
    """Frozen deterministic mode of the two-stage policy, one step at a time."""
    if type(episodes) is not int or episodes <= 0:
        raise ValueError("episodes must be a positive integer")
    if initial_states is not None and any(
        value.ndim == 0 or value.shape[0] != episodes for value in jax.tree.leaves(initial_states)
    ):
        raise ValueError("initial_states must have an episode axis matching episodes")
    num_actions = critic_params[-1].bias.shape[0]

    def episode(index):
        episode_key = jax.random.fold_in(key, index)
        reset_key, rollout_key = jax.random.split(episode_key)
        if initial_states is None:
            observation, env_state = env.reset(reset_key, global_step=global_step)
        else:
            observation, env_state = jax.tree.map(lambda value: value[index], initial_states)
            observation, env_state = env.with_global_step(observation, env_state, global_step)
        initial = (rollout_key, observation, env_state, jnp.int32(-1), jnp.float32(0),
                   jnp.int32(0), jnp.bool_(False), jnp.bool_(False))

        def step(carry):
            rng, obs, state, previous, reward_sum, length, _, _ = carry
            proposal = jnp.argmax(actor_logits(actor_params, obs, previous, num_actions)).astype(jnp.int32)
            q = apply_q(critic_params, obs)
            action = jnp.where((previous < 0) | (q[proposal] > q[jnp.maximum(previous, 0)]),
                               proposal, previous).astype(jnp.int32)
            rng, step_key = jax.random.split(rng)
            obs, state, reward, terminated, truncated, _ = env.step(step_key, state, action)
            return rng, obs, state, action, reward_sum + reward, length + 1, terminated, truncated

        final = jax.lax.while_loop(lambda carry: ~(carry[6] | carry[7]), step, initial)
        success, pit = outcome_fn(final[2])
        return EvaluationEpisodes(final[4], final[5], final[5], success, pit, final[6], final[7])

    return jax.vmap(episode)(jnp.arange(episodes, dtype=jnp.uint32))
