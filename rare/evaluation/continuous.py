"""Frozen deterministic continuous-policy evaluation across independent episodes."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from rare.agents.taac import select_taac_action
from rare.networks.continuous import apply_actor, apply_continuous_skip


class ContinuousEvaluation(NamedTuple):
    returns: jax.Array
    lengths: jax.Array
    decisions: jax.Array
    terminated: jax.Array
    truncated: jax.Array


def evaluate_continuous(env, config, learner, skip_learner, key, *, episodes: int,
                        uncertainty_factor=None):
    """Use a separate key stream; never modify training state or replay."""
    if type(episodes) is not int or episodes <= 0:
        raise ValueError("episodes must be positive")
    max_steps = env.spec.max_steps
    skip_algorithm = config.algorithm in ("TempoRL", "UTE", "RARe")
    episode_keys = jax.vmap(lambda index: jax.random.fold_in(key, index))(
        jnp.arange(episodes, dtype=jnp.uint32))

    def episode(episode_key):
        episode_key, reset_key = jax.random.split(episode_key)
        observation, env_state = env.reset(reset_key)
        zero_action = jnp.zeros(env.action_shape, jnp.float32)
        initial = (episode_key, observation, env_state, zero_action, jnp.bool_(False),
                   zero_action, jnp.int32(0), jnp.float32(0), jnp.int32(0),
                   jnp.int32(0), jnp.bool_(False), jnp.bool_(False), jnp.bool_(False))

        def one_step(carry, _):
            (key, obs, state, previous, has_previous, repeated, remaining,
             reward_sum, length, decisions, terminated, truncated, done) = carry

            def active(_):
                key, action_key = jax.random.split(carry[0])
                if config.algorithm == "TAAC":
                    action, _ = select_taac_action(learner, obs, previous, has_previous, action_key,
                                                   env.action_low, env.action_high,
                                                   jnp.float32(0), deterministic=True)
                    selected = jnp.int32(1)
                    decision = jnp.bool_(True)
                elif skip_algorithm:
                    decision = remaining == 0

                    def choose(_):
                        action = apply_actor(learner.actor, obs, env.action_low, env.action_high)
                        if config.algorithm == "UTE":
                            values = jax.vmap(apply_continuous_skip, in_axes=(0, None, None))(
                                skip_learner.params, obs, action)
                            factor = (config.uncertainty_factor if uncertainty_factor is None
                                      else uncertainty_factor)
                            scores = values.mean(0) + factor * values.std(0, ddof=1)
                        else:
                            scores = apply_continuous_skip(skip_learner.params, obs, action)
                        return action, jnp.argmax(scores).astype(jnp.int32) + 1

                    action, selected = jax.lax.cond(decision, choose,
                                                    lambda _: (repeated, remaining), operand=None)
                else:
                    action = apply_actor(learner.actor, obs, env.action_low, env.action_high)
                    selected = jnp.int32(1)
                    decision = jnp.bool_(True)
                key, env_key = jax.random.split(key)
                next_obs, next_state, reward, term, trunc, _ = env.step(env_key, state, action)
                return (key, next_obs, next_state, action, jnp.bool_(True), action,
                        selected - 1, reward_sum + reward, length + 1,
                        decisions + decision.astype(jnp.int32), term, trunc, term | trunc)

            return jax.lax.cond(done, lambda _: carry, active, operand=None), None

        final, _ = jax.lax.scan(one_step, initial, None, length=max_steps)
        return final[7], final[8], final[9], final[10], final[11]

    return ContinuousEvaluation(*jax.vmap(episode)(episode_keys))
