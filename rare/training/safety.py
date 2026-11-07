"""Run the continuous learners on transitions from the Safety-Gymnasium CPU process."""

import jax
import jax.numpy as jnp

from rare.agents.taac import select_taac_action
from rare.networks.continuous import apply_actor, apply_continuous_skip
from .continuous import _UTE_LAMBDAS


def prepare_step(runner, state):
    """Select the action exactly as the compiled JAX collector does."""
    skip = runner._is_skip()
    decision = state.repeat_remaining == 0 if skip else jnp.bool_(True)
    if skip:
        state = jax.lax.cond(decision, runner._begin_repeat, lambda current: current, state)
        return state, state.pending.action, jnp.bool_(True)
    action, switched, rngs = runner._choose_action(state)
    return state._replace(rngs=rngs), action, switched


def commit_step(runner, state, action, switched, next_obs, reset_obs, reward, terminated, truncated):
    """Apply learner and replay updates to one real transition."""
    env_stream, _ = jax.random.split(state.rngs.environment)
    rngs = state.rngs._replace(environment=env_stream)
    state, rngs = runner.apply_transition(state, rngs, action, switched, next_obs, reward,
                                          terminated, truncated)
    done = terminated | truncated
    reset_stream, _ = jax.random.split(rngs.environment)
    rngs = rngs._replace(environment=jnp.where(done, reset_stream, rngs.environment))
    state = state._replace(observation=jnp.where(done, reset_obs, next_obs))
    return runner._finish_episode(state, rngs, reward, done)


def evaluation_action(runner, state, observation, previous, has_previous, repeated, remaining, key):
    """Deterministic evaluation action, matching rare.evaluation.continuous."""
    cfg = runner.config
    low, high = runner.env.action_low, runner.env.action_high
    if cfg.algorithm == "TAAC":
        action, _ = select_taac_action(state.learner, observation, previous, has_previous, key,
                                       low, high, jnp.float32(0), deterministic=True)
        return action, action, jnp.int32(0), jnp.bool_(True)
    if runner._is_skip():
        decision = remaining == 0

        def choose(_):
            action = apply_actor(state.learner.actor, observation, low, high)
            if cfg.algorithm == "UTE":
                values = jax.vmap(apply_continuous_skip, in_axes=(0, None, None))(
                    state.skip_learner.params, observation, action)
                factor = (_UTE_LAMBDAS[state.ute_bandit.current]
                          if state.ute_bandit is not None else cfg.uncertainty_factor)
                scores = values.mean(0) + factor * values.std(0, ddof=1)
            else:
                scores = apply_continuous_skip(state.skip_learner.params, observation, action)
            selected = jnp.argmax(scores).astype(jnp.int32) + 1
            return action, selected - 1

        action, new_remaining = jax.lax.cond(
            decision, choose, lambda _: (repeated, remaining - 1), operand=None)
        return action, action, new_remaining, decision
    action = apply_actor(state.learner.actor, observation, low, high)
    return action, action, jnp.int32(0), jnp.bool_(True)
