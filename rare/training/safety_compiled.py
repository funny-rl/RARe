"""Compiled Safety-Gymnasium loops; the only side effect is one ordered environment step."""

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import io_callback

from .safety import commit_step, evaluation_action, prepare_step


def make_train_chunk(runner, bridge, seed_count, observations_shape, length):
    shape = (seed_count, *observations_shape)
    result_spec = (
        jax.ShapeDtypeStruct(shape, jnp.float32),
        jax.ShapeDtypeStruct(shape, jnp.float32),
        jax.ShapeDtypeStruct((seed_count,), jnp.float32),
        jax.ShapeDtypeStruct((seed_count,), jnp.bool_),
        jax.ShapeDtypeStruct((seed_count,), jnp.bool_),
    )

    def step_environment(actions):
        transition = bridge.call("step_train", np.asarray(actions))
        return (np.asarray(transition.observations, np.float32),
                np.asarray(transition.reset_observations, np.float32),
                np.asarray(transition.rewards, np.float32),
                np.asarray(transition.terminated, np.bool_),
                np.asarray(transition.truncated, np.bool_))

    prepare = jax.vmap(lambda current: prepare_step(runner, current))
    commit = jax.vmap(lambda current, action, switched, next_obs, reset_obs, reward, terminated,
                      truncated: commit_step(runner, current, action, switched, next_obs,
                                             reset_obs, reward, terminated, truncated))

    def body(state, _):
        state, actions, switched = prepare(state)
        values = io_callback(step_environment, result_spec, actions, ordered=True)
        return commit(state, actions, switched, *values), None

    return jax.jit(lambda state: jax.lax.scan(body, state, None, length=length)[0])


def make_evaluation(runner, bridge, seed_count, observations_shape, records):
    """Keep policy selection on device and record original-precision rewards on host."""
    shape = (seed_count, *observations_shape)
    step_spec = (jax.ShapeDtypeStruct(shape, jnp.float32),
                 jax.ShapeDtypeStruct((seed_count,), jnp.bool_),
                 jax.ShapeDtypeStruct((seed_count,), jnp.bool_))
    policy = jax.vmap(lambda state, obs, previous, has_previous, repeated, remaining, key:
                      evaluation_action(runner, state, obs, previous, has_previous, repeated,
                                        remaining, key))

    def step_environment(actions, decisions, active):
        transition = bridge.call("step_eval", np.asarray(actions))
        records.append((transition, np.asarray(decisions), np.asarray(active)))
        return (np.asarray(transition.reset_observations, np.float32),
                np.asarray(transition.terminated, np.bool_),
                np.asarray(transition.truncated, np.bool_))

    def run(state, initial_observation, keys):
        action_shape = (seed_count, *runner.env.action_shape)
        initial = (initial_observation,
                   jnp.zeros(action_shape, jnp.float32),
                   jnp.zeros(seed_count, jnp.bool_),
                   jnp.zeros(action_shape, jnp.float32),
                   jnp.zeros(seed_count, jnp.int32),
                   jnp.ones(seed_count, jnp.bool_),
                   jnp.int32(0))

        def condition(carry):
            return (carry[-1] < 1000) & jnp.any(carry[-2])

        def body(carry):
            observation, previous, has_previous, repeated, remaining, active, steps = carry
            action, repeated_next, remaining_next, decision = policy(
                state, observation, previous, has_previous, repeated, remaining, keys)
            action = jnp.where(active[:, None], action, 0.)
            reset_observation, terminated, truncated = io_callback(
                step_environment, step_spec, action, decision, active, ordered=True)
            newly_done = active & (terminated | truncated)
            active_next = active & ~newly_done
            return (reset_observation, action, has_previous | ~newly_done,
                    repeated_next, remaining_next, active_next, steps + 1)

        return jax.lax.while_loop(condition, body, initial)

    return jax.jit(run)
