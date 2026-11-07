"""Exact Risk on Grid: Eq. 4 is nonnegative and vanishes under a greedy policy."""

import jax
import jax.numpy as jnp
import numpy as np

from rare.envs.grid import make_grid_env
from rare.metrics.risk import decision_risks, grid_policy_probabilities, solve_grid_policy_values
from rare.networks import init_q_params


def setup(epsilon):
    env = make_grid_env("CliffWalking")
    params = init_q_params(jax.random.key(0), env.observation_shape[0], env.num_actions, 8)
    probabilities = grid_policy_probabilities(env, params, epsilon)
    return env, probabilities, solve_grid_policy_values(env, probabilities, .99)


def test_policy_values_start_at_zero_with_no_steps_left():
    _, _, values = setup(.5)
    np.testing.assert_array_equal(values[:, 0], 0.)


def test_risk_is_nonnegative():
    env, probabilities, values = setup(.5)
    _, state = env.reset(jax.random.key(1))
    risks, _ = decision_risks(env, values, probabilities, state, jnp.int32(0), 6)
    assert bool(jnp.all(risks >= -1e-4))


def test_risk_at_terminal_endpoints_is_zero():
    env, probabilities, values = setup(.5)
    _, state = env.reset(jax.random.key(1))
    risks, ended = decision_risks(env, values, probabilities, state, jnp.int32(2), 6)  # East: a pit.
    assert bool(ended[0]) and float(risks[0]) == 0.
