"""Targets of the methods: greedy and expected bootstraps, DDQN, and Monte Carlo expectation."""

import jax
import jax.numpy as jnp
import numpy as np

from rare.agents.bootstrap import expected_bootstrap, greedy_bootstrap
from rare.agents.ddpg import gaussian_expected_q, init_ddpg
from rare.agents.ddqn import ddqn_targets
from rare.agents.policies import epsilon_greedy_probs, sample_epsilon_greedy
from rare.agents.skip import skip_targets
from rare.agents.targets import update_target_params

ONLINE = jnp.array([[5., 5., 1.], [0., 4., 1.]])
TARGET = jnp.array([[-2., 50., 7.], [20., -3., 8.]])


def test_epsilon_greedy_probabilities_sum_to_one_and_favor_first_argmax():
    probabilities = epsilon_greedy_probs(ONLINE, .3)
    np.testing.assert_allclose(probabilities.sum(-1), 1., rtol=1e-6)
    np.testing.assert_allclose(probabilities[0], [.1 + .7, .1, .1], rtol=1e-6)


def test_greedy_bootstrap_selects_online_and_evaluates_target():
    np.testing.assert_array_equal(greedy_bootstrap(ONLINE, TARGET), [-2., -3.])


def test_expected_skip_target_recovers_greedy_target_at_epsilon_zero():
    """Proposition 3: the expectation equals the greedy value once pi_e is greedy."""
    expected = expected_bootstrap(epsilon_greedy_probs(ONLINE, 0.), TARGET)
    np.testing.assert_array_equal(expected, greedy_bootstrap(ONLINE, TARGET))


def test_expected_skip_target_weights_all_actions():
    expected = expected_bootstrap(epsilon_greedy_probs(ONLINE, .3), TARGET)
    np.testing.assert_allclose(expected, [.8 * -2 + .1 * 50 + .1 * 7, .1 * 20 + .8 * -3 + .1 * 8], rtol=1e-5)


def test_zero_discount_ignores_a_nonfinite_terminal_value():
    targets = skip_targets(jnp.array([1., -1.]), jnp.array([.5, 0.]), jnp.array([2., jnp.nan]))
    np.testing.assert_array_equal(targets, [2., -1.])


def test_ddqn_target_masks_episode_ends():
    targets = ddqn_targets(ONLINE, TARGET, jnp.array([1., 1.]), jnp.array([False, True]),
                           jnp.array([False, False]), .9)
    np.testing.assert_allclose(targets, [1. + .9 * -2., 1.], rtol=1e-6)


def test_polyak_update():
    updated = update_target_params(jnp.array([1.]), jnp.array([3.]), .5)
    np.testing.assert_array_equal(updated, [2.])


def test_gaussian_expectation_with_zero_noise_is_the_deterministic_value():
    learner = init_ddpg(jax.random.key(0), 3, 2, 8)
    observations = jnp.ones((4, 3))
    low, high = -jnp.ones(2), jnp.ones(2)
    noisy = gaussian_expected_q(jax.random.key(1), learner.actor, learner.critic, observations,
                                low, high, 0., 5)
    single = gaussian_expected_q(jax.random.key(2), learner.actor, learner.critic, observations,
                                 low, high, 0., 1)
    np.testing.assert_allclose(noisy, single, rtol=1e-5)


def test_exploration_uses_the_given_key_only():
    values = jnp.array([0., 1., 2.])
    key = jax.random.key(3)
    assert sample_epsilon_greedy(key, values, 0.) == 2
    assert sample_epsilon_greedy(key, values, .5) == sample_epsilon_greedy(key, values, .5)
