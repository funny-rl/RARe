"""Environment rules stated in Appendix B."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rare.envs.grid import make_grid_env, make_windy_goal_env
from rare.envs.mountain_car import make_mountain_car_env
from rare.envs.pendulum import make_pendulum_env

KEY = jax.random.key(0)


@pytest.mark.parametrize("name", ("CliffWalking", "Bridge", "ZigZag"))
def test_grid_layouts_are_eight_by_twelve_with_100_step_limit(name):
    env = make_grid_env(name)
    assert env.spec.shape == (8, 12) and env.spec.max_steps == 100 and env.num_actions == 4


@pytest.mark.parametrize("name,cases", (("CliffWalking", 16), ("Bridge", 256), ("ZigZag", 256)))
def test_randomized_versions_have_the_paper_case_counts(name, cases):
    spec = make_grid_env(name, random=True).spec
    assert len(spec.starts) * len(spec.goals) == cases


def test_entering_a_hazard_gives_minus_100_and_ends_the_episode():
    env = make_grid_env("CliffWalking")
    _, state = env.reset(KEY)
    _, _, reward, terminated, _, _ = env.step(None, state, jnp.int32(2))  # East of the start is a pit.
    assert float(reward) == -100. and bool(terminated)


@pytest.mark.parametrize("name,limit", (("WindyGoalTransition", 80), ("WindyGoalRandomStrong", 90),
                                         ("WindyGoalResponsive", 90)))
def test_windy_grids_have_the_paper_step_limits(name, limit):
    assert make_windy_goal_env(name).spec.max_steps == limit


def test_pendulum_episode_lasts_200_steps():
    env = make_pendulum_env()
    _, state = env.reset(KEY)
    truncated = False
    for _ in range(200):
        _, state, _, _, truncated, _ = env.step(KEY, state, jnp.zeros(1))
    assert bool(truncated)


def test_mountain_car_pays_the_control_cost():
    env = make_mountain_car_env()
    _, state = env.reset(KEY)
    _, _, reward, _, _, _ = env.step(KEY, state, jnp.array([1.]))
    np.testing.assert_allclose(reward, -.1, rtol=1e-6)
