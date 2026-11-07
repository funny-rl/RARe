"""Skip replay: contiguous sub-segments and the padding of Figure 3."""

import jax.numpy as jnp
import numpy as np

from rare.replay.replay import add_transition, init_replay
from rare.replay.skip import SkipSegment, add_skip_segment, init_skip_replay

GAMMA = .5


def segment(executed, selected, terminated_at=None):
    length = 3
    rewards = jnp.array([1., 2., 4.])
    ended = jnp.zeros(length, bool)
    if terminated_at is not None:
        ended = ended.at[terminated_at].set(True)
    observations = jnp.arange(length + 1, dtype=jnp.float32)[:, None]
    return SkipSegment(observations, jnp.int32(0), rewards, ended, jnp.zeros(length, bool),
                       jnp.int32(selected), jnp.int32(executed))


def records(state):
    return sorted((float(state.data.observations[i, 0]), int(state.data.skips[i]),
                   float(state.data.returns[i]), float(state.data.discounts[i]))
                  for i in range(int(state.size)))


def test_full_repeat_stores_every_contiguous_subsegment():
    state = add_skip_segment(init_skip_replay(16, (1,)), segment(2, 2), GAMMA)
    assert records(state) == [(0., 1, 1., .5), (0., 2, 2., .25), (1., 1, 2., .5)]


def test_early_termination_pads_longer_labels_with_zero_discount():
    state = add_skip_segment(init_skip_replay(16, (1,)), segment(1, 3, terminated_at=0), GAMMA)
    assert records(state) == [(0., 1, 1., 0.), (0., 2, 1., 0.), (0., 3, 1., 0.)]


def test_primitive_replay_overwrites_the_oldest_transition():
    state = init_replay(2, (1,))
    for value in range(3):
        state = add_transition(state, jnp.array([float(value)]), jnp.int32(0), 0., jnp.array([0.]),
                               False, False)
    assert int(state.size) == 2
    np.testing.assert_array_equal(np.sort(state.data.observations[:, 0]), [1., 2.])
