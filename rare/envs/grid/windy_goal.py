"""Non-stationary windy grids of Appendix B.2.

Paper names: WindShift is WindyGoalTransition, Random is WindyGoalRandomStrong, Episodic is
WindyGoalResponsive, and the shift-count sweep is WindyGoalTransitionFrequency.

WindyGoalTransitionFrequency matches WindyGoalTransition through its first
change; later scheduled changes alternate between equally strong west and
south winds.
"""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp


_WIND_VECTORS = jnp.asarray(((0, -1), (-1, 0), (0, 1), (1, 0)), dtype=jnp.int32)
_CORNERS = jnp.asarray((0, 9, 90, 99), dtype=jnp.int32)
_PRESETS = {
    "WindyGoalTransition": dict(max_steps=80, min_distance=12, winds=((2,), (0,)),
                                probabilities=(0.10, 0.35), local_change=False),
    "WindyGoalTransitionFrequency": dict(max_steps=80, min_distance=12, winds=((2,), (0,), (3,)),
                                         probabilities=(0.10, 0.35, 0.35), local_change=False,
                                         cycle=True),
    "WindyGoalRandomStrong": dict(max_steps=90, min_distance=9,
                                  winds=((2, 1), (2, 1, 0, 3)),
                                  probabilities=(0.15, 0.45), local_change=False),
    "WindyGoalResponsive": dict(max_steps=90, min_distance=9, winds=((2,), (1,)),
                                probabilities=(0.15, 0.35), local_change=True),
}


@dataclass(frozen=True)
class WindyGoalSpec:
    name: str
    shape: tuple[int, int] = (10, 10)
    max_steps: int = 90
    shift_step: int = 20_000
    step_reward: float = -0.01
    goal_reward: float = 1.0
    progress_scale: float = 0.1
    min_start_goal_distance: int = 9
    phase_wind_options: tuple[tuple[int, ...], ...] = ((2,), (0,))
    phase_wind_probabilities: tuple[float, ...] = (0.10, 0.35)
    local_change_step: int | None = None
    # Explicit global change points; None means the single shift_step.
    shift_steps: tuple[int, ...] | None = None
    # After the first change, cycle through the remaining phases instead of
    # staying in the last one.
    cycle_post_shift_phases: bool = False


class WindyGoalState(NamedTuple):
    position: jax.Array
    goal: jax.Array
    steps: jax.Array
    global_steps: jax.Array
    regime: jax.Array
    wind_index: jax.Array
    wind_probability: jax.Array
    local_changed: jax.Array
    clock_frozen: jax.Array


class WindyGoalEnv:
    """Hazard-free 10x10 navigation with observable wind and goal changes."""

    backend = "rare_jax"
    supports_exact_risk = False
    requires_global_step = True
    observation_codec = "raw"
    observation_storage_dtype = jnp.float32

    def __init__(self, spec: WindyGoalSpec):
        if spec.name not in _PRESETS or spec.shape != (10, 10):
            raise ValueError("WindyGoal requires a named 10x10 preset")
        if type(spec.shift_step) is not int or spec.shift_step <= 0:
            raise ValueError("shift_step must be a positive integer")
        shift_steps = (spec.shift_step,) if spec.shift_steps is None else tuple(spec.shift_steps)
        if (not shift_steps or any(type(step) is not int or step <= 0 for step in shift_steps)
                or any(a >= b for a, b in zip(shift_steps, shift_steps[1:]))
                or shift_steps[0] != spec.shift_step):
            raise ValueError("shift_steps must be strictly increasing positive integers starting at shift_step")
        if len(spec.phase_wind_options) != len(spec.phase_wind_probabilities):
            raise ValueError("each wind phase needs one probability")
        if len(shift_steps) > 1 and not spec.cycle_post_shift_phases and len(spec.phase_wind_options) < 3:
            raise ValueError("repeated shifts need cycling phases")
        self.spec = spec
        self._shift_steps = jnp.asarray(shift_steps, dtype=jnp.int32)
        positions = range(100)
        if spec.name.startswith("WindyGoalTransition"):
            pairs = ((90, 9),)
        else:
            starts = _CORNERS.tolist() if spec.name == "WindyGoalResponsive" else positions
            pairs = tuple((start, goal) for start in starts for goal in positions
                          if abs(start // 10 - goal // 10) + abs(start % 10 - goal % 10)
                          >= spec.min_start_goal_distance)
        self._pairs = jnp.asarray(pairs, dtype=jnp.int32)
        options = spec.phase_wind_options
        self._wind_choices = jnp.asarray(tuple(tuple(phase) + (phase[0],) * (4 - len(phase))
                                         for phase in options), dtype=jnp.int32)
        self._wind_counts = jnp.asarray(tuple(len(phase) for phase in options), dtype=jnp.int32)
        self._probabilities = jnp.asarray(spec.phase_wind_probabilities, dtype=jnp.float32)

    @property
    def observation_shape(self) -> tuple[int]:
        return (7,)

    @property
    def num_actions(self) -> int:
        return 4

    @staticmethod
    def _distance(left, right):
        return jnp.abs(left // 10 - right // 10) + jnp.abs(left % 10 - right % 10)

    @staticmethod
    def _move(position, wind_index):
        row, col = position // 10, position % 10
        move = _WIND_VECTORS[wind_index]
        next_row, next_col = row + move[0], col + move[1]
        valid = (next_row >= 0) & (next_row < 10) & (next_col >= 0) & (next_col < 10)
        return jnp.where(valid, next_row * 10 + next_col, position).astype(jnp.int32)

    def _regime(self, global_steps):
        """Wind phase after counting the scheduled changes already crossed."""
        count = jnp.sum(global_steps >= self._shift_steps).astype(jnp.int32)
        phases = len(self.spec.phase_wind_options)
        if self.spec.cycle_post_shift_phases:
            return jnp.where(count == 0, 0, 1 + (count - 1) % (phases - 1)).astype(jnp.int32)
        return jnp.minimum(count, phases - 1)

    def _sample_wind(self, key, regime):
        choice = jax.random.randint(key, (), 0, self._wind_counts[regime])
        return self._wind_choices[regime, choice]

    def observe(self, state: WindyGoalState):
        wind = _WIND_VECTORS[state.wind_index]
        return jnp.asarray((state.position // 10 / 9, state.position % 10 / 9,
                            state.goal // 10 / 9, state.goal % 10 / 9,
                            wind[0], wind[1], state.wind_probability), dtype=jnp.float32)

    def reset(self, key, *, global_step=0):
        pair_key, wind_key = jax.random.split(key)
        pair = self._pairs[jax.random.randint(pair_key, (), 0, self._pairs.shape[0])]
        global_steps = jnp.asarray(global_step, jnp.int32)
        regime = self._regime(global_steps)
        state = WindyGoalState(pair[0], pair[1], jnp.int32(0), global_steps, regime,
                               self._sample_wind(wind_key, regime), self._probabilities[regime],
                               jnp.bool_(False), jnp.bool_(False))
        return self.observe(state), state

    def evaluation_reset(self, key, *, global_step=0):
        observation, state = self.reset(key, global_step=global_step)
        return observation, state._replace(clock_frozen=jnp.bool_(True))

    def with_global_step(self, observation, state, global_step):
        del observation
        global_steps = jnp.asarray(global_step, jnp.int32)
        regime = self._regime(global_steps)
        wind_index = jnp.where(regime == state.regime, state.wind_index,
                               self._wind_choices[regime, 0])
        updated = state._replace(global_steps=global_steps, regime=regime,
                                 wind_index=wind_index,
                                 wind_probability=self._probabilities[regime],
                                 clock_frozen=jnp.bool_(True))
        return self.observe(updated), updated

    def episode_outcome(self, state):
        return state.position == state.goal, jnp.bool_(False)

    def step(self, key, state, action):
        global_key, local_key, gust_key = jax.random.split(key, 3)
        regime = self._regime(state.global_steps)
        global_changed = regime != state.regime
        wind_index = jnp.where(global_changed, self._sample_wind(global_key, regime),
                               state.wind_index)
        wind_probability = jnp.where(global_changed, self._probabilities[regime],
                                     state.wind_probability)
        local_changed = (jnp.bool_(self.spec.local_change_step is not None)
                         & ~state.local_changed
                         & (state.steps == (self.spec.local_change_step or 0)))
        wind_index = jnp.where(local_changed, (wind_index + 1) % 4, wind_index)

        distances = self._distance(_CORNERS, state.position)
        eligible = _CORNERS != state.goal
        farthest = jnp.max(jnp.where(eligible, distances, -1))
        logits = jnp.where(eligible & (distances == farthest), 0., -jnp.inf)
        relocated_goal = _CORNERS[jax.random.categorical(local_key, logits)]
        goal = jnp.where(local_changed, relocated_goal, state.goal)

        intended = self._move(state.position, jnp.asarray(action, jnp.int32))
        gust_applied = jax.random.bernoulli(gust_key, wind_probability)
        position = jnp.where(gust_applied, self._move(intended, wind_index), intended)
        next_steps = state.steps + 1
        success = position == goal
        truncated = (next_steps >= self.spec.max_steps) & ~success
        progress = self._distance(state.position, goal) - self._distance(position, goal)
        reward = jnp.where(success, self.spec.goal_reward, self.spec.step_reward)
        reward = reward + self.spec.progress_scale * progress
        next_state = WindyGoalState(position, goal, next_steps,
                                    state.global_steps + (~state.clock_frozen).astype(jnp.int32),
                                    regime, wind_index, wind_probability,
                                    state.local_changed | local_changed, state.clock_frozen)
        info = {"context_changed": global_changed | local_changed,
                "distance_progress": progress, "wind_applied": gust_applied,
                "regime": regime, "wind_index": wind_index,
                "wind_probability": wind_probability}
        return self.observe(next_state), next_state, reward, success, truncated, info


def make_windy_goal_env(name: str, *, shift_step: int = 20_000,
                        shift_steps: tuple[int, ...] | None = None) -> WindyGoalEnv:
    if name not in _PRESETS:
        raise ValueError(f"unknown WindyGoal environment: {name}")
    preset = _PRESETS[name]
    cycle = preset.get("cycle", False)
    if shift_steps is not None:
        shift_steps = tuple(shift_steps)
        if not cycle and len(shift_steps) > 1:
            raise ValueError(f"{name} supports one shift; use WindyGoalTransitionFrequency")
        shift_step = shift_steps[0]
    spec = WindyGoalSpec(name=name, max_steps=preset["max_steps"], shift_step=shift_step,
                         min_start_goal_distance=preset["min_distance"],
                         phase_wind_options=preset["winds"],
                         phase_wind_probabilities=preset["probabilities"],
                         local_change_step=12 if preset["local_change"] else None,
                         shift_steps=shift_steps, cycle_post_shift_phases=cycle)
    return WindyGoalEnv(spec)
