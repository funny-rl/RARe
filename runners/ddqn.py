"""DDQN primitive collection, replay and learning inside a compiled JAX loop."""

from dataclasses import dataclass
import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from algorithms import DDQNState, ddqn_train_step, init_ddqn_state
from buffers import ReplayState, add_transition, init_replay, sample_batch
from envs import FunctionalEnv
from networks import apply_q
from policies import sample_epsilon_greedy
from rng import RNGState, init_rngs
from .loop import collect_steps


@dataclass(frozen=True)
class DDQNRunConfig:
    """Static experiment settings; total_steps includes warmup interactions."""

    total_steps: int
    warmup_steps: int
    replay_capacity: int
    batch_size: int = 64
    hidden_dim: int = 64
    gamma: float = 0.99
    learning_rate: float = 0.001
    max_epsilon: float = 1.0
    min_epsilon: float = 0.0
    epsilon_decay_steps: int | None = None
    epsilon_schedule: str = "linear"
    use_lr_decay: bool = False
    tau: float = 0.005
    use_hard_update: bool = False
    update_interval: int = 1000

    def __post_init__(self):
        for name in ("total_steps", "replay_capacity", "batch_size", "hidden_dim", "update_interval"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive Python integer")
        if type(self.warmup_steps) is not int or not 0 <= self.warmup_steps <= self.total_steps:
            raise ValueError("warmup_steps must lie in [0, total_steps]")
        if self.epsilon_decay_steps is not None and (
            type(self.epsilon_decay_steps) is not int or self.epsilon_decay_steps <= 0
        ):
            raise ValueError("epsilon_decay_steps must be positive or None")
        if self.epsilon_schedule not in ("linear", "exponential"):
            raise ValueError("epsilon_schedule must be linear or exponential")
        if type(self.use_lr_decay) is not bool or type(self.use_hard_update) is not bool:
            raise TypeError("update/schedule flags must be bool")
        for name in ("gamma", "max_epsilon", "min_epsilon", "tau"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.min_epsilon > self.max_epsilon:
            raise ValueError("min_epsilon must not exceed max_epsilon")
        if not math.isfinite(self.learning_rate) or self.learning_rate < 0:
            raise ValueError("learning_rate must be finite and nonnegative")


class DDQNRunState(NamedTuple):
    """Checkpoint all fields, plus config/environment metadata kept separately."""

    learner: DDQNState
    replay: ReplayState
    rngs: RNGState
    env_state: Any
    observation: jax.Array
    training_steps: jax.Array
    episodes: jax.Array
    episode_return: jax.Array
    previous_action: jax.Array


class StepMetrics(NamedTuple):
    """Scalar metrics; unused buffer slots are identified by active=False."""

    active: jax.Array
    action: jax.Array
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    learned: jax.Array
    loss: jax.Array
    episode_return: jax.Array


@dataclass(frozen=True)
class DDQNRunner:
    """Bind a functional environment and immutable config; hold no rollout state.

    Supports JAX environments implementing FunctionalEnv, currently the six
    grid configurations. Plain Gymnasium needs a separate host collector.
    Jit bound initialize/run methods; num_steps is static for run. An outer
    vmap batches independent experiments, with separate replay and counters.
    """

    env: FunctionalEnv
    config: DDQNRunConfig

    def __post_init__(self):
        if len(self.env.observation_shape) != 1:
            raise ValueError("DDQNRunner requires a vector-observation adapter")

    def initialize(self, seed: int) -> DDQNRunState:
        """Initialize one seed, advance model/reset keys and preserve other streams."""
        rngs = init_rngs(seed)
        model_key, init_key = jax.random.split(rngs.model_init)
        environment_key, reset_key = jax.random.split(rngs.environment)
        observation, env_state = self.env.reset(reset_key)
        rngs = rngs._replace(model_init=model_key, environment=environment_key)
        learner = init_ddqn_state(init_key, self.env.observation_shape[0], self.env.num_actions, self.config.hidden_dim)
        return DDQNRunState(
            learner, init_replay(self.config.replay_capacity, self.env.observation_shape,
                                 observation_dtype=getattr(self.env, "observation_storage_dtype", jnp.float32)), rngs,
            env_state, observation, jnp.asarray(0, jnp.int32), jnp.asarray(0, jnp.int32),
            jnp.asarray(0., jnp.float32), jnp.asarray(-1, jnp.int32),
        )

    def _epsilon(self, steps):
        config = self.config
        decay = config.epsilon_decay_steps or max(config.total_steps - config.warmup_steps, 1)
        elapsed = jnp.maximum(steps - config.warmup_steps, 0).astype(jnp.float32)
        if config.epsilon_schedule == "linear":
            fraction = jnp.clip(elapsed / decay, 0., 1.)
            return config.max_epsilon - (config.max_epsilon - config.min_epsilon) * fraction
        return config.min_epsilon + (config.max_epsilon - config.min_epsilon) * jnp.exp(-elapsed / decay)

    def _learning_rate(self, steps):
        config = self.config
        if not config.use_lr_decay:
            return jnp.asarray(config.learning_rate, jnp.float32)
        progress = jnp.clip((steps - config.warmup_steps) / max(config.total_steps - config.warmup_steps, 1), 0., 1.)
        cosine = 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
        return config.learning_rate * (0.1 + 0.9 * cosine)

    def _collect(self, state):
        config = self.config
        warmup = state.training_steps < config.warmup_steps
        action_stream, action_key = jax.random.split(state.rngs.action)
        action = jax.lax.cond(
            warmup,
            lambda _: jax.random.randint(action_key, (), 0, self.env.num_actions),
            lambda _: sample_epsilon_greedy(
                action_key, apply_q(state.learner.online_params, state.observation),
                self._epsilon(state.training_steps),
            ), operand=None,
        )
        environment_stream, step_key = jax.random.split(state.rngs.environment)
        next_obs, env_state, reward, terminated, truncated, _ = self.env.step(step_key, state.env_state, action)
        replay = add_transition(
            state.replay, state.observation, action, reward, next_obs, terminated, truncated,
            previous_action=state.previous_action,
        )
        steps = state.training_steps + 1
        rngs = state.rngs._replace(action=action_stream, environment=environment_stream)

        def learn(_):
            replay_stream, sample_key = jax.random.split(rngs.replay)
            batch = sample_batch(sample_key, replay, config.batch_size)
            learner, loss = ddqn_train_step(
                state.learner, batch, config.gamma, self._learning_rate(state.training_steps), steps,
                tau=config.tau, use_hard_update=config.use_hard_update, update_interval=config.update_interval,
            )
            return learner, rngs._replace(replay=replay_stream), loss

        learner, rngs, loss = jax.lax.cond(
            ~warmup, learn, lambda _: (state.learner, rngs, jnp.asarray(0., jnp.float32)), operand=None,
        )
        done = terminated | truncated
        episode_return = state.episode_return + reward

        def reset_episode(keys):
            environment_stream, reset_key = jax.random.split(keys.environment)
            observation, reset_state = self.env.reset(reset_key)
            return observation, reset_state, keys._replace(environment=environment_stream)

        observation, env_state, rngs = jax.lax.cond(
            done, reset_episode, lambda keys: (next_obs, env_state, keys), rngs,
        )
        updated = DDQNRunState(
            learner, replay, rngs, env_state, observation, steps,
            state.episodes + done.astype(jnp.int32), jnp.where(done, 0., episode_return),
            jnp.where(done, -1, action).astype(jnp.int32),
        )
        metrics = StepMetrics(
            jnp.asarray(True), action, reward, terminated, truncated, ~warmup, loss,
            jnp.where(done, episode_return, 0.),
        )
        return updated, metrics

    def step(self, state: DDQNRunState) -> tuple[DDQNRunState, StepMetrics]:
        """Collect/store, learn if past warmup, then reset after an episode end.

        Store the actual endpoint before reset. Warmup uses seeded uniform
        actions. First learning occurs at environment step warmup_steps+1,
        matching legacy DDQN. Unlike the old root loop, the exact total budget
        is enforced: after it, state and every RNG stream stay unchanged.
        """
        return jax.lax.cond(
            state.training_steps < self.config.total_steps, self._collect,
            lambda current: (current, self._inactive_metrics()), state,
        )

    def _inactive_metrics(self):
        return StepMetrics(
            jnp.asarray(False), jnp.asarray(-1, jnp.int32), jnp.asarray(0., jnp.float32),
            jnp.asarray(False), jnp.asarray(False), jnp.asarray(False),
            jnp.asarray(0., jnp.float32), jnp.asarray(0., jnp.float32),
        )

    def run(self, state: DDQNRunState, num_steps: int, *, active_steps=None) -> tuple[DDQNRunState, StepMetrics]:
        """Collect a dynamic count with fixed-size metrics, not full trajectories.

        Use jax.jit(runner.run, static_argnames=('num_steps',)). State can be
        resumed between chunks, including mid-episode. Metric slots beyond the
        collected count are inactive; ignore them when aggregating metrics.
        Donate argument 0 when the previous state will not be reused, so XLA
        can reuse large replay storage. Keep donation off for branching runs.
        A dynamic active_steps reuses one executable at logging/model boundaries.
        The loop stops after collection; unused metric slots require no extra
        iterations or RNG draws. Pending repeats survive chunk boundaries.
        """
        if type(num_steps) is not int or num_steps <= 0:
            raise ValueError("num_steps must be a positive Python integer")
        if isinstance(active_steps, int) and not 0 <= active_steps <= num_steps:
            raise ValueError("active_steps must lie in [0, num_steps]")
        # The engine supplies one count for all seeds; keep its loop predicate unbatched.
        requested = self.config.total_steps - state.training_steps if active_steps is None else active_steps
        count = jnp.clip(requested, 0, num_steps)
        return collect_steps(
            self.step, state, self._inactive_metrics(), num_steps, count,
        )
