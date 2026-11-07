"""DDQN primitive collection, replay and learning inside a compiled JAX loop."""

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from rare.agents.ddqn import DDQNState, ddqn_train_step, init_ddqn_state
from rare.agents.policies import sample_epsilon_greedy
from rare.envs.base import FunctionalEnv
from rare.networks import apply_q
from rare.replay.replay import ReplayState, add_transition, init_replay, sample_batch
from rare.rng import RNGState, init_rngs
from .loop import run_steps


@dataclass(frozen=True)
class DDQNRunConfig:
    """Static experiment settings; total_steps includes warm-up interactions."""

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
    use_lr_decay: bool = False
    tau: float = 0.005


class DDQNRunState(NamedTuple):
    learner: DDQNState
    replay: ReplayState
    rngs: RNGState
    env_state: Any
    observation: jax.Array
    training_steps: jax.Array
    loss: jax.Array  # latest TD loss


@dataclass(frozen=True)
class DDQNRunner:
    """Bind an environment and config. Rollout state stays outside, so vmap batches the seeds."""

    env: FunctionalEnv
    config: DDQNRunConfig

    @property
    def _storage(self):
        return dict(observation_codec=self.env.observation_codec,
                    observation_dim=self.env.observation_shape[0])

    def initialize(self, seed: int) -> DDQNRunState:
        rngs = init_rngs(seed)
        model_key, init_key = jax.random.split(rngs.model_init)
        environment_key, reset_key = jax.random.split(rngs.environment)
        observation, env_state = self.env.reset(reset_key)
        rngs = rngs._replace(model_init=model_key, environment=environment_key)
        learner = init_ddqn_state(init_key, self.env.observation_shape[0], self.env.num_actions,
                                  self.config.hidden_dim)
        replay = init_replay(self.config.replay_capacity, self.env.observation_shape,
                             observation_dtype=self.env.observation_storage_dtype,
                             observation_codec=self.env.observation_codec)
        return DDQNRunState(learner, replay, rngs, env_state, observation, jnp.asarray(0, jnp.int32),
                            jnp.asarray(0., jnp.float32))

    def _epsilon(self, steps):
        config = self.config
        decay = config.epsilon_decay_steps or max(config.total_steps - config.warmup_steps, 1)
        elapsed = jnp.maximum(steps - config.warmup_steps, 0).astype(jnp.float32)
        fraction = jnp.clip(elapsed / decay, 0., 1.)
        return config.max_epsilon - (config.max_epsilon - config.min_epsilon) * fraction

    def _learning_rate(self, steps):
        config = self.config
        if not config.use_lr_decay:
            return jnp.asarray(config.learning_rate, jnp.float32)
        progress = jnp.clip((steps - config.warmup_steps) / max(config.total_steps - config.warmup_steps, 1), 0., 1.)
        cosine = 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
        return config.learning_rate * (0.1 + 0.9 * cosine)

    def _reset(self, keys, steps):
        stream, reset_key = jax.random.split(keys.environment)
        observation, state = (self.env.reset(reset_key, global_step=steps)
                              if self.env.requires_global_step else self.env.reset(reset_key))
        return observation, state, keys._replace(environment=stream)

    def _collect(self, state):
        """Act, store the real successor, learn after warm-up, then reset at an episode end."""
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
        replay = add_transition(state.replay, state.observation, action, reward, next_obs,
                                terminated, truncated, **self._storage)
        steps = state.training_steps + 1
        rngs = state.rngs._replace(action=action_stream, environment=environment_stream)

        def learn(_):
            replay_stream, sample_key = jax.random.split(rngs.replay)
            batch = sample_batch(sample_key, replay, config.batch_size, **self._storage)
            learner, loss = ddqn_train_step(state.learner, batch, config.gamma,
                                            self._learning_rate(state.training_steps), tau=config.tau)
            return learner, rngs._replace(replay=replay_stream), loss

        learner, rngs, loss = jax.lax.cond(~warmup, learn, lambda _: (state.learner, rngs, state.loss),
                                           operand=None)
        observation, env_state, rngs = jax.lax.cond(
            terminated | truncated, lambda keys: self._reset(keys, steps),
            lambda keys: (next_obs, env_state, keys), rngs)
        return DDQNRunState(learner, replay, rngs, env_state, observation, steps, loss)

    def run(self, state, count):
        """Collect `count` interactions, stopping at the exact total budget."""
        return run_steps(
            lambda current: jax.lax.cond(current.training_steps < self.config.total_steps,
                                         self._collect, lambda unchanged: unchanged, current),
            state, count)
