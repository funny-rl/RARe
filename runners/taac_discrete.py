"""Primitive-step discrete TAAC collector for the current JAX experiment loop."""

from dataclasses import dataclass
import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from algorithms.taac_discrete import init_taac_discrete, select_taac_action, taac_discrete_train_step
from buffers import add_transition, init_replay
from buffers.taac_discrete import sample_taac_sequence
from rng import init_rngs
from .ddqn import DDQNRunConfig, DDQNRunner


@dataclass(frozen=True)
class TAACDiscreteRunConfig(DDQNRunConfig):
    temperature: float = 0.1
    actor_temperature: float = 0.1
    target_entropy_delta: float = 0.05
    target_actor_entropy_ratio: float = 0.5
    compare_horizon: int = 10

    def __post_init__(self):
        super().__post_init__()
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not math.isfinite(self.actor_temperature) or self.actor_temperature <= 0:
            raise ValueError("actor_temperature must be finite and positive")
        if not math.isfinite(self.target_entropy_delta) or not 0 < self.target_entropy_delta < 1:
            raise ValueError("target_entropy_delta must lie in (0, 1)")
        if not math.isfinite(self.target_actor_entropy_ratio) or not 0 < self.target_actor_entropy_ratio < 1:
            raise ValueError("target_actor_entropy_ratio must lie in (0, 1)")
        if type(self.compare_horizon) is not int or not 1 <= self.compare_horizon <= self.replay_capacity:
            raise ValueError("compare_horizon must lie in [1, replay_capacity]")


class TAACDiscreteRunState(NamedTuple):
    learner: Any
    replay: Any
    rngs: Any
    env_state: Any
    observation: jax.Array
    training_steps: jax.Array
    episodes: jax.Array
    episode_return: jax.Array
    previous_action: jax.Array


class TAACDiscreteStepMetrics(NamedTuple):
    active: jax.Array
    action: jax.Array
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    learned: jax.Array
    loss: jax.Array
    episode_return: jax.Array
    action_changed: jax.Array
    actor_loss: jax.Array
    gate_entropy: jax.Array


@dataclass(frozen=True)
class TAACDiscreteRunner(DDQNRunner):
    config: TAACDiscreteRunConfig

    def _epsilon(self, steps):
        # The categorical proposal actor supplies exploration after warmup.
        return jnp.float32(0)

    def _inactive_metrics(self):
        return TAACDiscreteStepMetrics(
            jnp.bool_(False), jnp.int32(-1), jnp.float32(0), jnp.bool_(False),
            jnp.bool_(False), jnp.bool_(False), jnp.float32(0), jnp.float32(0),
            jnp.bool_(False), jnp.float32(0), jnp.float32(0))

    def initialize(self, seed):
        rngs = init_rngs(seed)
        model_key, init_key = jax.random.split(rngs.model_init)
        environment_key, reset_key = jax.random.split(rngs.environment)
        observation, env_state = self.env.reset(reset_key)
        rngs = rngs._replace(model_init=model_key, environment=environment_key)
        learner = init_taac_discrete(init_key, self.env.observation_shape[0], self.env.num_actions,
                                     self.config.hidden_dim, self.config.temperature,
                                     self.config.actor_temperature)
        replay = init_replay(self.config.replay_capacity, self.env.observation_shape,
                             observation_dtype=getattr(self.env, "observation_storage_dtype", jnp.float32),
                             observation_codec=getattr(self.env, "observation_codec", "raw"))
        return TAACDiscreteRunState(learner, replay, rngs, env_state, observation,
                                    jnp.int32(0), jnp.int32(0), jnp.float32(0), jnp.int32(-1))

    def _collect(self, state):
        config = self.config
        warmup = state.training_steps < config.warmup_steps
        action_stream, action_key = jax.random.split(state.rngs.action)
        def choose(_):
            return select_taac_action(state.learner, state.observation, state.previous_action,
                                      action_key)[0]
        action = jax.lax.cond(warmup, lambda _: jax.random.randint(action_key, (), 0, self.env.num_actions),
                              choose, operand=None)
        environment_stream, step_key = jax.random.split(state.rngs.environment)
        next_obs, env_state, reward, terminated, truncated, _ = self.env.step(step_key, state.env_state, action)
        replay = add_transition(state.replay, state.observation, action, reward, next_obs,
                                terminated, truncated, previous_action=state.previous_action,
                                observation_codec=getattr(self.env, "observation_codec", "raw"),
                                observation_dim=self.env.observation_shape[0])
        steps = state.training_steps + 1
        rngs = state.rngs._replace(action=action_stream, environment=environment_stream)

        def learn(_):
            replay_stream, sample_key = jax.random.split(rngs.replay)
            target_stream, gumbel_key = jax.random.split(rngs.target)
            batch = sample_taac_sequence(sample_key, replay, config.batch_size, config.compare_horizon,
                                         observation_codec=getattr(self.env, "observation_codec", "raw"),
                                         observation_dim=self.env.observation_shape[0])
            learner, loss, actor_loss, gate_entropy = taac_discrete_train_step(
                state.learner, batch, gumbel_key, config.gamma,
                self._learning_rate(state.training_steps), steps, config.target_entropy_delta,
                config.target_actor_entropy_ratio,
                tau=config.tau, use_hard_update=config.use_hard_update,
                update_interval=config.update_interval)
            return learner, rngs._replace(replay=replay_stream, target=target_stream), loss, actor_loss, gate_entropy

        learned = ~warmup & (replay.size >= config.compare_horizon)
        learner, rngs, loss, actor_loss, gate_entropy = jax.lax.cond(
            learned, learn,
            lambda _: (state.learner, rngs, jnp.float32(0), jnp.float32(0), jnp.float32(0)),
            operand=None)
        done = terminated | truncated
        episode_return = state.episode_return + reward

        def reset_episode(keys):
            environment_stream, reset_key = jax.random.split(keys.environment)
            observation, reset_state = (
                self.env.reset(reset_key, global_step=steps)
                if getattr(self.env, "requires_global_step", False) else self.env.reset(reset_key))
            return observation, reset_state, keys._replace(environment=environment_stream)

        observation, env_state, rngs = jax.lax.cond(
            done, reset_episode, lambda keys: (next_obs, env_state, keys), rngs)
        updated = TAACDiscreteRunState(
            learner, replay, rngs, env_state, observation, steps,
            state.episodes + done.astype(jnp.int32), jnp.where(done, 0., episode_return),
            jnp.where(done, -1, action).astype(jnp.int32))
        metrics = TAACDiscreteStepMetrics(
            jnp.bool_(True), action, reward, terminated, truncated, learned, loss,
            jnp.where(done, episode_return, 0.),
            (state.previous_action < 0) | (action != state.previous_action),
            actor_loss, gate_entropy)
        return updated, metrics
