"""Primitive-step collectors with action repeats retained across compiled chunks."""

from dataclasses import dataclass
import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from algorithms import ddqn_train_step, init_ddqn_state
from algorithms.bootstrap import expected_bootstrap, greedy_bootstrap
from algorithms.evaluation import evaluation_train_step
from algorithms.skip import SkipState, init_skip_state, skip_train_step
from algorithms.ute import init_ute_skip_state, ute_skip_train_step
from buffers import add_transition, sample_batch
from buffers.skip import SkipSegment, add_skip_segment, init_skip_replay, sample_skip_batch
from networks import apply_q, apply_skip, apply_skip_ensemble
from policies import epsilon_greedy_probs, sample_epsilon_greedy
from .ddqn import DDQNRunConfig, DDQNRunner


@dataclass(frozen=True)
class SkipRunConfig(DDQNRunConfig):
    algorithm: str = "TempoRL"
    max_skip: int = 6
    skip_replay_capacity: int = 36000
    skip_learning_rate: float | None = None
    use_data_aug: bool = True
    ensemble_size: int = 5
    uncertainty_factor: float = -1.5
    bootstrap_probability: float = .5
    max_eval_epsilon: float = .2
    min_eval_epsilon: float = .01
    eval_epsilon_cutoff: float = 1.
    eval_learning_rate: float | None = None
    rare_q_mode: str = "separate"
    separate_target_epsilon_source: str = "evaluation"
    shared_target_epsilon_source: str = "behavior"

    def __post_init__(self):
        super().__post_init__()
        if self.algorithm not in ("TempoRL", "UTE", "RARe"):
            raise ValueError("unsupported skip algorithm")
        if self.rare_q_mode not in ("separate", "shared"):
            raise ValueError("rare_q_mode must be 'separate' or 'shared'")
        if self.algorithm != "RARe" and self.rare_q_mode != "separate":
            raise ValueError("rare_q_mode='shared' requires RARe")
        if self.separate_target_epsilon_source not in ("behavior", "evaluation"):
            raise ValueError("separate_target_epsilon_source must be 'behavior' or 'evaluation'")
        if self.separate_target_epsilon_source != "evaluation" and (self.algorithm != "RARe" or self.rare_q_mode != "separate"):
            raise ValueError("separate_target_epsilon_source='behavior' requires separate RARe")
        if self.shared_target_epsilon_source not in ("behavior", "evaluation"):
            raise ValueError("shared_target_epsilon_source must be 'behavior' or 'evaluation'")
        if self.shared_target_epsilon_source != "behavior" and (self.algorithm != "RARe" or self.rare_q_mode != "shared"):
            raise ValueError("shared_target_epsilon_source='evaluation' requires shared RARe")
        for name in ("max_skip", "skip_replay_capacity", "ensemble_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.use_data_aug) is not bool:
            raise TypeError("use_data_aug must be bool")
        for name in ("skip_learning_rate", "eval_learning_rate"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative or null")
        if not math.isfinite(self.uncertainty_factor):
            raise ValueError("uncertainty_factor must be finite")
        if self.algorithm == "UTE" and self.ensemble_size < 2:
            raise ValueError("UTE requires ensemble_size >= 2 for sample standard deviation")
        if not 0 < self.bootstrap_probability <= 1:
            raise ValueError("bootstrap_probability must lie in (0,1]")
        if not 0 <= self.min_eval_epsilon <= self.max_eval_epsilon <= 1:
            raise ValueError("evaluation epsilons must satisfy 0 <= min <= max <= 1")
        if not math.isfinite(self.eval_epsilon_cutoff) or self.eval_epsilon_cutoff <= 0:
            raise ValueError("eval_epsilon_cutoff must be finite and positive")


class SkipRunState(NamedTuple):
    learner: Any
    replay: Any
    rngs: Any
    env_state: Any
    observation: jax.Array
    training_steps: jax.Array
    episodes: jax.Array
    episode_return: jax.Array
    previous_action: jax.Array
    skip_learner: SkipState
    skip_replay: Any
    evaluation: Any
    pending: SkipSegment
    repeat_remaining: jax.Array
    repeat_warmup: jax.Array
    decisions: jax.Array


class SkipStepMetrics(NamedTuple):
    active: jax.Array
    action: jax.Array
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    learned: jax.Array
    loss: jax.Array
    episode_return: jax.Array
    skip_learned: jax.Array
    skip_loss: jax.Array
    evaluation_loss: jax.Array
    decision: jax.Array
    selected_skip: jax.Array
    segment_completed: jax.Array
    executed_skip: jax.Array


@dataclass(frozen=True)
class SkipRunner(DDQNRunner):
    config: SkipRunConfig

    def _empty_segment(self, observation):
        length = self.config.max_skip
        return SkipSegment(
            jnp.zeros((length + 1, *self.env.observation_shape), jnp.float32).at[0].set(observation),
            jnp.asarray(0, jnp.int32), jnp.zeros(length, jnp.float32),
            jnp.zeros(length, bool), jnp.zeros(length, bool),
            jnp.asarray(1, jnp.int32), jnp.asarray(0, jnp.int32),
        )

    def initialize(self, seed):
        base = super().initialize(seed)
        model_stream, skip_key = jax.random.split(base.rngs.model_init)
        dimensions = (self.env.observation_shape[0], self.env.num_actions, self.config.max_skip)
        if self.config.algorithm == "UTE":
            skip = init_ute_skip_state(skip_key, *dimensions, ensemble_size=self.config.ensemble_size,
                                       hidden_dim=self.config.hidden_dim)
        else:
            skip = init_skip_state(skip_key, *dimensions, hidden_dim=self.config.hidden_dim)
        evaluation = None
        if self.config.algorithm == "RARe" and self.config.rare_q_mode == "separate":
            model_stream, eval_key = jax.random.split(model_stream)
            evaluation = init_ddqn_state(eval_key, dimensions[0], dimensions[1], self.config.hidden_dim)
        return SkipRunState(
            *base._replace(rngs=base.rngs._replace(model_init=model_stream)), skip,
            init_skip_replay(self.config.skip_replay_capacity, self.env.observation_shape,
                             observation_dtype=getattr(self.env, "observation_storage_dtype", jnp.float32),
                             observation_codec=getattr(self.env, "observation_codec", "raw")), evaluation,
            self._empty_segment(base.observation), jnp.asarray(0, jnp.int32), jnp.asarray(True),
            jnp.asarray(0, jnp.int32),
        )

    def _evaluation_epsilon(self, steps):
        source = (self.config.separate_target_epsilon_source if self.config.rare_q_mode == "separate"
                  else self.config.shared_target_epsilon_source)
        if source == "behavior":
            return self._epsilon(steps)
        duration = max(self.config.total_steps - self.config.warmup_steps, 1)
        progress = jnp.clip((steps - self.config.warmup_steps) /
                            (duration * self.config.eval_epsilon_cutoff), 0., 1.)
        return (self.config.max_eval_epsilon - progress *
                (self.config.max_eval_epsilon - self.config.min_eval_epsilon))

    def _rate(self, steps, initial):
        initial = self.config.learning_rate if initial is None else initial
        if not self.config.use_lr_decay:
            return jnp.asarray(initial, jnp.float32)
        progress = jnp.clip((steps - self.config.warmup_steps) / max(self.config.total_steps - self.config.warmup_steps, 1), 0., 1.)
        return initial * (.1 + .9 * .5 * (1. + jnp.cos(jnp.pi * progress)))

    def _begin_repeat(self, state):
        action_stream, action_key = jax.random.split(state.rngs.action)
        warmup = state.training_steps < self.config.warmup_steps
        action = jax.lax.cond(
            warmup, lambda _: jax.random.randint(action_key, (), 0, self.env.num_actions),
            lambda _: sample_epsilon_greedy(action_key, apply_q(state.learner.online_params, state.observation),
                                             self._epsilon(state.training_steps)), operand=None,
        )
        if self.config.algorithm == "UTE":
            values = apply_skip_ensemble(state.skip_learner.params, state.observation, action)
            # Torch std uses the sample standard deviation (correction=1).
            values = values.mean(axis=0) + self.config.uncertainty_factor * values.std(axis=0, ddof=1)
        else:
            values = apply_skip(state.skip_learner.params, state.observation, action)
        skip_stream, skip_key = jax.random.split(state.rngs.skip)
        selected = sample_epsilon_greedy(skip_key, values, self._epsilon(state.training_steps)) + 1
        segment = self._empty_segment(state.observation)._replace(action=action, selected_skip=selected)
        return state._replace(
            rngs=state.rngs._replace(action=action_stream, skip=skip_stream), pending=segment,
            repeat_remaining=selected, repeat_warmup=warmup, decisions=state.decisions + 1,
        )

    def _learn(self, state, replay, steps):
        config = self.config
        target_settings = dict(tau=config.tau, use_hard_update=config.use_hard_update, update_interval=config.update_interval)
        replay_stream, sample_key = jax.random.split(state.rngs.replay)
        learner, loss = ddqn_train_step(
            state.learner, sample_batch(
                sample_key, replay, config.batch_size,
                observation_codec=getattr(self.env, "observation_codec", "raw"),
                observation_dim=self.env.observation_shape[0]), config.gamma,
            self._learning_rate(state.training_steps), steps, **target_settings,
        )
        evaluation, eval_loss = state.evaluation, jnp.asarray(0., jnp.float32)
        if config.algorithm == "RARe" and config.rare_q_mode == "separate":
            replay_stream, sample_key = jax.random.split(replay_stream)
            evaluation, eval_loss = evaluation_train_step(
                evaluation, sample_batch(
                    sample_key, replay, config.batch_size,
                    observation_codec=getattr(self.env, "observation_codec", "raw"),
                    observation_dim=self.env.observation_shape[0]), config.gamma,
                self._evaluation_epsilon(steps), self._rate(state.training_steps, config.eval_learning_rate),
                steps, **target_settings,
            )
        rngs = state.rngs._replace(replay=replay_stream)

        def learn_skip(current_rngs):
            replay_stream, sample_key = jax.random.split(current_rngs.replay)
            batch = sample_skip_batch(
                sample_key, state.skip_replay, config.batch_size,
                observation_codec=getattr(self.env, "observation_codec", "raw"),
                observation_dim=self.env.observation_shape[0])
            current_rngs = current_rngs._replace(replay=replay_stream)
            bootstrap = self._skip_bootstrap(learner, evaluation, batch.next_observations, steps)
            rate = self._rate(state.training_steps, config.skip_learning_rate)
            if config.algorithm == "UTE":
                mask_stream, mask_key = jax.random.split(current_rngs.target)
                skip, skip_loss = ute_skip_train_step(state.skip_learner, batch, bootstrap, rate, mask_key,
                                                      bootstrap_probability=config.bootstrap_probability)
                current_rngs = current_rngs._replace(target=mask_stream)
            else:
                skip, skip_loss = skip_train_step(state.skip_learner, batch, bootstrap, rate)
            return skip, current_rngs, skip_loss

        skip_learned = state.skip_replay.size > 0
        skip, rngs, skip_loss = jax.lax.cond(
            skip_learned, learn_skip,
            lambda keys: (state.skip_learner, keys, jnp.asarray(0., jnp.float32)), rngs,
        )
        return learner, evaluation, skip, rngs, loss, eval_loss, skip_loss, skip_learned

    def _skip_bootstrap(self, learner, evaluation, observations, steps):
        if self.config.algorithm == "RARe":
            # Shared uses base Q, Separate its own critic; both use pi_e epsilon.
            critic = evaluation if self.config.rare_q_mode == "separate" else learner
            probabilities = epsilon_greedy_probs(apply_q(critic.online_params, observations),
                                                  self._evaluation_epsilon(steps))
            return expected_bootstrap(probabilities, apply_q(critic.target_params, observations))
        return greedy_bootstrap(apply_q(learner.online_params, observations),
                                 apply_q(learner.target_params, observations))

    def _collect(self, state):
        decision = state.repeat_remaining == 0
        state = jax.lax.cond(decision, self._begin_repeat, lambda current: current, state)
        action = state.pending.action
        stream, step_key = jax.random.split(state.rngs.environment)
        next_obs, env_state, reward, terminated, truncated, _ = self.env.step(step_key, state.env_state, action)
        state = state._replace(rngs=state.rngs._replace(environment=stream))
        replay = add_transition(state.replay, state.observation, action, reward, next_obs, terminated, truncated,
                                previous_action=state.previous_action,
                                observation_codec=getattr(self.env, "observation_codec", "raw"),
                                observation_dim=self.env.observation_shape[0])
        steps = state.training_steps + 1
        learned = ~state.repeat_warmup
        learner, evaluation, skip, rngs, loss, eval_loss, skip_loss, skip_learned = jax.lax.cond(
            learned, lambda _: self._learn(state, replay, steps),
            lambda _: (state.learner, state.evaluation, state.skip_learner, state.rngs,
                       jnp.asarray(0., jnp.float32), jnp.asarray(0., jnp.float32), jnp.asarray(0., jnp.float32), jnp.asarray(False)),
            operand=None,
        )
        index = state.pending.executed_steps
        segment = state.pending._replace(
            observations=state.pending.observations.at[index + 1].set(next_obs),
            rewards=state.pending.rewards.at[index].set(reward),
            terminated=state.pending.terminated.at[index].set(terminated),
            truncated=state.pending.truncated.at[index].set(truncated), executed_steps=index + 1,
        )
        done = terminated | truncated
        completed = done | (state.repeat_remaining == 1) | (steps == self.config.total_steps)
        # Match the old update-before-add_skip order. Chunk/model boundaries
        # never flush a partial repeat; only its actual end or budget does.
        skip_replay = jax.lax.cond(
            completed, lambda storage: add_skip_segment(storage, segment, self.config.gamma,
                                                        augment_subsegments=self.config.use_data_aug,
                                                        observation_codec=getattr(self.env, "observation_codec", "raw"),
                                                        observation_dim=self.env.observation_shape[0]),
            lambda storage: storage, state.skip_replay,
        )
        episode_return = state.episode_return + reward
        def reset(keys):
            stream, reset_key = jax.random.split(keys.environment)
            observation, reset_state = (
                self.env.reset(reset_key, global_step=steps)
                if getattr(self.env, "requires_global_step", False) else self.env.reset(reset_key)
            )
            return observation, reset_state, keys._replace(environment=stream)
        observation, env_state, rngs = jax.lax.cond(done, reset, lambda keys: (next_obs, env_state, keys), rngs)
        updated = state._replace(
            learner=learner, replay=replay, rngs=rngs, env_state=env_state, observation=observation,
            training_steps=steps, episodes=state.episodes + done.astype(jnp.int32),
            episode_return=jnp.where(done, 0., episode_return), previous_action=jnp.where(done, -1, action),
            skip_learner=skip, skip_replay=skip_replay, evaluation=evaluation, pending=segment,
            repeat_remaining=jnp.where(completed, 0, state.repeat_remaining - 1),
        )
        metrics = SkipStepMetrics(
            jnp.asarray(True), action, reward, terminated, truncated, learned, loss,
            jnp.where(done, episode_return, 0.), skip_learned, skip_loss, eval_loss, decision,
            jnp.where(decision, segment.selected_skip, 0), completed, jnp.where(completed, index + 1, 0),
        )
        return updated, metrics

    def step(self, state):
        return jax.lax.cond(state.training_steps < self.config.total_steps, self._collect,
                            lambda current: (current, self._inactive_metrics()), state)

    def _inactive_metrics(self):
        return SkipStepMetrics(
            jnp.asarray(False), jnp.asarray(-1, jnp.int32), jnp.asarray(0., jnp.float32),
            jnp.asarray(False), jnp.asarray(False), jnp.asarray(False), jnp.asarray(0., jnp.float32),
            jnp.asarray(0., jnp.float32), jnp.asarray(False), jnp.asarray(0., jnp.float32),
            jnp.asarray(0., jnp.float32), jnp.asarray(False), jnp.asarray(0, jnp.int32),
            jnp.asarray(False), jnp.asarray(0, jnp.int32),
        )
