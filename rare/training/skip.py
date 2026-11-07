"""Primitive-step collection for TempoRL, UTE and RARe on discrete actions."""

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from rare.agents.bootstrap import expected_bootstrap, greedy_bootstrap
from rare.agents.ddqn import ddqn_train_step, init_ddqn_state
from rare.agents.evaluation_critic import evaluation_train_step
from rare.agents.policies import epsilon_greedy_probs, sample_epsilon_greedy
from rare.agents.skip import SkipState, init_skip_state, skip_train_step
from rare.agents.ute import init_ute_skip_state, ute_skip_train_step
from rare.networks import apply_q, apply_skip, apply_skip_ensemble
from rare.replay.replay import add_transition, sample_batch
from rare.replay.skip import SkipSegment, add_skip_segment, init_skip_replay, sample_skip_batch
from .ddqn import DDQNRunConfig, DDQNRunner


@dataclass(frozen=True)
class SkipRunConfig(DDQNRunConfig):
    algorithm: str = "TempoRL"
    max_skip: int = 6
    skip_replay_capacity: int = 36000
    ensemble_size: int = 10  # UTE only
    uncertainty_factor: float = -1.5  # UTE only
    bootstrap_probability: float = .5  # UTE only
    rare_q_mode: str = "separate"  # RARe-se: separate; RARe-sh: shared
    max_eval_epsilon: float = .2  # pi_e schedule of RARe-se
    min_eval_epsilon: float = .01
    eval_epsilon_cutoff: float = 1.
    action_target_mode: str = "greedy"  # ablation: "expected" is Expected SARSA on the action critic
    skip_target_mode: str = "expected"  # ablation: "greedy" gives a greedy skip backup


class SkipRunState(NamedTuple):
    learner: Any
    replay: Any
    rngs: Any
    env_state: Any
    observation: jax.Array
    training_steps: jax.Array
    loss: jax.Array  # unused here; see losses
    skip_learner: SkipState
    skip_replay: Any
    evaluation: Any
    pending: SkipSegment
    repeat_remaining: jax.Array
    repeat_warmup: jax.Array
    # Latest (action, evaluation, skip) losses. XLA would otherwise drop them, which changes its
    # fusion and rounding, so removing this field changes the exact results.
    losses: jax.Array


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

    @property
    def _separate_critic(self):
        return self.config.algorithm == "RARe" and self.config.rare_q_mode == "separate"

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
        if self._separate_critic:
            model_stream, eval_key = jax.random.split(model_stream)
            evaluation = init_ddqn_state(eval_key, dimensions[0], dimensions[1], self.config.hidden_dim)
        return SkipRunState(
            *base._replace(rngs=base.rngs._replace(model_init=model_stream)), skip,
            init_skip_replay(self.config.skip_replay_capacity, self.env.observation_shape,
                             observation_dtype=self.env.observation_storage_dtype,
                             observation_codec=self.env.observation_codec), evaluation,
            self._empty_segment(base.observation), jnp.asarray(0, jnp.int32), jnp.asarray(True),
            jnp.zeros(3, jnp.float32),
        )

    def _evaluation_epsilon(self, steps):
        """Epsilon of pi_e: RARe-sh follows the behavior policy, RARe-se its own schedule."""
        if not self._separate_critic:
            return self._epsilon(steps)
        duration = max(self.config.total_steps - self.config.warmup_steps, 1)
        progress = jnp.clip((steps - self.config.warmup_steps) /
                            (duration * self.config.eval_epsilon_cutoff), 0., 1.)
        return (self.config.max_eval_epsilon - progress *
                (self.config.max_eval_epsilon - self.config.min_eval_epsilon))

    def _rate(self, steps):
        """Learning rate of the skip network and the evaluation critic."""
        initial = self.config.learning_rate
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
            values = values.mean(axis=0) + self.config.uncertainty_factor * values.std(axis=0, ddof=1)
        else:
            values = apply_skip(state.skip_learner.params, state.observation, action)
        skip_stream, skip_key = jax.random.split(state.rngs.skip)
        selected = sample_epsilon_greedy(skip_key, values, self._epsilon(state.training_steps)) + 1
        segment = self._empty_segment(state.observation)._replace(action=action, selected_skip=selected)
        return state._replace(
            rngs=state.rngs._replace(action=action_stream, skip=skip_stream), pending=segment,
            repeat_remaining=selected, repeat_warmup=warmup,
        )

    def _learn(self, state, replay):
        config = self.config
        steps = state.training_steps + 1
        replay_stream, sample_key = jax.random.split(state.rngs.replay)
        learner, loss = ddqn_train_step(
            state.learner, sample_batch(sample_key, replay, config.batch_size, **self._storage),
            config.gamma, self._learning_rate(state.training_steps), tau=config.tau,
            target_mode=config.action_target_mode, epsilon=self._epsilon(state.training_steps),
        )
        evaluation = state.evaluation
        eval_loss = jnp.asarray(0., jnp.float32)
        if self._separate_critic:
            replay_stream, sample_key = jax.random.split(replay_stream)
            evaluation, eval_loss = evaluation_train_step(
                evaluation, sample_batch(sample_key, replay, config.batch_size, **self._storage),
                config.gamma, self._evaluation_epsilon(steps), self._rate(state.training_steps),
                tau=config.tau, target_mode=config.skip_target_mode,
            )
        rngs = state.rngs._replace(replay=replay_stream)

        def learn_skip(current_rngs):
            replay_stream, sample_key = jax.random.split(current_rngs.replay)
            batch = sample_skip_batch(sample_key, state.skip_replay, config.batch_size, **self._storage)
            current_rngs = current_rngs._replace(replay=replay_stream)
            bootstrap = self._skip_bootstrap(learner, evaluation, batch.next_observations, steps)
            rate = self._rate(state.training_steps)
            if config.algorithm == "UTE":
                mask_stream, mask_key = jax.random.split(current_rngs.target)
                skip, skip_loss = ute_skip_train_step(state.skip_learner, batch, bootstrap, rate, mask_key,
                                              bootstrap_probability=config.bootstrap_probability)
                current_rngs = current_rngs._replace(target=mask_stream)
            else:
                skip, skip_loss = skip_train_step(state.skip_learner, batch, bootstrap, rate)
            return skip, current_rngs, skip_loss

        skip, rngs, skip_loss = jax.lax.cond(
            state.skip_replay.size > 0, learn_skip,
            lambda keys: (state.skip_learner, keys, jnp.asarray(0., jnp.float32)), rngs)
        return learner, evaluation, skip, rngs, jnp.stack((loss, eval_loss, skip_loss))

    def _skip_bootstrap(self, learner, evaluation, observations, steps):
        """Terminal value of a skip: greedy for TempoRL and UTE, expected under pi_e for RARe."""
        if self.config.algorithm == "RARe":
            critic = evaluation if self._separate_critic else learner
            online_next = apply_q(critic.online_params, observations)
            target_next = apply_q(critic.target_params, observations)
            if self.config.skip_target_mode == "greedy":
                return greedy_bootstrap(online_next, target_next)
            probabilities = epsilon_greedy_probs(online_next, self._evaluation_epsilon(steps))
            return expected_bootstrap(probabilities, target_next)
        return greedy_bootstrap(apply_q(learner.online_params, observations),
                                apply_q(learner.target_params, observations))

    def _collect(self, state):
        state = jax.lax.cond(state.repeat_remaining == 0, self._begin_repeat, lambda current: current, state)
        action = state.pending.action
        stream, step_key = jax.random.split(state.rngs.environment)
        next_obs, env_state, reward, terminated, truncated, _ = self.env.step(step_key, state.env_state, action)
        state = state._replace(rngs=state.rngs._replace(environment=stream))
        replay = add_transition(state.replay, state.observation, action, reward, next_obs,
                                terminated, truncated, **self._storage)
        steps = state.training_steps + 1
        learner, evaluation, skip, rngs, losses = jax.lax.cond(
            ~state.repeat_warmup, lambda _: self._learn(state, replay),
            lambda _: (state.learner, state.evaluation, state.skip_learner, state.rngs, state.losses),
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
        # The learner updates before the finished repeat enters skip replay.
        skip_replay = jax.lax.cond(
            completed, lambda storage: add_skip_segment(storage, segment, self.config.gamma, **self._storage),
            lambda storage: storage, state.skip_replay,
        )
        observation, env_state, rngs = jax.lax.cond(
            done, lambda keys: self._reset(keys, steps), lambda keys: (next_obs, env_state, keys), rngs)
        return state._replace(
            learner=learner, replay=replay, rngs=rngs, env_state=env_state, observation=observation,
            training_steps=steps, skip_learner=skip, skip_replay=skip_replay, evaluation=evaluation,
            pending=segment, repeat_remaining=jnp.where(completed, 0, state.repeat_remaining - 1),
            losses=losses,
        )
