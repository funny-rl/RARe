"""One compiled primitive-step collector for every continuous-control method."""

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from rare.agents.ddpg import (continuous_skip_train_step, ddpg_train_step,
                                   evaluation_critic_train_step, gaussian_expected_q,
                                   init_continuous_skip, init_ddpg, init_evaluation_critic)
from rare.agents.taac import init_taac, select_taac_action, taac_train_step
from rare.replay.continuous import (add_continuous_transition, init_continuous_replay,
                                sample_continuous_batch)
from rare.replay.skip import SkipSegment, add_skip_segment, init_skip_replay, sample_skip_batch
from rare.replay.sequence import sample_taac_sequence
from rare.envs.base import ContinuousFunctionalEnv
from rare.networks.continuous import apply_actor, apply_continuous_skip, apply_critic
from rare.rng import RNGState, init_rngs
from .loop import run_steps


@dataclass(frozen=True)
class ContinuousRunConfig:
    algorithm: str
    total_steps: int
    warmup_steps: int
    replay_capacity: int
    batch_size: int = 64
    hidden_dim: int = 64
    gamma: float = .99
    learning_rate: float = .001
    tau: float = .005
    use_lr_decay: bool = True
    exploration_noise: float = .1
    exploration_decay_steps: int | None = None
    max_skip: int = 10
    skip_replay_capacity: int = 50000
    skip_epsilon_decay_steps: int | None = None
    ensemble_size: int = 10  # UTE only
    uncertainty_factor: float = -1.5  # UTE only, when the lambda bandit is off
    ute_adaptive_lambda: bool = False
    bootstrap_probability: float = .5  # UTE only
    rare_q_mode: str = "separate"  # RARe-se: separate; RARe-sh: shared
    mc_samples: int = 30
    max_eval_sigma: float = .05  # sigma_e schedule of RARe-se
    min_eval_sigma: float = .001
    eval_sigma_cutoff: float = 1.
    taac_temperature: float = .1
    target_entropy_delta: float = .05
    compare_horizon: int = 3  # TAAC replay window


class ContinuousRunState(NamedTuple):
    learner: Any
    replay: Any
    rngs: RNGState
    env_state: Any
    observation: jax.Array
    training_steps: jax.Array
    episodes: jax.Array
    episode_return: jax.Array
    previous_action: jax.Array
    has_previous: jax.Array
    skip_learner: Any
    skip_replay: Any
    evaluation: Any
    pending: Any
    repeat_remaining: jax.Array
    repeat_warmup: jax.Array
    ute_bandit: Any
    # Latest (critic, actor, skip, evaluation, gate entropy) losses, kept for the reason given
    # in rare/training/skip.py.
    losses: jax.Array


class UTEBandit(NamedTuple):
    rewards: jax.Array
    arms: jax.Array
    write_index: jax.Array
    size: jax.Array
    choices: jax.Array
    current: jax.Array


_UTE_LAMBDAS = jnp.asarray((-1.5, -1., -.5, -.2, 0., .2, .5, 1., 1.5), jnp.float32)


def _record_ute_reward(bandit: UTEBandit, reward):
    index = bandit.write_index
    capacity = bandit.rewards.shape[0]
    return bandit._replace(
        rewards=bandit.rewards.at[index].set(reward),
        arms=bandit.arms.at[index].set(bandit.current),
        write_index=(index + 1) % capacity,
        size=jnp.minimum(bandit.size + 1, capacity),
    )


def _next_ute_arm(bandit: UTEBandit, key):
    """Choose the next lambda with the nine-arm, 500-step UCB rule."""
    valid = jnp.arange(bandit.rewards.shape[0]) < bandit.size
    matches = (bandit.arms[:, None] == jnp.arange(len(_UTE_LAMBDAS))[None, :]) & valid[:, None]
    counts = jnp.sum(matches, axis=0)
    means = jnp.sum(jnp.where(matches, bandit.rewards[:, None], 0.), axis=0) / (counts + 1e-10)
    scores = means + .5 * jnp.sqrt(1. / (counts + 1e-6))
    explore_key, choice_key = jax.random.split(key)
    greedy = jnp.argmax(scores).astype(jnp.int32)
    sampled = jax.random.randint(choice_key, (), 0, len(_UTE_LAMBDAS))
    arm = jnp.where(bandit.choices < len(_UTE_LAMBDAS), bandit.choices,
                    jnp.where(jax.random.uniform(explore_key) < .1, sampled, greedy))
    return bandit._replace(current=arm, choices=bandit.choices + 1)


@dataclass(frozen=True)
class ContinuousRunner:
    env: ContinuousFunctionalEnv
    config: ContinuousRunConfig

    def _is_skip(self):
        return self.config.algorithm in ("TempoRL", "UTE", "RARe")

    def _progress(self, steps):
        return jnp.clip((steps - self.config.warmup_steps) /
                        max(self.config.total_steps - self.config.warmup_steps, 1), 0., 1.)

    def _learning_rate(self, steps):
        if not self.config.use_lr_decay:
            return jnp.float32(self.config.learning_rate)
        return self.config.learning_rate * (.1 + .45 * (1. + jnp.cos(jnp.pi * self._progress(steps))))

    def _noise(self, steps):
        duration = self.config.exploration_decay_steps or (
            self.config.total_steps - self.config.warmup_steps)
        progress = jnp.clip((steps - self.config.warmup_steps) / max(duration, 1), 0., 1.)
        return self.config.exploration_noise * (1. - progress)

    def _skip_epsilon(self, steps):
        duration = self.config.skip_epsilon_decay_steps or (
            self.config.total_steps - self.config.warmup_steps)
        return 1. - jnp.clip(jnp.maximum(steps - self.config.warmup_steps, 0.) / duration, 0., 1.)

    def _eval_sigma(self, steps):
        progress = jnp.clip(self._progress(steps) / self.config.eval_sigma_cutoff, 0., 1.)
        return self.config.max_eval_sigma + progress * (
            self.config.min_eval_sigma - self.config.max_eval_sigma)

    def _target_sigma(self, steps):
        """sigma_e of pi_e: RARe-sh follows the behavior noise, RARe-se its own schedule."""
        return self._noise(steps) if self.config.rare_q_mode == "shared" else self._eval_sigma(steps)

    def _empty_segment(self, observation):
        length = self.config.max_skip
        return SkipSegment(
            jnp.zeros((length + 1, *self.env.observation_shape), jnp.float32).at[0].set(observation),
            jnp.zeros(self.env.action_shape, jnp.float32),
            jnp.zeros(length, jnp.float32), jnp.zeros(length, bool), jnp.zeros(length, bool),
            jnp.int32(1), jnp.int32(0),
        )

    def initialize(self, seed):
        rngs = init_rngs(seed)
        env_stream, reset_key = jax.random.split(rngs.environment)
        observation, env_state = self.env.reset(reset_key)
        model_stream, init_key = jax.random.split(rngs.model_init)
        obs_dim, action_dim = self.env.observation_shape[0], self.env.action_shape[0]
        if self.config.algorithm == "TAAC":
            learner = init_taac(init_key, obs_dim, action_dim, self.config.hidden_dim,
                                self.config.taac_temperature)
        else:
            learner = init_ddpg(init_key, obs_dim, action_dim, self.config.hidden_dim)
        skip, skip_replay, evaluation, pending = None, None, None, None
        if self._is_skip():
            model_stream, skip_key = jax.random.split(model_stream)
            skip = init_continuous_skip(skip_key, obs_dim, action_dim, self.config.max_skip,
                                        self.config.hidden_dim,
                                        self.config.ensemble_size if self.config.algorithm == "UTE" else 1)
            skip_replay = init_skip_replay(self.config.skip_replay_capacity,
                                           self.env.observation_shape,
                                           action_shape=self.env.action_shape,
                                           action_dtype=jnp.float32)
            pending = self._empty_segment(observation)
            if self.config.algorithm == "RARe" and self.config.rare_q_mode == "separate":
                model_stream, eval_key = jax.random.split(model_stream)
                evaluation = init_evaluation_critic(eval_key, obs_dim, action_dim,
                                                    self.config.hidden_dim)
        rngs = rngs._replace(environment=env_stream, model_init=model_stream)
        ute_bandit = None
        if self.config.algorithm == "UTE" and self.config.ute_adaptive_lambda:
            ute_bandit = UTEBandit(jnp.zeros(500, jnp.float32), jnp.zeros(500, jnp.int32),
                                   jnp.int32(0), jnp.int32(0), jnp.int32(1), jnp.int32(0))
        return ContinuousRunState(
            learner, init_continuous_replay(self.config.replay_capacity, obs_dim, action_dim),
            rngs, env_state, observation, jnp.int32(0), jnp.int32(0), jnp.float32(0),
            jnp.zeros(self.env.action_shape, jnp.float32), jnp.bool_(False),
            skip, skip_replay, evaluation, pending, jnp.int32(0), jnp.bool_(True), ute_bandit,
            jnp.zeros(5, jnp.float32),
        )

    def _choose_action(self, state):
        stream, key = jax.random.split(state.rngs.action)
        warmup = state.training_steps < self.config.warmup_steps
        low, high = self.env.action_low, self.env.action_high

        def random_action(_):
            action = jax.random.uniform(key, self.env.action_shape, minval=low, maxval=high)
            if self.config.algorithm == "TAAC":
                # Random stay-or-switch gate during warm-up; episode starts always switch.
                switched = jax.random.bernoulli(jax.random.fold_in(key, 1)) | ~state.has_previous
                return jnp.where(switched, action, state.previous_action), switched
            return action, jnp.bool_(True)

        def policy_action(_):
            if self.config.algorithm == "TAAC":
                return select_taac_action(state.learner, state.observation,
                                          state.previous_action, state.has_previous,
                                          key, low, high, self._noise(state.training_steps))
            mean = apply_actor(state.learner.actor, state.observation, low, high)
            action = jnp.clip(mean + self._noise(state.training_steps) *
                              jax.random.normal(key, self.env.action_shape), low, high)
            return action, jnp.bool_(True)

        action, switched = jax.lax.cond(warmup, random_action, policy_action, operand=None)
        return action, switched, state.rngs._replace(action=stream)

    def _begin_repeat(self, state):
        action, _, rngs = self._choose_action(state)
        stream, key = jax.random.split(rngs.skip)
        warmup = state.training_steps < self.config.warmup_steps

        def choose_skip(_):
            if self.config.algorithm == "UTE":
                values = jax.vmap(apply_continuous_skip, in_axes=(0, None, None))(
                    state.skip_learner.params, state.observation, action)
                factor = (_UTE_LAMBDAS[state.ute_bandit.current]
                          if state.ute_bandit is not None else self.config.uncertainty_factor)
                scores = values.mean(0) + factor * values.std(0, ddof=1)
            else:
                scores = apply_continuous_skip(state.skip_learner.params, state.observation, action)
            greedy = jnp.argmax(scores).astype(jnp.int32) + 1
            explore_key, choice_key = jax.random.split(key)
            explore = jax.random.uniform(explore_key) < self._skip_epsilon(state.training_steps)
            random = jax.random.randint(choice_key, (), 1, self.config.max_skip + 1)
            return jnp.where(explore, random, greedy)

        selected = jax.lax.cond(warmup,
                                lambda _: jax.random.randint(key, (), 1, self.config.max_skip + 1),
                                choose_skip, operand=None)
        pending = self._empty_segment(state.observation)._replace(action=action,
                                                                  selected_skip=selected)
        return state._replace(rngs=rngs._replace(skip=stream), pending=pending,
                              repeat_remaining=selected, repeat_warmup=warmup)

    def _skip_bootstrap(self, learner, evaluation, observations, rngs, steps):
        target_actions = apply_actor(learner.target_actor, observations,
                                     self.env.action_low, self.env.action_high)
        if self.config.algorithm != "RARe":
            values = apply_critic(learner.target_critic, observations, target_actions)
            return values, rngs
        stream, key = jax.random.split(rngs.target)
        critic = (evaluation.target if evaluation is not None else learner.target_critic)
        values = gaussian_expected_q(key, learner.target_actor, critic, observations,
                                     self.env.action_low, self.env.action_high,
                                     self._target_sigma(steps), self.config.mc_samples)
        return values, rngs._replace(target=stream)

    def _learn(self, state, replay, steps):
        cfg = self.config
        rngs = state.rngs
        replay_stream, sample_key = jax.random.split(rngs.replay)
        rngs = rngs._replace(replay=replay_stream)
        rate = self._learning_rate(state.training_steps)
        if cfg.algorithm == "TAAC":
            batch = sample_taac_sequence(sample_key, replay, cfg.batch_size,
                                         cfg.compare_horizon)
            learner, loss, actor_loss, gate_entropy = taac_train_step(
                state.learner, batch, cfg.gamma, rate, self.env.action_low, self.env.action_high,
                cfg.target_entropy_delta, tau=cfg.tau)
            zero = jnp.float32(0)
            return (learner, state.evaluation, state.skip_learner, rngs,
                    jnp.stack((loss, actor_loss, zero, zero, gate_entropy)))

        batch = sample_continuous_batch(sample_key, replay, cfg.batch_size)
        learner, loss, actor_loss = ddpg_train_step(state.learner, batch, cfg.gamma, rate,
                                                    self.env.action_low, self.env.action_high,
                                                    tau=cfg.tau)
        evaluation, eval_loss = state.evaluation, jnp.float32(0)
        if evaluation is not None:
            replay_stream, eval_sample_key = jax.random.split(rngs.replay)
            target_stream, target_key = jax.random.split(rngs.target)
            eval_batch = sample_continuous_batch(eval_sample_key, replay, cfg.batch_size)
            evaluation, eval_loss = evaluation_critic_train_step(
                evaluation, learner.target_actor, eval_batch, target_key, cfg.gamma, rate,
                self.env.action_low, self.env.action_high, self._eval_sigma(steps), cfg.mc_samples,
                tau=cfg.tau)
            rngs = rngs._replace(replay=replay_stream, target=target_stream)

        def update_skip(keys):
            replay_stream, skip_key = jax.random.split(keys.replay)
            skip_batch = sample_skip_batch(skip_key, state.skip_replay, cfg.batch_size)
            keys = keys._replace(replay=replay_stream)
            bootstrap, keys = self._skip_bootstrap(learner, evaluation,
                                                    skip_batch.next_observations, keys, steps)
            if cfg.algorithm == "UTE":
                target_stream, mask_key = jax.random.split(keys.target)
                keys = keys._replace(target=target_stream)
            else:
                mask_key = None
            skip, skip_loss = continuous_skip_train_step(
                state.skip_learner, skip_batch, bootstrap, rate,
                mask_key=mask_key, ensemble_size=cfg.ensemble_size if cfg.algorithm == "UTE" else 1,
                bootstrap_probability=cfg.bootstrap_probability)
            return skip, keys, skip_loss

        if self._is_skip():
            skip, rngs, skip_loss = jax.lax.cond(
                state.skip_replay.size > 0, update_skip,
                lambda keys: (state.skip_learner, keys, jnp.float32(0)), rngs)
        else:
            skip, skip_loss = state.skip_learner, jnp.float32(0)
        return (learner, evaluation, skip, rngs,
                jnp.stack((loss, actor_loss, skip_loss, eval_loss, jnp.float32(0))))

    def _collect(self, state):
        cfg = self.config
        decision = state.repeat_remaining == 0 if self._is_skip() else jnp.bool_(True)
        if self._is_skip():
            state = jax.lax.cond(decision, self._begin_repeat, lambda current: current, state)
            action, switched, rngs = state.pending.action, jnp.bool_(True), state.rngs
        else:
            action, switched, rngs = self._choose_action(state)
        env_stream, env_key = jax.random.split(rngs.environment)
        next_obs, env_state, reward, terminated, truncated, _ = self.env.step(
            env_key, state.env_state, action)
        rngs = rngs._replace(environment=env_stream)
        state, rngs = self.apply_transition(state, rngs, action, switched, next_obs, reward,
                                            terminated, truncated)
        done = terminated | truncated

        def reset_episode(keys):
            stream, reset_key = jax.random.split(keys.environment)
            obs, reset_state = self.env.reset(reset_key)
            return obs, reset_state, keys._replace(environment=stream)

        observation, env_state, rngs = jax.lax.cond(
            done, reset_episode, lambda keys: (next_obs, env_state, keys), rngs)
        return self._finish_episode(state._replace(env_state=env_state, observation=observation),
                                    rngs, reward, done)

    def apply_transition(self, state, rngs, action, switched, next_obs, reward, terminated, truncated):
        """Store one real transition, learn, and extend the pending repeat."""
        cfg = self.config
        replay = add_continuous_transition(state.replay, state.observation, action,
                                           reward, next_obs, terminated, truncated,
                                           state.previous_action, switched)
        steps = state.training_steps + 1
        warmup = (state.repeat_warmup if self._is_skip() else state.training_steps < cfg.warmup_steps)
        learned = ~warmup & (replay.size >= (cfg.compare_horizon if cfg.algorithm == "TAAC" else 1))
        learner, evaluation, skip, rngs, losses = jax.lax.cond(
            learned, lambda _: self._learn(state._replace(rngs=rngs), replay, steps),
            lambda _: (state.learner, state.evaluation, state.skip_learner, rngs, state.losses),
            operand=None)
        done = terminated | truncated
        if self._is_skip():
            index = state.pending.executed_steps
            pending = state.pending._replace(
                observations=state.pending.observations.at[index + 1].set(next_obs),
                rewards=state.pending.rewards.at[index].set(reward),
                terminated=state.pending.terminated.at[index].set(terminated),
                truncated=state.pending.truncated.at[index].set(truncated),
                executed_steps=index + 1)
            completed = done | (state.repeat_remaining == 1) | (steps == cfg.total_steps)
            skip_replay = jax.lax.cond(
                completed, lambda storage: add_skip_segment(storage, pending, cfg.gamma),
                lambda storage: storage, state.skip_replay)
            repeat_remaining = jnp.where(completed, 0, state.repeat_remaining - 1)
        else:
            pending, skip_replay, repeat_remaining = state.pending, state.skip_replay, jnp.int32(0)
        state = state._replace(
            learner=learner, replay=replay, training_steps=steps,
            previous_action=jnp.where(done, jnp.zeros_like(action), action), has_previous=~done,
            skip_learner=skip, skip_replay=skip_replay, evaluation=evaluation, pending=pending,
            repeat_remaining=repeat_remaining, losses=losses)
        return state, rngs

    def _finish_episode(self, state, rngs, reward, done):
        """Update the UTE lambda bandit; a new lambda is drawn at each episode end."""
        ute_bandit = state.ute_bandit
        if ute_bandit is not None:
            ute_bandit = _record_ute_reward(ute_bandit, reward)

            def on_episode_end(carry):
                bandit, keys = carry
                stream, choice_key = jax.random.split(keys.skip)
                return _next_ute_arm(bandit, choice_key), keys._replace(skip=stream)

            ute_bandit, rngs = jax.lax.cond(
                done, on_episode_end, lambda carry: carry, (ute_bandit, rngs))
        return state._replace(rngs=rngs, ute_bandit=ute_bandit)

    def run(self, state, count):
        """Collect `count` interactions, stopping at the exact total budget."""
        return run_steps(
            lambda current: jax.lax.cond(current.training_steps < self.config.total_steps,
                                         self._collect, lambda unchanged: unchanged, current),
            state, count)
