import torch
import torch.nn as nn
import torch.optim as optim

from copy import deepcopy

from torch import Tensor
from jaxtyping import Float

from algos.buffers.skip_buffer import SkipBuffer

from .maxminq import QNet as Expected_SARSA
from .ddpg import Critic as Expected_Critic


"""
Typing notations:
- SD: state dimension
- B: batch size
- K: expected-network ensemble size
- S: sample dimension
- A: action dimension
- R: skip / repetition dimension
"""


class Skip_ExpectedQ(nn.Module):

    def __init__(
        self,
        state_dim,
        action_dim,
        n_actions,
        hidden_dim,
        max_skip,
        is_continuous,
    ):
        super().__init__()

        self.is_continuous = is_continuous
        self.n_actions = n_actions

        act_dim = action_dim if self.is_continuous else self.n_actions

        self.fn = nn.Sequential(
            nn.Linear(state_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_skip),
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:

        if not self.is_continuous:
            if action.dim() == 1:
                action_t: Float[Tensor, "B A"] = torch.nn.functional.one_hot(
                    action.long(),
                    num_classes=self.n_actions,
                ).float().flatten()

            elif action.dim() == 2:
                action_t: Float[Tensor, "B A"] = torch.nn.functional.one_hot(
                    action.long().squeeze(-1),
                    num_classes=self.n_actions,
                ).float().squeeze(1)

            else:
                raise ValueError(
                    f"Action dimension {action.dim()} is not supported."
                )

        else:
            action_t = action

        x = torch.cat([state, action_t], dim=-1)
        skip_values = self.fn(x)

        return skip_values


class ExpectedSARSAEnsemble(nn.Module):

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dim: int,
        ensemble_size: int,
    ):
        super().__init__()

        self.ensemble_size = max(1, int(ensemble_size))

        self.models = nn.ModuleList(
            [
                Expected_SARSA(
                    state_dim=state_dim,
                    n_actions=n_actions,
                    hidden_dim=hidden_dim,
                )
                for _ in range(self.ensemble_size)
            ]
        )

    def forward(
        self,
        states: torch.Tensor,
        reduction: str = "min",
    ) -> torch.Tensor:
        q_stack = torch.stack(
            [model(states) for model in self.models],
            dim=0,
        )
        # [K, B, A]

        if reduction == "none":
            return q_stack

        if reduction == "mean":
            return q_stack.mean(dim=0)

        if reduction == "min":
            return q_stack.min(dim=0).values

        raise ValueError(f"Unknown reduction: {reduction}")


class ExpectedCriticEnsemble(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        ensemble_size: int,
    ):
        super().__init__()

        self.ensemble_size = max(1, int(ensemble_size))

        self.models = nn.ModuleList(
            [
                Expected_Critic(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    hidden_dim=hidden_dim,
                )
                for _ in range(self.ensemble_size)
            ]
        )

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        reduction: str = "min",
    ) -> torch.Tensor:
        q_stack = torch.stack(
            [model(states, actions) for model in self.models],
            dim=0,
        )
        # [K, B, 1]

        if reduction == "none":
            return q_stack

        if reduction == "mean":
            return q_stack.mean(dim=0)

        if reduction == "min":
            return q_stack.min(dim=0).values

        raise ValueError(f"Unknown reduction: {reduction}")


class RARe:
    def __init__(
        self,
        base_agent,
        name: str,
        max_skip: int,
        skip_buffer_size: int,
        n_sample: int,
        use_data_aug: bool,
        max_alpha: float,
        min_alpha: float,
        cutoff: float,
        use_es_target: bool,
        expected_ensemble_size: int = 1,
        expected_ensemble_reduction: str = "min",
    ):
        self.base_agent = base_agent
        self.name: str = name

        self.n_sample: int = n_sample
        self.state_dim: int = base_agent.state_dim
        self.action_dim: int = base_agent.action_dim

        self.n_actions: int | None = getattr(self.base_agent, "n_actions", None)
        self.is_continuous: bool = self.n_actions is None

        self.buffer_size: int = skip_buffer_size
        self.batch_size: int = self.base_agent.batch_size
        self.hidden_dim: int = base_agent.hidden_dim
        self.max_skip: int = max_skip

        self.lr: float = base_agent.lr
        self.initial_lr: float = self.lr
        self.final_lr: float = 0.1 * self.lr
        self.gamma: float = base_agent.gamma

        self.max_alpha: float = max_alpha
        self.min_alpha: float = min_alpha
        self.cutoff: float = cutoff
        self.use_es_target: bool = use_es_target

        self.expected_ensemble_size: int = max(
            1,
            int(expected_ensemble_size),
        )
        self.expected_ensemble_reduction: str = expected_ensemble_reduction

        if self.expected_ensemble_reduction not in ["mean", "min"]:
            raise ValueError(
                "expected_ensemble_reduction must be 'mean' or 'min', "
                f"but got {self.expected_ensemble_reduction}"
            )

        self.use_hard_update: bool = base_agent.use_hard_update
        self.tau: float = base_agent.tau
        self.update_interval: int = base_agent.update_interval

        self.e_greedy_type: str = base_agent.e_greedy_type
        self.e_decay: int = base_agent.e_decay
        self.max_epsilon: float = base_agent.max_epsilon
        self.min_epsilon: float = base_agent.min_epsilon
        self.skip_epsilon: float = base_agent.epsilon

        self.use_lr_decay: bool = base_agent.use_lr_decay
        self.device: str = base_agent.device

        if self.is_continuous:
            self.max_action = self.base_agent.max_action
            self.max_sigma_eps = self.max_alpha
            self.min_sigma_eps = self.min_alpha
            self.sigma_eps = self.max_sigma_eps

        self.skip_actor = Skip_ExpectedQ(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            n_actions=self.n_actions,
            hidden_dim=self.hidden_dim,
            max_skip=self.max_skip,
            is_continuous=self.is_continuous,
        ).to(self.device)

        self.loss_func = nn.SmoothL1Loss()

        self.skip_optimizer = optim.Adam(
            self.skip_actor.parameters(),
            lr=self.lr,
        )

        self.skip_replay_buffer = SkipBuffer(
            buffer_size=self.buffer_size,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            gamma=self.gamma,
            is_continuous=self.is_continuous,
            data_augmentation=use_data_aug,
            device=self.device,
        )

        if self.use_es_target:
            if self.is_continuous:
                self.expected_critic = ExpectedCriticEnsemble(
                    state_dim=self.state_dim,
                    action_dim=self.action_dim,
                    hidden_dim=self.hidden_dim,
                    ensemble_size=self.expected_ensemble_size,
                ).to(self.device)

                self.critic_optimizer = optim.Adam(
                    self.expected_critic.parameters(),
                    lr=self.lr,
                )

                self.target_expected_critic = deepcopy(self.expected_critic)

            else:
                self.max_sarsa_eps = self.max_alpha
                self.min_sarsa_eps = self.min_alpha

                self.expected_sarsa = ExpectedSARSAEnsemble(
                    state_dim=self.state_dim,
                    n_actions=self.n_actions,
                    hidden_dim=self.hidden_dim,
                    ensemble_size=self.expected_ensemble_size,
                ).to(self.device)

                self.sarsa_optimizer = optim.Adam(
                    self.expected_sarsa.parameters(),
                    lr=self.lr,
                )

                self.target_expected_sarsa = deepcopy(self.expected_sarsa)

        self.skip_num_parameters = sum(
            p.numel()
            for p in self.skip_actor.parameters()
            if p.requires_grad
        )

        if self.use_es_target:
            if self.is_continuous:
                self.skip_num_parameters += sum(
                    p.numel()
                    for p in self.expected_critic.parameters()
                    if p.requires_grad
                )
            else:
                self.skip_num_parameters += sum(
                    p.numel()
                    for p in self.expected_sarsa.parameters()
                    if p.requires_grad
                )

        print(
            f"[{self.__class__.__name__}] "
            f"Number of parameters: {self.skip_num_parameters}"
        )

        if self.use_es_target:
            print(
                f"[{self.__class__.__name__}] "
                f"Expected-Q ensemble size: {self.expected_ensemble_size}"
            )
            print(
                f"[{self.__class__.__name__}] "
                f"Expected-Q ensemble reduction: "
                f"{self.expected_ensemble_reduction}"
            )

    def select_action(
        self,
        state,
        deterministic: bool = False,
    ):
        return self.base_agent.select_action(state, deterministic)

    def select_skip(
        self,
        state,
        action,
        deterministic: bool = False,
    ):

        if deterministic or torch.rand(1).item() > self.skip_epsilon:
            with torch.no_grad():
                skip_values: Float[Tensor, "B R"] = self.skip_actor(
                    state,
                    action,
                )

                skip = skip_values.argmax(dim=-1).item() + 1

            if deterministic:
                return skip, skip_values.cpu().numpy()

        else:
            skip = torch.randint(
                low=1,
                high=self.max_skip + 1,
                size=(1,),
            ).item()

        return skip

    def epsilon_decay(
        self,
        train_steps,
    ):
        if hasattr(self.base_agent, "epsilon_decay"):
            self.base_agent.epsilon_decay(train_steps)
            self.skip_epsilon = self.base_agent.epsilon

        else:
            train_steps = torch.tensor(
                train_steps,
                dtype=torch.float32,
            )

            if self.e_greedy_type == "linear":
                self.skip_epsilon = self.max_epsilon - (
                    self.max_epsilon - self.min_epsilon
                ) * torch.clamp(
                    train_steps / self.e_decay,
                    0.0,
                    1.0,
                ).item()

            elif self.e_greedy_type == "exponential":
                self.skip_epsilon = self.min_epsilon + (
                    self.max_epsilon - self.min_epsilon
                ) * torch.exp(
                    -1.0 * train_steps / self.e_decay
                ).item()

            else:
                raise NotImplementedError(
                    f"Epsilon greedy type {self.e_greedy_type} "
                    f"is not supported."
                )

    def lr_decay(
        self,
        training_rate,
    ):
        self.base_agent.lr_decay(training_rate)
        self.lr = self.base_agent.lr

        for param_group in self.skip_optimizer.param_groups:
            param_group["lr"] = self.lr

        if self.use_es_target:
            if self.is_continuous:
                for param_group in self.critic_optimizer.param_groups:
                    param_group["lr"] = self.lr
            else:
                for param_group in self.sarsa_optimizer.param_groups:
                    param_group["lr"] = self.lr

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):
        self.base_agent.add(
            state,
            action,
            reward,
            next_state,
            done,
        )

    def add_skip(
        self,
        skip_states,
        action,
        skip_rewards,
        skip_dones,
        next_skip_states,
        skip,
    ):
        self.skip_replay_buffer = self.skip_replay_buffer.transform(
            skip_states=skip_states,
            action=action,
            skip_rewards=skip_rewards,
            skip_dones=skip_dones,
            next_skip_states=next_skip_states,
            skip=skip,
            buffer=self.skip_replay_buffer,
        )

    def alpha_update(
        self,
        training_rate: float,
    ):

        self.alpha = self.max_alpha - training_rate * (
            self.max_alpha - self.min_alpha
        )

        self.alpha = min(
            max(self.alpha, self.min_alpha),
            self.max_alpha,
        )

    def sarsa_epsilon_update(
        self,
        training_rate: float,
    ):
        
        progress = min(
            max(training_rate / self.cutoff, 0.0),
            1.0,
        )

        self.sarsa_eps = self.max_sarsa_eps - progress * (
            self.max_sarsa_eps - self.min_sarsa_eps
        )

        self.sarsa_eps = min(
            max(self.sarsa_eps, self.min_sarsa_eps),
            self.max_sarsa_eps,
        )

    @torch.no_grad()
    def compute_discrete_expected_value(
        self,
        next_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        online_next_q_values: Float[Tensor, "B A"] = (
            self.expected_sarsa(
                next_states,
                reduction=self.expected_ensemble_reduction,
            )
        )

        next_actions: Float[Tensor, "B 1"] = (
            online_next_q_values.argmax(
                dim=-1,
                keepdim=True,
            )
        )

        target_next_q_values: Float[Tensor, "B A"] = (
            self.target_expected_sarsa(
                next_states,
                reduction=self.expected_ensemble_reduction,
            )
        )

        max_q_values: Float[Tensor, "B 1"] = (
            target_next_q_values.gather(
                dim=-1,
                index=next_actions,
            )
        )

        # This is the action-wise mean in Expected-Q.
        # Do not replace this with ensemble min.
        am_q_values: Float[Tensor, "B 1"] = (
            target_next_q_values.mean(
                dim=-1,
                keepdim=True,
            )
        )

        expected_values: Float[Tensor, "B 1"] = (
            (1.0 - self.sarsa_eps) * max_q_values
            + self.sarsa_eps * am_q_values
        )

        return expected_values, next_actions, target_next_q_values

    def update(
        self,
        training_steps: int,
        training_rate: float,
    ) -> dict:
        log_dict = self.base_agent.update(
            training_steps,
            training_rate,
        )

        if not self.use_es_target:
            if not self.is_continuous:
                self.alpha_update(training_rate)
            else:
                self.sigma_update(training_rate)

        else:
            if self.is_continuous:
                self.train_expected_target_c(
                    training_steps=training_steps,
                    training_rate=training_rate,
                    log_dict=log_dict,
                )
            else:
                self.train_expected_target_d(
                    training_steps=training_steps,
                    training_rate=training_rate,
                    log_dict=log_dict,
                )

        (
            skip_states,
            actions,
            skips,
            rewards,
            next_skip_states,
            not_dones,
        ) = self.skip_replay_buffer.sample(self.batch_size)

        skip_idx = skips.long() - 1

        with torch.no_grad():
            if self.use_es_target:
                if self.is_continuous:
                    expected_values = self.compute_continuous_expected_value(
                        next_states=next_skip_states,
                        critic=self.target_expected_critic,
                        sigma=self.sigma_eps,
                    )

                else:
                    expected_values, _, _ = self.compute_discrete_expected_value(
                        next_skip_states,
                    )

                    self.alpha = self.sarsa_eps

            else:
                if self.is_continuous:
                    next_actions = self.base_agent.target_actor(
                        next_skip_states,
                    )

                    noises = torch.randn(
                        self.n_sample,
                        next_actions.shape[0],
                        self.action_dim,
                        device=self.device,
                        dtype=next_actions.dtype,
                    ) * self.sigma_eps

                    sampled_next_actions = (
                        next_actions.unsqueeze(0)
                        + noises
                    ).clamp(
                        -self.max_action,
                        self.max_action,
                    )

                    S, B, A = sampled_next_actions.shape

                    aug_next_states = next_skip_states.unsqueeze(0).expand(
                        S,
                        -1,
                        -1,
                    ).reshape(S * B, -1)

                    flat_sampled_next_actions = sampled_next_actions.reshape(
                        S * B,
                        A,
                    )

                    next_q_values = self.base_agent.target_critic(
                        aug_next_states,
                        flat_sampled_next_actions,
                    )

                    expected_values = next_q_values.view(
                        S,
                        B,
                        -1,
                    ).mean(dim=0)

                else:
                    next_actions = self.base_agent.actor(
                        next_skip_states,
                    ).argmax(
                    dim=-1,
                    keepdim=True,
                    )
                    next_q_values = self.base_agent.target_actor(
                        next_skip_states,
                    )
                    
                    max_q_values: Float[Tensor, "B 1"] = next_q_values.gather(
                        dim=-1,
                        index=next_actions,
                    )
                    
                    am_q_values: Float[Tensor, "B 1"] = next_q_values.mean(
                        dim=-1,
                        keepdim=True,
                    )
                    
                    expected_values  = (1 - self.alpha) * max_q_values + self.alpha * am_q_values
                    
                    

            target_skip_values: Float[Tensor, "B 1"] = (
                rewards
                + not_dones * (self.gamma ** skips) * expected_values
            )

        skip_values: Float[Tensor, "B R"] = self.skip_actor(
            skip_states,
            actions,
        )

        pred_skip_values: Float[Tensor, "B 1"] = skip_values.gather(
            dim=-1,
            index=skip_idx,
        )

        skip_loss = self.loss_func(
            pred_skip_values,
            target_skip_values,
        )

        self.skip_optimizer.zero_grad()
        skip_loss.backward()
        self.skip_optimizer.step()

        log_dict.update(
            {
                "skip_q_loss": skip_loss.detach().cpu().item(),
                "skip_td_error": (
                    target_skip_values - pred_skip_values
                ).detach().cpu().mean().item(),
                "predicted_skip_q_values": pred_skip_values.detach().cpu().mean().item(),
                "target_skip_q_values": target_skip_values.detach().cpu().mean().item(),
            }
        )

        return log_dict

    def train_expected_target_d(
        self,
        training_steps: int,
        training_rate: float,
        log_dict: dict,
    ):

        (
            states,
            actions,
            rewards,
            next_states,
            not_dones,
        ) = self.base_agent.replay_buffer.sample(self.batch_size)

        actions = actions.long()
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)

        self.sarsa_epsilon_update(training_rate)

        with torch.no_grad():
            expectation, _, _ = self.compute_discrete_expected_value(
                next_states,
            )

            target_q_values: Float[Tensor, "B 1"] = (
                rewards
                + not_dones * self.gamma * expectation
            )

        pred_q_values_all: Float[Tensor, "K B A"] = (
            self.expected_sarsa(
                states,
                reduction="none",
            )
        )

        expanded_actions = actions.unsqueeze(0).expand(
            self.expected_ensemble_size,
            -1,
            -1,
        )

        pred_q_selected_all: Float[Tensor, "K B 1"] = (
            pred_q_values_all.gather(
                dim=-1,
                index=expanded_actions,
            )
        )

        target_q_values_all: Float[Tensor, "K B 1"] = (
            target_q_values.unsqueeze(0).expand_as(
                pred_q_selected_all,
            )
        )

        sarsa_loss = self.loss_func(
            pred_q_selected_all,
            target_q_values_all,
        )

        self.sarsa_optimizer.zero_grad()
        sarsa_loss.backward()
        self.sarsa_optimizer.step()

        if self.use_hard_update:
            if training_steps % self.update_interval == 0:
                self.target_expected_sarsa.load_state_dict(
                    self.expected_sarsa.state_dict()
                )
        else:
            for target_param, param in zip(
                self.target_expected_sarsa.parameters(),
                self.expected_sarsa.parameters(),
            ):
                target_param.data.copy_(
                    self.tau * param.data
                    + (1.0 - self.tau) * target_param.data
                )

        with torch.no_grad():
            pred_q_values_mean = pred_q_selected_all.mean(dim=0)
            pred_q_values_min = pred_q_selected_all.min(dim=0).values

            if self.expected_ensemble_reduction == "mean":
                pred_q_values_reduced = pred_q_values_mean
            else:
                pred_q_values_reduced = pred_q_values_min

        log_dict.update(
            {
                "skip_alpha": self.sarsa_eps,
                "sarsa_loss": sarsa_loss.detach().cpu().item(),
                "sarsa_td_error": (
                    target_q_values - pred_q_values_reduced
                ).detach().cpu().mean().item(),
                "predicted_sarsa_q_values": pred_q_values_reduced.detach().cpu().mean().item(),
                "target_sarsa_q_values": target_q_values.detach().cpu().mean().item(),
                "expected_ensemble_size": self.expected_ensemble_size,
                "predicted_sarsa_q_values_mean": pred_q_values_mean.detach().cpu().mean().item(),
                "predicted_sarsa_q_values_min": pred_q_values_min.detach().cpu().mean().item(),
            }
        )

        return log_dict

    def compute_continuous_expected_value(
        self,
        next_states,
        critic,
        sigma: float,
    ):
        with torch.no_grad():
            next_actions = self.base_agent.target_actor(next_states)

            noises = torch.randn(
                self.n_sample,
                next_actions.shape[0],
                self.action_dim,
                device=self.device,
                dtype=next_actions.dtype,
            ) * sigma

            sampled_actions = (
                next_actions.unsqueeze(0) + noises
            ).clamp(
                -self.max_action,
                self.max_action,
            )

            S, B, A = sampled_actions.shape

            aug_next_states = next_states.unsqueeze(0).expand(
                S,
                -1,
                -1,
            ).reshape(S * B, -1)

            flat_sampled_actions = sampled_actions.reshape(S * B, A)

            q_values = critic(
                aug_next_states,
                flat_sampled_actions,
                reduction=self.expected_ensemble_reduction,
            )

            expected_values = q_values.view(
                S,
                B,
                -1,
            ).mean(dim=0)

        return expected_values

    def sigma_update(
        self,
        training_rate: float,
    ):
        progress = training_rate / self.cutoff

        self.sigma_eps = self.max_sigma_eps - progress * (
            self.max_sigma_eps - self.min_sigma_eps
        )

        self.sigma_eps = min(
            max(self.sigma_eps, self.min_sigma_eps),
            self.max_sigma_eps,
        )

    def train_expected_target_c(
        self,
        training_steps: int,
        training_rate: float,
        log_dict: dict,
    ):

        self.sigma_update(training_rate)

        (
            states,
            actions,
            rewards,
            next_states,
            not_dones,
        ) = self.base_agent.replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            expected_values = self.compute_continuous_expected_value(
                next_states=next_states,
                critic=self.target_expected_critic,
                sigma=self.sigma_eps,
            )

            target_q_values: Float[Tensor, "B 1"] = (
                rewards
                + not_dones * self.gamma * expected_values
            )

        pred_q_values_all: Float[Tensor, "K B 1"] = (
            self.expected_critic(
                states,
                actions,
                reduction="none",
            )
        )

        target_q_values_all: Float[Tensor, "K B 1"] = (
            target_q_values.unsqueeze(0).expand_as(
                pred_q_values_all,
            )
        )

        critic_loss = self.loss_func(
            pred_q_values_all,
            target_q_values_all,
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.use_hard_update:
            if training_steps % self.update_interval == 0:
                self.target_expected_critic.load_state_dict(
                    self.expected_critic.state_dict()
                )
        else:
            for target_param, param in zip(
                self.target_expected_critic.parameters(),
                self.expected_critic.parameters(),
            ):
                target_param.data.copy_(
                    self.tau * param.data
                    + (1.0 - self.tau) * target_param.data
                )

        with torch.no_grad():
            pred_q_values_mean = pred_q_values_all.mean(dim=0)
            pred_q_values_min = pred_q_values_all.min(dim=0).values

            if self.expected_ensemble_reduction == "mean":
                pred_q_values_reduced = pred_q_values_mean
            else:
                pred_q_values_reduced = pred_q_values_min

        log_dict.update(
            {
                "skip_sigma": self.sigma_eps,
                "expected_critic_loss": critic_loss.detach().cpu().item(),
                "expected_critic_td_error": (
                    target_q_values - pred_q_values_reduced
                ).detach().cpu().mean().item(),
                "predicted_expected_q_values": pred_q_values_reduced.detach().cpu().mean().item(),
                "expected_target_q_values": target_q_values.detach().cpu().mean().item(),
                "expected_ensemble_size": self.expected_ensemble_size,
                "predicted_expected_q_values_mean": pred_q_values_mean.detach().cpu().mean().item(),
                "predicted_expected_q_values_min": pred_q_values_min.detach().cpu().mean().item(),
            }
        )

        return log_dict