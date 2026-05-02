import torch
import torch.nn as nn
import torch.optim as optim

from torch import Tensor
from jaxtyping import Float

from algos.buffers.skip_buffer import SkipBuffer


class Skip_MAXQ(nn.Module):
    """
    Single skip-value network.

    Input:
        state  : [B, state_dim]
        action : [B, 1] for discrete actions
                 [B, action_dim] for continuous actions

    Output:
        skip_values : [B, max_skip]

    skip_values[:, j - 1] corresponds to Q_J(s, a, j).
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        n_actions,
        hidden_dim,
        max_skip,
    ):
        super().__init__()

        self.is_continuous = n_actions is None
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
        """
        Returns:
            skip_values: [B, max_skip]
        """

        if not self.is_continuous:
            if action.dim() == 1:
                action_t = torch.nn.functional.one_hot(
                    action.long(),
                    num_classes=self.n_actions,
                ).float().flatten()

            elif action.dim() == 2:
                action_t = torch.nn.functional.one_hot(
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


class TempoRL:
    def __init__(
        self,
        base_agent,
        name: str,
        max_skip: int,
        skip_buffer_size: int,
        use_data_aug: bool,
    ):
        self.base_agent = base_agent
        self.name: str = name

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

        # ------------------------------------------------------------
        # Single skip-value network.
        #
        # The base_agent may internally use MAXMINQ / DDQN-style
        # MAXMINQ, but the skip-value estimator itself does not need
        # an additional ensemble here.
        # ------------------------------------------------------------
        self.skip_actor = Skip_MAXQ(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            n_actions=self.n_actions,
            hidden_dim=self.hidden_dim,
            max_skip=self.max_skip,
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

        self.skip_num_parameters = sum(
            p.numel()
            for p in self.skip_actor.parameters()
            if p.requires_grad
        )

        print(
            f"[{self.__class__.__name__}] "
            f"Number of skip parameters: {self.skip_num_parameters}"
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
        """
        Epsilon-greedy skip selection.

        Greedy:
            j* = argmax_j Q_J(s, a, j)

        Random:
            j ~ Uniform({1, ..., max_skip})
        """
        if deterministic or torch.rand(1).item() > self.skip_epsilon:
            with torch.no_grad():
                skip_values: Float[Tensor, "bs skip_dim"] = self.skip_actor(
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

    def update(
        self,
        training_steps: int,
        training_rate: float,
    ) -> dict:
        log_dict = self.base_agent.update(
            training_steps,
            training_rate,
        )

        (
            skip_states,
            actions,
            skips,
            rewards,
            next_skip_states,
            not_dones,
        ) = self.skip_replay_buffer.sample(self.batch_size)

        # Expected shapes:
        #   skips    : [B, 1] or [B]
        #   skip_idx : [B, 1]
        skip_idx = skips.long() - 1
        if skip_idx.dim() == 1:
            skip_idx = skip_idx.unsqueeze(-1)

        # ------------------------------------------------------------
        # Skip target:
        #
        #   y_J = R_{t:t+j}
        #         + gamma^j * Q_base^-(s_{t+j}, a*)
        #
        # Discrete base agent:
        #   a* = argmax_a Q_base(s_{t+j}, a)
        #   Q_base^-(s_{t+j}, a*) is evaluated by target_actor.
        #
        # If base_agent is Double-style MAXMINQ:
        #   base_agent.actor(next_skip_states) returns min_i Q_i(s, a)
        #   base_agent.target_actor(next_skip_states) returns min_i Q_i^-(s, a)
        #
        # Therefore the discrete target becomes:
        #   a* = argmax_a min_i Q_i(s_{t+j}, a)
        #   y_J = R + gamma^j * min_i Q_i^-(s_{t+j}, a*)
        # ------------------------------------------------------------
        with torch.no_grad():
            if self.is_continuous:
                next_actions = self.base_agent.target_actor(
                    next_skip_states,
                )

                next_q_values = self.base_agent.target_critic(
                    next_skip_states,
                    next_actions,
                )

            else:
                next_actions = self.base_agent.actor(
                    next_skip_states,
                ).argmax(
                    dim=-1,
                    keepdim=True,
                )

                next_q_values = self.base_agent.target_actor(
                    next_skip_states,
                ).gather(
                    dim=-1,
                    index=next_actions,
                )

            target_skip_values = (
                rewards
                + not_dones * (self.gamma ** skips) * next_q_values
            )
            # [B, 1]

        # ------------------------------------------------------------
        # Current skip-value prediction:
        #
        #   Q_J(s, a, j)
        # ------------------------------------------------------------
        skip_values: Float[Tensor, "bs skip_dim"] = self.skip_actor(
            skip_states,
            actions,
        )
        # [B, max_skip]

        pred_skip_values = skip_values.gather(
            dim=-1,
            index=skip_idx,
        )
        # [B, 1]

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