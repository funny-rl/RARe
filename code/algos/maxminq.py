import torch
import torch.nn as nn
import torch.optim as optim

from copy import deepcopy

from algos.buffers.naive_buffer import ReplayBuffer


class QNet(nn.Module):
    def __init__(
        self,
        state_dim,
        n_actions,
        hidden_dim,
    ):
        super(QNet, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x):
        return self.fc(x)


class QNetEnsemble(nn.Module):
    """
    Wrapper for multiple Q networks.

    reduction="min":
        returns min_i Q_i(s, a)

    reduction="none":
        returns all Q values with shape [N, B, A]

    This wrapper preserves old-style usage:
        self.actor(states)
        self.target_actor(states)

    Therefore RARe can still call:
        self.base_agent.actor(states)
        self.base_agent.target_actor(states)

    By default:
        self.actor(states)        -> min_i Q_i(s, a)
        self.target_actor(states) -> min_i Q_i^-(s, a)
    """

    def __init__(
        self,
        q_nets: nn.ModuleList,
        default_reduction: str = "min",
    ):
        super().__init__()
        self.q_nets = q_nets
        self.default_reduction = default_reduction

    def forward(
        self,
        states: torch.Tensor,
        reduction: str | None = None,
    ) -> torch.Tensor:
        reduction = self.default_reduction if reduction is None else reduction

        q_stack = torch.stack(
            [q_net(states) for q_net in self.q_nets],
            dim=0,
        )
        # q_stack: [N, B, A] or [N, A]

        if reduction == "none":
            return q_stack
        elif reduction == "min":
            return q_stack.min(dim=0).values
        elif reduction == "mean":
            return q_stack.mean(dim=0)
        elif reduction == "max":
            return q_stack.max(dim=0).values
        elif reduction == "first":
            return q_stack[0]
        else:
            raise ValueError(f"Unknown ensemble reduction: {reduction}")


class MAXMINQ:
    def __init__(
        self,
        state_dim,
        action_dim,
        n_actions,
        is_continuous,
        lr,
        gamma,
        buffer_size,
        batch_size,
        hidden_dim,
        e_greedy_type,
        e_decay,
        max_epsilon,
        min_epsilon,
        use_lr_decay,
        use_hard_update,
        update_interval,
        tau,
        n_target,
        device,
    ):
        super().__init__()

        self.state_dim: int = state_dim
        self.action_dim: int = action_dim
        self.n_actions: int = n_actions
        self.buffer_size: int = buffer_size
        self.batch_size: int = batch_size
        self.hidden_dim: int = hidden_dim
        self.update_interval: int = update_interval
        self.e_decay: int = e_decay

        # In this implementation, n_target is the number of Q estimators.
        #
        # n_target == 1:
        #   Double DQN-style target learning.
        #
        # n_target > 1:
        #   Double-style MAXMINQ target learning.
        self.n_target: int = max(1, int(n_target))

        self.lr: float = lr
        self.initial_lr: float = lr
        self.final_lr: float = lr * 0.1

        self.tau: float = tau
        self.gamma: float = gamma
        self.epsilon: float = max_epsilon
        self.max_epsilon: float = max_epsilon
        self.min_epsilon: float = min_epsilon
        self.e_greedy_type: str = e_greedy_type
        self.device: str = device

        self.use_lr_decay: bool = use_lr_decay
        self.use_hard_update: bool = use_hard_update
        self.is_continuous: bool = is_continuous

        if self.is_continuous:
            raise ValueError("MAXMINQ only supports discrete action spaces.")

        # ------------------------------------------------------------
        # Online Q ensemble
        # ------------------------------------------------------------
        self.actors = nn.ModuleList(
            [
                QNet(
                    state_dim=self.state_dim,
                    n_actions=self.n_actions,
                    hidden_dim=self.hidden_dim,
                )
                for _ in range(self.n_target)
            ]
        ).to(self.device)

        # Callable online ensemble.
        #
        # self.actor(states) returns min_i Q_i(states, a) by default.
        self.actor = QNetEnsemble(
            q_nets=self.actors,
            default_reduction="min",
        ).to(self.device)

        # Separate optimizers are cleaner for random single-network updates.
        self.actor_optimizers = [
            optim.Adam(
                actor.parameters(),
                lr=self.lr,
            )
            for actor in self.actors
        ]

        # ------------------------------------------------------------
        # Target Q ensemble
        # ------------------------------------------------------------
        self.target_actors = nn.ModuleList(
            [deepcopy(actor) for actor in self.actors]
        ).to(self.device)

        # Callable target ensemble.
        #
        # self.target_actor(states) returns min_i Q_i^-(states, a)
        # by default.
        self.target_actor = QNetEnsemble(
            q_nets=self.target_actors,
            default_reduction="min",
        ).to(self.device)

        self.loss_func = nn.SmoothL1Loss()

        self.replay_buffer = ReplayBuffer(
            buffer_size=self.buffer_size,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            device=self.device,
        )

        self.num_parameters = sum(
            p.numel() for p in self.actors.parameters() if p.requires_grad
        )

        print(
            f"[{self.__class__.__name__}] Number of Q networks: {self.n_target}"
        )
        print(
            f"[{self.__class__.__name__}] Number of parameters: {self.num_parameters}"
        )

    def select_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ):
        """
        Epsilon-greedy action selection.

        n_target == 1:
            action = argmax_a Q_1(s, a)

        n_target > 1:
            action = argmax_a min_i Q_i(s, a)

        This function always uses the online ensemble self.actor.
        """

        if deterministic or torch.rand(1).item() > self.epsilon:
            with torch.no_grad():
                q_values = self.actor(
                    state,
                    reduction="min",
                )
                action = q_values.argmax(dim=-1)

            if deterministic:
                return action.cpu().numpy(), q_values.cpu().numpy()

        else:
            action = torch.randint(
                0, 
                self.n_actions, 
                (1,)
            )

        return action.cpu().numpy()

    def select_skip(
        self,
        state,
        action,
        deterministic=False,
    ) -> tuple[int, None] | int:
        if deterministic:
            return 1, None
        else:
            return 1

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):
        self.replay_buffer.add(
            state,
            action,
            reward,
            next_state,
            done,
        )

    def lr_decay(self, training_rate):
        cosine = 0.5 * (
            1 + torch.cos(torch.pi * torch.tensor(training_rate))
        )
        self.lr = self.final_lr + (
            self.initial_lr - self.final_lr
        ) * cosine.item()

        for optimizer in self.actor_optimizers:
            for param_group in optimizer.param_groups:
                param_group["lr"] = self.lr

    def epsilon_decay(self, train_steps):
        train_steps = torch.tensor(train_steps, dtype=torch.float32)

        if self.e_greedy_type == "linear":
            self.epsilon = self.max_epsilon - (
                self.max_epsilon - self.min_epsilon
            ) * torch.clamp(
                train_steps / self.e_decay,
                0.0,
                1.0,
            ).item()

        elif self.e_greedy_type == "exponential":
            self.epsilon = self.min_epsilon + (
                self.max_epsilon - self.min_epsilon
            ) * torch.exp(
                -1.0 * train_steps / self.e_decay
            ).item()

        else:
            raise NotImplementedError(
                f"Epsilon greedy type {self.e_greedy_type} is not supported."
            )

    def _online_q_stack(
        self,
        states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns:
            q_stack: [N, B, A]
        """
        return self.actor(
            states,
            reduction="none",
        )

    def _target_q_stack(
        self,
        states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns:
            q_stack: [N, B, A]
        """
        return self.target_actor(
            states,
            reduction="none",
        )

    def _sync_target_networks(self):
        """
        Hard or soft update for every target network.

        Q_i -> Q_i^-
        """
        if self.use_hard_update:
            for target_actor, actor in zip(
                self.target_actors,
                self.actors,
            ):
                target_actor.load_state_dict(actor.state_dict())

        else:
            for target_actor, actor in zip(
                self.target_actors,
                self.actors,
            ):
                for target_param, param in zip(
                    target_actor.parameters(),
                    actor.parameters(),
                ):
                    target_param.data.copy_(
                        self.tau * param.data
                        + (1.0 - self.tau) * target_param.data
                    )

    @torch.no_grad()
    def compute_maxmin_target_value(
        self,
        next_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Double-style MAXMINQ target:

            a* = argmax_a min_i Q_i(s', a)
            y  = r + gamma * min_i Q_i^-(s', a*)

        Difference from vanilla MAXMINQ:

            Vanilla MAXMINQ-style target:
                y = r + gamma * max_a min_i Q_i^-(s', a)

            Double-style MAXMINQ target:
                a* = argmax_a min_i Q_i(s', a)
                y  = r + gamma * min_i Q_i^-(s', a*)

        In other words:
            - self.actor selects the next action.
            - self.target_actor evaluates the selected action.

        Returns:
            next_q_values: [B, 1]
            next_actions:  [B, 1]
            target_min_q_values_all_actions: [B, A]
        """

        # ------------------------------------------------------------
        # 1) Action selection by online MAXMIN ensemble
        #
        #   a* = argmax_a min_i Q_i(s', a)
        # ------------------------------------------------------------
        online_min_q_values_all_actions = self.actor(
            next_states,
            reduction="min",
        )
        # [B, A]

        next_actions = online_min_q_values_all_actions.argmax(
            dim=-1,
            keepdim=True,
        )
        # [B, 1]

        # ------------------------------------------------------------
        # 2) Action evaluation by target MAXMIN ensemble
        #
        #   min_i Q_i^-(s', a*)
        # ------------------------------------------------------------
        target_min_q_values_all_actions = self.target_actor(
            next_states,
            reduction="min",
        )
        # [B, A]

        next_q_values = target_min_q_values_all_actions.gather(
            dim=-1,
            index=next_actions,
        )
        # [B, 1]

        return next_q_values, next_actions, target_min_q_values_all_actions

    def update(
        self,
        training_steps: int,
        training_rate: float,
    ) -> dict:
        (
            states,
            actions,
            rewards,
            next_states,
            not_dones,
        ) = self.replay_buffer.sample(self.batch_size)

        actions = actions.long()
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)

        with torch.no_grad():
            next_q_values, next_actions, target_min_q_all_actions = (
                self.compute_maxmin_target_value(next_states)
            )
            target_q_values = rewards + not_dones * self.gamma * next_q_values

        # ------------------------------------------------------------
        # MAXMINQ update:
        #
        # Randomly update one Q-network per gradient step.
        #
        # This is important. If every Q_i is updated with the same
        # batch and same target at every step, the ensemble collapses
        # into nearly identical estimators.
        # ------------------------------------------------------------
        update_idx = torch.randint(
            low=0,
            high=self.n_target,
            size=(1,),
            device=self.device,
        ).item()

        pred_q_values = self.actors[update_idx](states).gather(
            dim=-1,
            index=actions,
        )
        # [B, 1]

        loss = self.loss_func(
            pred_q_values,
            target_q_values,
        )

        optimizer = self.actor_optimizers[update_idx]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ------------------------------------------------------------
        # Target network update
        # ------------------------------------------------------------
        if self.use_hard_update:
            if training_steps % self.update_interval == 0:
                self._sync_target_networks()
        else:
            self._sync_target_networks()

        with torch.no_grad():
            pred_q_stack = self._online_q_stack(states)
            # [N, B, A]

            expanded_actions = actions.unsqueeze(0).expand(
                self.n_target,
                -1,
                -1,
            )
            # [N, B, 1]

            pred_q_all = pred_q_stack.gather(
                dim=-1,
                index=expanded_actions,
            )
            # [N, B, 1]

            online_q_min = pred_q_all.min(dim=0).values
            online_q_mean = pred_q_all.mean(dim=0)

            online_q_std = (
                pred_q_all.std(dim=0, unbiased=False)
                if self.n_target > 1
                else torch.zeros_like(online_q_min)
            )

            policy_q_values = self.actor(
                states,
                reduction="min",
            ).gather(
                dim=-1,
                index=actions,
            )

            td_error = target_q_values - pred_q_values

        log_dict: dict = {
            "q_loss": loss.detach().cpu().item(),

            # Updated network diagnostics.
            "td_error": td_error.detach().cpu().mean().item(),
            "predicted_q_values": pred_q_values.detach().cpu().mean().item(),
            "updated_q_index": update_idx,

            # Online ensemble diagnostics.
            "online_q_min": online_q_min.detach().cpu().mean().item(),
            "online_q_mean": online_q_mean.detach().cpu().mean().item(),
            "online_q_std": online_q_std.detach().cpu().mean().item(),
            "policy_q_values": policy_q_values.detach().cpu().mean().item(),

            # Actual bootstrap value used in the Double-style MAXMIN target.
            "double_maxmin_next_q_values": next_q_values.detach().cpu().mean().item(),

            # Mean over all target min-Q values before selecting next_actions.
            "target_min_q_all_actions": target_min_q_all_actions.detach().cpu().mean().item(),

            "n_target": self.n_target,
        }

        return log_dict