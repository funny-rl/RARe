import torch
import random
import collections
import numpy as np
import torch.nn as nn
import torch.optim as optim

from torch import Tensor
from jaxtyping import Float

from algos.buffers.skip_buffer import SkipBuffer

class Ensemble_DQN(nn.Module):
    def __init__(
        self, 
        state_dim,
        action_dim,
        n_actions,
        hidden_dim,
        max_skip,
        num_ensemble,
        is_continuous
        
    ):
        super().__init__()
        self.is_continuous = is_continuous
        if self.is_continuous:
            act_dim = action_dim
        else:
            self.n_actions = n_actions
            act_dim = self.n_actions
        
        self.models = nn.ModuleList(
            [
                self._create_model(
                    state_dim,
                    act_dim,
                    hidden_dim,
                    max_skip
                    
                ) for _ in range(num_ensemble)
            ]
        )
        
    def _create_model(
        self,
        state_dim,
        act_dim,
        hidden_dim,
        max_skip
    ):
        encoder = nn.Sequential(
            nn.Linear(state_dim + act_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),

        )
        mixing_layer = nn.Sequential(                
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, max_skip)
        )
        
        return nn.ModuleDict({
            "encoder": encoder,
            "mixing_layer": mixing_layer
        })
    
    def forward(self, state, action):
        outputs = []
        for model in self.models:
            if not self.is_continuous:
                # transform to one-hot
                if action.dim() == 1:
                    action_t: Float[Tensor, "act_dim"] = torch.nn.functional.one_hot(action.long(), num_classes=self.n_actions).float().flatten()
                elif action.dim() == 2:
                    action_t: Float[Tensor, "bs act_dim"] = torch.nn.functional.one_hot(action.long(), num_classes=self.n_actions).float().squeeze(1)
                else:
                    raise ValueError(f"Action dimension {action_t.dim()} is not supported.")
            else:
                action_t = action

            x = torch.cat([state, action_t], dim=-1)
            x = model["encoder"](x)
            x = model["mixing_layer"](x)
            outputs.append(x)
            
        return torch.stack(outputs, dim=0)
    
class UCB:
    """
    Determine the index of the arms in terms of solving a multi-armed bandit problem
    Attributes:
      data           : list that stores the index and average reward of the arms
      num_arms  (int): number of arms used in multi-armed bandit problem
      epsilon (float): probability to select the index of the arms used in multi-armed bandit problem
      beta    (float): weight between frequency and mean reward
      count     (int): if count is less than num_arms, index is count because of trying to pick every arm at least once
    """

    def __init__(self, num_arms, window_size, epsilon, beta):
        """
        num_arms    (int): number of arms used in multi-armed bandit problem
        window_size (int): size of window used in multi-armed bandit problem
        epsilon   (float): probability to select the index of the arms used in multi-armed bandit problem
        beta      (float): weight between frequency and mean reward
        """
        
        self.data = collections.deque(maxlen=window_size)
        self.num_arms = num_arms
        self.epsilon = epsilon
        self.beta = beta
        self.count = 0

    def pull_index(self):
        """
        pull index to determine value of betas and gammas
        Returns:
          index (float): index of arms 
        """
        
        if self.count < self.num_arms:
            index = self.count
            self.count += 1
            
        else:
            if random.random() > self.epsilon:
                N = np.zeros(self.num_arms)
                mu = np.zeros(self.num_arms)
                
                for j, reward in self.data:
                    N[j] += 1
                    mu[j] += reward
                mu = mu / (N + 1e-10)
                index = np.argmax(mu + self.beta * np.sqrt(1 / (N + 1e-6)))
                
            else:
                index = np.random.choice(self.num_arms)
        return index

    def push_data(self, datas):
        """
        push datas to UCB's data list
        Args:
          datas :store index of arms and resulting reward         
        """
        
        self.data += [(j, reward) for j, reward in datas]
        
class UTE:
    def __init__(
        self,
        base_agent,
        name: str,
        max_skip: int,
        skip_buffer_size: int,
        ensemble_size: int,
        use_adaptive_lambda: bool = False,
        uncertainty_factor: float = -1.5,
        use_data_aug: bool = False
    ):
        self.base_agent = base_agent
        self.state_dim: int = base_agent.state_dim
        self.action_dim: int = base_agent.action_dim
        
        self.n_actions: int | None = getattr(self.base_agent, "n_actions", None)
        self.is_continuous: bool = self.n_actions is None   
        
        self.buffer_size: int = skip_buffer_size
        self.batch_size: int = self.base_agent.batch_size    
        self.hidden_dim: int = base_agent.hidden_dim
        self.max_skip: int = max_skip
        self.ensemble_size: int = ensemble_size
        
        self.lr: float = base_agent.lr
        self.initial_lr: float = self.lr
        self.final_lr: float = 0.1 * self.lr
        self.gamma: float = base_agent.gamma
        
        self.use_hard_update: bool = base_agent.use_hard_update
        self.use_adaptive_lambda: bool = use_adaptive_lambda
        
        self.tau: float = base_agent.tau
        self.update_interval: int = base_agent.update_interval
        
        self.e_greedy_type: str = base_agent.e_greedy_type
        self.e_decay: int = base_agent.e_decay
        self.max_epsilon: float = base_agent.max_epsilon
        self.min_epsilon: float = base_agent.min_epsilon
        self.skip_epsilon: float = base_agent.epsilon
        
        self.use_lr_decay: bool = base_agent.use_lr_decay
        self.device: str = base_agent.device
        
        self.skip_actors = Ensemble_DQN(
                self.state_dim,
                self.action_dim,
                self.n_actions,
                self.hidden_dim,
                self.max_skip,
                self.ensemble_size,
                self.is_continuous
        ).to(self.device)
        
        self.loss_func = nn.SmoothL1Loss()
        
        self.skip_optimizer = optim.Adam(
            self.skip_actors.parameters(),
            lr=self.lr
        )
        
        self.skip_replay_buffer = SkipBuffer(
            buffer_size=self.buffer_size,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            gamma=self.gamma,
            is_continuous=self.is_continuous,
            data_augmentation=use_data_aug, 
            device=self.device
        )
        
        if self.use_adaptive_lambda:
            self.lambdas = [-1.5, -1.0, -0.5, -0.2, 0.0, +0.2, +0.5, +1.0, +1.5]
            num_arms = len(self.lambdas)
            self.ucb = UCB(
                num_arms = num_arms, 
                window_size = 500, 
                epsilon = 0.1, 
                beta = 0.5
            )
        else:
            self.uncertainty_factor: float = uncertainty_factor
        
        self.skip_num_parameters = sum(p.numel() for p in self.skip_actors.parameters() if p.requires_grad)
        print(f"[{self.__class__.__name__}] Number of parameters: {self.skip_num_parameters}") 
        
    def adaptive_lambda(self):
        self.j = self.ucb.pull_index()
        self.uncertainty_factor = self.lambdas[self.j]
        self.ucb_datas: list = []  
        
    def select_action(self, state, deterministic=False):
        return self.base_agent.select_action(state, deterministic)

    def select_skip( 
        self, 
        state, 
        action,
        deterministic: bool = False,
    ):
        if (deterministic or torch.rand(1).item() > self.skip_epsilon):
            with torch.no_grad():
                e_skip_values: Float[Tensor, "E skip_dim"] = self.skip_actors(state, action)
                mean_q_values = torch.mean(e_skip_values, dim=0)
                std_q_values = torch.std(e_skip_values, dim=0)
                
                skip_values = mean_q_values + self.uncertainty_factor * std_q_values
                
                skip = skip_values.argmax(dim=-1).item() + 1
            if deterministic:
                return skip, skip_values.cpu().numpy()
        else:
            skip = torch.randint(1, self.max_skip + 1, (1,)).item()
        return skip

    def epsilon_decay(self, train_steps):
        if hasattr(self.base_agent, "epsilon_decay"):
            self.base_agent.epsilon_decay(train_steps)
            self.skip_epsilon = self.base_agent.epsilon
        else: 
            train_steps = torch.tensor(train_steps, dtype=torch.float32)
            if self.e_greedy_type == "linear":
                self.skip_epsilon = self.max_epsilon - (self.max_epsilon - self.min_epsilon) * torch.clamp(train_steps / self.e_decay, 0.0, 1.0).item()
            elif self.e_greedy_type == "exponential":
                self.skip_epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * torch.exp(-1.0 * train_steps / self.e_decay).item()
            else:
                raise NotImplementedError(f"Epsilon greedy type {self.e_greedy_type} is not supported.")

    def lr_decay(self, training_rate):
        self.base_agent.lr_decay(training_rate)
        self.lr = self.base_agent.lr
            
        for param_group in self.skip_optimizer.param_groups:
            param_group['lr'] = self.lr
      
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
            done
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
            skip_states = skip_states,
            action = action,
            skip_rewards = skip_rewards,
            skip_dones = skip_dones,
            next_skip_states = next_skip_states,
            skip = skip,
            buffer = self.skip_replay_buffer,
        )
        
        return 
    
    def update(self, training_steps: int, training_rate: float) -> dict:
        log_dict = self.base_agent.update(training_steps, training_rate)

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
            if self.is_continuous:
                next_actions = self.base_agent.target_actor(next_skip_states)
                next_q_values = self.base_agent.target_critic(next_skip_states, next_actions)
            else:
                next_actions = self.base_agent.actor(next_skip_states).argmax(dim=-1, keepdim=True)
                next_q_values = self.base_agent.target_actor(next_skip_states).gather(-1, next_actions)

            target_skip_values = rewards + not_dones * (self.gamma ** skips) * next_q_values
        
        skip_values = self.skip_actors(skip_states, actions)
        masks = torch.bernoulli(torch.zeros((self.batch_size, self.ensemble_size), device=self.device) + 0.5)
        
        cnt_losses = 0.0
        for k in range(self.ensemble_size):
            num_update = masks[:, k].sum()
            if num_update > 0:
                current_Q = skip_values[k].gather(-1, index=skip_idx)
                loss = self.loss_func(current_Q * masks[:, k], target_skip_values * masks[:, k]) / num_update
                cnt_losses += loss

        skip_losses = cnt_losses / self.ensemble_size
            
        self.skip_optimizer.zero_grad()
        skip_losses.backward()
        self.skip_optimizer.step()
            
        log_dict.update({
            "skip_q_loss": skip_losses.clone().cpu().item(),
            "predicted_skip_q_values": skip_values.clone().cpu().mean().item(),
        })
        
        return log_dict