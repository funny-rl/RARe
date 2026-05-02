import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

from algos.buffers.naive_buffer import ReplayBuffer

class Actor(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim,
        max_action
    ):
        super(Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        self.max_action = max_action
        
    def forward(self, x):
        x = self.fc(x)
        return x * self.max_action
    
class Critic(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim
    ):
        super(Critic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = self.fc(x)
        return x



class DDPG:
    def __init__(
        self,
        state_dim,
        action_dim, 
        max_action,
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
        expl_noise,
        device  
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.state_dim: int = state_dim
        self.action_dim: int = action_dim
        self.max_action: int = max_action
        self.buffer_size: int = buffer_size
        self.batch_size: int = batch_size
        self.hidden_dim: int = hidden_dim
        self.update_interval: int = update_interval
        self.e_decay: int = e_decay
        
        self.lr: float = lr
        self.initial_lr: float = lr
        self.final_lr: float = lr * 0.1 
        
        self.tau: float = tau
        self.gamma: float = gamma
        self.init_expl_noise: float = expl_noise
        self.expl_noise: float = expl_noise
        
        self.e_greedy_type: str = e_greedy_type
        self.max_epsilon: float = max_epsilon
        self.min_epsilon: float = min_epsilon
        self.epsilon: float = max_epsilon
        self.device: str = device
        
        self.use_lr_decay: bool = use_lr_decay
        self.use_hard_update: bool = use_hard_update
        self.is_continuous: bool = is_continuous
        
        self.actor = Actor(
            self.state_dim,
            self.action_dim,
            self.hidden_dim,
            self.max_action
        ).to(self.device)
        
        self.critic = Critic(
            self.state_dim,
            self.action_dim,
            self.hidden_dim
        ).to(self.device)
        
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), 
            lr=self.lr
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), 
            lr=self.lr
        )

        self.target_actor = deepcopy(self.actor).to(self.device)
        self.target_critic = deepcopy(self.critic).to(self.device)
        
        self.loss_fn = nn.MSELoss()
        
        self.replay_buffer = ReplayBuffer(
            buffer_size=self.buffer_size,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            device=self.device
        )
            
        self.num_actor_parameters = sum(p.numel() for p in self.actor.parameters() if p.requires_grad)
        self.num_critic_parameters = sum(p.numel() for p in self.critic.parameters() if p.requires_grad)
        
        print(f"[{self.__class__.__name__}] Number of actor parameters: {self.num_actor_parameters}")
        print(f"[{self.__class__.__name__}] Number of critic parameters: {self.num_critic_parameters}")
        
    
    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            orig_action = self.actor(state)
            flat_orig_action = orig_action.flatten()
            if deterministic:
                q_values = self.critic(state, orig_action)
                return flat_orig_action.cpu().numpy(), q_values.cpu().numpy()
            else:
                noised_action = flat_orig_action + torch.normal(0, self.expl_noise, size=flat_orig_action.shape).to(self.device)
                action = torch.clamp(noised_action, -self.max_action, self.max_action)
                return action.cpu().numpy()
    
    def decay_sigma(self, training_rate):
        self.expl_noise = self.init_expl_noise * (1 - training_rate)

    def select_skip(
        self,
        state,
        action,
        deterministic = False
    ) -> int:
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
        cosine = 0.5 * (1 + torch.cos(torch.pi * torch.tensor(training_rate)))
        self.lr = self.final_lr + (self.initial_lr - self.final_lr) * cosine.item()
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = self.lr
        
    def update(self, training_steps: int, training_rate: float) -> dict:
        (
            states, 
            actions, 
            rewards, 
            next_states, 
            not_dones,
        ) = self.replay_buffer.sample(self.batch_size)
        
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_q_values = self.target_critic(next_states, next_actions)
            target_q_values = rewards + self.gamma * not_dones * target_q_values
            
        current_q_values = self.critic(states, actions)
        critic_loss = self.loss_fn(current_q_values, target_q_values)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        if  self.use_hard_update:
            if training_steps % self.update_interval == 0:
                self.target_actor.load_state_dict(self.actor.state_dict())
                self.target_critic.load_state_dict(self.critic.state_dict())
        else:
            for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
        log_dict = {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "td_error": (target_q_values - current_q_values).clone().cpu().mean().item(),
            "predicted_q_values": current_q_values.clone().cpu().mean().item(),
            "exlr_noise": self.expl_noise,  
        }
        return log_dict