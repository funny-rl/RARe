import torch
import numpy as np
import torch.nn as nn

from copy import deepcopy

import torch.nn.functional as F

from algos.buffers.taac_buffer import TAACBuffer

class Actor(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim,
        max_action
    ):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
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
        super().__init__()
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

class TAAC:
    def __init__(
        self,
        base_agent,
        name,
        temperature,
        target_entropy_delta,
        seq_len,
        skip_buffer_size,
    ):
        self.device: str = base_agent.device
        self.base_agent = base_agent
        self.state_dim: int = base_agent.state_dim
        self.action_dim: int = base_agent.action_dim
        self.hidden_dim: int = base_agent.hidden_dim
        self.seq_len: int = seq_len
        self.buffer_size: int = skip_buffer_size
        self.batch_size = base_agent.batch_size
        self.use_hard_update: bool = base_agent.use_hard_update
        self.update_interval: int = base_agent.update_interval
        
        self.tau: float = base_agent.tau
        self.expl_noise: float = base_agent.expl_noise
        self.init_expl_noise: float = base_agent.expl_noise
        self.max_action: float = base_agent.max_action
        self.gamma: float = base_agent.gamma
        self.lr: float = base_agent.lr
        self.initial_lr: float = base_agent.lr
        self.final_lr: float = base_agent.lr * 0.1 
        self.temperature: float = temperature
        self.log_temperature = torch.tensor(np.log(temperature), dtype=torch.float32, device=self.device, requires_grad=True)
        self.target_entropy_delta: float = target_entropy_delta
        self.target_entropy : float = -self.target_entropy_delta * np.log(self.target_entropy_delta) \
                                    - (1 - self.target_entropy_delta)*np.log(1 - self.target_entropy_delta)
                                    
        self.is_continuous: bool = True                     
        
        
        self.actor = Actor(
            self.state_dim,
            self.action_dim,
            self.hidden_dim,
            self.max_action
        ).to(self.device)
        
        self.critic = Critic(
            self.state_dim,
            self.action_dim,
            self.hidden_dim,
        ).to(self.device)
        
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), 
            lr=self.lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), 
            lr=self.lr
        )
        self.temp_optimizer = torch.optim.Adam(
            [self.log_temperature],
            lr=self.lr
        )
        self.target_actor = deepcopy(self.actor).to(self.device)
        self.target_critic = deepcopy(self.critic).to(self.device)
        
        self.loss_fn = nn.SmoothL1Loss()
        self.replay_buffer = TAACBuffer(
            buffer_size=self.buffer_size,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            gamma=self.gamma,
            device=self.device
        )
        
    def decay_sigma(self, training_rate):
        self.expl_noise = self.init_expl_noise * (1 - training_rate)
        
    def lr_decay(self, training_rate):
        cosine = 0.5 * (1 + torch.cos(torch.pi * torch.tensor(training_rate)))
        self.lr = self.final_lr + (self.initial_lr - self.final_lr) * cosine.item()
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = self.lr
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = self.lr

    def select_action(
        self, 
        state,
        prev_action = None,
        deterministic = False
    ) -> torch.Tensor:
        with torch.no_grad():
            none_flag = False

            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            if state_t.ndim == 1:
                state_t = state_t.unsqueeze(0)
            if prev_action is None:
                prev_action = np.zeros((state_t.shape[0], self.action_dim))
                none_flag = True

            prev_action_t = torch.as_tensor(prev_action, dtype=torch.float32, device=self.device)
            if prev_action_t.ndim == 1:
                prev_action_t = prev_action_t.unsqueeze(0)
            actor_input = torch.cat([state_t, prev_action_t], dim=-1)
            new_action = self.actor(actor_input)


            if not deterministic:
                new_action += torch.normal(0, self.expl_noise, size=new_action.shape).to(self.device)
            new_action = torch.clamp(new_action, -self.max_action, self.max_action)

            if none_flag:
                return new_action.flatten().cpu().numpy(), 1
             
            q_stay = self.critic(state_t, prev_action_t)
            q_switch = self.critic(state_t, new_action)
            
            if deterministic:
                beta = (q_stay < q_switch).float().item() # 0: stay, 1: switch
            else:
                q_cat = torch.cat([q_stay, q_switch], dim=-1)
                beta_probs = F.softmax(q_cat/self.temperature, dim=-1)

                dist = torch.distributions.Categorical(beta_probs)
                beta = dist.sample().item()  # 0: stay, 1: switch
            
            action = new_action if beta > 0.5 else prev_action_t
            return action.flatten().cpu().numpy(), beta

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
        prev_action,
        reward,
        next_state,
        beta,
        done,
    ):
        self.replay_buffer.add(
            state,
            action,
            prev_action,
            reward,
            next_state,
            beta,
            done
        )
        
    def update(
        self, 
        training_steps: int, 
        training_rate: float
    ) -> dict:

        (
            states, 
            actions, 
            prev_actions,
            rewards, 
            next_states, 
            beta,
            dones,
        ) = self.replay_buffer.sample(self.batch_size, self.seq_len )

        with torch.no_grad():
            next_prev_actions = actions
            target_actor_input = torch.cat([next_states, next_prev_actions], dim=-1)
            next_actions = self.target_actor(target_actor_input)

            target_Q_stay = self.target_critic(next_states, next_prev_actions)
            target_Q_switch = self.target_critic(next_states, next_actions)

            target_Q_values = torch.max(target_Q_stay, target_Q_switch)
            target_beta = (target_Q_switch > target_Q_stay).float()

            bootstrap_mask = ((beta > 0.5) | (target_beta > 0.5) | (dones > 0.5)).float()

            
            target_Q = torch.zeros_like(rewards)
            next_return = target_Q_values[:, -1] 

            for t in reversed(range(self.seq_len)):
                future_val = bootstrap_mask[:, t] * target_Q_values[:, t]*(1 - dones[:, t]) + \
                              (1 - bootstrap_mask[:, t]) * next_return
                current_val = rewards[:, t] + (self.gamma * future_val)
                
                target_Q[:, t] = current_val
                next_return = current_val    
        
        current_Q = self.critic(states, actions)
        critic_loss = self.loss_fn(current_Q, target_Q.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        with torch.no_grad():
            current_q_stay = self.critic(states, prev_actions)

        new_actions = self.actor(torch.cat([states, prev_actions], dim=-1))
        current_q_switch = self.critic(states, new_actions)

        q_cat = torch.cat([current_q_stay, current_q_switch], dim=-1)
        beta_probs = F.softmax(q_cat/self.temperature, dim=-1) 
        beta_switch = beta_probs[:, :, 1].unsqueeze(-1) 


        actor_loss = -(beta_switch.detach() * current_q_switch).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_for_grad = self.log_temperature.exp() 
        q_cat_detached = q_cat.detach()
        
        beta_probs_grad = F.softmax(q_cat_detached / alpha_for_grad, dim=-1)
        
        current_entropy = -torch.sum(beta_probs_grad * torch.log(beta_probs_grad + 1e-10), dim=-1).mean()
        
        temperature_loss = (self.log_temperature * (current_entropy - self.target_entropy).detach()).mean()

        self.temp_optimizer.zero_grad()
        temperature_loss.backward()
        self.temp_optimizer.step()

        self.temperature = self.log_temperature.exp().item()

        if self.use_hard_update:
            if training_steps % self.update_interval == 0:
                self.target_actor.load_state_dict(self.actor.state_dict())
                self.target_critic.load_state_dict(self.critic.state_dict())
        else:
            for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        log_dict = {
            "critic_loss": critic_loss.item(),
            "td_error": torch.abs(target_Q - current_Q).clone().cpu().mean().item(),
            "actor_loss": actor_loss.item(),
        }
        return log_dict