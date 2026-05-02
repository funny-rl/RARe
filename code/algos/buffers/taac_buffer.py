import torch
import numpy as np

class TAACBuffer:
    def __init__(
        self, 
        buffer_size, 
        state_dim, 
        action_dim, 
        gamma,
        device
    ):
        self.buffer_size: int = buffer_size
        self.device: str = device
        self.gamma: float = gamma
        
        self.ptr: int = 0
        self.size: int = 0
        if isinstance(state_dim, int):
            state_dim = (state_dim,)
            
        self.state_buffer = torch.zeros((buffer_size, *state_dim))
        self.action_buffer = torch.zeros((buffer_size, action_dim)) 
        self.prev_action_buffer = torch.zeros((buffer_size, action_dim))
        self.reward_buffer = torch.zeros((buffer_size, 1))
        self.next_state_buffer = torch.zeros((buffer_size, *state_dim))
        self.beta_buffer = torch.zeros((buffer_size, 1))
        self.done = torch.zeros((buffer_size, 1))
        
    def add(
        self,
        state,
        action,
        prev_action,
        reward,
        next_state,
        beta,
        done
    ):
        """
        self.state_buffer[self.ptr] = state
        self.action_buffer[self.ptr] = action if torch.is_tensor(action) else torch.tensor(action)
        if prev_action is None:
            self.prev_action_buffer[self.ptr] = torch.zeros_like(self.action_buffer[self.ptr])
        else:
            self.prev_action_buffer[self.ptr] = prev_action if torch.is_tensor(prev_action) else torch.tensor(prev_action)
        """

        self.state_buffer[self.ptr] = torch.as_tensor(state, dtype=torch.float32, device='cpu')
        self.action_buffer[self.ptr] = torch.as_tensor(action, dtype=torch.float32, device='cpu')
        if prev_action is None:
            self.prev_action_buffer[self.ptr] = torch.zeros_like(self.action_buffer[self.ptr])
        else:
            self.prev_action_buffer[self.ptr] = prev_action if torch.is_tensor(prev_action) else torch.tensor(prev_action)
        self.reward_buffer[self.ptr] = torch.as_tensor(reward, dtype=torch.float32, device='cpu')
        
        self.next_state_buffer[self.ptr] = torch.as_tensor(next_state, dtype=torch.float32, device='cpu')
        self.beta_buffer[self.ptr] = torch.as_tensor(beta, dtype=torch.float32, device='cpu')
        self.done[self.ptr] = torch.as_tensor(done, dtype=torch.float32, device='cpu')
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
        
    def sample(self, batch_size, seq_len = 5):
        idxs = torch.randint(0, self.size - seq_len + 1, (batch_size,))

        chunk_idxs = idxs[:, None] + np.arange(seq_len)[None, :]
        chunk_idxs = chunk_idxs % self.buffer_size 


        # 데이터 추출 (Batch, Seq, Dim)
        return (
            self.state_buffer[chunk_idxs].to(self.device),
            self.action_buffer[chunk_idxs].to(self.device),
            self.prev_action_buffer[chunk_idxs].to(self.device),
            self.reward_buffer[chunk_idxs].to(self.device),
            self.next_state_buffer[chunk_idxs].to(self.device),
            self.beta_buffer[chunk_idxs].to(self.device),
            self.done[chunk_idxs].to(self.device), 
        )