import torch

class ReplayBuffer():
    def __init__(
        self, 
        buffer_size, 
        state_dim, 
        action_dim, 
        device
    ):
        self.buffer_size: int = buffer_size
        self.device: str = device
        
        self.ptr: int = 0
        self.size: int = 0
        if isinstance(state_dim, int):
            state_dim = (state_dim,)
            
        self.state_buffer = torch.zeros((buffer_size, *state_dim))
        self.action_buffer = torch.zeros((buffer_size, action_dim))        
        self.reward_buffer = torch.zeros((buffer_size, 1))
        self.next_state_buffer = torch.zeros((buffer_size, *state_dim))
        self.not_done = torch.zeros((buffer_size, 1))
        
    def add(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):
        self.state_buffer[self.ptr] = state
        self.action_buffer[self.ptr] = action if torch.is_tensor(action) else torch.tensor(action)
        self.reward_buffer[self.ptr] = reward
        self.next_state_buffer[self.ptr] = next_state
        self.not_done[self.ptr] = 1. - done
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
        
    def sample(self, batch_size):
        idxs = torch.randint(0, self.size, (batch_size,))

        return (
            torch.FloatTensor(self.state_buffer[idxs]).to(self.device),
            torch.FloatTensor(self.action_buffer[idxs]).to(self.device),
            torch.FloatTensor(self.reward_buffer[idxs]).to(self.device),
            torch.FloatTensor(self.next_state_buffer[idxs]).to(self.device),
            torch.FloatTensor(self.not_done[idxs]).to(self.device),
        )