import torch
import numpy as np
import gymnasium as gym

from gymnasium import spaces

class Core(gym.Env):
    def __init__(
        self,
        shape: tuple[int,int],
        start_idices: list[tuple[int,int]], 
        action_dict: dict[int, tuple[int,int]],
        max_decision: int,
        reward_mean: float,
        reward_std: float
    ):
        self.shape = shape
        self.ACT_DICT = action_dict
        self.start_idices = start_idices
        self.max_decision = max_decision
        
        self.reward_mean = reward_mean
        self.reward_std = reward_std
        
        self.state_size: int = np.prod(self.shape, dtype=int)
        self.action_size = len(self.ACT_DICT.keys())
        self.init_state = np.zeros(self.state_size)
        
        self.observation_space = spaces.Box(low=0, high=3, shape=(self.state_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.action_size)
        
    def get_obs(self, current_pos: tuple[int,int]):
        init_state = np.zeros(self.state_size)
            
        init_state[np.ravel_multi_index(current_pos, self.shape)] += 1.0
        return init_state

    def is_valid(self, pos: tuple[int,int]):
        if pos[0] < 0 or pos[0] >= self.shape[0] or pos[1] < 0 or pos[1] >= self.shape[1]:
            return False
        else:
            return True
    
    
    def reset(self, seed: int | None = None):
        super().reset(seed=seed)
        self._decision = 0
        # start
        start_index = np.random.choice(len(self.start_idices))
        self.start_pos = self.start_idices[start_index]
        self.curr_pos = self.start_pos
        
        obs: np.ndarray = self.get_obs(current_pos = self.start_pos)
            
        return obs, {}

    def step(self, action: int):
        if isinstance(action, np.ndarray):
            action = action.item()
        elif isinstance(action, torch.Tensor):
            action = action.item()
        act = self.ACT_DICT[action]
        next_pos = (self.curr_pos[0] + act[0], self.curr_pos[1] + act[1])
        
        if self.is_valid(next_pos):
            self.curr_pos = next_pos
        
        obs = self.get_obs(current_pos=self.curr_pos)
        
        terminated = False
        truncated = False
        
        reward = np.random.normal(self.reward_mean, self.reward_std)
        
        if self._decision >= self.max_decision:
            truncated = True
            
        return obs, reward, terminated, truncated, {}
    
    def count_decision(self):
        self._decision += 1
        
    def render(self):
        H, W = self.shape
        cell_size = 40
        img = np.full((H*cell_size, W*cell_size, 3), 255, dtype=np.uint8)
        
        color_map = {
            "agent": (0, 0, 0),
            "start": (200, 255, 200), 
            "grid": (220, 220, 220)
        }
        sr, sc = self.start_pos
        img[sr*cell_size:(sr+1)*cell_size, sc*cell_size:(sc+1)*cell_size] = color_map["start"]

        for i in range(1, H):
            img[i * cell_size - 1 : i * cell_size + 1, :] = color_map["grid"]
        for j in range(1, W):
            img[:, j * cell_size - 1 : j * cell_size + 1] = color_map["grid"]

        ar, ac = self.curr_pos
        center_y = ar * cell_size + cell_size // 2
        center_x = ac * cell_size + cell_size // 2
        radius = cell_size // 3
        
        Y, X = np.ogrid[:H * cell_size, :W * cell_size]
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        img[dist_from_center <= radius] = color_map["agent"]
        
        return img

class DiscreteNoisyRewards(Core):
    
    def __init__(self, num_decision: int, reward_mean: float):
        shape: tuple[int, int] = (21, 21)
        start_idices: list[tuple[int, int]] = [
            (10, 10)
        ]

        max_decision: int = num_decision
        
        reward_std: float = 1

        # left, right
        ACT_DICT = {
            0: (0, -1),
            1: (-1, 0),
            2: (0, 1),
            3: (1, 0)
        }


        super().__init__(
            shape=shape,
            start_idices=start_idices,
            action_dict=ACT_DICT,
            max_decision=max_decision,
            reward_mean=reward_mean,
            reward_std=reward_std
        )