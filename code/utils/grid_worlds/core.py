import torch
import numpy as np
import gymnasium as gym

from gymnasium import spaces

"""
This module's base is row-col coordinates, which is more intuitive for gridworlds.
The state is represented as a vetor of size (H*W), where the index corresponds to the row-col position in the grid.

This grid world assumes deterministic transitions (Deterministic-MDP). 

The left-top corner is (0, 0) and the right-bottom corner is (H-1, W-1).

For example, in a 4x4 grid:
(0,0) -> index 0
(0,3) -> index 3
(1,0) -> index 4
(1,2) -> index 6
-> W * row + col

Reward is two version:
    1. Sparse reward: end for reaching the goal, -1 otherwise.
    2. Dense reward: BFS based available manhattan distance.
    
Start and goal state can be randomized or fixed.

"""


class GridEnv(gym.Env):
    def __init__(
        self,
        pits: list[tuple[int,int]],
        shape: tuple[int,int],
        start_idices: list[tuple[int,int]],
        goal_idices: list[tuple[int,int]],
        action_dict: dict[int, tuple[int,int]],
        max_steps: int,
        sparse_reward_order: dict[str, float],
    ):
        self.pits = pits
        self.shape = shape
        self.ACT_DICT = action_dict
        self.start_idices = start_idices
        self.goal_idices = goal_idices
        self.max_steps = max_steps
        self.sparse_reward_order = sparse_reward_order
        self.state_size: int = np.prod(self.shape, dtype=int)
        self.action_size = len(self.ACT_DICT.keys())
        self.init_state = np.zeros(self.state_size)
        
        self.observation_space = spaces.Box(low=0, high=3, shape=(self.state_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.action_size)
        
        self.color_map = {
            "start": (200, 255, 200), 
            "goal": (200, 200, 255), 
            "pit": (255, 200, 200),   
            "grid": (220, 220, 220), 
            "agent": (0, 0, 0)      
        }
    
    def get_obs(self, current_pos: tuple[int,int]):
        init_state = np.zeros(self.state_size)
        init_state[np.ravel_multi_index(self.goal_pos, self.shape)] = 2.0
        # # pits
        # for pit in self.pits:
        #     init_state[np.ravel_multi_index(pit, self.shape)] = 3.0
            
        # current position
        init_state[np.ravel_multi_index(current_pos, self.shape)] += 1.0
        return init_state
    
    def reset(self, seed: int | None = None):
        super().reset(seed=seed)
        self._steps = 0
        # start
        start_index = np.random.choice(len(self.start_idices))
        self.start_pos = self.start_idices[start_index]
        self.curr_pos = self.start_pos
        
        # goal
        goal_index = np.random.choice(len(self.goal_idices))
        self.goal_pos = self.goal_idices[goal_index]
        
        obs: np.ndarray = self.get_obs(current_pos = self.start_pos)
        
            
        return obs, {}
        
    def is_valid(self, pos: tuple[int,int]):
        # check if pos is within the grid
        if pos[0] < 0 or pos[0] >= self.shape[0] or pos[1] < 0 or pos[1] >= self.shape[1]:
            return False
        else:
            return True
    
    def step(self, action: int):
        self._steps += 1
        if isinstance(action, np.ndarray):
            action = action.item()
        elif isinstance(action, torch.Tensor):
            action = action.item()
        
        # action
        move = self.ACT_DICT[action]
        next_pos = (self.curr_pos[0] + move[0], self.curr_pos[1] + move[1])
        
        if self.is_valid(next_pos):
            self.curr_pos = next_pos
            
        
        obs = self.get_obs(current_pos=self.curr_pos)
        
        terminated = False
        truncated = False
        
        reward = self.sparse_reward_order["normal"]
            
        # if not valid_transition:
        #     reward += -1.0
            
        if self.curr_pos in self.pits:
            terminated = True
            reward = self.sparse_reward_order["pit"]
        elif self.curr_pos == self.goal_pos:
            terminated = True
            reward = self.sparse_reward_order["goal"]
            # reward = 1.0 - self._steps / self.max_steps
        elif self._steps >= self.max_steps:
            truncated = True
                
        return obs, reward, terminated, truncated, {}
    
    def render(self):
        """
        RGB 배열로 그리드 월드를 시각화합니다.
        - 배경: 흰색 / 시작: 녹색 / 함정: 빨간색 / 목표: 파란색
        - 에이전트: 검은색 원
        - 격자선: 회색
        """
        H, W = self.shape
        cell_size = 40
        
        img = np.full((H * cell_size, W * cell_size, 3), 255, dtype=np.uint8)

        sr, sc = self.start_pos
        img[sr*cell_size:(sr+1)*cell_size, sc*cell_size:(sc+1)*cell_size] = self.color_map["start"]
        
        gr, gc = self.goal_pos
        img[gr*cell_size:(gr+1)*cell_size, gc*cell_size:(gc+1)*cell_size] = self.color_map["goal"]

        for pr, pc in self.pits:
            img[pr*cell_size:(pr+1)*cell_size, pc*cell_size:(pc+1)*cell_size] = self.color_map["pit"]

        for i in range(1, H):
            img[i * cell_size - 1 : i * cell_size + 1, :] = self.color_map["grid"]
        for j in range(1, W):
            img[:, j * cell_size - 1 : j * cell_size + 1] = self.color_map["grid"]
        ar, ac = self.curr_pos
        center_y = ar * cell_size + cell_size // 2
        center_x = ac * cell_size + cell_size // 2
        radius = cell_size // 3

        Y, X = np.ogrid[:H * cell_size, :W * cell_size]
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        img[dist_from_center <= radius] = self.color_map["agent"]

        return img
    