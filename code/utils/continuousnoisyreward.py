import torch
import numpy as np
import gymnasium as gym

from gymnasium import spaces

class ContinuousNoisyRewards(gym.Env):
    def __init__(
        self,
        num_decision: int,
        state_dim: int = 2,
        action_dim: int = 2,
        max_action: float = 1.0,
        reward_mean: float = 0.0,
        reward_std: float = 1.0,
        random_state: bool = False,
        state_bound: float = 10.0
    ):
        super().__init__()

        assert state_dim == 2, "This coordinate environment assumes state_dim=2."
        assert action_dim == 2, "This coordinate environment assumes action_dim=2."

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

        self.max_decision = num_decision
        self.reward_mean = reward_mean
        self.reward_std = reward_std
        self.random_state = random_state
        
        self.state_bound = state_bound
        self.state_low = -self.state_bound
        self.state_high = self.state_bound
        
        self.observation_space = spaces.Box(
            low=self.state_low,
            high=self.state_high,
            shape=(self.state_dim,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-self.max_action,
            high=self.max_action,
            shape=(self.action_dim,),
            dtype=np.float32,
        )

        self._decision = 0
        self.curr_state = np.zeros(self.state_dim, dtype=np.float32)

    def reset(self, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self._decision = 0

        if self.random_state:
            self.curr_state = self.np_random.uniform(
                low=-1.0,
                high=1.0,
                size=(self.state_dim,),
            ).astype(np.float32)
        else:
            self.curr_state = np.zeros(self.state_dim, dtype=np.float32)

        return self.curr_state.copy(), {}

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()

        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -self.max_action, self.max_action)

        next_state = (
            self.curr_state + action
        ).clip(
            self.state_low,
            self.state_high,
        ).astype(np.float32)

        self.curr_state = np.clip(
            next_state,
            self.state_low,
            self.state_high,
        ).astype(np.float32)

        reward = self.np_random.normal(
            loc=self.reward_mean,
            scale=self.reward_std,
        )

        truncated = self._decision >= self.max_decision

        info = {}

        return self.curr_state.copy(), float(reward), False, truncated, info

    def count_decision(self):
        self._decision += 1

    def render(self):
        return None