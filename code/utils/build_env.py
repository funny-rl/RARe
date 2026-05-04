import numpy as np
import gymnasium as gym

from utils.grid_worlds import ENVS_REGISTRY

GRID = "grid"
CNR = "cnr"
CLASSIC = "classic"
ROBOTICS = "robotics"
MUJOCO = "mujoco"
SAFE = "safe"

class SparsePendulumWrapper(gym.RewardWrapper):
    def __init__(self, env, angle_threshold_deg=12.5): # 12.5 25
        super().__init__(env)
        self.angle_threshold_rad = np.deg2rad(angle_threshold_deg)

    def reward(self, _unused_reward):
        theta, _ = self.env.unwrapped.state
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
        abs_theta = abs(theta)
        if abs_theta < self.angle_threshold_rad:
            return 1.0 - abs_theta / self.angle_threshold_rad

        return 0.0

class BottomSpawnWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        new_theta = self.env.unwrapped.np_random.uniform(low=np.pi/2, high=1.5*np.pi)
        if new_theta > np.pi:
            new_theta -= 2 * np.pi
            
        new_thetadot = self.env.unwrapped.np_random.uniform(low=-1.0, high=1.0)
        self.env.unwrapped.state = np.array([new_theta, new_thetadot], dtype=np.float32)
        return self.env.unwrapped._get_obs(), info
    
class SafetyGymnasiumRewardCostWrapper(gym.Wrapper):
    def __init__(self, env, reward_weight: float = 10.0, cost_weight: float = 0.1):
        super().__init__(env)
        self.reward_weight = float(reward_weight)
        self.cost_weight = float(cost_weight)

    def step(self, action):
        out = self.env.step(action)
        
        if len(out) == 6:
            obs, reward, cost, terminated, truncated, info = out
            merged_reward = self.reward_weight * reward - self.cost_weight * cost
            return obs, merged_reward, terminated, truncated, info
        else:

            raise ValueError(f"Unexpected safety env.step output length: {len(out)}")

def build_env(envs_args, render = False):
    env_name: str = envs_args.name
    env_domain: str = envs_args.domain
    if env_domain == CLASSIC:
        if env_name == "Sparse_Pendulum-v1":
            env = gym.make(
                "Pendulum-v1",
                render_mode="rgb_array" if render else None,
            )
            if envs_args.pendulum.custom_reward:
                env = SparsePendulumWrapper(env)
                env = BottomSpawnWrapper(env)
        else:
            env = gym.make(
                env_name,
                render_mode="rgb_array" if render else None,
            )
        env_info = {}
        is_continuous = isinstance(env.action_space, gym.spaces.Box)
        state_dim: int = env.observation_space.shape[0]
        env_info["state_dim"] = state_dim
        if is_continuous:
            action_dim: int = env.action_space.shape[0] 
            env_info["action_dim"] = action_dim
            env_info["max_action"] = env.action_space.high[0]
            env_info["is_continuous"] = True
        else:
            raise NotImplementedError("Only Pendulum-v1 is supported for classic control environments.")
    
    elif env_domain == GRID:
        if env_name == "DiscreteNoisyRewards":
            num_decision = int(envs_args.skip_mdp.num_decision)
            reward_mean = float(envs_args.skip_mdp.reward_mean)
            env = ENVS_REGISTRY[env_name](num_decision=num_decision, reward_mean=reward_mean)
        else:
            env = ENVS_REGISTRY[env_name]()
        
        state_dim: int = env.observation_space.shape[0]
        n_actions: int = env.action_space.n
        
        env_info = {
            "state_dim": int(state_dim),
            "action_dim": 1,
            "n_actions": int(n_actions),
            "is_continuous": False
        }
    
    elif env_domain == ROBOTICS:
        import gymnasium_robotics
        gym.register_envs(gymnasium_robotics)
        if "PointMaze" in env_name:
            env = gym.make(
                env_name,
                render_mode="rgb_array" if render else None,
            )
            env_info = {}
            obs_space = env.observation_space
            state_dim: int = obs_space["observation"].shape[0] + \
                obs_space["desired_goal"].shape[0] + \
                obs_space["achieved_goal"].shape[0]
            action_dim: int = env.action_space.shape[0] 
            env_info["state_dim"] = state_dim
            env_info["action_dim"] = action_dim
            env_info["max_action"] = env.action_space.high[0]
            env_info["is_continuous"] = True
        else:
            raise NotImplementedError(f"Robotics environment {env_name} not implemented.")
        
    elif env_domain == CNR:
        from utils.continuousnoisyreward import ContinuousNoisyRewards
        num_decision = int(envs_args.num_decision)
        env = ContinuousNoisyRewards(num_decision=num_decision)
        env_info = {
            "state_dim": env.observation_space.shape[0],
            "action_dim": env.action_space.shape[0],
            "max_action": env.action_space.high[0],
            "is_continuous": True
        }

    elif env_domain == MUJOCO:
        import mujoco
        env = gym.make(
            env_name, 
            render_mode="rgb_array" if render else None,
        )
        env_info = {}
        state_dim: int = env.observation_space.shape[0]
        action_dim: int = env.action_space.shape[0]
        env_info["state_dim"] = state_dim
        env_info["action_dim"] = action_dim
        env_info["max_action"] = env.action_space.high[0]
        env_info["is_continuous"] = True
    
    elif SAFE == env_domain:
        import safety_gymnasium
        env = safety_gymnasium.make(
            env_name,
            render_mode="rgb_array" if render else None,
        )
        env = SafetyGymnasiumRewardCostWrapper(env)
        env_info = {
            "state_dim": env.observation_space.shape[0],
            "action_dim": env.action_space.shape[0],
            "max_action": env.action_space.high[0],
            "is_continuous": True
        }
        
    else:
        raise NotImplementedError(f"Environment domain {env_domain} not implemented.")
    
    return env, env_info