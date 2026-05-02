import torch 
import numpy as np
from utils.grid_worlds.core import GridEnv

class ChainMDP(GridEnv):
    def __init__(self):
        shape: tuple[int,int] = (1, 25)
        start_idices: list[tuple[int, int]] = [
            (0, 0)
        ]
        goal_idices: list[tuple[int, int]] = [
            (0, 24)
        ]
        
        dense_setting = True
        
        pits = []
        max_steps: int = 100
        if dense_setting:
            sparse_reward_order = {
                "normal": -1.0,
                "goal": -1.0,
            }
        else:
            
            sparse_reward_order = {
                "normal": 0.0,
                "goal": 1.0,
            }
            
        # left, right
        ACT_DICT = {
            0: (0, -1),
            1: (0, 1)
        }
            
        super().__init__(
            pits=pits,
            shape=shape,
            start_idices=start_idices,
            goal_idices=goal_idices,
            action_dict=ACT_DICT,
            max_steps=max_steps,
            sparse_reward_order=sparse_reward_order
        )
        
        