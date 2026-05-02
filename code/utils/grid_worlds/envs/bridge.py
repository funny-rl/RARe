from utils.grid_worlds.core import GridEnv

class Bridge(GridEnv):
    def __init__(self):
        
        # start and gaal should be at least 1 steps away
        shape: tuple[int,int] = (8, 12)
        start_idices: list[tuple[int, int]] = [ # if randomized, there are more then 1 start state
            (0, 0)
        ]
        goal_idices: list[tuple[int, int]] = [ # if randomized, there are more then 1 goal state
            (7, 10)
        ]
        
        pits = []
        pits += [(0, i) for i in range(2, 10)]
        pits += [(1, i) for i in range(2, 10)]
        pits += [(2, i) for i in range(2, 10)]
        
        pits += [(5, i) for i in range(2, 10)]
        pits += [(6, i) for i in range(2, 10)]
        pits += [(7, i) for i in range(2, 10)]
        
        
        max_steps: int = 100
        
        # left, up, right, down
        ACT_DICT = {
            0: (0, -1),
            1: (-1, 0),
            2: (0, 1),
            3: (1, 0)
        }
        
        sparse_reward_order = {
            "normal": -1.0,
            "goal": -1.0,
            "pit": -100.0,
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