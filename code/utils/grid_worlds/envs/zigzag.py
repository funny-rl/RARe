from utils.grid_worlds.core import GridEnv

class ZigZag(GridEnv):
    def __init__(self):
        
        # start and gaal should be at least 1 steps away
        shape: tuple[int,int] = (8, 12)
        start_idices: list[tuple[int, int]] = [ # if randomized, there are more then 1 start state
            (0, 0)
        ]
        goal_idices: list[tuple[int, int]] = [ # if randomized, there are more then 1 goal state
            (7, 10)
        ]
        
        pits : list[list[int,int]] = [
            (0,2),
            (1,2),
            (2,2),
            (3,2),
            (0,3),
            (1,3),
            (2,3),
            (3,3),
            (0,4),
            (1,4),
            (2,4),
            (3,4),

            (4,7),
            (5,7),
            (6,7),
            (7,7),
            (4,8),
            (5,8),
            (6,8),
            (7,8),
            (4,9),
            (5,9),
            (6,9),
            (7,9),
        ]
        
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

class RZigZag(GridEnv):
    def __init__(self):
        
        # start and gaal should be at least 1 steps away
        shape: tuple[int,int] = (8, 12)
        start_idices: list[tuple[int, int]] = [ # if randomized, there are more then 1 start state
            (0, 0), (7, 0)
        ]
        goal_idices: list[tuple[int, int]] = [ # if randomized, there are more then 1 goal state
            (7, 10), (0, 10)
        ]
        
        pits : list[list[int,int]] = [
            (0,2),
            (1,2),
            (2,2),
            (3,2),
            (0,3),
            (1,3),
            (2,3),
            (3,3),
            (0,4),
            (1,4),
            (2,4),
            (3,4),

            (4,7),
            (5,7),
            (6,7),
            (7,7),
            (4,8),
            (5,8),
            (6,8),
            (7,8),
            (4,9),
            (5,9),
            (6,9),
            (7,9),
        ]
        
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