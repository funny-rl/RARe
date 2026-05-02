import torch 
import random
import numpy as np


def set_seed(seed: int) -> None:
    """Seed the program."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
def state_transform(state):
    if isinstance(state, np.ndarray):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        if state_tensor.dim() == 2:
            state_tensor = state_tensor.view(-1) # flatten the state tensor
    elif isinstance(state, dict):
        obs_tensor = torch.tensor(state["observation"], dtype=torch.float32)
        desired_goal_tensor = torch.tensor(state["desired_goal"], dtype=torch.float32)
        achieved_goal_tensor = torch.tensor(state["achieved_goal"], dtype=torch.float32)
        state_tensor = torch.cat([obs_tensor, desired_goal_tensor, achieved_goal_tensor], dim=-1)
    else:
        raise NotImplementedError(f"State type {type(state)} is not supported.")

    # if state_tensor.dim() == 1:
    #     state_tensor = state_tensor.unsqueeze(0) # add batch dimension
    # else:
    #     raise NotImplementedError(f"State dimension {state_tensor.dim()} is not supported.")
    return state_tensor

def action_transform(
    action, 
    is_continuous
):
    if is_continuous:
        if isinstance(action, np.ndarray):
            action_tensor = torch.tensor(action, dtype=torch.float32)
        
        else:
            raise NotImplementedError(f"Action type {type(action)} is not supported.")
        
    else:
        if isinstance(action, np.ndarray):
            if action.ndim == 0:
                action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
            elif action.ndim == 1:
                action_tensor = torch.tensor(action, dtype=torch.float32)
            else:
                raise NotImplementedError(f"Action dimension {action.ndim} is not supported.")
    
        elif isinstance(action, np.int64):
            action_tensor = torch.tensor([action], dtype=torch.float32)

        else:
            raise NotImplementedError(f"Action type {type(action)} is not supported.")
    
    return action_tensor