import os 
import json
import time

import hydra
import torch
import wandb

import imageio

import numpy as np

from omegaconf import OmegaConf
from collections import defaultdict

from utils.utils import (
    set_seed, 
    state_transform,
    action_transform
)
from utils.build_env import build_env
from utils.set_hyperparams import set_hyperparams

from algos import BASE_REGISTRY, ALGO_REGISTRY

@hydra.main(config_path="configs/", config_name="config", version_base=None)
def main(args):
    envs_args = args.envs
    common_args = args.common
    base_args = args.base_agent
    algo_args = base_args.get("algo", None)
    
    use_wandb: bool = args.use_wandb
    use_eval_render: bool = args.use_eval_render
    use_lr_decay: bool = common_args.use_lr_decay
    
    seed: int = args.seed
    warmup_steps: int = args.warmup_steps 
    eval_interval: int = args.eval_interval
    num_eval_episodes: int = args.num_eval_episodes
    traj_log_interval: int = args.traj_log_interval
    total_training_steps: int = args.total_training_steps
    eval_render_interval: int = args.eval_render_interval
    
    
    env_name: str = envs_args.name
    base_name: str = base_args.name
    traj_log_dir: str = args.traj_log_dir
    device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    set_seed(seed = seed)
    env, env_info = build_env(envs_args = envs_args)
    
    base_config, algo_config = set_hyperparams(
        common_args = common_args,
        base_args = base_args,
        algo_args = algo_args,
        env_info = env_info,
        device = device
    )
    
    base_agent = BASE_REGISTRY[base_name](**base_config)
    has_base_agent: bool = algo_args is not None
    
    is_continuous: bool = env_info["is_continuous"]
    if has_base_agent:
        algo_name: str = algo_args.name
        skip_agent = ALGO_REGISTRY[algo_name](
            base_agent=base_agent,
            **algo_config
        )
    else:
        algo_name = None
        skip_agent = base_agent

    is_TAAC: bool = algo_name == "TAAC"
    ute_adaptive_lambda: bool = algo_config.get("use_adaptive_lambda", False) and algo_name == "UTE"
    
    if use_wandb:
        group_name = args.group_name
        use_offline_wandb = args.use_offline_wandb
        unique_id = unique_id = f"{algo_name}_{base_name}_{env_name}_{seed}_{time.time()}"
        wandb.init(
            mode="offline" if use_offline_wandb else "online",
            project=env_name,
            id=unique_id,
            name=f"{algo_name}_{base_name}_{env_name}_{seed}",
            group=group_name,
            config=OmegaConf.to_container(args, resolve=True),
        )
    else:
        group_name = None
    
    num_episodes = 0
    training_steps = 0
    
    
    while training_steps < total_training_steps:
        log: dict[list] = defaultdict(list)
        done = False
        prev_action = None
        state, _ =  env.reset()
        state = state_transform(state)
        
        if ute_adaptive_lambda:
            skip_agent.adaptive_lambda()
        
        while not done:
            warmup: bool = training_steps < warmup_steps
            if warmup: 
                action = env.action_space.sample()
                if is_TAAC:
                    beta = 1.0
            else:
                if is_continuous:
                    if is_TAAC:
                        action, beta = skip_agent.select_action(state.to(device), prev_action)
                    else:
                        action = skip_agent.select_action(state = state.to(device))
                else:
                    action = skip_agent.select_action(state = state.to(device))
                    
            action = action_transform(
                action, 
                is_continuous = is_continuous, 
            )
            
            skip: int = skip_agent.select_skip(
                state.to(device), 
                action.to(device)
            )
            
            log["skip"].append(skip)
            
            (
                skip_states,
                skip_rewards,
                skip_dones,
                next_skip_states,
            ) = [], [], [], []
            
            if hasattr(env, "count_decision"):
                env.count_decision()
            
            for _ in range(skip):
                next_state, reward, terminated, truncated, _ = env.step(action)
                
                if isinstance(reward, np.float32):
                    reward = reward.item()

                training_steps += 1
                log["reward"].append(reward)
                done = terminated or truncated
                
                next_state = state_transform(next_state)
                skip_states.append(state)
                skip_rewards.append(reward)
                skip_dones.append(done)
                next_skip_states.append(next_state)
                
                if algo_name == "TAAC":
                    skip_agent.add(
                        state = state,
                        action = action,
                        prev_action = prev_action,
                        reward = reward,
                        next_state = next_state,
                        beta = beta,
                        done = done
                    )
                else:
                    skip_agent.add(
                        state = state,
                        action = action,
                        reward = reward,
                        next_state = next_state,
                        done = done
                    )

                state = next_state
                prev_action = action
                
                training_rate = max(
                    0.0, 
                    min(
                        1.0, 
                        (training_steps - warmup_steps) / (total_training_steps - warmup_steps)
                    )
                )
                if not warmup:
                    log_dict: dict = skip_agent.update(
                        training_steps, 
                        training_rate
                    )
                    for key, value in log_dict.items():
                        log[key].append(value)

                if eval_interval > 0 and training_steps % eval_interval == 0:
                    eval(
                        seed = seed,
                        use_wandb = use_wandb,
                        is_continuous = is_continuous,
                        is_TAAC = is_TAAC,
                        skip_agent = skip_agent,
                        traj_log_dir = traj_log_dir,
                        traj_log_interval = traj_log_interval,
                        use_eval_render = use_eval_render,
                        eval_render_interval = eval_render_interval,
                        training_steps = training_steps,
                        num_eval_episodes= num_eval_episodes,
                        envs_args = envs_args,
                        env_name = env_name,
                        base_name = base_name,
                        algo_name = algo_name,
                        group_name = group_name,
                        device = device
                    )
                    
                if ute_adaptive_lambda:
                    skip_agent.ucb_datas.append(
                        (
                            skip_agent.j, 
                            reward
                        )
                    )
                
                train_steps = max(training_steps - warmup_steps, 0.0)
                if has_base_agent and not is_TAAC:
                    log["epsilon"].append(skip_agent.base_agent.epsilon)
                    log["skip_epsilon"].append(skip_agent.skip_epsilon)
                    skip_agent.epsilon_decay(train_steps)
                else:
                    if not is_continuous:
                        log["epsilon"].append(skip_agent.epsilon)
                        skip_agent.epsilon_decay(train_steps)
                
                log["lr"].append(skip_agent.lr)
                if use_lr_decay:
                    skip_agent.lr_decay(training_rate)
                    
                if is_continuous:
                    if is_TAAC or not has_base_agent:
                        skip_agent.decay_sigma(training_rate)
                    elif has_base_agent and not is_TAAC:
                        skip_agent.base_agent.decay_sigma(training_rate)
                    else:
                        raise NotImplementedError("Decay sigma is only implemented for continuous action spaces.")
                    
                if done:
                    num_episodes += 1
                    total_reward = sum(log["reward"])
                    log["total_reward"].append(total_reward)
                    log.pop("reward")
                    
                    msg = f"Training steps: {training_steps} | Episode: {num_episodes} | Rewards: {total_reward} | LR: {skip_agent.lr} |Env: {env_name} | Backbone: {base_name} | Algo: {algo_name} | Seed: {seed} "
                    if hasattr(skip_agent, "epsilon"):
                        msg += f" | Epsilon: {skip_agent.epsilon}"
                    
                    if use_wandb:
                        msg += f" | Group name: {group_name}"
                    print(msg)
                    
                    if use_wandb:
                        _log = {}
                        for key, value in log.items():
                            _log[f"train/{key}"] = np.mean(value)
                        wandb.log(_log, step=training_steps)
                    log.clear()
                    break       
            if has_base_agent and not is_TAAC:
                skip_agent.add_skip(
                    skip_states = skip_states,
                    action = action, 
                    skip_rewards = skip_rewards,
                    skip_dones = skip_dones,
                    next_skip_states = next_skip_states,
                    skip = skip
                )
            if ute_adaptive_lambda:
                skip_agent.ucb.push_data(skip_agent.ucb_datas)
                                           

def eval(
    seed: int,
    use_wandb: bool,
    is_continuous: bool,
    is_TAAC: bool,  
    skip_agent: object,
    traj_log_dir: str,
    traj_log_interval: int,
    use_eval_render: bool,
    eval_render_interval: int,
    training_steps: int,
    num_eval_episodes: int,
    envs_args: dict,
    env_name: str,
    base_name: str,
    algo_name: str,
    group_name: str,
    device: str,
):
    print("\n\nStarting Evaluation...")
    use_traj_log: bool = traj_log_dir is not None
    
    if use_traj_log:
        traj_log_directory = os.path.join(traj_log_dir,  f"{env_name}/{base_name}/{algo_name}/{group_name}/seed_{seed}", str(training_steps))
    
    total_eval_log: dict[list] = defaultdict(list)
    
    
    eval_env, _ = build_env(
        envs_args = envs_args, 
        render = use_eval_render
    )
    
    for ep in range(num_eval_episodes):
        done = False
        traj_log: list = []
        frames: list = []
        epi_eval_log: dict[list] = defaultdict(list)    
        test_seed = seed + 100 * ep
        state, _ = eval_env.reset(seed = test_seed)
        step = 0
        prev_action = None  
        
        while not done:
            state = state_transform(state)
            if is_TAAC:
                action_t, _ = skip_agent.select_action(
                    state.to(device), 
                    prev_action, 
                    deterministic=True
                )
            else:
                action_t, q_values = skip_agent.select_action(
                    state.to(device),
                    deterministic=True
                )
            action = action_transform(
                action_t,
                is_continuous = is_continuous,
            )
            skip, skip_values = skip_agent.select_skip(
                state.to(device), 
                action.to(device), 
                deterministic=True
            )
            
            epi_eval_log["action"].append(action_t)
            epi_eval_log["skip"].append(skip)
            epi_eval_log["num_decisions"].append(1)

            if use_traj_log and training_steps % traj_log_interval == 0:
                traj_log.append({
                    "step": step,
                    "state": state.tolist(),
                    "action": action_t.tolist(),
                    "q_values": q_values.tolist() if not is_TAAC else None,
                    "skip_values": skip_values.tolist() if skip_values is not None else None,
                    "skip": skip
                })
                
            if hasattr(eval_env, "count_decision"):
                eval_env.count_decision()
                
            for _ in range(skip):
                next_state, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                epi_eval_log["total_reward"].append(reward)
                
                if use_eval_render and training_steps % eval_render_interval == 0:
                    frame = eval_env.render()
                    frames.append(frame)

                state = next_state
                prev_action = action
                step += 1
                
                if done:
                    if use_traj_log and training_steps % traj_log_interval == 0:
                        log_path = f"{traj_log_directory}/{test_seed}/eval_log.json"
                        os.makedirs(os.path.dirname(log_path), exist_ok=True)
                        with open(log_path, "w") as f:
                            json.dump(traj_log, f, indent=4)
                            print(f"Saved evaluation log at {log_path}")
                    
                    if use_eval_render and training_steps % eval_render_interval == 0:
                        video_path = f"{traj_log_directory}/{test_seed}/eval_video.mp4"
                        os.makedirs(os.path.dirname(video_path), exist_ok=True)
                        imageio.mimsave(video_path, frames, fps=30)
                        print(f"Saved evaluation video at {video_path}...!")
                    
                    epi_eval_log["total_reward"] = sum(epi_eval_log["total_reward"])
                    epi_eval_log["num_decisions"] = sum(epi_eval_log["num_decisions"])
                    
                    for key, value in epi_eval_log.items():
                        if value is not None or isinstance(value, list) and value[0] != None:
                            try:
                                total_eval_log[key].append(np.mean(np.array(value)))
                            except Exception as e:
                                breakpoint()
                        else:
                            print(f"Key {key} has None values, skipping...")
                        
                    epi_eval_log.clear()
                    break

    _total_eval_log = {}
    for key, value in total_eval_log.items():
        _total_eval_log[f"eval/{key}"] = np.mean(value)
        _total_eval_log[f"eval/{key}_std"] = np.std(value)
    msg = f"Eval Rewards: {_total_eval_log['eval/total_reward']}\n\n"
    print(msg)
    
    if use_wandb:
        wandb.log(_total_eval_log, step=training_steps)
        

    eval_env.close()
    
if __name__ == "__main__":    
    main()