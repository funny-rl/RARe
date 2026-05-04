def set_hyperparams(
    common_args,
    base_args,
    algo_args,
    env_info,
    device
):
    base_name: str = base_args.name
    if base_name == "MAXMINQ":
        n_target = base_args.n_target
        base_config = {
            **env_info,
            **common_args, 
            "n_target": n_target,
            "device": device,
        }
    elif base_name == "DDPG":
        expl_noise = base_args.expl_noise
        base_config = {
            **env_info,
            **common_args, 
            "expl_noise": expl_noise,
            "device": device
        }
    else:
        raise NotImplementedError(f"Unsupported backbone: {base_name}")
    
    if algo_args is not None:
        algo_config = {**algo_args}
    else:
        algo_config = {}
    
    return base_config, algo_config