"""Unified grid experiments: python -m experiments.train --help."""

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

from .config import ALGORITHM_FIELDS, COMMON_FIELDS, DEFAULT_CONFIG, resolve_config, resolve_evaluation_config
from .device import configure_device
from .engine import run_experiment


def parse_seeds(value):
    """Accept comma-separated integers or start:stop with an exclusive stop."""
    try:
        if ":" in value:
            start, stop = map(int, value.split(":"))
            seeds = list(range(start, stop))
        else:
            seeds = list(map(int, value.split(",")))
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("use comma-separated seeds or start:stop") from error
    if not seeds or len(set(seeds)) != len(seeds) or any(not 0 <= seed < 2 ** 32 for seed in seeds):
        raise argparse.ArgumentTypeError("seeds must be distinct uint32 values and nonempty")
    return seeds


def parse_evaluation_episodes(value):
    if value == "all":
        return value
    try:
        episodes = int(value)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("use a positive episode count or 'all'") from error
    if episodes <= 0:
        raise argparse.ArgumentTypeError("use a positive episode count or 'all'")
    return episodes


def parse_switch_steps(value):
    try:
        steps = tuple(int(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("use comma-separated positive switch steps") from error
    if not steps or any(step <= 0 for step in steps) or any(a >= b for a, b in zip(steps, steps[1:])):
        raise argparse.ArgumentTypeError("switch steps must be strictly increasing positive integers")
    return steps

def parser():
    result = argparse.ArgumentParser(allow_abbrev=False, description="JAX grid DDQN/TempoRL/UTE/RARe/TAACDiscrete with common and algorithm-specific YAML settings. Physical GPUs 0..3 only.")
    result.add_argument("--env", choices=("CliffWalking", "Bridge", "ZigZag", "WindShift"), default="CliffWalking")
    result.add_argument("--random", action="store_true", help="resample start/goal at every reset")
    result.add_argument("--wind-switch-interval", type=int,
                        help="WindShift: global primitive steps per direction; 0 keeps a fixed direction")
    result.add_argument("--wind-switch-steps", type=parse_switch_steps,
                        help="WindShift: explicit global steps for direction reversals, e.g. 20000,30000")
    result.add_argument("--wind-probability", type=float, help="WindShift: gust probability; default 0.5")
    result.add_argument("--wind-layout", choices=("descending", "ascending"),
                        help="WindShift: top-left to bottom-right or its vertical mirror")
    result.add_argument("--wind-initial", choices=("north", "south"),
                        help="WindShift: direction at global step zero")
    result.add_argument("--wind-observation", choices=("hidden", "visible"),
                        help="WindShift: whether the current direction is appended to the observation")
    result.add_argument("--seeds", type=parse_seeds, default=[0], help="e.g. 0,7,19 or 0:20 (stop exclusive)")
    result.add_argument("--algorithm", choices=tuple(ALGORITHM_FIELDS), default="DDQN")
    result.add_argument("--config", type=Path, help=f"optional YAML overrides of {DEFAULT_CONFIG}")
    result.add_argument("--total-steps", type=int, default=argparse.SUPPRESS, help="primitive interactions PER seed, including warmup")
    for name in ("warmup_steps", "replay_capacity", "batch_size", "hidden_dim", "epsilon_decay_steps", "target_interval"):
        result.add_argument("--" + name.replace("_", "-"), type=int, default=argparse.SUPPRESS)
    for name in ("gamma", "learning_rate", "max_epsilon", "min_epsilon", "tau"):
        result.add_argument("--" + name.replace("_", "-"), type=float, default=argparse.SUPPRESS)
    result.add_argument("--epsilon-schedule", choices=("linear", "exponential"), default=argparse.SUPPRESS)
    result.add_argument("--lr-decay", action="store_true", default=argparse.SUPPRESS)
    result.add_argument("--hard-target", action="store_true", default=argparse.SUPPRESS)
    result.add_argument("--chunk-steps", type=int, default=256, help="compiled interactions per logging chunk")
    result.add_argument("--platform", choices=("cuda", "cpu"), default="cuda")
    result.add_argument("--gpu", type=int, help="physical CUDA index 0..3; default: 0, overrides inherited visibility")
    result.add_argument("--output", type=Path, help="new run directory; default: runs/<unique id>")
    result.add_argument("--eval-interval-fraction", type=float, default=argparse.SUPPRESS,
                        help="evaluation interval as a fraction of total primitive steps, starting after warmup; default: 0.01")
    result.add_argument("--eval-interval-steps", type=int, default=argparse.SUPPRESS,
                        help="evaluate every N primitive steps after warmup plus final; 0 disables")
    result.add_argument("--eval-episodes", type=parse_evaluation_episodes, default=argparse.SUPPRESS,
                        help="'all' evaluates every grid start-goal pair once; positive N samples N episodes; default: fixed=1, random=all")
    result.add_argument("--risk-diagnostics", dest="eval_risk_diagnostics", action=argparse.BooleanOptionalAction,
                        default=argparse.SUPPRESS,
                        help="exact Grid skip-value and V*-minus-V-policy risk errors; skip algorithms only; default: enabled")
    result.add_argument("--model-save-interval", type=int,
                        help="override default saves at 20/40/60/80/100%% of total steps per seed (rounded up; fewer than 5 steps deduplicate); positive N saves every N steps plus final, 0 disables saves")
    result.add_argument("--wandb-mode", choices=("disabled", "offline", "online"), default="online",
                        help="W&B logging mode; offline records locally without uploading")
    result.add_argument("--wandb-project", help="fixed to RARe-<environment>; WindShift adds its effective direction-reversal count")
    result.add_argument("--wandb-entity", default="singfor7012", help="W&B team or user name")
    result.add_argument("--wandb-name", help="W&B run-name prefix; each seed adds -seed<N>")
    result.add_argument("--wandb-group", help="W&B group; defaults to the algorithm label, override for distinct ablations")
    for name in ("max_skip", "skip_replay_capacity", "ensemble_size", "compare_horizon"):
        result.add_argument("--" + name.replace("_", "-"), type=int, default=argparse.SUPPRESS)
    for name in ("skip_learning_rate", "uncertainty_factor", "bootstrap_probability",
                 "max_eval_epsilon", "min_eval_epsilon", "eval_epsilon_cutoff", "eval_learning_rate",
                 "temperature", "actor_temperature", "target_entropy_delta", "target_actor_entropy_ratio"):
        result.add_argument("--" + name.replace("_", "-"), type=float, default=argparse.SUPPRESS)
    result.add_argument("--data-aug", dest="use_data_aug", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    result.add_argument("--rare-q-mode", choices=("separate", "shared"), default=argparse.SUPPRESS,
                        help="RARe only: separate Expected SARSA Q (default), or reuse base DDQN Q for skip targets")
    result.add_argument("--separate-target-epsilon-source", choices=("evaluation", "behavior"),
                        default=argparse.SUPPRESS,
                        help="separate RARe only: independent target epsilon (default), or match behavior epsilon")
    result.add_argument("--shared-target-epsilon-source", choices=("behavior", "evaluation"),
                        default=argparse.SUPPRESS,
                        help="shared RARe only: target uses behavior epsilon (default), or the independent evaluation schedule")
    result.add_argument("--no-lr-decay", dest="lr_decay", action="store_false", default=argparse.SUPPRESS)
    result.add_argument("--soft-target", dest="hard_target", action="store_false", default=argparse.SUPPRESS)
    return result


def main(argv=None):
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    if args.wandb_group is not None:
        args.wandb_group = args.wandb_group.strip()
        if not args.wandb_group:
            argument_parser.error("wandb-group must not be empty")
    args.launch_argv = list(sys.argv[1:] if argv is None else argv)
    renamed = {"lr_decay": "use_lr_decay", "hard_target": "use_hard_update", "target_interval": "update_interval"}
    overrides = {renamed.get(key, key): value for key, value in vars(args).items()
                 if renamed.get(key, key) in COMMON_FIELDS | set().union(*ALGORITHM_FIELDS.values())}
    try:
        if args.chunk_steps <= 0 or (args.model_save_interval is not None and args.model_save_interval < 0):
            raise ValueError("chunk-steps must be positive and model-save-interval nonnegative")
        configure_device(args)
        common, specific, inputs = resolve_config(args.config, args.env, args.algorithm, overrides)
        project = f"RARe-{'R' if args.random else ''}{args.env}"
        if args.env == "WindShift":
            if args.wind_switch_steps is not None:
                shifts = sum(step <= common["total_steps"] for step in args.wind_switch_steps)
                project = f"{project}-{shifts}"
            elif args.wind_switch_interval is not None:
                shifts = (common["total_steps"] // args.wind_switch_interval
                          if args.wind_switch_interval > 0 else 0)
                project = f"{project}-{shifts}"
        if args.wandb_project is not None and args.wandb_project != project:
            raise ValueError(f"wandb-project is fixed to {project}")
        args.wandb_project = project
        args.evaluation_config = resolve_evaluation_config(inputs, args.env, args.random, {
            key.removeprefix("eval_"): value for key, value in vars(args).items()
            if key in ("eval_interval_fraction", "eval_interval_steps", "eval_episodes", "eval_risk_diagnostics")
        })
        import jax
        import jax.numpy as jnp
        import numpy as np
        from envs.grid import make_grid_env, make_wind_shift_env
        from runners import (DDQNRunConfig, DDQNRunner, SkipRunConfig, SkipRunner,
                             TAACDiscreteRunConfig, TAACDiscreteRunner)
        wind_options = ("wind_switch_interval", "wind_switch_steps", "wind_probability", "wind_layout",
                        "wind_initial", "wind_observation")
        if args.env == "WindShift":
            if args.random:
                raise ValueError("WindShift uses fixed corner endpoints; --random is not supported")
            if args.wind_switch_interval is None and args.wind_switch_steps is None:
                raise ValueError("WindShift requires --wind-switch-interval or --wind-switch-steps")
            env = make_wind_shift_env(
                switch_interval=args.wind_switch_interval or 0,
                switch_steps=args.wind_switch_steps or (),
                wind_probability=0.5 if args.wind_probability is None else args.wind_probability,
                layout=args.wind_layout or "descending", initial_wind=args.wind_initial or "north",
                observe_wind=args.wind_observation == "visible",
            )
        else:
            if any(getattr(args, name) is not None for name in wind_options):
                raise ValueError("--wind-* options require --env WindShift")
            env = make_grid_env(args.env, random=args.random)
        if args.algorithm == "DDQN":
            config, runner = DDQNRunConfig(**common), DDQNRunner
        elif args.algorithm == "TAACDiscrete":
            config, runner = TAACDiscreteRunConfig(**common, **specific), TAACDiscreteRunner
        else:
            config, runner = SkipRunConfig(**common, algorithm=args.algorithm, **specific), SkipRunner
    except (ValueError, TypeError, OSError) as error:
        argument_parser.error(str(error))
    args.configuration = {"common": common, "algorithm": specific,
                          "evaluation": asdict(args.evaluation_config),
                          "inputs": [{key: value for key, value in source.items() if key != "content"} for source in inputs]}
    args.configuration_inputs = inputs
    return run_experiment(args, config, env, jax, jnp, np, runner)


if __name__ == "__main__":
    main()
