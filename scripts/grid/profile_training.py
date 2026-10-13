"""Short GPU timing probe; writes no models, experiment logs, or W&B runs."""

import argparse
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.config import resolve_config
from experiments.device import configure_device


def main():
    parser = argparse.ArgumentParser(description="Time a short, unlogged Grid training window on GPU 0–3.")
    parser.add_argument("--env", choices=("CliffWalking", "Bridge", "ZigZag"), default="Bridge")
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--algorithm", choices=("DDQN", "TempoRL", "UTE", "RARe"), default="RARe")
    parser.add_argument("--rare-q-mode", choices=("separate", "shared"), default="separate")
    parser.add_argument("--storage", choices=("compact", "raw"), default="compact")
    parser.add_argument("--total-steps", type=int, default=150000)
    parser.add_argument("--replay-capacity", type=int, default=100000)
    parser.add_argument("--skip-replay-capacity", type=int, default=300000)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--chunk-steps", type=int, default=128)
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    if (not 1 <= args.seeds <= 20 or not 1 <= args.chunk_steps <= 256 or
            not 2 <= args.chunks <= 8 or args.chunk_steps * args.chunks > 2048):
        parser.error("profile limits: 1–20 seeds, 1–256 steps/chunk, 2–8 chunks, at most 2048 steps")
    configure_device(argparse.Namespace(platform="cuda", gpu=args.gpu))

    import jax
    import jax.numpy as jnp
    from envs.grid import make_grid_env
    from runners import DDQNRunConfig, DDQNRunner, SkipRunConfig, SkipRunner

    overrides = {"total_steps": args.total_steps, "warmup_steps": 0,
                 "replay_capacity": args.replay_capacity}
    if args.algorithm != "DDQN":
        overrides["skip_replay_capacity"] = args.skip_replay_capacity
    if args.algorithm == "RARe":
        overrides["rare_q_mode"] = args.rare_q_mode
    common, specific, _ = resolve_config(None, args.env, args.algorithm, overrides)
    config = (DDQNRunConfig(**common) if args.algorithm == "DDQN" else
              SkipRunConfig(**common, **specific, algorithm=args.algorithm))
    env = make_grid_env(args.env, random=args.random)
    if args.storage == "raw":
        env.observation_codec = "raw"
    Runner = DDQNRunner if args.algorithm == "DDQN" else SkipRunner
    runner = Runner(env, config)
    started = time.perf_counter()
    state = jax.jit(jax.vmap(runner.initialize))(jnp.arange(args.seeds, dtype=jnp.uint32))
    jax.block_until_ready(state)
    initialization = time.perf_counter() - started
    started = time.perf_counter()
    execute = jax.jit(
        jax.vmap(lambda current: runner.run(current, args.chunk_steps, active_steps=args.chunk_steps)),
        donate_argnums=(0,),
    ).lower(state).compile()
    compilation = time.perf_counter() - started
    durations = []
    for _ in range(args.chunks):
        started = time.perf_counter()
        state, metrics = execute(state)
        jax.block_until_ready((state, metrics))
        durations.append(time.perf_counter() - started)
    stable = durations[1:]
    print(json.dumps({
        "environment": ("R" if args.random else "") + args.env,
        "algorithm": args.algorithm, "rare_q_mode": args.rare_q_mode if args.algorithm == "RARe" else None,
        "storage": args.storage, "gpu": args.gpu, "device": str(jax.devices()[0]),
        "seeds": args.seeds, "warmup_override": 0, "chunk_steps": args.chunk_steps,
        "chunks": args.chunks, "replay_capacity": args.replay_capacity,
        "skip_replay_capacity": args.skip_replay_capacity if args.algorithm != "DDQN" else None,
        "initialization_seconds": initialization, "compilation_seconds": compilation,
        "chunk_seconds": durations,
        "steady_ms_per_parallel_step": 1000 * sorted(stable)[len(stable) // 2] / args.chunk_steps,
    }, indent=2))


if __name__ == "__main__":
    main()
