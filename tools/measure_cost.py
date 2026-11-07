"""Training and inference time of each method on dummy data (Tables 14 and 15).

The cost of a method comes from its networks and updates, not from the task, so a
dummy environment replaces the simulator: random observations and rewards at no
simulation cost. Each method keeps the network and batch sizes of RBridge or Goal.

Every job is a fresh process, alone on one GPU that reports no other compute
process, and jobs run strictly one after another. Each seed round runs in a
shuffled order, so slow drift cannot favor a method.

Training is milliseconds per interaction step of the real runner after warm-up,
so a step includes action selection, replay writes and all updates. Compilation
is excluded. One seed runs without vmap, so lax.cond keeps its branches.
Inference is the latency of one frozen policy decision at batch size 1: `host`
includes dispatch, `device` is the per-decision time of a compiled loop.

python -m tools.measure_cost run --gpu 0 --seeds 0:5
python -m tools.measure_cost report
"""

import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from experiments.config import resolve

OUTPUT = Path("runs/_cost")
# Setting -> (environment supplying the configuration, observation size, action size, arms).
SETTINGS = {"Discrete": ("RBridge", 96, 4, ("DDQN", "TempoRL", "UTE", "RARe-sh", "RARe-se")),
            "Continuous": ("Goal", 60, 2, ("DDPG", "TAAC", "TempoRL", "UTE", "RARe-sh", "RARe-se"))}
PHASES = ("train", "inference")
WARMUP_STEPS = 2000
TIMED_STEPS = 20000
CHUNK_STEPS = 1000
EPISODE_STEPS = 100
INFERENCE_CALLS = 5000
FUSED_CALLS = 2000
FUSED_REPEATS = 5


def result_path(root, setting, arm, phase, seed):
    return root / setting / arm / f"{phase}-seed{seed}.json"


def dummy_env(observation_size, action_size, discrete):
    import jax
    import jax.numpy as jnp

    class DummyEnv:
        name = "Dummy"
        requires_global_step = False
        observation_shape = (observation_size,)
        num_actions = action_size
        action_shape = (action_size,)
        action_low = -jnp.ones(action_size, jnp.float32)
        action_high = jnp.ones(action_size, jnp.float32)
        # Grid replay stores two cell indices, so a dummy Grid state keeps that storage cost.
        observation_codec = "grid_indices" if discrete else "raw"
        observation_storage_dtype = jnp.uint8 if discrete else jnp.float32

        def observe(self, key):
            if not discrete:
                return jax.random.uniform(key, self.observation_shape)
            agent, goal = jax.random.randint(key, (2,), 0, observation_size)
            return (jax.nn.one_hot(agent, observation_size, dtype=jnp.float32)
                    + 2. * jax.nn.one_hot(goal, observation_size, dtype=jnp.float32))

        def reset(self, key):
            return self.observe(key), jnp.int32(0)

        def step(self, key, state, action):
            del action
            observation_key, reward_key = jax.random.split(key)
            state = state + 1
            return (self.observe(observation_key), state, jax.random.normal(reward_key),
                    jnp.bool_(False), state >= EPISODE_STEPS, {})

    return DummyEnv()


def build(setting, arm):
    """Real runner on the dummy environment, plus a builder of the frozen decision."""
    import jax.numpy as jnp
    environment, observation_size, action_size, _ = SETTINGS[setting]
    discrete = setting == "Discrete"
    env = dummy_env(observation_size, action_size, discrete)
    values = {**resolve(environment, arm).config, "total_steps": WARMUP_STEPS + TIMED_STEPS,
              "warmup_steps": WARMUP_STEPS}
    if discrete:
        from experiments.grid import make_runner
        from rare.evaluation.grid import greedy_decision
        runner = make_runner(env, values)
        algorithm = values["algorithm"]

        def decision(state):
            skip = None if algorithm == "DDQN" else state.skip_learner.params
            factor = getattr(runner.config, "uncertainty_factor", -1.5)
            return lambda observation: greedy_decision(algorithm, state.learner.online_params, skip,
                                                       observation, factor)[:2]
        return runner, env, decision
    from rare.training.continuous import ContinuousRunConfig, ContinuousRunner
    from rare.training.safety import evaluation_action
    runner = ContinuousRunner(env, ContinuousRunConfig(**values))

    def decision(state):
        policy = state._replace(replay=None, skip_replay=None, pending=None, evaluation=None)
        zero = jnp.zeros(env.action_shape, jnp.float32)
        key = policy.rngs.evaluation

        def decide(observation):
            action, _, remaining, _ = evaluation_action(runner, policy, observation, zero,
                                                        jnp.bool_(False), zero, jnp.int32(0), key)
            return action, remaining  # Returning the skip keeps its network out of dead-code removal.
        return decide
    return runner, env, decision


def train(runner, seed):
    import jax
    import jax.numpy as jnp
    state = jax.block_until_ready(jax.jit(runner.initialize)(jnp.uint32(seed)))
    started = time.perf_counter()
    step = jax.jit(runner.run, donate_argnums=(0,)).lower(state, jnp.int32(CHUNK_STEPS)).compile()
    compilation = time.perf_counter() - started
    count = jnp.int32(CHUNK_STEPS)
    for _ in range(WARMUP_STEPS // CHUNK_STEPS + 1):  # Warm-up plus one discarded learning chunk.
        state = jax.block_until_ready(step(state, count))
    durations = []
    for _ in range(TIMED_STEPS // CHUNK_STEPS - 1):
        started = time.perf_counter()
        state = jax.block_until_ready(step(state, count))
        durations.append(time.perf_counter() - started)
    timed = len(durations) * CHUNK_STEPS
    return {"ms_per_step": 1e3 * sum(durations) / timed, "timed_steps": timed,
            "compilation_seconds": compilation, "chunk_seconds": durations}


def inference(runner, env, decision, seed):
    import jax
    import jax.numpy as jnp
    import numpy as np
    state = jax.jit(runner.initialize)(jnp.uint32(seed))
    decide = decision(state)
    batch = jax.block_until_ready(jax.random.uniform(jax.random.PRNGKey(seed), (FUSED_CALLS, *env.observation_shape)))
    samples = [batch[index] for index in range(256)]
    single = jax.jit(decide)
    jax.block_until_ready(single(samples[0]))
    host = []
    for index in range(INFERENCE_CALLS):
        observation = samples[index % len(samples)]
        started = time.perf_counter()
        jax.block_until_ready(single(observation))
        host.append(time.perf_counter() - started)
    fused = jax.jit(lambda values: jax.lax.map(decide, values))
    jax.block_until_ready(fused(batch))
    device = []
    for _ in range(FUSED_REPEATS):
        started = time.perf_counter()
        jax.block_until_ready(fused(batch))
        device.append((time.perf_counter() - started) / FUSED_CALLS)
    return {"host_us": float(np.median(host) * 1e6), "host_p95_us": float(np.percentile(host, 95) * 1e6),
            "device_us": float(np.median(device) * 1e6), "calls": INFERENCE_CALLS}


def worker(args):
    os.environ.update(JAX_PLATFORMS="cuda", CUDA_DEVICE_ORDER="PCI_BUS_ID",
                      CUDA_VISIBLE_DEVICES=str(args.gpu), XLA_PYTHON_CLIENT_PREALLOCATE="false")
    import jax
    runner, env, decision = build(args.setting, args.arm)
    result = train(runner, args.seed) if args.phase == "train" else inference(runner, env, decision, args.seed)
    result.update(seed=args.seed, phase=args.phase, device=jax.devices()[0].device_kind, jax=jax.__version__)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


def gpu_busy(gpu):
    query = ("nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader")
    return bool(subprocess.run(query, capture_output=True, text=True, check=True).stdout.strip())


def schedule(args):
    for seed in range(*map(int, args.seeds.split(":"))):
        jobs = [(setting, arm, phase) for setting, values in SETTINGS.items() for arm in values[3] for phase in PHASES]
        random.Random(seed).shuffle(jobs)
        for setting, arm, phase in jobs:
            output = result_path(args.output, setting, arm, phase, seed)
            if output.exists():
                continue
            while gpu_busy(args.gpu):
                print(f"waiting for GPU {args.gpu}", flush=True)
                time.sleep(30)
            command = [sys.executable, "-m", "tools.measure_cost", "worker", "--setting", setting, "--arm", arm,
                       "--seed", str(seed), "--phase", phase, "--output", str(output), "--gpu", str(args.gpu)]
            done = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            print(f"{'done' if done.returncode == 0 else 'FAILED'} {setting} {arm} {phase} seed{seed}", flush=True)
            if done.returncode:
                print(done.stderr[-2000:], flush=True)


def collect(root, setting, arm, phase, key):
    return {path.stem: json.loads(path.read_text())[key]
            for path in sorted((root / setting / arm).glob(f"{phase}-seed*.json"))}


def report(args):
    summary = {setting: {arm: {name: collect(args.output, setting, arm, phase, name)
                               for name, phase in (("ms_per_step", "train"), ("host_us", "inference"),
                                                   ("device_us", "inference"))}
                         for arm in values[3]}
               for setting, values in SETTINGS.items()}
    (args.output / "cost.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {args.output / 'cost.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--gpu", type=int, default=0)
    run.add_argument("--seeds", default="0:5", help="start:stop")
    run.add_argument("--output", type=Path, default=OUTPUT)
    job = commands.add_parser("worker")
    job.add_argument("--setting", choices=SETTINGS, required=True)
    job.add_argument("--arm", required=True)
    job.add_argument("--seed", type=int, required=True)
    job.add_argument("--phase", choices=PHASES, required=True)
    job.add_argument("--output", type=Path, required=True)
    job.add_argument("--gpu", type=int, required=True)
    summary = commands.add_parser("report")
    summary.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    {"run": schedule, "worker": worker, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
