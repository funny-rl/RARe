"""HalfCheetah reference scores of Table 16: uniform-random floor and Brax PPO expert.

python -m tools.halfcheetah_reference floor
python -m tools.halfcheetah_reference ppo --seed 0    (seeds 0, 1 and 2 in the paper)

Runs the unmodified Brax task on the spring backend. The expert score is the mean
over seeds of each seed's final evaluation (128 deterministic episodes), written
to runs/_brax_ppo/BraxHalfcheetah/seed<seed>.json.
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from brax import envs
from brax.training.agents.ppo import train as ppo

OUTPUT = Path("runs/_brax_ppo/BraxHalfcheetah")
FLOOR_EPISODES = 256
PPO_SETTINGS = dict(num_timesteps=50_000_000, num_evals=20, reward_scaling=1., unroll_length=20,
                    num_minibatches=32, num_updates_per_batch=8, discounting=.95, entropy_cost=.001,
                    num_envs=2048, batch_size=512)


def environment():
    return envs.get_environment("halfcheetah", backend="spring")


def floor():
    env = environment()
    low = jnp.asarray(env.sys.actuator.ctrl_range[:, 0])
    high = jnp.asarray(env.sys.actuator.ctrl_range[:, 1])

    def episode(key):
        reset_key, rollout_key = jax.random.split(key)

        def step(carry, _):
            state, total, stream = carry
            stream, action_key = jax.random.split(stream)
            action = jax.random.uniform(action_key, (env.action_size,), minval=low, maxval=high)
            state = env.step(state, action)
            return (state, total + state.reward, stream), None

        return jax.lax.scan(step, (env.reset(reset_key), jnp.float32(0), rollout_key), None, 1000)[0][1]

    returns = jax.jit(jax.vmap(episode))(jax.random.split(jax.random.PRNGKey(0), FLOOR_EPISODES))
    print(f"uniform-random return: {float(np.mean(returns)):.1f}")


def train(seed):
    history = []

    def progress(step, metrics):
        history.append({"step": int(step), "reward": float(metrics["eval/episode_reward"])})
        print(history[-1], flush=True)

    ppo.train(environment=environment(), episode_length=1000, action_repeat=1,
              normalize_observations=True, learning_rate=3e-4, deterministic_eval=True, seed=seed,
              progress_fn=progress, **PPO_SETTINGS)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / f"seed{seed}.json").write_text(
        json.dumps({"settings": PPO_SETTINGS, "history": history}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("floor", "ppo"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    floor() if args.command == "floor" else train(args.seed)


if __name__ == "__main__":
    main()
