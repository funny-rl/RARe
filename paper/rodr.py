"""Score RODR and replay per-case returns from the Grid decision traces (CPU-heavy stage).

Every decision in evaluation_decisions.jsonl is scored against the shared oracle,
and the deterministic layout is stepped through the logged actions and skips to
recover each episode's return. Replayed outcomes must reproduce evaluations.jsonl,
so an incomplete or misaligned trace fails here rather than in a table.

Usage: python -m paper.rodr --runs runs --output results [--workers N]
Outputs: <output>/rodr/<Env>/<Arm>.npz with seeds, steps, rodr (seeds, steps) and
         returns (seeds, steps, episodes)
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import os
from pathlib import Path

import numpy as np

from paper.data import GRIDS, Cell, grid_spec, read_jsonl
from rare.envs.grid.core import GridEnv
from rare.metrics.rodr import build_grid_rodr_oracle, score_rodr_decisions

# Repeat horizon J of the Grid environments. Every arm, DDQN included, uses this oracle.
ORACLE_HORIZON = 6


def replay_episode(decisions, spec, oracle):
    """Step one episode through the layout; returns (return, length, pit, success)."""
    position = int(decisions[0]["observation"]["agent_index"])
    goal = int(decisions[0]["observation"]["goal_index"])
    reward_sum, length, pit, success = 0., 0, False, False
    for index, decision in enumerate(decisions):
        if int(decision["primitive_step"]) != length or int(decision["observation"]["agent_index"]) != position:
            raise ValueError("decision does not continue the replayed episode")
        for _ in range(int(decision["skip"])):
            position = int(oracle.transitions[position, int(decision["action"])])
            length += 1
            pit, success = bool(oracle.pits[position]), position == goal
            reward_sum += spec.pit_reward if pit else spec.goal_reward if success else spec.step_reward
            if pit or success or length >= spec.max_steps:
                if index != len(decisions) - 1:
                    raise ValueError("decisions recorded after the episode ended")
                return reward_sum, length, pit, success and not pit
    raise ValueError("trace stops before the episode ended")


def score_cell(root, output, environment, arm):
    """Write one cell's scores and return its path."""
    cell = Cell(root, environment, arm)
    spec = grid_spec(cell.spec)
    env = GridEnv(spec)
    oracle = build_grid_rodr_oracle(env, ORACLE_HORIZON)
    width = spec.shape[1]
    episodes = cell.evaluation["episodes"]
    cases = [(start[0] * width + start[1], goal[0] * width + goal[1]) for start in spec.starts for goal in spec.goals]
    seed_index = {seed: index for index, seed in enumerate(cell.seeds)}
    step_index = {step: index for index, step in enumerate(cell.schedule)}
    shape = len(cell.seeds), len(cell.schedule), episodes
    rodr, returns = np.full(shape, np.nan), np.full(shape, np.nan)
    lengths, pits, successes, decision_counts = (np.zeros(shape) for _ in range(4))
    for path in cell.paths:
        for row in read_jsonl(path / "evaluation_decisions.jsonl"):
            index = seed_index[row["seed"]], step_index[row["env_step"]], row["episode"] - 1
            decisions = row["decisions"]
            first = decisions[0]["observation"]
            start_goal = int(first["agent_index"]), int(first["goal_index"])
            if len(cases) > 1 and start_goal != cases[row["case_index"]]:
                raise ValueError(f"{path}: episode {index} does not start at its case")
            if np.isfinite(rodr[index]) or len(decisions) != row["decision_count"]:
                raise ValueError(f"{path}: duplicate or truncated episode {index}")
            rodr[index] = score_rodr_decisions(decisions, oracle)["rodr"]
            returns[index], lengths[index], pits[index], successes[index] = replay_episode(decisions, spec, oracle)
            decision_counts[index] = len(decisions)
    if not np.isfinite(rodr).all():
        raise ValueError(f"{environment}/{arm}: {int(np.isnan(rodr).sum())} evaluation episodes have no trace")
    for name, values in (("reward_sum", returns), ("length_mean", lengths), ("pit_rate", pits),
                         ("success_rate", successes), ("decisions_mean", decision_counts)):
        logged = cell.curves(lambda row: row[name], cell.schedule)
        if not np.allclose(values.mean(axis=-1), logged, rtol=1e-6, atol=1e-5):
            raise ValueError(f"{environment}/{arm}: replayed {name} disagrees with evaluations.jsonl")
    destination = Path(output) / "rodr" / environment / f"{arm}.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, seeds=np.asarray(cell.seeds), steps=np.asarray(cell.schedule),
                        rodr=rodr.mean(axis=-1), returns=returns)
    return destination


def trace_size(directory):
    """Bytes of the cell's decision traces; 0 if any shard has none."""
    files = [*directory.glob("evaluation_decisions.jsonl"), *directory.glob("*/evaluation_decisions.jsonl")]
    sizes = [file.stat().st_size for file in files]
    return sum(sizes) if sizes and all(sizes) else 0


def traced_cells(root):
    """Grid cells with decision traces, largest first; arms without traces have no RODR to report."""
    cells = [(trace_size(directory), environment, directory.name) for environment in GRIDS
             for directory in (Path(root) / environment).glob("*") if directory.is_dir()]
    return [cell[1:] for cell in sorted(cells, reverse=True) if cell[0]]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count()))
    args = parser.parse_args(argv)
    # Spawn, not fork: workers build JAX environments.
    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn")) as pool:
        jobs = [pool.submit(score_cell, args.runs, args.output, *cell) for cell in traced_cells(args.runs)]
        for job in jobs:
            print(job.result(), flush=True)


if __name__ == "__main__":
    main()
