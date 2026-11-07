"""Load finished runs from runs/<Env>/<Arm>/ and turn them into seed-level curves.

A cell holds run.json, evaluations.jsonl and evaluation_decisions.jsonl, or splits
them into shards runs/<Env>/<Arm>/<shard>/ that are merged by seed. Arm identity
comes from the directory name only. Every curve is an array (seeds, checkpoints)
with seeds sorted, so bootstrap resamples do not depend on the shard layout.

Randomized Grid rewards are normalized per start-goal case, which needs per-case
returns. paper.rodr replays them from the decision traces.
"""

from functools import cached_property
import json
from pathlib import Path

import numpy as np

from rare.envs.grid.core import GridSpec
from rare.metrics.auc import grid_reward_ceiling, normalize_rewards, normalize_to_reference, span_auc

STANDARD_GRIDS = ("CliffWalking", "Bridge", "ZigZag")
RANDOMIZED_GRIDS = ("RCliffWalking", "RBridge", "RZigZag")
GRIDS = STANDARD_GRIDS + RANDOMIZED_GRIDS
GRID_TITLES = {"CliffWalking": "Cliff", "Bridge": "Bridge", "ZigZag": "ZigZag",
               "RCliffWalking": "RCliff", "RBridge": "RBridge", "RZigZag": "RZigZag"}
GRID_ARMS = ("DDQN", "TempoRL", "UTE", "RARe-sh", "RARe-se")
GRID_FLOOR = -100.  # Hazard penalty of the Grid layouts (paper Eq. in Appendix D.1).

CONTINUOUS = ("MountainCar", "PointMaze", "Goal", "Button")
CONTINUOUS_ARMS = ("DDPG", "TAAC", "TempoRL", "UTE", "RARe-sh", "RARe-se")
LOOP_ARMS = ("TAAC", "RARe-se")  # The only arms compared on Pendulum and HalfCheetah (Table 17).
# Environment -> (random, expert) of the normalized AUC. Random and expert scores come
# from the original RARe evaluation protocol (paper Table 16).
REFERENCES = {"MountainCar": (-1.7, 99.), "PointMaze": (17., 400.), "Goal": (-6., 100.),
              "Button": (-12., 50.), "Pendulum": (-1400., -173.)}
HALFCHEETAH_FLOOR = -65.5  # Mean uniform-random return over 256 episodes on the Brax spring backend.
PPO_REFERENCE = "_brax_ppo/BraxHalfcheetah"  # Final scores of three Brax PPO seeds set the expert.

NONSTATIONARY = ("WindShift", "Random", "Episodic", "Shift1", "Shift3", "Shift5")
NONSTATIONARY_ARMS = ("UTE", "RARe-sh", "RARe-se")


def seed_count(environment):
    """Seeds every cell must hold: 20, except HalfCheetah (3) and the shift sweep (10)."""
    return 3 if environment == "HalfCheetah" else 10 if environment.startswith("Shift") else 20


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def grid_spec(data):
    """Rebuild a GridSpec from run.json, where tuples were stored as lists."""
    resolved = dict(data, shape=tuple(data["shape"]))
    for name in ("starts", "goals", "pits", "actions", "walls"):
        if name in resolved:
            resolved[name] = tuple(tuple(point) for point in resolved[name])
    return GridSpec(**resolved)


class Cell:
    """All shards of one (environment, arm), merged by seed."""

    def __init__(self, root, environment, arm):
        self.environment, self.arm = environment, arm
        directory = Path(root) / environment / arm
        self.paths = ([directory] if (directory / "run.json").exists()
                      else sorted(path.parent for path in directory.glob("*/run.json")))
        if not self.paths:
            raise FileNotFoundError(f"no run.json under {directory}")
        self.runs = [json.loads((path / "run.json").read_text()) for path in self.paths]
        seeds = [seed for run in self.runs for seed in run["seeds"]]
        self.seeds = sorted(seeds)
        self.config = self.runs[0]["config"]
        self.spec = self.runs[0]["environment"].get("spec")
        self.evaluation = self.runs[0]["evaluation"]
        self.schedule = self.evaluation["steps"]
        if any(run["status"] != "completed" for run in self.runs):
            raise ValueError(f"{directory}: every shard must be completed")
        if len(set(seeds)) != len(seeds) or len(seeds) != seed_count(environment):
            raise ValueError(f"{directory}: expected {seed_count(environment)} distinct seeds, got {sorted(seeds)}")
        if any(run["evaluation"]["steps"] != self.schedule for run in self.runs):
            raise ValueError(f"{directory}: shards disagree on the evaluation schedule")
        if arm.startswith("RARe-sh") and any(
                self.config.get(key, "behavior") != "behavior"
                for key in ("shared_target_epsilon_source", "shared_target_sigma_source")):
            raise ValueError(f"{directory}: RARe-sh must target the behavior exploration level")

    @cached_property
    def rows(self):
        """Evaluation records keyed by (seed, env_step)."""
        rows = {}
        for path, run in zip(self.paths, self.runs):
            for row in read_jsonl(path / "evaluations.jsonl"):
                key = row["seed"], row["env_step"]
                if key in rows or row["seed"] not in run["seeds"]:
                    raise ValueError(f"{path}: unexpected or duplicate evaluation {key}")
                rows[key] = row
        return rows

    @property
    def checkpoints(self):
        """Post-warm-up evaluation steps, the span of every AUC except the non-stationary one."""
        return np.asarray([step for step in self.schedule if step > self.config["warmup_steps"]])

    def curves(self, value, steps=None):
        """Apply value(row) at each checkpoint of each seed."""
        steps = self.checkpoints if steps is None else steps
        return np.asarray([[value(self.rows[seed, int(step)]) for step in steps] for seed in self.seeds],
                          dtype=np.float64)

    def __getitem__(self, name):
        """A config value that must agree across shards."""
        values = {run["config"][name] for run in self.runs}
        if len(values) != 1:
            raise ValueError(f"{self.environment}/{self.arm}: {name} differs across shards: {values}")
        return values.pop()


class Runs:
    """Cell cache over one runs directory; output holds the scores written by paper.rodr."""

    def __init__(self, root, output):
        self.root, self.output, self._cells = Path(root), Path(output), {}

    def cell(self, environment, arm):
        if (environment, arm) not in self._cells:
            self._cells[environment, arm] = Cell(self.root, environment, arm)
        return self._cells[environment, arm]

    def scores_path(self, environment, arm):
        return self.output / "rodr" / environment / f"{arm}.npz"

    def scores(self, environment, arm, name):
        """Array written by paper.rodr, restricted to post-warm-up checkpoints."""
        cell = self.cell(environment, arm)
        with np.load(self.scores_path(environment, arm)) as scores:
            if scores["seeds"].tolist() != cell.seeds or scores["steps"].tolist() != cell.schedule:
                raise ValueError(f"{environment}/{arm}: paper.rodr scores do not match the runs")
            return scores[name][:, np.isin(scores["steps"], cell.checkpoints)]


def case_returns(runs, environment, arm):
    """Per-case returns (seeds, checkpoints, cases), replayed by paper.rodr."""
    return runs.scores(environment, arm, "returns")


def grid_reward(runs, environment, arm):
    """Reward normalized per start-goal case, then averaged over cases."""
    cell = runs.cell(environment, arm)
    ceilings = [case["optimal_reward"] for case in grid_reward_ceiling(cell.spec)["cases"]]
    scores = (case_returns(runs, environment, arm) if len(ceilings) > 1
              else cell.curves(lambda row: row["reward_sum"])[..., None])
    return cell.checkpoints, normalize_rewards(scores, GRID_FLOOR, ceilings).mean(axis=-1)


def risk_per_step(runs, environment, arm):
    """Direct Risk summed over each episode, divided by its primitive steps."""
    cell = runs.cell(environment, arm)
    return cell.checkpoints, cell.curves(lambda row: row["risk_direct_sum_mean"] / row["length_mean"])


def decisions_per_step(runs, environment, arm):
    cell = runs.cell(environment, arm)
    return cell.checkpoints, cell.curves(lambda row: row["decisions_mean"] / row["length_mean"])


def risk_per_decision_auc(runs, environment, arm):
    """AUC per seed over the checkpoints with at least one valid (nonterminal) endpoint."""
    cell = runs.cell(environment, arm)
    aucs = []
    for seed in cell.seeds:
        rows = [(step, cell.rows[seed, int(step)]) for step in cell.checkpoints]
        rows = [(step, row) for step, row in rows if row["risk_direct_valid_decisions_mean"] > 0]
        values = [row["risk_direct_sum_mean"] / row["risk_direct_valid_decisions_mean"] for _, row in rows]
        aucs.append(span_auc([step for step, _ in rows], [values])[0])
    return np.asarray(aucs)


def rodr(runs, environment, arm):
    if not runs.scores_path(environment, arm).exists():
        raise FileNotFoundError(f"{environment}/{arm}: run python -m paper.rodr first")
    return runs.cell(environment, arm).checkpoints, runs.scores(environment, arm, "rodr")


def halfcheetah_expert(runs):
    """Mean final score of the three Brax PPO reference seeds."""
    paths = sorted((runs.root / PPO_REFERENCE).glob("seed*.json"))
    if len(paths) != 3:
        raise ValueError(f"expected 3 PPO references under {runs.root / PPO_REFERENCE}, got {len(paths)}")
    return float(np.mean([json.loads(path.read_text())["history"][-1]["reward"] for path in paths]))


def references(runs, environment):
    return (HALFCHEETAH_FLOOR, halfcheetah_expert(runs)) if environment == "HalfCheetah" else REFERENCES[environment]


def continuous_reward(runs, environment, arm):
    cell = runs.cell(environment, arm)
    return cell.checkpoints, normalize_to_reference(cell.curves(lambda row: row["reward_sum"]),
                                                    *references(runs, environment))


def post_change_auc(runs, environment, arm):
    """AUC of 1 - length/limit from the first change to the end of training.

    A timeout scores 0 because its length equals the limit, so the checkpoint mean
    of the per-episode score is 1 - length_mean/limit. These grids have no hazards.
    """
    cell = runs.cell(environment, arm)
    change, limit = cell.spec["shift_step"], cell.spec["max_steps"]
    steps = np.asarray([step for step in cell.schedule if step >= change])
    if steps[0] != change or steps[-1] != cell.config["total_steps"]:
        raise ValueError(f"{environment}/{arm}: evaluation does not span {change}..{steps[-1]}")
    if any(abs(row["success_rate"] + row["timeout_rate"] - 1) > 1e-9 for row in cell.rows.values()):
        raise ValueError(f"{environment}/{arm}: an episode neither succeeded nor timed out")
    return span_auc(steps, cell.curves(lambda row: 1. - row["length_mean"] / limit, steps))
