"""Every cell of the paper resolves, and its budgets match Tables 7, 8 and 9."""

import json
from types import SimpleNamespace

import pytest

from experiments import reproduce
from experiments.config import arms as all_arms, checkpoints, resolve
from experiments.reproduce import EXPERIMENTS


def cells():
    for environments, arms in EXPERIMENTS.values():
        for environment in environments:
            for arm in arms or all_arms(environment):
                yield environment, arm


@pytest.mark.parametrize("environment,arm", sorted(set(cells())))
def test_cell_resolves(environment, arm):
    cell = resolve(environment, arm)
    assert cell.config["total_steps"] > cell.config["warmup_steps"]
    assert checkpoints(cell.config["total_steps"], cell.config["warmup_steps"], cell.evaluation)[-1] \
        == cell.config["total_steps"]


@pytest.mark.parametrize("environment,steps,warmup,replay,skip_replay", (
    ("CliffWalking", 15000, 2000, 15000, 45000), ("Bridge", 40000, 3000, 40000, 120000),
    ("ZigZag", 50000, 3000, 50000, 150000), ("RCliffWalking", 50000, 2000, 50000, 150000),
    ("RBridge", 150000, 3000, 100000, 300000), ("RZigZag", 200000, 3000, 100000, 300000),
    ("MountainCar", 50000, 5000, 50000, 120000), ("PointMaze", 200000, 5000, 200000, 305000),
    ("Goal", 52000, 2000, 52000, 100000), ("Button", 65000, 5000, 65000, 120000),
    ("Pendulum", 17000, 2000, 17000, 50000), ("HalfCheetah", 3000000, 50000, 100000, 300000),
    ("WindShift", 40000, 2000, 40000, 120000),
))
def test_budgets_match_the_paper(environment, steps, warmup, replay, skip_replay):
    config = resolve(environment, "RARe-se").config
    assert (config["total_steps"], config["warmup_steps"], config["replay_capacity"],
            config["skip_replay_capacity"]) == (steps, warmup, replay, skip_replay)


def test_rare_sh_has_no_separate_schedule_and_rare_se_does():
    assert "max_eval_epsilon" not in resolve("ZigZag", "RARe-sh").config
    assert resolve("ZigZag", "RARe-se").config["max_eval_epsilon"] == .3


def test_taac_uses_the_shared_network_and_batch_sizes():
    config = resolve("PointMaze", "TAAC").config
    assert (config["hidden_dim"], config["batch_size"], config["learning_rate"],
            config["compare_horizon"]) == (128, 64, .001, 3)


def test_reproduce_restarts_incomplete_runs_and_reports_failures(tmp_path, monkeypatch):
    for arm, status in (("TAAC", "failed"), ("RARe-se", "completed")):
        (tmp_path / "Pendulum" / arm).mkdir(parents=True)
        (tmp_path / "Pendulum" / arm / "run.json").write_text(json.dumps({"status": status}))
    commands = []
    monkeypatch.setattr(reproduce.subprocess, "run",
                        lambda command, **_: commands.append(command) or SimpleNamespace(returncode=1))
    with pytest.raises(SystemExit, match="failed runs"):
        reproduce.main(["pendulum", "--runs", str(tmp_path)])
    assert len(commands) == 1 and "TAAC" in commands[0]
    assert not (tmp_path / "Pendulum" / "TAAC").exists()


def test_reproduce_requires_the_safety_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("RARE_SAFETY_PYTHON", raising=False)
    with pytest.raises(SystemExit, match="RARE_SAFETY_PYTHON"):
        reproduce.main(["continuous", "--runs", str(tmp_path)])
