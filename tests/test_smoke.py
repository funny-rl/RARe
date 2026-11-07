"""Short end-to-end runs on the CPU: a run directory appears with evaluation rows."""

import json
import subprocess
import sys

import pytest


@pytest.mark.parametrize("environment,arm,stop", (
    ("CliffWalking", "DDQN", 2300), ("CliffWalking", "RARe-se", 2300), ("WindShift", "UTE", 3000),
    ("Pendulum", "TAAC", 2300), ("MountainCar", "RARe-sh", 5300),
))
def test_run_writes_evaluations(tmp_path, environment, arm, stop):
    output = tmp_path / "run"
    subprocess.run([sys.executable, "-m", "experiments.run", "--env", environment, "--arm", arm,
                    "--seeds", "0:2", "--cpu", "--stop-step", str(stop), "--output", str(output)],
                   check=True)
    assert json.loads((output / "run.json").read_text())["status"] == "completed"
    rows = (output / "evaluations.jsonl").read_text().splitlines()
    assert rows and {json.loads(row)["seed"] for row in rows} == {0, 1}
