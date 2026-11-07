"""Run every cell of a paper experiment across GPUs: python -m experiments.reproduce --help."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys

from experiments.config import arms, resolve

# Paper experiment -> environments and arms; None means every arm of the environment's suite.
GRID = ("CliffWalking", "Bridge", "ZigZag", "RCliffWalking", "RBridge", "RZigZag")
MAIN_GRID_ARMS = ("DDQN", "TempoRL", "UTE", "RARe-sh", "RARe-se")
ABLATION_ARMS = ("RARe-sh-AE-SG", "RARe-sh-AE-SE", "RARe-se-AE-SG", "RARe-se-AE-SE", "RARe-se-AG-SG")
EXPERIMENTS = {
    "grid": (GRID, MAIN_GRID_ARMS),
    "ablation": (GRID, ABLATION_ARMS),
    "nonstationary": (("WindShift", "Random", "Episodic", "Shift1", "Shift3", "Shift5"), None),
    "continuous": (("MountainCar", "PointMaze", "Goal", "Button"), None),
    "pendulum": (("Pendulum",), ("TAAC", "RARe-se")),
    "halfcheetah": (("HalfCheetah",), ("TAAC", "RARe-se")),
}


def jobs(names, root):
    """Yield (arguments, output, needs Safety-Gymnasium) for every run that has not completed."""
    for name in names:
        environments, selected = EXPERIMENTS[name]
        for environment in environments:
            for arm in selected or arms(environment):
                cell = resolve(environment, arm)
                for seeds in cell.seeds:
                    output = root / environment / arm
                    if len(cell.seeds) > 1:
                        output = output / f"seeds-{seeds.replace(':', '-')}"
                    status = output / "run.json"
                    if status.exists() and json.loads(status.read_text())["status"] == "completed":
                        continue
                    yield (["--env", environment, "--arm", arm, "--seeds", seeds, "--output", str(output)],
                           output, cell.settings.get("safety", False))


def worker(gpu, pending, extra):
    """Run queued jobs on one GPU and return the outputs that failed."""
    failed = []
    while True:
        try:
            arguments, output, _ = pending.get_nowait()
        except queue.Empty:
            return failed
        # Runs have no resume, so an interrupted run starts again.
        shutil.rmtree(output, ignore_errors=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        log = output.with_name(output.name + ".log")
        with log.open("w", encoding="utf-8") as stream:
            code = subprocess.run([sys.executable, "-m", "experiments.run", "--gpu", str(gpu), *arguments, *extra],
                                  stdout=stream, stderr=subprocess.STDOUT).returncode
        print(f"{'done' if code == 0 else 'FAILED'} {output} (GPU {gpu})", flush=True)
        if code:
            failed.append(log)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiments", nargs="+", choices=(*EXPERIMENTS, "all"))
    parser.add_argument("--gpus", default="0", help="comma-separated CUDA indices")
    parser.add_argument("--per-gpu", type=int, default=1, help="concurrent runs per GPU")
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--stop-step", type=int, help="short pipeline check; use with a separate --runs")
    parser.add_argument("--dry-run", action="store_true", help="print the commands and exit")
    args = parser.parse_args(argv)
    # HalfCheetah needs its own conda environment, so "all" leaves it out.
    names = (tuple(name for name in EXPERIMENTS if name != "halfcheetah") if "all" in args.experiments
             else tuple(args.experiments))
    planned = list(jobs(names, args.runs))
    if args.dry_run:
        for arguments, *_ in planned:
            print("python -m experiments.run", *arguments)
        return
    if any(safety for *_, safety in planned) and "RARE_SAFETY_PYTHON" not in os.environ:
        raise SystemExit("Goal and Button need RARE_SAFETY_PYTHON; see Installation in README.md")
    pending = queue.Queue()
    for job in planned:
        pending.put(job)
    gpus = [int(gpu) for gpu in args.gpus.split(",")] * args.per_gpu
    extra = ["--stop-step", str(args.stop_step)] if args.stop_step else []
    with ThreadPoolExecutor(len(gpus)) as pool:
        failed = [log for logs in pool.map(lambda gpu: worker(gpu, pending, extra), gpus) for log in logs]
    if failed:
        raise SystemExit("failed runs:\n" + "\n".join(map(str, failed)))


if __name__ == "__main__":
    main()
