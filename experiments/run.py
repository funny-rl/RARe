"""Run one (environment, arm) cell of the paper: python -m experiments.run --help."""

import argparse
import os
from pathlib import Path


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument("--env", required=True, help="for example CliffWalking, WindShift, MountainCar")
    result.add_argument("--arm", required=True, help="for example RARe-se, TempoRL, TAAC")
    result.add_argument("--seeds", help="start:stop; default: the seeds used in the paper")
    result.add_argument("--output", type=Path, help="new run directory; default: runs/<env>/<arm>")
    result.add_argument("--gpu", type=int, default=0, help="CUDA device index")
    result.add_argument("--cpu", action="store_true", help="run JAX on the CPU")
    result.add_argument("--stop-step", type=int,
                        help="stop at the last checkpoint not after this step; schedules still use "
                             "the full budget, so the run is a prefix of the full run")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    from experiments.config import parse_seeds, resolve
    cell = resolve(args.env, args.arm)
    safety = cell.settings.get("safety", False)
    # Device visibility must be set before JAX is imported.
    if args.cpu:
        os.environ["JAX_PLATFORMS"] = "cpu"
    else:
        # The Safety-Gymnasium loop calls back to the host from compiled code, which needs the CPU backend.
        os.environ["JAX_PLATFORMS"] = "cuda,cpu" if safety else "cuda"
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if args.seeds is None and len(cell.seeds) != 1:
        raise SystemExit(f"{args.env} runs in seed shards {cell.seeds}: pass --seeds and --output, "
                         "or use python -m experiments.reproduce")
    seeds = parse_seeds(args.seeds or cell.seeds[0])
    output = args.output or Path("runs") / args.env / args.arm
    if cell.suite == "continuous":
        from experiments import continuous, safety as safety_module
        module = safety_module if safety else continuous
    else:
        from experiments import grid as module
    module.train(cell, seeds, output, args.stop_step)


if __name__ == "__main__":
    main()
