"""Offline AUC comparison, gated on a completed environment cohort."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from diagnostics.auc import auc_statistics, grid_reward_ceiling, normalize_rewards, seed_auc


ALGORITHMS = ("DDQN", "TempoRL", "UTE", "RARe-sh", "RARe-se")
SUPPORTED_ALGORITHMS = (*ALGORITHMS, "RARe-sh-eval", "TAACDiscrete")


def _integer(value, name):
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def _algorithm(metadata):
    algorithm = metadata["algorithm"]
    if algorithm == "RARe":
        mode = metadata["config"].get("rare_q_mode", "separate")
        if mode not in ("shared", "separate"):
            raise ValueError(f"unknown RARe mode: {mode}")
        if mode == "separate":
            return "RARe-se"
        return ("RARe-sh-eval" if metadata["config"].get("shared_target_epsilon_source") == "evaluation"
                else "RARe-sh")
    if algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"unsupported algorithm: {algorithm}")
    return algorithm


def load_completed_cohort(paths, *, algorithms=ALGORITHMS):
    """Read each log once; reject incomplete, mismatched or duplicated seeds."""
    algorithms = tuple(algorithms)
    if not algorithms or len(set(algorithms)) != len(algorithms) or set(algorithms) - set(SUPPORTED_ALGORITHMS):
        raise ValueError("algorithms must be distinct supported comparison labels")
    paths = [Path(path).resolve() for path in paths]
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate run directory")
    entries = []
    for path in paths:
        content = (path / "run.json").read_bytes()
        metadata = json.loads(content)
        if metadata.get("status") != "completed":
            raise ValueError(f"all experiments must be completed before analysis: {path}")
        label = _algorithm(metadata)
        if label not in algorithms:
            raise ValueError(f"unexpected algorithm {label}; declare the complete cohort with --algorithms")
        entries.append((path, metadata, label, hashlib.sha256(content).hexdigest()))
    missing = set(algorithms) - {entry[2] for entry in entries}
    if missing:
        raise ValueError(f"environment cohort is incomplete; missing algorithms: {', '.join(sorted(missing))}")
    environment, protocol, checkpoint_steps = None, None, None
    curves = {label: {} for label in algorithms}
    case_curves = {label: {} for label in algorithms}
    cases = None
    sources = []
    for path, metadata, label, metadata_hash in entries:
        config, evaluation = metadata["config"], metadata["evaluation"]
        total = _integer(config["total_steps"], "total_steps")
        warmup = _integer(config["warmup_steps"], "warmup_steps")
        if not 0 <= warmup < total:
            raise ValueError(f"no post-warmup learning interval: {path}")
        seeds = metadata["seeds"]
        if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
            raise ValueError(f"invalid seed manifest: {path}")
        counters = metadata.get("summary", {}).get("per_seed", [])
        if (len(counters) != len(seeds) or {row["seed"] for row in counters} != set(seeds)
                or any(row["env_steps"] != total for row in counters)):
            raise ValueError(f"seed training budgets are incomplete: {path}")
        if evaluation.get("enabled") is not True or evaluation.get("policy") != "greedy":
            raise ValueError(f"completed frozen greedy evaluations are required: {path}")
        schedule = evaluation["steps"]
        if (not schedule or any(type(step) is not int or not 0 < step <= total for step in schedule)
                or schedule != sorted(set(schedule)) or schedule[-1] != total):
            raise ValueError(f"invalid evaluation schedule: {path}")
        retained = tuple(step for step in schedule if step > warmup)
        schedule_set = set(schedule)
        if len(retained) < 2:
            raise ValueError(f"at least two post-warmup evaluation points are required: {path}")
        current_protocol = {"total_steps": total, "warmup_steps": warmup,
                            "episodes": evaluation["episodes"], "policy": evaluation["policy"],
                            "reset_mode": evaluation.get("reset_mode", "sampled"),
                            "case_order": evaluation.get("case_order"),
                            "unit": evaluation.get("unit", "primitive_steps_per_seed")}
        if environment is None:
            environment, protocol, checkpoint_steps = metadata["environment"], current_protocol, retained
            spec = environment.get("spec", {})
            cases = [{"start": start, "goal": goal} for start in spec.get("starts", [])
                     for goal in spec.get("goals", [])] or [{}]
            if len(cases) > 1 and (protocol["reset_mode"] != "all_start_goal_pairs"
                                  or protocol["episodes"] != len(cases)
                                  or protocol["case_order"] != "starts_major_goals_minor"):
                raise ValueError("case-normalized AUC requires exhaustive start-goal evaluation in a fixed case order")
        if (metadata["environment"] != environment or current_protocol != protocol
                or retained != checkpoint_steps):
            raise ValueError("all algorithms must share the environment, budget, warmup and post-warmup evaluation protocol/grid")
        if set(seeds) & set(curves[label]):
            raise ValueError(f"duplicate {label} seeds across input runs")
        values = {seed: {} for seed in seeds}
        seen = set()
        log_hash = hashlib.sha256()
        with (path / "evaluations.jsonl").open("rb") as stream:
            for line_number, line in enumerate(stream, 1):
                log_hash.update(line)
                row = json.loads(line)
                seed = _integer(row["seed"], "seed")
                step = _integer(row["env_step"], "env_step")
                if seed not in values or step not in schedule_set or (seed, step) in seen:
                    raise ValueError(f"unexpected or duplicate evaluation at {path}:{line_number}")
                seen.add((seed, step))
                if row["episodes"] != evaluation["episodes"]:
                    raise ValueError(f"evaluation episode counts differ: {path}:{line_number}")
                # Older runs called the same undiscounted episode average return_mean.
                score = row.get("reward_sum", row.get("return_mean"))
                if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                    raise ValueError(f"finite evaluation reward_sum/return_mean required: {path}:{line_number}")
                if "reward_sum" in row and "return_mean" in row and row["reward_sum"] != row["return_mean"]:
                    raise ValueError(f"conflicting legacy/current reward fields: {path}:{line_number}")
                if step > warmup:
                    values[seed][step] = float(score)
        for seed, observations in values.items():
            if set(observations) != set(checkpoint_steps):
                raise ValueError(f"missing post-warmup evaluation points for {label} seed {seed}")
            curves[label][seed] = [observations[step] for step in checkpoint_steps]
        episode_hash = None
        if len(cases) > 1:
            # Preserve the case axis until after clipping and integration.
            observations = np.full((len(seeds), len(retained), len(cases)), np.nan)
            seed_indices = {seed: index for index, seed in enumerate(seeds)}
            step_indices = {step: index for index, step in enumerate(retained)}
            episode_hash = hashlib.sha256()
            with (path / "evaluation_episodes.jsonl").open("rb") as stream:
                for line_number, line in enumerate(stream, 1):
                    episode_hash.update(line)
                    row = json.loads(line)
                    seed = _integer(row["seed"], "seed")
                    step = _integer(row["env_step"], "env_step")
                    if seed not in seed_indices or step not in schedule_set:
                        raise ValueError(f"unexpected case evaluation: {path}:{line_number}")
                    if step <= warmup:
                        continue
                    case = _integer(row["case_index"], "case_index")
                    if (not 0 <= case < len(cases) or row.get("start") != cases[case]["start"]
                            or row.get("goal") != cases[case]["goal"]):
                        raise ValueError(f"case index/coordinates disagree: {path}:{line_number}")
                    score = row.get("reward_sum", row.get("return"))
                    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                        raise ValueError(f"finite case reward_sum/return required: {path}:{line_number}")
                    if "reward_sum" in row and "return" in row and row["reward_sum"] != row["return"]:
                        raise ValueError(f"conflicting case reward fields: {path}:{line_number}")
                    index = (seed_indices[seed], step_indices[step], case)
                    if np.isfinite(observations[index]):
                        raise ValueError(f"duplicate case evaluation: {path}:{line_number}")
                    observations[index] = score
            if not np.isfinite(observations).all():
                raise ValueError(f"missing post-warmup case evaluations: {path}")
            for index, seed in enumerate(seeds):
                if not np.allclose(observations[index].mean(axis=-1), curves[label][seed], rtol=1e-6, atol=1e-5):
                    raise ValueError(f"case rewards disagree with aggregate evaluation: {path}, seed {seed}")
                case_curves[label][seed] = observations[index]
        else:
            case_curves[label].update({seed: np.asarray(curves[label][seed])[:, None] for seed in seeds})
        sources.append({"path": str(path), "algorithm": label, "seeds": seeds,
                        "run_json_sha256": metadata_hash, "evaluations_sha256": log_hash.hexdigest(),
                        "evaluation_episodes_sha256": episode_hash.hexdigest() if episode_hash is not None else None,
                        "training_source_sha256": metadata.get("source", {}).get("sha256")})
    seed_ids = sorted(curves[algorithms[0]])
    if len(seed_ids) < 2 or any(set(curves[label]) != set(seed_ids) for label in algorithms):
        raise ValueError("all algorithms must contain the same seed set, with at least two seeds")
    return {"environment": environment, "protocol": protocol, "steps": list(checkpoint_steps),
            "seeds": seed_ids, "cases": cases,
            "scores": {label: np.asarray([case_curves[label][seed] for seed in seed_ids], np.float64)
                                         for label in algorithms}, "sources": sources}


def analyze_cohort(cohort, *, reward_min=-100., reward_max=None):
    """Normalize/integrate each case, then summarize the complete seed means."""
    steps = cohort["steps"]
    duration = steps[-1] - steps[0]
    ceiling = None
    case_count = len(cohort["cases"])
    if case_count > 1 and reward_max is not None:
        raise ValueError("random grids require exact per-case ceilings; omit --reward-max")
    if reward_max is None:
        environment = cohort["environment"]
        spec = environment["spec"]
        if environment["name"] not in ("CliffWalking", "Bridge", "ZigZag"):
            raise ValueError("automatic reward ceiling is available for RARe grids; supply --reward-max for other environments")
        ceiling = grid_reward_ceiling(spec)
        reward_max = [case["optimal_reward"] for case in ceiling["cases"]]
    upper = np.broadcast_to(np.asarray(reward_max, dtype=np.float64), (case_count,))
    results, per_seed, per_case = {}, [], []
    for label, scores in cohort["scores"].items():
        normalized = normalize_rewards(scores, reward_min, upper)
        case_values = seed_auc(steps, normalized.transpose(0, 2, 1).reshape(-1, len(steps)))
        case_values = case_values.reshape(len(cohort["seeds"]), case_count) / duration
        values = case_values.mean(axis=-1)
        raw_values = seed_auc(steps, scores.mean(axis=-1))
        statistics = auc_statistics(values)
        results[label] = {"seed_count": len(values), "auc": statistics,
                          "lower_clipped_points": int((scores < reward_min).sum()),
                          "evaluation_points": int(scores.size),
                          "lower_clipped_fraction": float((scores < reward_min).mean())}
        per_seed.extend({"algorithm": label, "seed": seed, "auc": float(value), "raw_reward_auc": float(raw_value),
                         "lower_clipped_points": int((scores[index] < reward_min).sum())}
                        for index, (seed, value, raw_value) in enumerate(zip(cohort["seeds"], values, raw_values)))
        per_case.extend({"algorithm": label, "seed": seed, "case_index": case_index,
                         "start": case.get("start"), "goal": case.get("goal"),
                         "reward_min": float(reward_min), "reward_max": float(upper[case_index]),
                         "auc": float(case_values[seed_index, case_index])}
                        for seed_index, seed in enumerate(cohort["seeds"])
                        for case_index, case in enumerate(cohort["cases"]))
    return {"schema_version": 2, "status": "completed", "environment": cohort["environment"],
            "evaluation_protocol": cohort["protocol"], "seeds": cohort["seeds"],
            "integration": {"metric": "per-case undiscounted episode reward (reward_sum; legacy return)",
                            "method": "trapezoidal, linear between observed checkpoints",
                            "axis": "original primitive env_step", "warmup_excluded": True,
                            "start_step": steps[0], "end_step": steps[-1], "duration": duration,
                            "checkpoint_steps": steps, "extrapolation": False,
                            "formula": "mean_cases[integral clip((case_reward - L)/(U_case-L), 0, 1) d(env_step) / actual_duration]",
                            "reward_min": float(reward_min), "reward_max": upper.tolist(),
                            "case_count": case_count, "case_weighting": "uniform within each seed, before seed statistics",
                            "clipping": "clip below reward_min before integration; reject rewards above theoretical ceiling",
                            "ceiling": ceiling or {"method": "explicit common --reward-max"},
                            "direction": "0..1; higher is better"},
            "statistics": {"input": "one mean of normalized case AUCs per independently trained seed",
                           "std_ddof": 1, "metrics": ["mean", "std"],
                           "seed_selection": "all completed declared seeds; no trimming",
                           "scope": "one environment; independent training seeds are the statistical unit"},
            "results": results, "per_seed": per_seed, "per_case": per_case, "inputs": cohort["sources"],
            "runtime": {"python": sys.version, "numpy": np.__version__},
            "analysis_sources": {str(path.relative_to(Path(__file__).resolve().parents[1])): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in (Path(__file__).resolve(), Path(__file__).resolve().parents[1] / "diagnostics/auc.py")}}


def plot_auc(report, output):
    """Optionally show the mean and sample standard deviation across seeds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(report["results"])
    fig, axis = plt.subplots(figsize=(7, 4.8), layout="constrained")
    colors = plt.get_cmap("tab10")(np.arange(len(labels)))
    for index, label in enumerate(labels):
        values = report["results"][label]["auc"]
        axis.errorbar(values["mean"], index, xerr=values["std"], fmt="o", capsize=4, color=colors[index])
    axis.set_title("AUC mean ± sample standard deviation")
    axis.set_yticks(np.arange(len(labels)), labels)
    axis.set_xlabel("Normalized AUC (0–1)")
    axis.grid(axis="x", alpha=.25)
    axis.invert_yaxis()
    window = report["integration"]
    name = ("R" if report["environment"]["random"] else "") + report["environment"]["name"]
    fig.suptitle(f"{name}: {window['start_step']:,}–{window['end_step']:,} steps, "
                 f"{len(report['seeds'])} seeds, warmup excluded")
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"auc.{suffix}", dpi=180)
    plt.close(fig)


def write_report(report, output, *, plots=False):
    """Publish a new analysis directory; never change experiment records."""
    started = time.perf_counter()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for label, result in report["results"].items():
        rows.append({"algorithm": label, "seed_count": result["seed_count"],
                     **{f"auc_{key}": value for key, value in result["auc"].items()},
                     "lower_clipped_points": result["lower_clipped_points"],
                     "evaluation_points": result["evaluation_points"]})
    for name, records in (("auc_summary", rows), ("seed_auc", report["per_seed"]), ("case_auc", report["per_case"])):
        with (output / f"{name}.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    if plots:
        plot_auc(report, output)
    report = dict(report, generated_at=datetime.now(timezone.utc).isoformat(),
                  export_seconds=time.perf_counter() - started)
    (output / "auc_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline AUC analysis after every algorithm in an environment finishes.")
    parser.add_argument("runs", nargs="+", type=Path, help="explicit completed run directories, including all seed shards")
    parser.add_argument("--output", required=True, type=Path, help="new analysis directory; refuses overwriting")
    parser.add_argument("--reward-min", type=float, default=-100., help="common reward floor (default: -100); lower rewards clip to zero")
    parser.add_argument("--reward-max", type=float, help="single-case ceiling override; random grids always use exact per-case optima")
    parser.add_argument("--algorithms", nargs="+", choices=SUPPORTED_ALGORITHMS, default=ALGORITHMS,
                        help="complete comparison cohort; default: all five algorithms/RARe variants")
    plots = parser.add_mutually_exclusive_group()
    plots.add_argument("--plot", dest="plots", action="store_true", help="also export a mean/std figure; disabled by default")
    plots.add_argument("--no-plot", dest="plots", action="store_false", help="export data only (default)")
    parser.set_defaults(plots=False)
    args = parser.parse_args(argv)
    try:
        started = time.perf_counter()
        cohort = load_completed_cohort(args.runs, algorithms=args.algorithms)
        loaded = time.perf_counter()
        report = analyze_cohort(cohort, reward_min=args.reward_min, reward_max=args.reward_max)
        report["timing"] = {"loading_seconds": loaded - started,
                            "statistics_seconds": time.perf_counter() - loaded}
        output = write_report(report, args.output, plots=args.plots)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    window = report["integration"]
    print(f"Normalized AUC [0,1]: floor {window['reward_min']}, "
          f"case ceilings {min(window['reward_max'])}..{max(window['reward_max'])} ({window['case_count']} cases); "
          f"window {window['start_step']}..{window['end_step']} env_step; warmup excluded")
    print("Algorithm    Seeds    AUC mean ± sample std")
    for label, result in report["results"].items():
        stats = result["auc"]
        print(f"{label:<12} {result['seed_count']:>3}      {stats['mean']:.6g} ± {stats['std']:.6g}")
    print(f"Analysis: {output}")
    return output


if __name__ == "__main__":
    main()
