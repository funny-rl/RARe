"""Summarize matched-seed WindShift learning curves after a completed campaign."""

import argparse
import json
from pathlib import Path

import numpy as np


BASE_ALGORITHMS = ("DDQN", "TempoRL", "UTE", "TAACDiscrete", "RARe-se", "RARe-sh")
OPTIONAL_ALGORITHMS = ("RARe-se-behavior",)
CONDITIONS = (0, 2, 4)
BLOCKS = (("descending", "north"), ("descending", "south"),
          ("ascending", "north"), ("ascending", "south"))
SWITCH_STEPS = {0: (), 2: (20000, 40000), 4: (20000, 30000, 40000, 50000)}
STEPS = np.arange(4000, 60001, 1000)


def mean_sd(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "sd": float(values.std(ddof=1)),
            "n": int(values.size)}


def load_condition(campaign, shifts, algorithm):
    seeds = {}
    for layout, initial in BLOCKS:
        run_dir = campaign / f"shift{shifts}" / algorithm / f"{layout}-{initial}"
        run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        spec = run["environment"]["spec"]
        assert run["status"] == "completed" and not run["model_saved"]
        assert run["wandb"]["project"] == "RARe-WindShift"
        assert run["wandb"]["group"] == algorithm and len(run["wandb"]["runs"]) == 5
        assert run["config"]["total_steps"] == 60000 and run["config"]["warmup_steps"] == 3000
        if algorithm == "RARe-se-behavior":
            assert run["config"]["rare_q_mode"] == "separate"
            assert run["config"]["separate_target_epsilon_source"] == "behavior"
            assert run["config"]["max_epsilon"] == 1.0 and run["config"]["min_epsilon"] == 0.0
            assert run["config"]["epsilon_schedule"] == "linear"
            assert run["config"]["epsilon_decay_steps"] == 57000
        assert run["evaluation"]["episodes"] == 20
        assert spec.get("switch_steps", []) == list(SWITCH_STEPS[shifts]) and spec["observe_wind"]
        assert spec["switch_interval"] == 0
        assert spec["wind_probability"] == 0.5 and spec["initial_wind"] == initial
        expected_start = [0, 0] if layout == "descending" else [4, 0]
        assert spec["starts"] == [expected_start]
        for line in (run_dir / "evaluations.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            seeds.setdefault(row["seed"], []).append(row)
    assert set(seeds) == set(range(20)), (shifts, algorithm, sorted(seeds))
    data = {}
    for seed, rows in sorted(seeds.items()):
        rows.sort(key=lambda row: row["env_step"])
        np.testing.assert_array_equal([row["env_step"] for row in rows], STEPS)
        data[seed] = {key: np.asarray([row[key] for row in rows], dtype=np.float64)
                      for key in ("success_rate", "reward_sum", "pit_rate")}
    return data


def score_condition(data, shifts):
    metrics = {}
    for key, title in (("success_rate", "success_auc"), ("reward_sum", "reward_auc"),
                       ("pit_rate", "pit_auc")):
        # Uniform 1,000-step checkpoints; integrate each independent seed once.
        curves = np.stack([data[seed][key] for seed in range(20)])
        auc = (0.5 * curves[:, 0] + curves[:, 1:-1].sum(axis=1)
               + 0.5 * curves[:, -1]) / (len(STEPS) - 1)
        post = curves[:, STEPS >= 20000]
        post_auc = (0.5 * post[:, 0] + post[:, 1:-1].sum(axis=1)
                    + 0.5 * post[:, -1]) / (post.shape[1] - 1)
        metrics[title] = mean_sd(auc)
        metrics["post20k_" + title] = mean_sd(post_auc)
        if key == "success_rate":
            metrics["per_seed_success_auc"] = auc.tolist()
            metrics["per_seed_post20k_success_auc"] = post_auc.tolist()
            metrics["final_5k_success"] = mean_sd(curves[:, -5:].mean(axis=1))
            metrics["block_success_auc"] = {
                f"{layout}-{initial}": mean_sd(auc[5 * block:5 * (block + 1)])
                for block, (layout, initial) in enumerate(BLOCKS)
            }
            metrics["mean_curve"] = curves.mean(axis=0).tolist()
            metrics["sd_curve"] = curves.std(axis=0, ddof=1).tolist()
    if shifts:
        changes = SWITCH_STEPS[shifts]
        curves = np.stack([data[seed]["success_rate"] for seed in range(20)])
        immediate = []
        for change in changes:
            before = np.flatnonzero(STEPS < change)[-1]
            after = np.flatnonzero(STEPS >= change)[0]
            immediate.append({"change_step": change, "before_step": int(STEPS[before]),
                              "after_step": int(STEPS[after]),
                              "success_delta": mean_sd(curves[:, after] - curves[:, before])})
        metrics["immediate_switches"] = immediate
    return metrics


def render(campaign, summary, algorithms):
    lines = ["# WindShift: visible wind, matched 20-seed comparison", "",
             "Each algorithm uses seeds 0–19. Five seeds cover each of the four layout/initial-wind blocks.",
             "The wind gust probability is 0.5. Training uses 60,000 primitive steps, including 3,000 warmup steps.",
             "Frozen greedy evaluation runs 20 stochastic episodes every 1,000 steps, at steps 4,000–60,000.",
             "Exact deterministic-grid risk diagnostics are disabled for this stochastic wind environment; this report compares policy performance.",
             "No models are saved. W&B project: [RARe-WindShift](https://wandb.ai/singfor7012/RARe-WindShift).", "",
             "The 2-shift condition reverses at steps 20,000 and 40,000; the 4-shift condition at 20,000, 30,000, 40,000, and 50,000. Both have the same first change point.",
             "For each algorithm and seed, all three schedules have identical evaluation records before step 20,000; the analyzer verifies this.",
             "The primary post-change AUC is each seed's trapezoidal time average over steps 20,000–60,000, including the immediate evaluation at the first switch. Full AUC spans 4,000–60,000 as a secondary learning-speed measure.",
             "Each table reports the mean and sample standard deviation across 20 independent seeds.",
             "The standard deviation includes differences between the four matched layout/initial-wind blocks; block-specific scores are in summary.json.",
             "Success and pit AUC lie in [0, 1]; reward AUC retains return units. Final success averages each seed's last five evaluations before computing cross-seed statistics.", ""]
    if "RARe-se-behavior" in algorithms:
        lines += ["RARe-se-behavior retains the separate Expected SARSA critic and matches its skip-target epsilon to behavior epsilon (1 to 0). RARe-se keeps the independent 0.2 to 0.01 target epsilon; RARe-sh shares base Q and behavior epsilon.", ""]
    for shifts in CONDITIONS:
        lines += [f"## {shifts} wind reversals", "",
                  "| Algorithm | Post-20k success AUC | Full success AUC | Post-20k reward AUC | Final 5k success |",
                  "|---|---:|---:|---:|---:|"]
        for algorithm in algorithms:
            row = summary[str(shifts)][algorithm]
            def fmt(key, digits=3):
                item = row[key]
                return f"{item['mean']:.{digits}f} ± {item['sd']:.{digits}f}"
            lines.append(f"| {algorithm} | {fmt('post20k_success_auc')} | {fmt('success_auc')} | {fmt('post20k_reward_auc', 1)} | {fmt('final_5k_success')} |")
        lines.append("")
    lines += ["## Paired change from stationary wind", "",
              "Each difference subtracts the same seed's stationary post-20k success AUC, preserving the matched layout and initial wind. Negative values mean reduced success under shifts.",
              "", "| Algorithm | 2 shifts − stationary | 4 shifts − stationary |",
              "|---|---:|---:|"]
    for algorithm in algorithms:
        def difference(shifts):
            item = summary[str(shifts)][algorithm]["delta_vs_stationary"]
            return f"{item['mean']:+.3f} ± {item['sd']:.3f}"
        lines.append(f"| {algorithm} | {difference(2)} | {difference(4)} |")
    lines.append("")
    if "RARe-se-behavior" in algorithms:
        lines += ["## RARe target-epsilon ablation", "",
                  "Paired differences in post-20k success AUC use matching seeds and environment blocks. Positive values favor the first named group.",
                  "", "| Contrast | 0 shifts | 2 shifts | 4 shifts |",
                  "|---|---:|---:|---:|"]
        for first, second in (("RARe-se-behavior", "RARe-se"),
                              ("RARe-sh", "RARe-se-behavior")):
            cells = []
            for shifts in CONDITIONS:
                left = np.asarray(summary[str(shifts)][first]["per_seed_post20k_success_auc"])
                right = np.asarray(summary[str(shifts)][second]["per_seed_post20k_success_auc"])
                item = mean_sd(left - right)
                cells.append(f"{item['mean']:+.3f} ± {item['sd']:.3f}")
            lines.append(f"| {first} − {second} | " + " | ".join(cells) + " |")
        lines.append("")
    (campaign / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(campaign, summary, algorithms):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True, layout="constrained")
    for axis, shifts in zip(axes, CONDITIONS):
        for algorithm in algorithms:
            row = summary[str(shifts)][algorithm]
            mean = np.asarray(row["mean_curve"])
            axis.plot(STEPS / 1000, mean, label=algorithm, linewidth=1.7)
        for index in range(1, shifts + 1):
            axis.axvline(SWITCH_STEPS[shifts][index - 1] / 1000, color="gray", linestyle=":", linewidth=0.9)
        axis.set(xlabel="Training steps (thousands)", title=f"{shifts} wind reversals", xlim=(4, 60), ylim=(0, 1))
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Greedy evaluation success rate")
    axes[-1].legend(loc="lower right", fontsize=8)
    fig.savefig(campaign / "success_curves.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", type=Path)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    algorithms = list(BASE_ALGORITHMS)
    for algorithm in OPTIONAL_ALGORITHMS:
        run_files = [campaign / f"shift{shifts}" / algorithm / f"{layout}-{initial}" / "run.json"
                     for shifts in CONDITIONS for layout, initial in BLOCKS]
        if any(path.exists() for path in run_files):
            if not all(path.exists() for path in run_files):
                raise ValueError(f"{algorithm} has incomplete WindShift conditions")
            algorithms.append(algorithm)
    data = {shifts: {algorithm: load_condition(campaign, shifts, algorithm)
                     for algorithm in algorithms} for shifts in CONDITIONS}
    before_first_shift = STEPS < 20000
    for algorithm in algorithms:
        for seed in range(20):
            for key in ("success_rate", "reward_sum", "pit_rate"):
                for shifts in (2, 4):
                    np.testing.assert_array_equal(
                        data[0][algorithm][seed][key][before_first_shift],
                        data[shifts][algorithm][seed][key][before_first_shift],
                    )
    summary = {str(shifts): {algorithm: score_condition(data[shifts][algorithm], shifts)
                             for algorithm in algorithms} for shifts in CONDITIONS}
    for shifts in (2, 4):
        for algorithm in algorithms:
            shifted = np.asarray(summary[str(shifts)][algorithm]["per_seed_post20k_success_auc"])
            stationary = np.asarray(summary["0"][algorithm]["per_seed_post20k_success_auc"])
            summary[str(shifts)][algorithm]["delta_vs_stationary"] = mean_sd(shifted - stationary)
    (campaign / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    render(campaign, summary, algorithms)
    plot(campaign, summary, algorithms)
    print(campaign / "report.md")


if __name__ == "__main__":
    main()
