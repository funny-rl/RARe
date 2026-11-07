"""Draw Figures 1, 4, 5, 6 and 7 of the paper from finished runs.

Learning curves show post-warm-up checkpoints: the line is the IQM over seeds, the
band its 95% percentile bootstrap CI, both smoothed by an edge-padded moving average
of five checkpoints. Run paper.rodr first: Figures 4 and 6 use its scores.

Usage: python -m paper.figures --runs runs --output results
Outputs: <output>/figures/figureNN_<topic>.pdf
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np

from paper import data
from paper.data import CONTINUOUS, CONTINUOUS_ARMS, GRID_ARMS, GRID_TITLES, GRIDS, Runs
from rare.envs.grid.layouts import make_grid_env
from rare.metrics.stats import bootstrap_ci, iqm

# Embed TrueType fonts; camera-ready checks reject Type 3 fonts.
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

COLORS = {"DDQN": "#666666", "DDPG": "#666666", "TAAC": "#e07b39", "TempoRL": "#463dbe", "UTE": "#35af35",
          "RARe-sh": "#f2b705", "RARe-se": "#c53a3a"}
MARKERS = {"DDQN": "o", "DDPG": "o", "TAAC": "v", "TempoRL": "^", "UTE": "s", "RARe-sh": "P", "RARe-se": "D"}
SMOOTHING_WINDOW = 5
BASE_LAYOUTS = ("CliffWalking", "Bridge", "ZigZag")
START, GOAL, PIT, GRID_LINE = "#2ecc71", "#3498db", "#8c2323", "#c8c8c8"


def smooth_curve(curve, window=SMOOTHING_WINDOW):
    if window <= 1 or len(curve) < window:
        return curve
    padded = np.pad(curve, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def plot_curve(ax, steps, curves, arm):
    """Draw one arm's (seeds, steps) curves: smoothed IQM and 95% bootstrap band."""
    center, (lower, upper) = iqm(curves), bootstrap_ci(curves)
    emphasize = arm.startswith("RARe")
    zorder = 5 if emphasize else 3
    line, = ax.plot(steps, smooth_curve(center), color=COLORS[arm], linewidth=2.7 if emphasize else 2.2,
                    marker=MARKERS[arm], markersize=9, markerfacecolor=COLORS[arm], markeredgecolor="black",
                    markeredgewidth=.8, markevery=max(1, len(steps) // 6), zorder=zorder)
    ax.fill_between(steps, smooth_curve(lower), smooth_curve(upper), color=COLORS[arm],
                    alpha=.20 if emphasize else .14, linewidth=0, zorder=zorder - 1)
    return line


def style_axis(ax, steps, title):
    ax.set_title(title, fontsize=20, fontweight="bold", pad=8)
    ax.set_xlim(steps[0], steps[-1])
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5, steps=[1, 2, 5, 10]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1e4:.1f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1f}"))
    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.grid(True, which="major", alpha=.6, linestyle="-", linewidth=.8, color="gray")


def draw_panels(axes, runs, metric, environments, arms, titles):
    """Fill one axis per environment and return the legend handles."""
    handles = {}
    for ax, environment in zip(axes, environments):
        for arm in arms:
            steps, curves = metric(runs, environment, arm)
            handles.setdefault(arm, plot_curve(ax, steps, curves, arm))
        style_axis(ax, steps, titles[environment])
    return handles


def finish(fig, handles, ylabel, legend_height, path):
    fig.supylabel(ylabel, x=.01, fontweight="bold", fontsize=20)
    fig.supxlabel(r"$\mathbf{Training\ Steps\ (\times 10^4)}$", y=.01, fontsize=18)
    fig.legend(handles.values(), handles.keys(), loc="upper center", bbox_to_anchor=(.5, legend_height),
               ncol=len(handles), frameon=False, fontsize=18)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def grid_figure(runs, metric, ylabel, arms, path):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    handles = draw_panels(axes.flat, runs, metric, GRIDS, arms, GRID_TITLES)
    # Fixed margins (no tight_layout) keep the axes at the same place in every figure.
    fig.subplots_adjust(left=.075, right=.99, bottom=.1, top=.88, wspace=.2, hspace=.32)
    finish(fig, handles, ylabel, 1., path)


def continuous_figure(runs, path):
    fig, axes = plt.subplots(1, len(CONTINUOUS), figsize=(16, 4.6))
    handles = draw_panels(axes, runs, data.continuous_reward, CONTINUOUS, CONTINUOUS_ARMS,
                          {environment: environment for environment in CONTINUOUS})
    fig.subplots_adjust(left=.06, right=.99, bottom=.19, top=.77, wspace=.2)
    finish(fig, handles, "Reward", 1.03, path)


def draw_cells(ax, cells, color):
    ax.add_collection(PatchCollection([Rectangle((col, row), 1, 1) for row, col in cells],
                                      facecolor=color, edgecolor=GRID_LINE, linewidth=.8))


def draw_layout(ax, spec, title):
    rows, cols = spec.shape
    every_cell = [(row, col) for row in range(rows) for col in range(cols)]
    for cells, color in ((every_cell, "white"), (spec.pits, PIT), (spec.starts, START), (spec.goals, GOAL)):
        draw_cells(ax, cells, color)
    for cells, label in ((spec.starts, "S"), (spec.goals, "G")):
        for row, col in cells:
            ax.text(col + .5, row + .5, label, color="white", fontsize=14, fontweight="bold",
                    ha="center", va="center_baseline")
    ax.set(xlim=(0, cols), ylim=(rows, 0), aspect="equal", xticks=[], yticks=[])
    ax.set_title(title, fontsize=15, fontweight="bold", pad=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.5)


def gridworlds_figure(path):
    # Fixed margins: tight_layout ignores the equal aspect and overlaps the rows.
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2))
    fig.subplots_adjust(left=.01, right=.99, top=.94, bottom=.01, wspace=.04, hspace=.18)
    for row, random in zip(axes, (False, True)):
        for ax, name in zip(row, BASE_LAYOUTS):
            draw_layout(ax, make_grid_env(name, random=random).spec, GRID_TITLES[f"R{name}" if random else name])
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    runs = Runs(args.runs, args.output)
    output = args.output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    gridworlds_figure(output / "figure01_gridworlds.pdf")
    grid_figure(runs, data.grid_reward, "Reward", GRID_ARMS, output / "figure04_grid_reward.pdf")
    grid_figure(runs, data.risk_per_step, "Risk", GRID_ARMS[1:], output / "figure05_grid_risk.pdf")
    grid_figure(runs, data.rodr, "RODR", GRID_ARMS, output / "figure06_grid_rodr.pdf")
    continuous_figure(runs, output / "figure07_continuous_reward.pdf")
    print(output)


if __name__ == "__main__":
    main()
