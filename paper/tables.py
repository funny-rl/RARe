"""Write Tables 1-18 and the paired comparisons of the paper from finished runs.

Every setting is read from run.json, so a table cannot drift from the runs behind
the results, and parameter counts initialize the real networks. Run paper.rodr
first: Table 3 and every randomized Grid reward use its scores.

Usage: python -m paper.tables --runs runs --output results
Outputs: <output>/tables/tableNN_<topic>.tex, <output>/paired_comparisons.md
"""

import argparse
import json
from pathlib import Path

import jax
import numpy as np

from paper import data
from paper.data import (CONTINUOUS, CONTINUOUS_ARMS, GRID_ARMS, GRID_FLOOR, GRID_TITLES, GRIDS, LOOP_ARMS,
                        NONSTATIONARY, NONSTATIONARY_ARMS, Runs)
from rare.agents.ddpg import init_continuous_skip, init_ddpg, init_evaluation_critic
from rare.agents.taac import init_taac
from rare.metrics.auc import grid_reward_ceiling, span_auc
from rare.metrics.stats import bootstrap_ci, iqm, paired
from rare.networks.q import init_q_params
from rare.networks.skip import init_skip_params

SKIP_ARMS = GRID_ARMS[1:]
# Ablation rows per critic, as (table arm, action backup, skip backup); None marks the main runs.
ABLATION = {"sh": (("RARe-sh-AE-SG", "E", "G"), ("RARe-sh-AE-SE", "E", "E"), ("TempoRL", None, None), ("RARe-sh", None, None)),
            "se": (("RARe-se-AE-SG", "E", "G"), ("RARe-se-AE-SE", "E", "E"), ("RARe-se-AG-SG", "G", "G"), ("RARe-se", None, None))}
BACKUP_MODES = {"E": "expected", "G": "greedy"}
SETTINGS_COLUMNS = (*CONTINUOUS, "Pendulum", "HalfCheetah")
# Environment -> (observation size, action size), fixed by rare/envs.
SHAPES = {"MountainCar": (2, 1), "PointMaze": (8, 2), "Goal": (60, 2), "Button": (76, 2),
          "Pendulum": (3, 1), "HalfCheetah": (17, 6)}
# Episode limits, fixed by rare/envs; run.json has no continuous limit to read.
EPISODE_LIMITS = {"MountainCar": 999, "PointMaze": 600, "Goal": 1000, "Button": 1000,
                  "Pendulum": 200, "HalfCheetah": 1000}
GRID_OBSERVATIONS, GRID_ACTIONS = 96, 4  # 8 x 12 cells, shared by the six layouts.
WINDY_OBSERVATIONS, WINDY_ACTIONS = 7, 4  # rare/envs/grid/windy_goal.py.
WIND_NAMES = ("west", "north", "east", "south")  # Wind index order of rare/envs/grid/windy_goal.py.
NONSTATIONARY_NAMES = {"WindShift": "WindShift", "Random": "Random", "Episodic": "Episodic",
                       "Shift1": "1 shift", "Shift3": "3 shifts", "Shift5": "5 shifts"}


# Formatting.

def decimal(value, places):
    return f"{value:.{places}f}".removeprefix("0")


def number(value):
    return f"{value:,}".replace(",", "{,}")


def trim(value):
    return decimal(value, 3).rstrip("0").rstrip(".") or "0"


def schedule(high, low, cutoff):
    text = f"{trim(high)}$\\to${trim(low)}"
    return text if cutoff == 1 else f"{text} (first half)"


def interval_cell(center, low, high, bold, places=3, ci_places=2):
    """Center with its interval as a subscript, without leading zeros as in the paper."""
    text = decimal(center, places)
    text = f"\\boldsymbol{{{text}}}" if bold else text
    bounds = decimal(low, ci_places), decimal(high, ci_places)
    if bounds[0] == bounds[1] and ci_places < 3:  # Keep a degenerate interval visible.
        bounds = decimal(low, 3), decimal(high, 3)
    return f"${text}_{{[{bounds[0]},\\,{bounds[1]}]}}$"


def column(stats, best=max, places=3, ci_places=2):
    """Format {arm: (center, low, high)}, bolding every arm tied for best after rounding."""
    top = best(round(value[0], places) for value in stats.values())
    return {arm: interval_cell(*value, round(value[0], places) == top, places, ci_places)
            for arm, value in stats.items()}


def seed_stats(values):
    return (float(iqm(values)), *map(float, bootstrap_ci(values)))


def float_table(caption, label, columns, lines):
    """An appendix table: plain tabular in a floating table."""
    return "\n".join((r"\begin{table}[t]", r"\centering", r"\small", f"\\caption{{{caption}}}",
                      f"\\label{{{label}}}", f"\\begin{{tabular}}{{{columns}}}", r"\toprule", *lines,
                      r"\bottomrule", r"\end{tabular}", r"\end{table}", ""))


def results_table(caption, label, columns, lines, *, separation="2.5pt", size=r"\small", environment="table*"):
    """A results table in small capitals."""
    opening = r"\begin{table}[t]" if environment == "table" else r"\begin{table*}"
    return "\n".join((opening, r"\centering", size, f"\\caption{{{caption}}}", r"\begin{sc}",
                      f"\\setlength{{\\tabcolsep}}{{{separation}}}", f"\\begin{{tabular}}{{{columns}}}", r"\toprule",
                      *lines, r"\bottomrule", r"\end{tabular}", r"\end{sc}", f"\\label{{{label}}}",
                      f"\\end{{{environment}}}", ""))


# Main results.

def grid_aucs(runs, metric, arm):
    return {environment: span_auc(*metric(runs, environment, arm)) for environment in GRIDS}


def grid_columns(runs, metric, arms, best=max, places=3, ci_places=2):
    """{environment: {arm: formatted cell}}."""
    aucs = {arm: grid_aucs(runs, metric, arm) for arm in arms}
    return {environment: column({arm: seed_stats(aucs[arm][environment]) for arm in arms}, best, places, ci_places)
            for environment in GRIDS}


def grid_header(first_column):
    names = " & ".join(f"\\textbf{{{GRID_TITLES[environment]}}}" for environment in GRIDS)
    padding = "& " * first_column
    return (r"& \multicolumn{3}{c}{\textbf{Standard}} & \multicolumn{3}{c}{\textbf{Randomized}} \\",
            f"\\cmidrule(lr){{{first_column + 1}-{first_column + 3}}} "
            f"\\cmidrule(lr){{{first_column + 4}-{first_column + 6}}}",
            f"{padding}{names} \\\\", r"\midrule")


def grid_loop_table(columns, caption, label):
    """Tables 1 and 3: DDQN and the open-loop arms."""
    def row(arm):
        cells = [columns[environment][arm] for environment in GRIDS]
        return " & ".join(cells[:3]) + " &\n" + " & ".join(cells[3:]) + " \\\\"

    return results_table(caption, label, "cl ccc ccc", (
        r"\multirow{2}{*}{\textbf{Loop}} & \multirow{2}{*}{\textbf{Model}}", *grid_header(2),
        r"{\color{red}\ding{55}} & DDQN &", row("DDQN"), r"\midrule",
        r" & TempoRL &", row("TempoRL"), r" & UTE &", row("UTE"),
        r"\cmidrule(lr){2-8}", r"\addlinespace[1pt]", r"\rowcolor{gray!12}",
        r"\cellcolor{white} & RARe-sh &", row("RARe-sh"), r"\rowcolor{gray!12}",
        r"\cellcolor{white}\multirow{-4}{*}{Open} & RARe-se &", row("RARe-se")))


def table_grid_auc(runs):
    return grid_loop_table(
        grid_columns(runs, data.grid_reward, GRID_ARMS),
        r"Normalized reward AUC on grid environments (IQM over 20 seeds with 95\% bootstrap CI; higher is better). "
        r"Scores are scaled between the hazard penalty and the optimal return.", "tab:grid_reward_auc")


def table_grid_risk(runs):
    columns = grid_columns(runs, data.risk_per_step, SKIP_ARMS, best=min, places=2, ci_places=1)

    def row(arm):
        cells = [columns[environment][arm] for environment in GRIDS]
        return f"{arm} &\n" + " & ".join(cells[:3]) + " &\n" + " & ".join(cells[3:]) + " \\\\"

    return results_table(
        r"Risk per step on grid environments, averaged over training (IQM over 20 seeds with 95\% bootstrap CI; "
        r"lower is better). DDQN is omitted because it does not repeat actions.", "tab:grid_risk", "l ccc ccc",
        (r"\multirow{2}{*}{\textbf{Model}}", *grid_header(1), row("TempoRL"), row("UTE"),
         r"\cmidrule(lr){1-7}", r"\addlinespace[1pt]", r"\rowcolor{gray!12}", row("RARe-sh"),
         r"\rowcolor{gray!12}", row("RARe-se")))


def table_grid_rodr(runs):
    return grid_loop_table(
        grid_columns(runs, data.rodr, GRID_ARMS),
        r"Risk-aware Optimal Decision Ratio on grid environments, averaged over training (IQM over 20 seeds with "
        r"95\% bootstrap CI; higher is better).", "tab:grid_rodr")


def table_continuous_auc(runs):
    columns = {environment: column({arm: seed_stats(span_auc(*data.continuous_reward(runs, environment, arm)))
                                    for arm in CONTINUOUS_ARMS}, ci_places=3) for environment in CONTINUOUS}

    def row(arm):
        cells = [columns[environment][arm] for environment in CONTINUOUS]
        return " & ".join(cells[:2]) + " &\n" + " & ".join(cells[2:]) + " \\\\"

    return results_table(
        r"Normalized reward AUC on continuous-control environments (IQM over 20 seeds with 95\% bootstrap CI; higher "
        r"is better). Scores are scaled between random and expert reference scores; the expert score is a reference, "
        r"not an upper bound.", "tab:continuous_reward_auc", "cl cc cc",
        (r"\multirow{2}{*}{\textbf{Loop}} & \multirow{2}{*}{\textbf{Model}}",
         r"& \multicolumn{2}{c}{\textbf{Continuous Control}} & \multicolumn{2}{c}{\textbf{Safe RL}} \\",
         r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}", r"& & \textbf{MountainCar} & \textbf{PointMaze}",
         r"  & \textbf{Goal} & \textbf{Button} \\", r"\midrule",
         r"{\color{red}\ding{55}} & DDPG &", row("DDPG"), r"\midrule",
         r"Closed & TAAC &", row("TAAC"), r"\midrule",
         r"& TempoRL &", row("TempoRL"), r"& UTE &", row("UTE"),
         r"\cmidrule(lr){2-6}", r"\addlinespace[1pt]", r"\rowcolor{gray!12}",
         r"\cellcolor{white} & RARe-sh &", row("RARe-sh"), r"\rowcolor{gray!12}",
         r"\cellcolor{white}\multirow{-4}{*}{Open} & RARe-se &", row("RARe-se")), separation="4pt")


def table_td_backup_ablation(runs):
    columns = {critic: grid_columns(runs, data.grid_reward, [arm for arm, *_ in rows])
               for critic, rows in ABLATION.items()}

    def row(critic, arm):
        return " & ".join(columns[critic][environment][arm] for environment in GRIDS) + " \\\\"

    def block(critic, name):
        (expected_greedy, *_), (expected_expected, *_), (greedy, action, _), (rare, *_) = ABLATION[critic]
        third = "G & G" if action else r"\multicolumn{2}{c}{TempoRL}"
        return (f" & E & G & {row(critic, expected_greedy)}", f" & E & E & {row(critic, expected_expected)}",
                f" & {third} & {row(critic, greedy)}", r"\rowcolor{gray!12}",
                f"\\cellcolor{{white}}\\multirow{{-4}}{{*}}{{{name}}} & \\multicolumn{{2}}{{c}}{{{rare}}} & "
                f"{row(critic, rare)}")

    width = r">{\centering\arraybackslash}p{2.2em}"
    return results_table(
        r"TD-backup ablation of RARe on grid environments: normalized AUC (IQM over 20 seeds with 95\% bootstrap CI). "
        r"Action and Skip give the backup of the action critic and of the skip value: G (greedy) or E (expected under "
        r"the epsilon-greedy policy). Shared bootstraps skips from the action critic, so its G/G arm is TempoRL and its "
        r"G/E arm RARe-sh; Separate uses a separate endpoint critic, and its G/E arm is RARe-se. Shaded rows are RARe; "
        r"bold marks the best arm per critic.", "tab:target_ablation", f"c {width} {width} ccc ccc",
        (r"\multirow{2}{*}{\textbf{Critic}} & \multirow{2}{*}{\textbf{Act.}} & \multirow{2}{*}{\textbf{Skip}}",
         *grid_header(3), *block("sh", "Shared"), r"\midrule", *block("se", "Separate")), size=r"\footnotesize")


def table_loop_limitation(runs):
    pendulum = {arm: seed_stats(span_auc(*data.continuous_reward(runs, "Pendulum", arm))) for arm in LOOP_ARMS}
    halfcheetah = {}
    for arm in LOOP_ARMS:  # Three seeds: the mean with the seed range (the IQM of 3 is their mean).
        values = span_auc(*data.continuous_reward(runs, "HalfCheetah", arm))
        halfcheetah[arm] = float(values.mean()), float(values.min()), float(values.max())
    columns = column(pendulum), column(halfcheetah)
    pendulum_steps = runs.cell("Pendulum", "RARe-se")["total_steps"]
    halfcheetah_steps = runs.cell("HalfCheetah", "RARe-se")["total_steps"]
    return results_table(
        r"Normalized reward AUC of closed-loop TAAC and open-loop {\em RARe} across action dimensions (higher is "
        r"better). Pendulum: IQM over 20 seeds with 95\% bootstrap CI. HalfCheetah: mean over 3 seeds with the seed "
        r"range. Reference scores are listed in Table~\ref{tab:auc_references}.", "tab:loop_limitation", "cl cc",
        (r"\textbf{Loop} & \textbf{Model} & \textbf{Pendulum} & \textbf{HalfCheetah} \\",
         f"\\multicolumn{{2}}{{l}}{{\\textit{{Action dim.}}}} & {SHAPES['Pendulum'][1]} & {SHAPES['HalfCheetah'][1]} \\\\",
         f"\\multicolumn{{2}}{{l}}{{\\textit{{Steps}}}} & {pendulum_steps // 1000}k & {halfcheetah_steps // 10 ** 6}M \\\\",
         r"\midrule", f"Closed & TAAC & {columns[0]['TAAC']} & {columns[1]['TAAC']} \\\\", r"\midrule",
         r"\rowcolor{gray!12}", f"\\cellcolor{{white}}Open & RARe-se & {columns[0]['RARe-se']} & {columns[1]['RARe-se']} \\\\"),
        separation="5pt", environment="table")


def table_nonstationary_auc(runs):
    columns = {condition: column({arm: seed_stats(data.post_change_auc(runs, condition, arm))
                                  for arm in NONSTATIONARY_ARMS}) for condition in NONSTATIONARY}

    def row(arm):
        return " & ".join(columns[condition][arm] for condition in NONSTATIONARY) + " \\\\"

    return results_table(
        r"Post-change AUC of $1 - \text{length}/\text{max steps}$ over 20k--40k steps on non-stationary grids (IQM with "
        r"95\% bootstrap CI; higher is better). Environment columns use 20 seeds and the shift-count sweep uses 10.",
        "tab:nonstationary", "l ccc ccc",
        (r"\multirow{2}{*}{\textbf{Model}} & \multicolumn{3}{c}{\textbf{Environment}}"
         r" & \multicolumn{3}{c}{\textbf{Number of Shifts}} \\",
         r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
         r"& \textbf{WindShift} & \textbf{Random} & \textbf{Episodic} & \textbf{1} & \textbf{3} & \textbf{5} \\",
         r"\midrule", f"UTE & {row('UTE')}", r"\rowcolor{gray!12}", f"RARe-sh & {row('RARe-sh')}",
         r"\rowcolor{gray!12}", f"RARe-se & {row('RARe-se')}"))


# Environment and method settings.

def table_nonstationary_conditions(runs):
    specs = {condition: runs.cell(condition, "UTE").spec for condition in NONSTATIONARY[:3]}

    def wind(spec, phase):
        directions = spec["phase_wind_options"][phase]
        name = "any direction" if len(directions) == len(WIND_NAMES) else " or ".join(WIND_NAMES[i] for i in directions)
        return f"{name}, ${decimal(spec['phase_wind_probabilities'][phase], 2)}$"

    distance = {name: spec["min_start_goal_distance"] for name, spec in specs.items()}
    start_goal = {"WindShift": "opposite corners, fixed", "Random": f"Manhattan distance $\\geq {distance['Random']}$",
                  "Episodic": f"corner start, distance $\\geq {distance['Episodic']}$"}
    change = {spec["shift_step"] for spec in specs.values()}.pop()
    lines = [r"Condition & Start and goal & Wind before & Wind after & Limit \\", r"\midrule"]
    lines += [f"{name} & {start_goal[name]} & {wind(spec, 0)} & {wind(spec, 1)} & {spec['max_steps']} \\\\"
              for name, spec in specs.items()]
    return float_table(f"Non-stationary Grid conditions. Probabilities are gust probabilities before and after the "
                       f"change at step ${number(change)}$.", "tab:nonstationary_envs", "lllll", lines)


def table_nonstationary_settings(runs):
    lines = [r"Condition & Steps & Warm-up & Replay & Skip replay & Interval & Episodes & Seeds \\", r"\midrule"]
    for condition, name in NONSTATIONARY_NAMES.items():
        cell = runs.cell(condition, "UTE")
        lines.append(f"{name} & {number(cell['total_steps'])} & {number(cell['warmup_steps'])} & "
                     f"{number(cell['replay_capacity'])} & {number(cell['skip_replay_capacity'])} & "
                     f"{number(cell.schedule[1] - cell.schedule[0])} & {cell.evaluation['episodes']} & "
                     f"{len(cell.seeds)} \\\\")
    return float_table("Non-stationary settings, shared by UTE and RARe.", "tab:nonstationary_implementation_details",
                       "lrrrrrrr", lines)


def table_grid_settings(runs):
    lines = [r"Environment & Steps & Warm-up & Replay & Skip replay & Cases & $\epsilon$ of $\pi_e$ (RARe-se) \\",
             r"\midrule"]
    for environment in GRIDS:
        cell = runs.cell(environment, "RARe-se")
        cases = len(cell.spec["starts"]) * len(cell.spec["goals"])
        lines.append(f"{environment} & {number(cell['total_steps'])} & {number(cell['warmup_steps'])} & "
                     f"{number(cell['replay_capacity'])} & {number(cell['skip_replay_capacity'])} & {cases} & "
                     f"{schedule(cell['max_eval_epsilon'], cell['min_eval_epsilon'], cell['eval_epsilon_cutoff'])} \\\\")
    return float_table(r"Grid settings. $\epsilon$ of $\pi_e$ decays linearly after warm-up, over the first half of "
                       r"training where noted and over all of it otherwise. RARe-sh uses the behavior $\epsilon$. "
                       r"The CliffWalking and ZigZag schedules were selected on the same seeds used for reporting.",
                       "tab:grid_implementation_details", "lrrrrrl", lines)


def table_continuous_settings(runs):
    cells = {environment: runs.cell(environment, "RARe-se") for environment in SETTINGS_COLUMNS}

    def row(name, value):
        return f"{name} & " + " & ".join(str(value(cells[environment])) for environment in SETTINGS_COLUMNS) + r" \\"

    lines = [" & " + " & ".join(SETTINGS_COLUMNS) + r" \\", r"\midrule",
             row("Training steps", lambda c: number(c["total_steps"])),
             row("Warm-up steps", lambda c: number(c["warmup_steps"])),
             row("Episode limit", lambda c: number(EPISODE_LIMITS[c.environment])),
             row("Hidden width", lambda c: c["hidden_dim"]),
             row("Maximum skip $J$", lambda c: c["max_skip"]),
             row("Learning rate", lambda c: "cosine" if c["use_lr_decay"] else "constant"),
             row("Replay size", lambda c: number(c["replay_capacity"])),
             row("Skip replay size", lambda c: number(c["skip_replay_capacity"])),
             row("Skip $\\epsilon$ decay steps",
                 lambda c: number(c["skip_epsilon_decay_steps"] or c["total_steps"] - c["warmup_steps"])),
             row("$\\sigma_e$ (RARe-se)",
                 lambda c: schedule(c["max_eval_sigma"], c["min_eval_sigma"], c["eval_sigma_cutoff"])),
             row("Evaluation interval", lambda c: number(c.schedule[1] - c.schedule[0])),
             row("Evaluation episodes", lambda c: c.evaluation["episodes"])]
    return float_table(r"Continuous settings. RARe-sh uses the behavior exploration noise as $\sigma_e$. HalfCheetah "
                       "uses three seeds.", "tab:continuous_implementation_details",
                       "l" + "r" * len(SETTINGS_COLUMNS), lines)


def only(cells, key):
    """The single value of a hyperparameter shared by every run, or fail loudly."""
    values = {run["config"][key] for cell in cells for run in cell.runs}
    if len(values) != 1:
        raise ValueError(f"{key} differs across runs: {values}")
    return values.pop()


def table_hyperparameters(runs):
    grid = {arm: [runs.cell(environment, arm) for environment in GRIDS] for arm in SKIP_ARMS}
    windy = {arm: [runs.cell(condition, arm) for condition in NONSTATIONARY] for arm in NONSTATIONARY_ARMS}
    continuous = {arm: [runs.cell(environment, arm) for environment in (*CONTINUOUS, "Pendulum")]
                  for arm in CONTINUOUS_ARMS}
    domains = sum(grid.values(), []), sum(windy.values(), []), sum(continuous.values(), [])
    grid_all, windy_all, continuous_all = domains
    settings = r"Table~\ref{tab:continuous_implementation_details}"

    def shared(name, key, text=str):
        return f"{name} & " + " & ".join(text(only(cells, key)) for cells in domains) + r" \\"

    def group(name):
        return r"\midrule", f"\\multicolumn{{4}}{{l}}{{\\textit{{{name}}}}} \\\\"

    def rate(value):
        return f"${value * 1000:g}\\times 10^{{-3}}$"

    def lr_schedule(cells):
        return "cosine" if only(cells, "use_lr_decay") else "constant"

    def taac(name, value):
        return f"{name} & -- & -- & {value} \\\\"

    behavior = (f"$\\epsilon$: {only(grid_all, 'max_epsilon'):g}$\\to${only(grid_all, 'min_epsilon'):g}",
                f"$\\epsilon$: {only(windy_all, 'max_epsilon'):g}$\\to${trim(only(windy_all, 'min_epsilon'))}",
                f"$\\sigma$: {trim(only(continuous_all, 'exploration_noise'))}$\\to$0")
    ute = grid["UTE"], windy["UTE"], continuous["UTE"]
    lines = [r"Hyperparameter & Grid & Non-stationary & Continuous \\", *group("Shared"),
             r"Optimizer & Adam & Adam & Adam \\",
             shared("Learning rate", "learning_rate", rate),
             f"Learning-rate schedule & {lr_schedule(grid_all)} & {lr_schedule(windy_all)} & {settings} \\\\",
             shared("Batch size", "batch_size"), shared("Discount $\\gamma$", "gamma"),
             shared("Polyak $\\tau$", "tau", lambda value: f"{value:g}"),
             r"Hidden layers & 2 & 2 & 2 \\",
             f"Hidden width & {only(grid_all, 'hidden_dim')} & {only(windy_all, 'hidden_dim')} & {settings} \\\\",
             r"Critic loss & Huber & Huber & Huber \\",
             "Behavior exploration & " + " & ".join(behavior) + r" \\", *group("Open-loop methods"),
             f"Maximum skip $J$ & {only(grid_all, 'max_skip')} & {only(windy_all, 'max_skip')} & {settings} \\\\",
             f"Skip exploration & behavior $\\epsilon$ & behavior $\\epsilon$ & {settings} \\\\", *group("UTE"),
             "Skip networks & " + " & ".join(str(only(cells, "ensemble_size")) for cells in ute) + r" \\",
             "Bootstrap mask probability & " + " & ".join(
                 f"{only(cells, 'bootstrap_probability'):g}" for cells in ute) + r" \\",
             f"$\\lambda$ & ${only(ute[0], 'uncertainty_factor'):g}$ & ${only(ute[1], 'uncertainty_factor'):g}$ & "
             r"UCB over 9 values \\", *group("RARe"),
             r"$\pi_e$ of RARe-sh & behavior $\epsilon$ & behavior $\epsilon$ & behavior $\sigma$ \\",
             r"$\pi_e$ of RARe-se & Table~\ref{tab:grid_implementation_details} & "
             f"$\\epsilon$: {trim(only(windy['RARe-se'], 'max_eval_epsilon'))}$\\to$"
             f"{trim(only(windy['RARe-se'], 'min_eval_epsilon'))} & {settings} \\\\",
             f"Monte Carlo samples $N$ & -- & -- & {only(continuous['RARe-se'], 'mc_samples')} \\\\", *group("TAAC"),
             taac("Hidden width", settings),
             taac("Batch size", only(continuous["TAAC"], "batch_size")),
             taac("Learning rate", rate(only(continuous["TAAC"], "learning_rate"))),
             taac("Replay window", only(continuous["TAAC"], "compare_horizon")),
             taac("Initial temperature", f"{only(continuous['TAAC'], 'taac_temperature'):g}"),
             taac("Target entropy $\\delta$", f"{only(continuous['TAAC'], 'target_entropy_delta'):g}")]
    return float_table("Hyperparameters. Non-stationary means the non-stationary Grid environments. Dashes mark "
                       "settings that do not apply.", "tab:hyperparameters", "llll", lines)


def table_ablation_arms(runs):
    """Check every TD-backup arm against its main-run budget, then describe the eight arms."""
    for family, rows in ABLATION.items():
        for environment in GRIDS:
            main = runs.cell(environment, f"RARe-{family}")
            for arm, action, skip in rows:
                if action is None:
                    continue
                cell = runs.cell(environment, arm)
                if (cell["action_target_mode"], cell["skip_target_mode"]) != (BACKUP_MODES[action], BACKUP_MODES[skip]):
                    raise ValueError(f"{environment}/{arm}: backups differ from the arm name")
                for key in ("total_steps", "warmup_steps", "replay_capacity", "skip_replay_capacity", "max_skip",
                            "hidden_dim", "learning_rate", "batch_size", "max_epsilon", "min_epsilon"):
                    if cell[key] != main[key]:
                        raise ValueError(f"{environment}/{arm}: {key} differs from the main run")
    lines = [r"Critic & Action backup & Skip backup & Skip bootstrap & Arm \\", r"\midrule",
             r"Shared & Expected SARSA & greedy & action critic & -- \\",
             r"Shared & Expected SARSA & expected & action critic & -- \\",
             r"Shared & DDQN & greedy & action critic & TempoRL \\",
             r"Shared & DDQN & expected & action critic & RARe-sh \\", r"\midrule",
             r"Separate & Expected SARSA & greedy & evaluation critic & -- \\",
             r"Separate & Expected SARSA & expected & evaluation critic & -- \\",
             r"Separate & DDQN & greedy & evaluation critic & -- \\",
             r"Separate & DDQN & expected & evaluation critic & RARe-se \\"]
    return float_table(r"Arms of the TD-backup ablation. All arms use the budgets of "
                       r"Table~\ref{tab:grid_implementation_details}. Expected SARSA uses the behavior $\epsilon$, and "
                       r"expected skip backups use the $\pi_e$ of the matching RARe variant.",
                       "tab:ablation_settings", "lllll", lines)


# Parameter counts and cost.

def size(tree):
    return sum(leaf.size for leaf in jax.tree.leaves(tree))


def discrete_counts(observations, actions, cell):
    """Inference and training sizes per arm; training adds target networks and the RARe-se critic."""
    key = jax.random.key(0)
    base = size(init_q_params(key, observations, actions, cell["hidden_dim"]))
    skip = size(init_skip_params(key, observations, actions, cell["max_skip"], hidden_dim=cell["hidden_dim"]))
    inference = {"DDQN": base, "TempoRL": base + skip, "UTE": base + cell["ensemble_size"] * skip,
                 "RARe-sh": base + skip, "RARe-se": base + skip}
    training = {arm: value + base for arm, value in inference.items()}
    training["RARe-se"] += 2 * base
    return inference, training


def table_parameters_discrete(runs):
    def rows(name, observations, actions, cell, arms):
        return [f"{name} ({stage}) & " + " & ".join(number(counts[arm]) if arm in arms else "--" for arm in GRID_ARMS)
                + r" \\" for stage, counts in zip(("inference", "training"), discrete_counts(observations, actions, cell))]

    lines = [r"Setting & DDQN & TempoRL & UTE & RARe-sh & RARe-se \\", r"\midrule",
             *rows("Grid", GRID_OBSERVATIONS, GRID_ACTIONS, runs.cell("Bridge", "UTE"), GRID_ARMS),
             *rows("Non-stationary", WINDY_OBSERVATIONS, WINDY_ACTIONS, runs.cell("WindShift", "UTE"), NONSTATIONARY_ARMS)]
    return float_table("Parameters on discrete environments. Training counts online and target networks, excluding "
                       "optimizer states. The six Grid environments share one network size, and every ablation arm "
                       "matches the RARe variant of its critic. Dashes mark methods outside the comparison.",
                       "tab:parameter_grid", "lrrrrr", lines)


def continuous_counts(arm, environment, cell):
    """Inference and training sizes of one arm, with the widths of its own run."""
    key = jax.random.key(0)
    observations, actions = SHAPES[environment]
    hidden = cell["hidden_dim"]
    if arm == "TAAC":
        taac = init_taac(key, observations, actions, hidden, 1.)
        return (size(taac.actor) + size(taac.critic),
                size(taac.actor) + size(taac.target_actor) + size(taac.critic) + size(taac.target_critic)
                + size(taac.log_temperature))
    ddpg = init_ddpg(key, observations, actions, hidden)
    actor = size(ddpg.actor)
    base = actor + size(ddpg.target_actor) + size(ddpg.critic) + size(ddpg.target_critic)
    if arm == "DDPG":
        return actor, base
    skip = size(init_continuous_skip(key, observations, actions, cell["max_skip"], hidden,
                                     cell["ensemble_size"] if arm == "UTE" else 1).params)
    if arm == "RARe-se":
        evaluation = init_evaluation_critic(key, observations, actions, hidden)
        return actor + skip, base + skip + size(evaluation.online) + size(evaluation.target)
    return actor + skip, base + skip


def table_parameters_continuous(runs):
    def shown(arm, environment):
        return environment in CONTINUOUS or arm in LOOP_ARMS

    counts = {(arm, environment): continuous_counts(arm, environment, runs.cell(environment, arm))
              for arm in CONTINUOUS_ARMS for environment in SETTINGS_COLUMNS if shown(arm, environment)}
    lines = ["Model & " + " & ".join(SETTINGS_COLUMNS) + r" \\"]
    for stage, name in enumerate(("Inference", "Training")):
        lines += [r"\midrule", f"\\multicolumn{{{len(SETTINGS_COLUMNS) + 1}}}{{l}}{{\\textit{{{name}}}}} \\\\"]
        lines += [f"{arm} & " + " & ".join(number(counts[arm, environment][stage]) if shown(arm, environment) else "--"
                                           for environment in SETTINGS_COLUMNS) + r" \\" for arm in CONTINUOUS_ARMS]
    return float_table("Parameters on the continuous environments. Training counts online and target networks, "
                       "excluding optimizer states. HalfCheetah compares TAAC and RARe-se only.",
                       "tab:parameter_continuous", "l" + "r" * len(SETTINGS_COLUMNS), lines)


def cost_table(runs, setting, arms, source):
    """Cost on dummy data, paired by seed against the skip baselines."""
    rows = json.loads((runs.root / "_cost/cost.json").read_text())[setting]
    train = {arm: rows[arm]["ms_per_step"] for arm in arms}

    def spread(values, places):
        values = list(values)
        return f"${np.mean(values):.{places}f} \\pm {np.std(values, ddof=1):.{places}f}$"

    def relative(arm, baseline):
        if arm == baseline:
            return "--"
        change = 100 * (np.mean([train[arm][seed] / train[baseline][seed] for seed in train[arm]]) - 1)
        return f"${change:+.1f}\\%$"

    lines = [r"Model & Training (ms/step) & vs.\ TempoRL & vs.\ UTE & Inference ($\mu$s/decision) \\", r"\midrule"]
    lines += [f"{arm} & {spread(train[arm].values(), 3)} & {relative(arm, 'TempoRL')} & {relative(arm, 'UTE')} & "
              f"{spread(rows[arm]['device_us'].values(), 2)} \\\\" for arm in arms]
    return float_table(f"{setting} method cost on dummy data with the {source} network sizes, mean $\\pm$ SD over 5 "
                       "seeds. Simulation is excluded.", f"tab:cost_{setting.lower()}", "lrrrr", lines)


def table_cost_discrete(runs):
    return cost_table(runs, "Discrete", GRID_ARMS, "RBridge")


def table_cost_continuous(runs):
    return cost_table(runs, "Continuous", CONTINUOUS_ARMS, "Goal")


def table_reference_scores(runs):
    """References of the normalized AUC; randomized Grids show their per-case range."""
    def grid_expert(environment):
        values = [case["optimal_reward"] for case in grid_reward_ceiling(runs.cell(environment, "RARe-se").spec)["cases"]]
        low, high = min(values), max(values)
        return f"${low:g}$" if low == high else f"$[{low:g},\\,{high:g}]$"

    scores = {environment: data.references(runs, environment) for environment in SETTINGS_COLUMNS}
    lines = [" & " + " & ".join(GRIDS) + r" \\", r"\midrule",
             "Floor & " + " & ".join([f"${GRID_FLOOR:g}$"] * len(GRIDS)) + r" \\",
             "Expert & " + " & ".join(map(grid_expert, GRIDS)) + r" \\", r"\midrule",
             " & " + " & ".join(SETTINGS_COLUMNS) + r" \\", r"\midrule",
             "Random & " + " & ".join(f"${floor:.4g}$" for floor, _ in scores.values()) + r" \\",
             "Expert & " + " & ".join(f"${expert:.4g}$" for _, expert in scores.values()) + r" \\"]
    return float_table(r"Reference scores of the normalized AUC. On grid environments, the expert score is the optimal "
                       r"return of each start--goal case, so randomized versions show the range over cases. The "
                       r"HalfCheetah expert is the final score of a Brax PPO reference and is not an upper bound.",
                       "tab:auc_references", "l" + "r" * len(GRIDS), lines)


TABLES = (("table01_grid_auc", table_grid_auc), ("table02_grid_risk", table_grid_risk),
          ("table03_grid_rodr", table_grid_rodr), ("table04_continuous_auc", table_continuous_auc),
          ("table05_td_backup_ablation", table_td_backup_ablation),
          ("table06_nonstationary_conditions", table_nonstationary_conditions),
          ("table07_nonstationary_settings", table_nonstationary_settings), ("table08_grid_settings", table_grid_settings),
          ("table09_continuous_settings", table_continuous_settings), ("table10_hyperparameters", table_hyperparameters),
          ("table11_ablation_arms", table_ablation_arms), ("table12_parameters_discrete", table_parameters_discrete),
          ("table13_parameters_continuous", table_parameters_continuous), ("table14_cost_discrete", table_cost_discrete),
          ("table15_cost_continuous", table_cost_continuous), ("table16_reference_scores", table_reference_scores),
          ("table17_loop_limitation", table_loop_limitation), ("table18_nonstationary_auc", table_nonstationary_auc))


# Paired comparisons quoted in the text.

def section(title, quote, metric, environments, pairs, places=3):
    """One group of comparisons: left and right values per environment, as a Markdown table."""
    rows, excluded = [], []
    for left, right in pairs:
        for environment in environments:
            item = paired(metric(environment, left), metric(environment, right))
            low, high = item["ci"]
            mark = "*" if low > 0 or high < 0 else ""
            if mark:
                excluded.append(f"{left} - {right} on {environment}")
            rows.append(f"| {environment} | {left} - {right} | {item['difference']:+.{places}f}{mark} | "
                        f"[{low:+.{places}f}, {high:+.{places}f}] | {item['win_rate']:.0%} |")
    summary = f"CI excludes zero in {len(excluded)} of {len(rows)}" + (f": {'; '.join(excluded)}." if excluded else ".")
    return [f"## {title}", "", f"Paper: {quote}", "", summary, "",
            "| Environment | Pair | IQM difference | 95% CI | Seed wins |", "|---|---|---:|---:|---:|", *rows, ""]


def paired_comparisons(runs):
    def grid(metric):
        return lambda environment, arm: span_auc(*metric(runs, environment, arm))

    def continuous(environment, arm):
        return span_auc(*data.continuous_reward(runs, environment, arm))

    def risk_per_decision(environment, arm):
        return data.risk_per_decision_auc(runs, environment, arm)

    def windy(environment, arm):
        return data.post_change_auc(runs, environment, arm)

    versus_skip = [(rare, baseline) for rare in ("RARe-sh", "RARe-se") for baseline in ("TempoRL", "UTE")]
    versus_tempo = [("RARe-sh", "TempoRL"), ("RARe-se", "TempoRL")]
    lines = ["# Paired comparisons quoted in the paper", "",
             "Each row resamples seeds jointly (10,000 bootstrap resamples) and reports the difference of interquartile "
             "means of the seed AUCs, left minus right, with its 95% percentile CI. `*` marks a CI excluding zero. Seed "
             "wins count seeds where the left arm scores higher, ties as half.", ""]
    lines += section("Section 5.2: reward against TempoRL",
                     "the paired 95% confidence interval excludes zero for RARe-sh on RCliffWalking and RBridge, and for "
                     "RARe-se on RBridge and RZigZag; on the fixed layouts the differences are not significant.",
                     grid(data.grid_reward), GRIDS, versus_tempo)
    lines += section("Section 5.2: Risk per step", "both RARe variants incur lower Risk per step than TempoRL and UTE in "
                     "all six environments, and the paired 95% confidence interval excludes zero in every comparison.",
                     grid(data.risk_per_step), GRIDS, versus_skip)
    lines += section("Section 5.2: RODR", "both RARe variants achieve higher RODR than TempoRL and UTE in all six "
                     "environments, and the paired 95% confidence interval excludes zero in every comparison.",
                     grid(data.rodr), GRIDS, versus_skip)
    lines += section("Section 5.2: decisions per step", "RARe also makes fewer decisions per step in all these comparisons.",
                     grid(data.decisions_per_step), GRIDS, versus_skip)
    lines += section("Section 5.2: Risk per decision", "Risk per decision is significantly lower in 13 of the 24.",
                     risk_per_decision, GRIDS, versus_skip)
    lines += section("Section 5.3: continuous reward", "RARe-se has a significant gain over DDPG on MountainCar and "
                     "PointMaze, and its differences from TempoRL, UTE and TAAC there are not significant. On Goal and "
                     "Button it has significant gains over DDPG and TAAC, and its differences from TempoRL and UTE are "
                     "not significant. RARe-sh scores lower than RARe-se on all four tasks, but significantly exceeds "
                     "DDPG on all of them and TAAC on Goal and Button.", continuous, CONTINUOUS,
                     [(rare, baseline) for rare in ("RARe-se", "RARe-sh") for baseline in ("DDPG", "TAAC", "TempoRL", "UTE")]
                     + [("RARe-se", "RARe-sh")])
    lines += section("Section 5.4: Expected SARSA on the action critic", "compared with RARe, the expected action backup "
                     "lowers the normalized AUC by .18 to .33 in every environment, and the paired 95% confidence "
                     "interval excludes zero in all twelve comparisons.", grid(data.grid_reward), GRIDS,
                     [("RARe-sh", "RARe-sh-AE-SE"), ("RARe-se", "RARe-se-AE-SE")])
    lines += section("Section 5.4: expected skip backup", "with a greedy action critic, the expected skip backup has "
                     "significant gains on RCliff and RBridge for the shared critic and on CliffWalking and RBridge for "
                     "the separate critic.", grid(data.grid_reward), GRIDS,
                     [("RARe-sh", "TempoRL"), ("RARe-se", "RARe-se-AG-SG")])
    lines += section("Section 5.4: separate critic, greedy skip backups", "separating the critic changes the normalized "
                     "AUC significantly in no environment.", grid(data.grid_reward), GRIDS,
                     [("RARe-se-AG-SG", "TempoRL")])
    lines += section("Section 5.4: separate critic, expected skip backups, reward", "RARe-se and RARe-sh reach close "
                     "rewards without a consistent winner.", grid(data.grid_reward), GRIDS, [("RARe-se", "RARe-sh")])
    lines += section("Section 5.4: separate critic, expected skip backups, RODR", "RARe-se achieves higher RODR on "
                     "CliffWalking, Bridge, ZigZag, and RBridge.", grid(data.rodr), GRIDS, [("RARe-se", "RARe-sh")])
    lines += section("Section 5.4: separate critic, expected skip backups, Risk per step", "Risk per step shows no "
                     "consistent difference.", grid(data.risk_per_step), GRIDS, [("RARe-se", "RARe-sh")])
    lines += section("Appendix F: non-stationary grids", "the paired 95% confidence interval excludes zero for RARe-se on "
                     "Random and three shifts, and for RARe-sh at one, three, and five shifts. Differences on WindShift "
                     "and Episodic are not significant, and neither RARe variant significantly outperforms the other.",
                     windy, NONSTATIONARY, [("RARe-se", "UTE"), ("RARe-sh", "UTE"), ("RARe-se", "RARe-sh")])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    runs = Runs(args.runs, args.output)
    (args.output / "tables").mkdir(parents=True, exist_ok=True)
    for name, build in TABLES:
        (args.output / "tables" / f"{name}.tex").write_text(build(runs))
        print(f"{name}.tex", flush=True)
    (args.output / "paired_comparisons.md").write_text(paired_comparisons(runs))
    print("paired_comparisons.md")


if __name__ == "__main__":
    main()
