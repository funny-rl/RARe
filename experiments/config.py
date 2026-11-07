"""Resolve one (environment, arm) cell of the paper from the YAML files in configs/."""

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import yaml

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SUITES = ("grid", "nonstationary", "continuous")
SKIP_ALGORITHMS = ("TempoRL", "UTE", "RARe")
# Settings the learners do not read for a given algorithm.
_SKIP_ONLY = ("max_skip", "skip_replay_capacity")
_SHARED_ARMS = ("RARe-sh",)


@dataclass(frozen=True)
class Cell:
    suite: str
    environment: str
    arm: str
    config: dict  # keyword arguments of the run config
    settings: dict  # environment entry of the YAML file
    evaluation: dict
    seeds: tuple[str, ...]  # one "start:stop" per process


def parse_seeds(text):
    start, stop = map(int, text.split(":"))
    return list(range(start, stop))


def load(suite):
    return yaml.safe_load((CONFIGS / f"{suite}.yaml").read_text(encoding="utf-8"))


def suite_of(environment):
    for suite in SUITES:
        if environment in load(suite)["environments"]:
            return suite
    raise ValueError(f"unknown environment: {environment}")


def resolve(environment, arm):
    suite = suite_of(environment)
    document = load(suite)
    settings = document["environments"][environment]
    if arm not in document["arms"]:
        raise ValueError(f"{arm} is not an arm of {suite}: {sorted(document['arms'])}")
    values = {**document["common"], **settings.get("common", {}), **document["arms"][arm]}
    if values["algorithm"] in SKIP_ALGORITHMS:
        values = {**document.get("skip_defaults", {}), **values}
    elif suite != "continuous":
        for name in _SKIP_ONLY:
            values.pop(name, None)
    if values.get("rare_q_mode") == "separate":
        values.update(settings.get("rare_se", {}))
    evaluation = {**document.get("evaluation", {}), **settings.get("evaluation", {})}
    if suite == "grid":
        evaluation["episodes"] = "all" if settings["random"] else 1
    if values["algorithm"] == "DDQN":
        evaluation["risk"] = False  # DDQN does not repeat actions, so the metric is not reported.
    seeds = tuple(settings.get("seed_shards", ())) or (settings.get("seeds", document.get("seeds")),)
    return Cell(suite, environment, arm, values, settings, evaluation, seeds)


def arms(environment):
    return tuple(load(suite_of(environment))["arms"])


def checkpoints(total_steps, warmup_steps, evaluation):
    """Evaluation steps: fixed intervals after warm-up, always ending at the budget."""
    interval = evaluation.get("interval_steps")
    if interval is not None:
        return tuple(range(warmup_steps + interval, total_steps, interval)) + (total_steps,)
    fraction = Fraction(str(evaluation["interval_fraction"]))
    numerator, denominator = fraction.numerator, fraction.denominator
    last = (total_steps - warmup_steps) * denominator // (total_steps * numerator)
    steps = {warmup_steps + (total_steps * numerator * k + denominator - 1) // denominator
             for k in range(1, last + 1)}
    return tuple(sorted(steps | {total_steps}))
