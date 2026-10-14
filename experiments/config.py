"""Strict YAML composition without importing JAX or selecting a backend."""

from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import math
from pathlib import Path

import yaml


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "grid.yaml"
COMMON_FIELDS = {
    "total_steps", "warmup_steps", "replay_capacity", "batch_size", "hidden_dim", "gamma",
    "learning_rate", "max_epsilon", "min_epsilon", "epsilon_decay_steps", "epsilon_schedule",
    "use_lr_decay", "tau", "use_hard_update", "update_interval",
}
SKIP_FIELDS = {"max_skip", "skip_replay_capacity", "skip_learning_rate", "use_data_aug"}
ALGORITHM_FIELDS = {
    "DDQN": set(), "TempoRL": SKIP_FIELDS,
    "TAACDiscrete": {"temperature", "actor_temperature", "target_entropy_delta",
                     "target_actor_entropy_ratio", "compare_horizon"},
    "UTE": SKIP_FIELDS | {"ensemble_size", "uncertainty_factor", "bootstrap_probability"},
    "RARe": SKIP_FIELDS | {"max_eval_epsilon", "min_eval_epsilon", "eval_epsilon_cutoff",
                            "eval_learning_rate", "rare_q_mode", "separate_target_epsilon_source",
                            "shared_target_epsilon_source"},
}
ENVIRONMENTS = {"CliffWalking", "Bridge", "ZigZag", "WindShift"}
EVALUATION_FIELDS = {"interval_fraction", "interval_steps", "episodes", "random_episodes", "risk_diagnostics"}


@dataclass(frozen=True)
class EvaluationConfig:
    """Frozen performance and Grid diagnostics, separate from learner settings."""

    interval_fraction: float = 0.01
    interval_steps: int | None = None
    episodes: int | str = 1
    risk_diagnostics: bool = True

    def __post_init__(self):
        if (isinstance(self.interval_fraction, bool) or not isinstance(self.interval_fraction, (int, float))
                or not math.isfinite(self.interval_fraction) or not 0 < self.interval_fraction <= 1):
            raise ValueError("evaluation.interval_fraction must be in (0, 1]")
        if self.interval_steps is not None and (type(self.interval_steps) is not int or self.interval_steps < 0):
            raise ValueError("evaluation.interval_steps must be null or a nonnegative integer")
        if self.episodes != "all" and (type(self.episodes) is not int or self.episodes <= 0):
            raise ValueError("evaluation.episodes must be a positive integer or 'all'")
        if type(self.risk_diagnostics) is not bool:
            raise ValueError("evaluation.risk_diagnostics must be a boolean")

    def checkpoints(self, total_steps, warmup_steps=0):
        """Start intervals after warmup; retain the total-budget interval size."""
        if type(total_steps) is not int or total_steps <= 0:
            raise ValueError("total_steps must be a positive integer")
        if type(warmup_steps) is not int or not 0 <= warmup_steps <= total_steps:
            raise ValueError("warmup_steps must lie in [0, total_steps]")
        if self.interval_steps == 0 or warmup_steps == total_steps:
            return ()
        if self.interval_steps is not None:
            return tuple(range(warmup_steps + self.interval_steps, total_steps, self.interval_steps)) + (total_steps,)
        fraction = Fraction(str(self.interval_fraction))
        if fraction * total_steps <= 1:
            return tuple(range(warmup_steps + 1, total_steps + 1))
        numerator, denominator = fraction.numerator, fraction.denominator
        last = (total_steps - warmup_steps) * denominator // (total_steps * numerator)
        steps = {warmup_steps + (total_steps * numerator * k + denominator - 1) // denominator
                 for k in range(1, last + 1)}
        return tuple(sorted(steps | {total_steps}))


class UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if not isinstance(key, str) or key in result:
            raise ValueError(f"YAML keys must be unique strings: {key!r}")
        result[key] = loader.construct_object(value_node)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def _check_keys(value, allowed, location):
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be a mapping")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown setting in {location}: {sorted(unknown)}")


def _validate(document):
    _check_keys(document, {"common", "algorithms", "environments", "evaluation"}, "root")
    _check_keys(document.get("evaluation", {}), EVALUATION_FIELDS, "evaluation")
    _check_keys(document.get("common", {}), COMMON_FIELDS, "common")
    _check_keys(document.get("algorithms", {}), set(ALGORITHM_FIELDS), "algorithms")
    for algorithm, settings in document.get("algorithms", {}).items():
        _check_keys(settings, ALGORITHM_FIELDS[algorithm], f"algorithms.{algorithm}")
    _check_keys(document.get("environments", {}), ENVIRONMENTS, "environments")
    for name, settings in document.get("environments", {}).items():
        _check_keys(settings, {"common", "algorithms", "evaluation"}, f"environments.{name}")
        _check_keys(settings.get("evaluation", {}), EVALUATION_FIELDS, f"environments.{name}.evaluation")
        _check_keys(settings.get("common", {}), COMMON_FIELDS, f"environments.{name}.common")
        _check_keys(settings.get("algorithms", {}), set(ALGORITHM_FIELDS), f"environments.{name}.algorithms")
        for algorithm, parameters in settings.get("algorithms", {}).items():
            _check_keys(parameters, ALGORITHM_FIELDS[algorithm], f"environments.{name}.algorithms.{algorithm}")


def _merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        result[key] = _merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else deepcopy(value)
    return result


def resolve_config(path, environment, algorithm, overrides):
    """Bundled defaults < custom YAML < environment values < explicit CLI.

    A custom YAML can override only the settings it needs. Capture exact
    input text now so later edits cannot change the provenance of this run.
    """
    documents, inputs = [], []
    paths = [DEFAULT_CONFIG]
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG:
        paths.append(Path(path).resolve())
    for source in paths:
        content = source.read_text(encoding="utf-8")
        document = yaml.load(content, Loader=UniqueLoader)
        _validate(document)
        documents.append(document)
        inputs.append({"path": str(source), "sha256": hashlib.sha256(content.encode()).hexdigest(), "content": content})
    document = documents[0]
    for override in documents[1:]:
        document = _merge(document, override)
    per_env = document["environments"][environment]
    common = _merge(document["common"], per_env.get("common", {}))
    specific = _merge(document["algorithms"][algorithm], per_env.get("algorithms", {}).get(algorithm, {}))
    _check_keys(overrides, COMMON_FIELDS | ALGORITHM_FIELDS[algorithm], "command line")
    common.update({key: value for key, value in overrides.items() if key in COMMON_FIELDS})
    specific.update({key: value for key, value in overrides.items() if key in ALGORITHM_FIELDS[algorithm]})
    if algorithm == "TAACDiscrete":
        ignored = {"max_epsilon", "min_epsilon", "epsilon_decay_steps", "epsilon_schedule"} & overrides.keys()
        if ignored:
            raise ValueError(f"TAACDiscrete samples its categorical actor; epsilon settings are unsupported: {sorted(ignored)}")
        common["max_epsilon"] = common["min_epsilon"] = 0.0
        common["epsilon_decay_steps"] = None
    if common.get("replay_capacity") is None:
        common["replay_capacity"] = common["total_steps"]
    if common.get("epsilon_decay_steps") is None and algorithm != "TAACDiscrete":
        common["epsilon_decay_steps"] = max(common["total_steps"] - common["warmup_steps"], 1)
    if algorithm in ("TempoRL", "UTE", "RARe"):
        if specific.get("skip_replay_capacity") is None:
            specific["skip_replay_capacity"] = 3 * common["total_steps"]
        for name in ("skip_learning_rate", "eval_learning_rate"):
            if name in specific and specific[name] is None:
                specific[name] = common["learning_rate"]
    return common, specific, inputs


def resolve_evaluation_config(inputs, environment, random, overrides):
    """Compose captured YAML inputs, then apply explicit evaluation CLI values."""
    document = {}
    for source in inputs:
        document = _merge(document, yaml.load(source["content"], Loader=UniqueLoader))
    settings = _merge(document.get("evaluation", {}),
                      document.get("environments", {}).get(environment, {}).get("evaluation", {}))
    random_episodes = settings.pop("random_episodes", "all")
    if random_episodes != "all" and (type(random_episodes) is not int or random_episodes <= 0):
        raise ValueError("evaluation.random_episodes must be a positive integer or 'all'")
    if random:
        settings["episodes"] = random_episodes
    _check_keys(overrides, EVALUATION_FIELDS - {"random_episodes"}, "evaluation command line")
    settings.update(overrides)
    return EvaluationConfig(**settings)
