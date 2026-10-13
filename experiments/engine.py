"""Shared host-side experiment execution, logging and model persistence."""

from bisect import bisect_right
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import os
from pathlib import Path
import sys
import time
import uuid

from .logging import EpisodeTracker, ExperimentLogger
from .models import default_model_save_steps, save_models
from .tracking import WandbTracker


def runtime_metadata(jax):
    versions = {name: importlib.metadata.version(name) for name in ("jax", "jaxlib", "numpy", "optax")}
    flags = (
        "JAX_PLATFORMS", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "XLA_FLAGS",
        "JAX_ENABLE_X64", "XLA_PYTHON_CLIENT_PREALLOCATE", "NVIDIA_TF32_OVERRIDE",
    )
    return {
        "python": sys.version, "python_executable": sys.executable, "versions": versions,
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]), "device_kind": jax.devices()[0].device_kind,
        "environment": {name: os.environ.get(name) for name in flags},
        "jax_config": {name: str(jax.config.values[name]) for name in (
            "jax_enable_x64", "jax_default_matmul_precision", "jax_threefry_partitionable",
        )},
    }


def rng_records(jax, state, seeds, phase, *, snapshot=None):
    streams = state.rngs._asdict()
    implementations = {name: str(jax.random.key_impl(key)) for name, key in streams.items()}
    data, steps = snapshot if snapshot is not None else jax.device_get((
        {name: jax.random.key_data(key) for name, key in streams.items()}, state.training_steps))
    return [{
        "seed": seed, "env_step": int(steps[index]), "phase": phase,
        "rng": {name: {"impl": implementations[name], "data": values[index].tolist()}
                for name, values in data.items()},
    } for index, seed in enumerate(seeds)]


def policy_models(state, algorithm, *, with_risk=False, shared_target_epsilon_source="behavior"):
    models = {"online": state.learner.online_params}
    if algorithm != "DDQN":
        models["skip"] = state.skip_learner.params
    if with_risk:
        # True Risk always measures the policy that actually chooses actions.
        models["risk_policy"] = state.learner.online_params
        if algorithm == "RARe" and state.evaluation is not None:
            models["target_risk_policy"] = state.evaluation.online_params
        elif algorithm == "RARe" and shared_target_epsilon_source == "evaluation":
            # Same greedy critic, but the target epsilon still defines a distinct policy.
            models["target_risk_policy"] = state.learner.online_params
    return models


def evaluation_records(jax, results, keys, seeds, env_step, *, cases=None, key_data=None):
    """Log mean/std of undiscounted episode reward sums and raw episodes."""
    if key_data is None:
        results, key_data = jax.device_get((results, jax.random.key_data(keys)))
    aggregates, episodes = [], []
    for index, seed in enumerate(seeds):
        values = {name: field[index] for name, field in results._asdict().items()}
        aggregates.append({
            "seed": seed, "env_step": env_step, "episodes": len(values["returns"]),
            "reward_sum": float(values["returns"].mean()), "reward_sum_std": float(values["returns"].std()),
            "length_mean": float(values["lengths"].mean()), "decisions_mean": float(values["decisions"].mean()),
            "success_rate": float(values["success"].mean()), "pit_rate": float(values["pit"].mean()),
            "timeout_rate": float(values["truncated"].mean()),
            "rng": {"impl": str(jax.random.key_impl(keys)), "data": key_data[index].tolist()},
        })
        for episode in range(len(values["returns"])):
            episodes.append({
                "seed": seed, "env_step": env_step, "episode": episode + 1,
                "reward_sum": float(values["returns"][episode]), "length": int(values["lengths"][episode]),
                "decisions": int(values["decisions"][episode]),
                **{name: bool(values[name][episode]) for name in ("success", "pit", "terminated", "truncated")},
            })
            if cases is not None:
                episodes[-1].update(case_index=episode, **cases[episode])
    return aggregates, episodes


def compiled_chunk_steps(total_steps, requested, save_steps, evaluation_steps, save_interval):
    """Cap metric buffer capacity at the largest possible scheduled chunk."""
    boundaries = sorted({0, total_steps, *(save_steps or ()), *evaluation_steps})
    largest_gap = max(right - left for left, right in zip(boundaries, boundaries[1:]))
    return min(requested, largest_gap, save_interval or total_steps)


def run_experiment(args, config, env, jax, jnp, np, Runner):
    from .metric_definitions import METRIC_DEFINITIONS

    root = Path(__file__).resolve().parents[1]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    algorithm = args.algorithm
    separate_rare = algorithm == "RARe" and config.rare_q_mode == "separate"
    shared_eval = (algorithm == "RARe" and config.rare_q_mode == "shared"
                   and config.shared_target_epsilon_source == "evaluation")
    distinct_target_policy = separate_rare or shared_eval
    if algorithm == "RARe":
        algorithm_name = "RARe-se" if separate_rare else "RARe-sh-eval" if shared_eval else "RARe-sh"
    else:
        algorithm_name = algorithm
    environment_name = ("R" if args.random else "") + args.env
    launch_id = uuid.uuid4().hex[:8]
    run_name = args.wandb_name or f"{algorithm_name}-{environment_name}-{stamp}-{launch_id}"
    output = args.output or root / "runs" / f"{stamp}-{algorithm_name.lower()}-{environment_name}-{launch_id}"
    resolved = asdict(config)
    save_steps = default_model_save_steps(config.total_steps) if args.model_save_interval is None else None
    evaluation_steps = args.evaluation_config.checkpoints(config.total_steps, config.warmup_steps)
    compiled_steps = compiled_chunk_steps(config.total_steps, args.chunk_steps, save_steps,
                                          evaluation_steps, args.model_save_interval)
    save_step_set = frozenset(save_steps or ())
    evaluation_step_set = frozenset(evaluation_steps)
    from envs.grid import GridEnv
    risk_enabled = (bool(evaluation_steps) and args.evaluation_config.risk_diagnostics
                    and algorithm != "DDQN" and isinstance(env, GridEnv))
    requested_episodes = args.evaluation_config.episodes
    evaluation_cases = None
    evaluation_initial_states = None
    if requested_episodes == "all":
        if not isinstance(env, GridEnv):
            raise ValueError("evaluation episodes='all' requires a GridEnv adapter")
        evaluation_cases = [{"start": list(start), "goal": list(goal)}
                            for start in env.spec.starts for goal in env.spec.goals]
        evaluation_episode_count = len(evaluation_cases)
        if evaluation_steps:
            evaluation_initial_states = env.evaluation_resets()
    else:
        evaluation_episode_count = requested_episodes
    metadata = {
        "algorithm": algorithm, "seeds": args.seeds, "config": resolved,
        "environment": {"name": args.env, "random": args.random, "spec": asdict(env.spec)},
        "execution": {"seed_batching": "vmap", "device_count": 1,
                      "chunk_steps": compiled_steps, "requested_chunk_steps": args.chunk_steps,
                      "logging_snapshot": "one synchronized transfer per training chunk",
                      "collection_loop": "dynamic while loop; inactive metric slots are not executed"},
        "runtime": runtime_metadata(jax),
        "command": [sys.executable, "-m", "experiments.train", *args.launch_argv],
        "working_directory": str(root),
        "metrics_scope": "training and separate frozen greedy-policy evaluation" if evaluation_steps else "training",
        "metric_definitions": METRIC_DEFINITIONS,
        "evaluation": {
            **asdict(args.evaluation_config), "enabled": bool(evaluation_steps),
            "requested_episodes": requested_episodes, "episodes": evaluation_episode_count,
            "reset_mode": "all_start_goal_pairs" if evaluation_cases is not None else "sampled",
            "case_order": "starts_major_goals_minor" if evaluation_cases is not None else None,
            "steps": evaluation_steps, "unit": "primitive_steps_per_seed",
            "schedule_origin": "warmup_end", "warmup_steps": config.warmup_steps,
            "policy": "greedy", "training_state_mutated": False,
            "rng": "fold_in(evaluation_root, env_step), then fold_in(checkpoint_key, episode_index_0based)",
            "ute_skip_selector": "mean + uncertainty_factor * sample_std(ddof=1)",
            "risk": {
                "enabled": risk_enabled, "requested": args.evaluation_config.risk_diagnostics,
                "scope": "GridEnv skip algorithms only", "reference": "exact finite-horizon V_star minus V_pi",
                "policy_source": "base_online_for_every_algorithm",
                "epsilon_source": "behavior_epsilon_including_warmup_for_every_algorithm",
                "shared_rare_target_policy": ("base_online_with_independent_evaluation_epsilon" if shared_eval
                                               else "base_online_with_behavior_epsilon"),
                "separate_rare_target_policy": "evaluation_online_with_independent_evaluation_epsilon",
                "distinct_target_metrics": "target_* for RARe-se or RARe-sh-eval; never pooled with behavior-risk errors",
                "policy_frozen_per_checkpoint": True, "state_distribution": "own greedy evaluation decision points",
                "lengths": "all 1..max_skip at every decision, including unselected and terminal lengths",
                "true_risk": "max_b Q_star(s_j,b) - sum_b pi(b|s_j) Q^pi(s_j,b)",
                "paper_status": "proposed experiment definition; manuscript formula unchanged",
                "estimated_risk": "(exact_optimal_skip_value - predicted_skip_value) / gamma**j",
                "error_identity": "absolute_risk_error = abs(exact_policy_skip_value - predicted_skip_value) / gamma**j",
                "skip_value_reference": "exact expected skip value under the same pi",
                "ute_prediction": "ensemble_mean_without_selection_penalty",
                "terminal_rule": "absorbing; value errors retained; risk errors exclude the ending length and all later lengths",
                "aggregation": "decision-weighted MAE per length, then equal-length mean and population variance over lengths with valid samples",
                "risk_coverage": "per-length valid counts/excluded fractions and valid-length count recorded locally; empty lengths=null",
                "critic_values_used_for_truth": False, "arithmetic": "float32 oracle; float64 host aggregation",
                "interpretation": "implied risk error includes skip-value prediction error; no separate risk head",
                "local_details": "risk_diagnostics.jsonl",
            },
        },
        "protocol": {"exact_primitive_budget": True, "seeded_warmup": True, "time_limit_bootstrap": False},
        "rng_algorithm": "threefry2x32",
        "wandb": {"mode": args.wandb_mode, "project": args.wandb_project,
                  "entity": args.wandb_entity, "name": run_name,
                  "group": algorithm_name, "run_count": 0 if args.wandb_mode == "disabled" else len(args.seeds)},
        "model_saving": {"mode": "fifths" if save_steps is not None else "interval",
                         "interval": args.model_save_interval, "steps": save_steps,
                         "unit": "primitive_steps_per_seed", "save_final": args.model_save_interval != 0,
                         "contents": ["online_params", "target_params"], "full_training_state": False},
    }
    if algorithm != "DDQN":
        metadata["model_saving"]["contents"].append("skip_params")
        metadata["protocol"].update(repeat_warmup="fixed_at_decision", skip_replay_insertion="after_learning_at_repeat_end")
    if algorithm == "RARe":
        if separate_rare:
            metadata["model_saving"]["contents"].extend(["evaluation_online_params", "evaluation_target_params"])
        metadata["protocol"].update(
            rare_q_mode=config.rare_q_mode,
            shared_target_epsilon_source=config.shared_target_epsilon_source,
            evaluation_policy="own_online_epsilon_greedy" if separate_rare else "base_online_epsilon_greedy",
            evaluation_ensemble_size=1,
            skip_bootstrap_critic="separate_expected_sarsa" if separate_rare else "base_ddqn",
        )
    if hasattr(args, "configuration"):
        metadata["configuration"] = args.configuration
        metadata["runtime"]["versions"]["PyYAML"] = importlib.metadata.version("PyYAML")
    runner = Runner(env, config)
    tracker = EpisodeTracker(args.seeds)
    print(f"Experiment: {run_name}; independent seeds ({len(args.seeds)}): {','.join(map(str, args.seeds))}", flush=True)
    print(f"W&B: {args.wandb_mode}; project={args.wandb_project}; group={algorithm_name}; runs={metadata['wandb']['run_count']}", flush=True)
    with ExperimentLogger(output, metadata, source_root=root) as logger, WandbTracker(
        logger.run_dir, logger.metadata, mode=args.wandb_mode, project=args.wandb_project,
        entity=args.wandb_entity, name=run_name,
    ) as tracking:
        if tracking.runs:
            logger.metadata["wandb"].update(
                runs=[{"seed": seed, "id": run.id, "name": run.name} for seed, run in tracking.runs.items()],
                version=importlib.metadata.version("wandb"),
            )
        print(f"Run directory: {logger.run_dir}", flush=True)
        if hasattr(args, "configuration_inputs"):
            import yaml
            inputs_dir = logger.run_dir / "configuration"
            inputs_dir.mkdir()
            for index, source in enumerate(args.configuration_inputs):
                (inputs_dir / f"input_{index}.yaml").write_text(source["content"], encoding="utf-8")
            (logger.run_dir / "resolved.yaml").write_text(yaml.safe_dump({
                "algorithm": algorithm, "environment": metadata["environment"], "seeds": args.seeds,
                "common": args.configuration["common"], "specific": args.configuration["algorithm"],
                "execution": metadata["execution"], "model_saving": metadata["model_saving"],
                "evaluation": metadata["evaluation"],
                "wandb": metadata["wandb"],
            }, sort_keys=False), encoding="utf-8")
        started = time.perf_counter()
        state = jax.jit(jax.vmap(runner.initialize))(jnp.asarray(args.seeds, dtype=jnp.uint32))
        jax.block_until_ready(state)
        initialization_seconds = time.perf_counter() - started
        logger.append("rng", rng_records(jax, state, args.seeds, "initialized"))
        evaluate = None
        evaluation_compilation_seconds = 0.
        evaluation_seconds = 0.
        if evaluation_steps:
            from runners.evaluation import evaluate_policy
            compile_start = time.perf_counter()
            if risk_enabled:
                from diagnostics.grid_risk import evaluate_grid_risk
                evaluate = jax.jit(jax.vmap(lambda models, key, eps, target_eps: evaluate_grid_risk(
                    env, models["online"], models["skip"], models["risk_policy"], eps, key,
                    gamma=config.gamma, algorithm=algorithm, episodes=evaluation_episode_count,
                    uncertainty_factor=getattr(config, "uncertainty_factor", -1.5),
                    initial_states=evaluation_initial_states,
                    target_policy_params=models.get("target_risk_policy"),
                    target_epsilon=target_eps if distinct_target_policy else None,
                ))).lower(policy_models(state, algorithm, with_risk=True,
                                       shared_target_epsilon_source=config.shared_target_epsilon_source),
                          state.rngs.evaluation,
                          state.training_steps.astype(jnp.float32),
                          state.training_steps.astype(jnp.float32)).compile()
            else:
                evaluate = jax.jit(jax.vmap(lambda models, key: evaluate_policy(
                    env, models["online"], key, episodes=evaluation_episode_count,
                    algorithm=algorithm, outcome_fn=env.episode_outcome, skip_params=models.get("skip"),
                    uncertainty_factor=getattr(config, "uncertainty_factor", -1.5),
                    initial_states=evaluation_initial_states,
                ))).lower(policy_models(state, algorithm), state.rngs.evaluation).compile()
            evaluation_compilation_seconds = time.perf_counter() - compile_start
        compilation_seconds = 0.
        execution_and_transfer_seconds = 0.
        training_start = time.perf_counter()
        compile_start = time.perf_counter()

        def execute_chunk(current, count):
            updated, metrics = runner.run(current, compiled_steps, active_steps=count)
            snapshot = {
                "counters": (updated.training_steps, updated.learner.optimizer_state.count,
                             updated.episodes, updated.episode_return, updated.replay.size),
                "epsilon": runner._epsilon(updated.training_steps),
                "learning_rate": runner._learning_rate(updated.training_steps),
                "rng": {name: jax.random.key_data(key) for name, key in updated.rngs._asdict().items()},
            }
            if algorithm != "DDQN":
                snapshot["skip"] = (updated.skip_replay.size, updated.skip_learner.optimizer_state.count,
                                    updated.decisions, runner._rate(updated.training_steps, config.skip_learning_rate))
            if algorithm == "RARe":
                snapshot["evaluation_epsilon"] = runner._evaluation_epsilon(updated.training_steps)
                if separate_rare:
                    snapshot["evaluation_learning_rate"] = runner._rate(updated.training_steps, config.eval_learning_rate)
            if risk_enabled:
                # Greedy evaluation selects visited states, but this frozen
                # counterfactual policy matches the current training action policy.
                snapshot["risk_epsilon"] = jnp.where(updated.training_steps < config.warmup_steps,
                                                     1., snapshot["epsilon"])
            return updated, metrics, snapshot

        execute = jax.jit(jax.vmap(execute_chunk, in_axes=(0, None)), donate_argnums=(0,)).lower(
            state, jnp.asarray(compiled_steps, jnp.int32)).compile()
        compilation_seconds = time.perf_counter() - compile_start
        start_step = 0
        while start_step < config.total_steps:
            chunk_steps = min(args.chunk_steps, config.total_steps - start_step)
            if save_steps is not None:
                next_save = save_steps[bisect_right(save_steps, start_step)]
                chunk_steps = min(chunk_steps, next_save - start_step)
            elif args.model_save_interval:
                next_save = (start_step // args.model_save_interval + 1) * args.model_save_interval
                chunk_steps = min(chunk_steps, next_save - start_step)
            if evaluation_steps:
                next_evaluation = evaluation_steps[bisect_right(evaluation_steps, start_step)]
                chunk_steps = min(chunk_steps, next_evaluation - start_step)
            chunk_start = time.perf_counter()
            state, metrics, device_snapshot = execute(state, jnp.asarray(chunk_steps, jnp.int32))
            metrics, snapshot = jax.device_get((metrics, device_snapshot))
            counters = snapshot["counters"]
            chunk_seconds = time.perf_counter() - chunk_start
            execution_and_transfer_seconds += chunk_seconds
            episode_records = tracker.records(metrics, start_step)
            epsilon, rate = snapshot["epsilon"], snapshot["learning_rate"]
            if algorithm != "DDQN":
                skip_sizes, skip_updates, decisions, skip_rates = snapshot["skip"]
            if algorithm == "RARe":
                eval_epsilons = snapshot["evaluation_epsilon"]
                if separate_rare:
                    eval_rates = snapshot["evaluation_learning_rate"]
            records = []
            for index, seed in enumerate(args.seeds):
                active, learned = metrics.active[index], metrics.learned[index]
                ended = active & (metrics.terminated[index] | metrics.truncated[index])
                steps, updates, episodes, partial_return, replay_size = (values[index] for values in counters)
                records.append({
                    "seed": seed, "env_step_start": start_step + 1, "env_step": int(steps),
                    "interactions": int(active.sum()), "base_q_updates": int(learned.sum()),
                    "base_q_updates_total": int(updates),
                    "loss_mean": float(metrics.loss[index][learned].mean()) if learned.any() else None,
                    "reward_sum": float(metrics.reward[index][active].sum()),
                    "episodes_finished": int(ended.sum()), "episodes_total": int(episodes),
                    "episode_return_mean": float(metrics.episode_return[index][ended].mean()) if ended.any() else None,
                    "partial_episode_return": float(partial_return), "partial_episode_length": tracker.lengths[index],
                    "replay_size": int(replay_size),
                    "epsilon": 1. if int(steps) < config.warmup_steps else float(epsilon[index]),
                    "next_epsilon": 1. if int(steps) < config.warmup_steps else float(epsilon[index]),
                    "next_learning_rate": float(rate[index] if np.ndim(rate) else rate),
                    "chunk_seconds": chunk_seconds,
                })
                if algorithm != "DDQN":
                    skip_learned = metrics.skip_learned[index]
                    selected = metrics.selected_skip[index][metrics.decision[index] & active]
                    executed = metrics.executed_skip[index][metrics.segment_completed[index] & active]
                    records[-1].update(
                        skip_loss_mean=float(metrics.skip_loss[index][skip_learned].mean()) if skip_learned.any() else None,
                        skip_q_updates=int(skip_learned.sum()),
                        skip_q_updates_total=int(skip_updates[index]),
                        skip_replay_size=int(skip_sizes[index]), decisions=int(metrics.decision[index].sum()),
                        decisions_total=int(decisions[index]), selected_skip_mean=float(selected.mean()) if selected.size else None,
                        executed_skip_mean=float(executed.mean()) if executed.size else None,
                        next_skip_epsilon=float(epsilon[index]),
                        next_skip_learning_rate=float(skip_rates[index] if np.ndim(skip_rates) else skip_rates),
                    )
                if algorithm == "RARe":
                    records[-1]["expected_policy_epsilon"] = float(eval_epsilons[index])
                    records[-1]["next_evaluation_epsilon"] = float(eval_epsilons[index])
                if separate_rare:
                    records[-1].update(
                        evaluation_loss_mean=float(metrics.evaluation_loss[index][learned].mean()) if learned.any() else None,
                        next_evaluation_learning_rate=float(eval_rates[index] if np.ndim(eval_rates) else eval_rates),
                    )
                if int(episodes) != tracker.completed[index]:
                    raise RuntimeError("episode logs disagree with runner counters")
            logger.append("metrics", records)
            logger.append("episodes", episode_records)
            finished = int(counters[0][0]) == config.total_steps
            logger.append("rng", rng_records(jax, state, args.seeds, "completed" if finished else "progress",
                                             snapshot=(snapshot["rng"], counters[0])))
            end_step = int(counters[0][0])
            should_save = end_step in save_step_set if save_steps is not None else (
                args.model_save_interval and (end_step % args.model_save_interval == 0 or finished)
            )
            if should_save:
                relative_path = f"models/step_{end_step:09d}.npz"
                models = {"online": state.learner.online_params, "target": state.learner.target_params}
                if algorithm != "DDQN":
                    models["skip"] = state.skip_learner.params
                if separate_rare:
                    models.update(evaluation_online=state.evaluation.online_params, evaluation_target=state.evaluation.target_params)
                save_models(
                    logger.run_dir / relative_path,
                    models,
                    seeds=args.seeds, env_step=end_step, source_sha256=logger.metadata["source"]["sha256"],
                )
                logger.record_model(relative_path, end_step)
            tracking.log(records, episode_records)
            if end_step in evaluation_step_set:
                evaluation_start = time.perf_counter()
                # Derive checkpoint keys without advancing any training stream.
                keys = jax.vmap(lambda key: jax.random.fold_in(key, end_step))(state.rngs.evaluation)
                if risk_enabled:
                    risk_epsilon = device_snapshot["risk_epsilon"]
                    target_risk_epsilon = device_snapshot.get("evaluation_epsilon", risk_epsilon)
                    risk_results, key_data = jax.device_get((
                        evaluate(policy_models(state, algorithm, with_risk=True,
                                               shared_target_epsilon_source=config.shared_target_epsilon_source), keys,
                                 risk_epsilon, target_risk_epsilon),
                        jax.random.key_data(keys)))
                    results = risk_results.episodes
                else:
                    results, key_data = jax.device_get((evaluate(policy_models(state, algorithm), keys),
                                                       jax.random.key_data(keys)))
                evaluation_rows, evaluation_episodes = evaluation_records(
                    jax, results, keys, args.seeds, end_step, cases=evaluation_cases, key_data=key_data,
                )
                for index, row in enumerate(evaluation_rows):
                    row["policy_epsilon"] = 0.0
                    if risk_enabled:
                        row["risk_epsilon"] = float(snapshot["risk_epsilon"][index])
                        if distinct_target_policy:
                            row["target_risk_epsilon"] = float(snapshot["evaluation_epsilon"][index])
                if risk_enabled:
                    from diagnostics.grid_risk import aggregate_risk_errors, aggregate_target_risk_errors
                    detail_rows = []
                    for index, seed in enumerate(args.seeds):
                        result = jax.tree.map(lambda value: value[index], risk_results)
                        risk_metrics, details = aggregate_risk_errors(result)
                        details["policy_source"] = "base_online"
                        if distinct_target_policy:
                            target_metrics, target_details = aggregate_target_risk_errors(result)
                            risk_metrics.update(target_metrics)
                            details["target_policy"] = dict(
                                target_details, policy_source="evaluation_online" if separate_rare else "base_online")
                        evaluation_rows[index].update(risk_metrics)
                        detail_rows.append({"seed": seed, "env_step": end_step, **details})
                    logger.append("risk_diagnostics", detail_rows)
                logger.append("evaluations", evaluation_rows)
                logger.append("evaluation_episodes", evaluation_episodes)
                tracking.log_evaluation(evaluation_rows)
                evaluation_seconds += time.perf_counter() - evaluation_start
            print(f"Steps per seed: {int(counters[0][0])}/{config.total_steps}; seeds: {len(args.seeds)}", flush=True)
            start_step = end_step
        if not np.all(counters[0] == config.total_steps):
            raise RuntimeError("training ended before the exact interaction budget")
        summary = {
            "initialization_seconds": initialization_seconds, "compilation_seconds": compilation_seconds,
            "compilation_count": 1, "compiled_chunk_steps": compiled_steps,
            "execution_and_transfer_seconds": execution_and_transfer_seconds,
            "training_and_logging_seconds": time.perf_counter() - training_start - compilation_seconds - evaluation_seconds,
            "evaluation_seconds": evaluation_seconds,
            "evaluation_compilation_seconds": evaluation_compilation_seconds,
            "evaluation_compilation_count": int(bool(evaluation_steps)),
            "evaluations_per_seed": len(evaluation_steps),
            "per_seed": [{"seed": seed, "env_steps": int(counters[0][i]),
                          "updates": int(counters[1][i]), "episodes": int(counters[2][i])}
                         for i, seed in enumerate(args.seeds)],
        }
        tracking.complete(summary)
        logger.complete(summary)
    return logger.run_dir
