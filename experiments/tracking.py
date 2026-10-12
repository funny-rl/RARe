"""Optional host-side W&B runs, one per independent training seed."""

import importlib.metadata


class WandbTracker:
    """Keep batched training in separate W&B runs grouped by algorithm.

    Disabled mode never imports the SDK. Source, paper, model files and RNG
    dumps are not uploaded. Offline mode writes only local W&B records.
    """

    def __init__(self, run_dir, metadata, *, mode="disabled", project="RARe", entity=None, name=None):
        self.runs = {}
        self._finished = set()
        if mode == "disabled":
            return
        import wandb

        config = {key: value for key, value in metadata.items() if key in (
            "algorithm", "seeds", "config", "environment", "execution", "runtime",
            "metrics_scope", "protocol", "rng_algorithm", "model_saving", "evaluation",
        )}
        config["source_sha256"] = metadata["source"]["sha256"]
        config["git_commit"] = metadata["source"]["git_commit"]
        config["wandb_version"] = importlib.metadata.version("wandb")
        if "configuration" in metadata:
            config["hyperparameters"] = {key: metadata["configuration"][key] for key in ("common", "algorithm")}
        environment = metadata["environment"]
        environment_name = ("R" if environment["random"] else "") + environment["name"]
        group = metadata.get("wandb", {}).get("group", metadata["algorithm"])
        base_name = name or f"{group}-{environment_name}-{metadata['source']['sha256'][:8]}"
        config["launch_name"] = base_name
        try:
            for seed in metadata["seeds"]:
                run = wandb.init(
                    mode=mode, project=project, entity=entity,
                    name=f"{base_name}-seed{seed}", group=group, job_type="train",
                    reinit="create_new", dir=str(run_dir), config=dict(config, seed=seed), save_code=False,
                    settings={"disable_git": True, "disable_code": True, "console": "off",
                              "x_disable_stats": True, "x_disable_meta": True},
                )
                self.runs[seed] = run
                run.define_metric("env_step", hidden=True, summary="none")
                run.define_metric("train/*", step_metric="env_step")
                run.define_metric("eval/*", step_metric="env_step")
        except BaseException:
            self._finish_runs(exit_code=1)
            raise

    def log(self, metrics, episodes):
        """Flatten all training records under train/* with an environment-step axis."""
        if not self.runs:
            return
        events = []
        for record in metrics:
            events.append((record["seed"], {
                "env_step": record["env_step"], **{
                    "train/" + key: value for key, value in record.items()
                    if key not in ("seed", "env_step", "env_step_start") and value is not None
                },
            }))
        for record in episodes:
            events.append((record["seed"], {"env_step": record["env_step"], **{
                "train/" + ("episode_index" if key == "episode" else key): value
                for key, value in record.items() if key not in ("seed", "env_step")
            }}))
        # Episode endings precede the chunk summary when their step is earlier.
        # Separate rows retain all episodes, including equal-step records.
        for seed, payload in sorted(events, key=lambda event: event[1]["env_step"]):
            self.runs[seed].log(payload)

    def log_evaluation(self, records):
        """Route frozen-policy performance and risk metrics to each seed."""
        if not self.runs:
            return
        for record in records:
            self.runs[record["seed"]].log({"env_step": record["env_step"], **{
                "eval/" + key: value for key, value in record.items()
                if key not in ("seed", "env_step", "rng") and value is not None
            }})

    def _finish_runs(self, exit_code):
        errors = []
        for seed, run in self.runs.items():
            if seed not in self._finished:
                try:
                    run.finish(exit_code=exit_code)
                except BaseException as error:
                    errors.append(error)
                else:
                    self._finished.add(seed)
        if errors:
            raise errors[0]

    def complete(self, summary):
        common = {key: value for key, value in summary.items() if key != "per_seed"}
        per_seed = {record["seed"]: record for record in summary["per_seed"]}
        for seed, run in self.runs.items():
            run.summary.update(dict(common, **per_seed[seed]))
        self._finish_runs(exit_code=0)

    def __enter__(self):
        return self

    def __exit__(self, error_type, error, traceback):
        self._finish_runs(exit_code=1)
        return False
