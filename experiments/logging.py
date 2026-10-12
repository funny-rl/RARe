"""Local experiment records and source snapshots, outside compiled training."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


_SOURCE_DIRS = ("algorithms", "buffers", "envs", "networks", "optimizers", "policies", "runners", "experiments", "diagnostics", "configs", "scripts")
_CHANNELS = ("metrics", "episodes", "rng", "evaluations", "evaluation_episodes", "risk_diagnostics")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)


def _atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(_json(value) + "\n", encoding="utf-8")
    temporary.replace(path)


def snapshot_source(source_root, run_dir):
    """Archive executable project sources, including uncommitted/untracked code.

    Only the listed code directories, rng.py and .gitignore are included.
    Paper, tests, credentials, Git internals and experiment outputs are excluded.
    Store per-file SHA256 plus a deterministic aggregate fingerprint.
    """
    source_root, run_dir = Path(source_root).resolve(), Path(run_dir)
    paths = [source_root / name for name in ("rng.py", ".gitignore") if (source_root / name).is_file()]
    for name in _SOURCE_DIRS:
        paths.extend((source_root / name).rglob("*.py"))
    for pattern in ("*.yaml", "*.yml"):
        paths.extend((source_root / "configs").rglob(pattern))
    paths.extend((source_root / "scripts" / "grid").rglob("*.sh"))
    hashes = {}
    with zipfile.ZipFile(run_dir / "source.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            if path.resolve() != path.absolute():
                raise ValueError(f"source snapshot requires regular project paths: {path}")
            relative = path.relative_to(source_root).as_posix()
            content = path.read_bytes()
            hashes[relative] = hashlib.sha256(content).hexdigest()
            archive.writestr(relative, content)
    fingerprint = hashlib.sha256(_json(hashes).encode("utf-8")).hexdigest()
    def git(*args):
        result = subprocess.run(
            ["git", *args], cwd=source_root, text=True, capture_output=True, check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    status = git("status", "--porcelain", "--", *_SOURCE_DIRS, "rng.py", ".gitignore")
    return {
        "archive": "source.zip", "sha256": fingerprint, "files": hashes,
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status), "git_source_status": status,
    }


class ExperimentLogger:
    """Create a fresh run directory and append strict JSONL in flushed batches.

    run.json is replaced atomically on status changes. Existing directories
    are refused, so repeating a command cannot overwrite a previous run.
    RNG logs identify draws; they are not full replay/optimizer checkpoints.
    """

    def __init__(self, run_dir, metadata, *, source_root):
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.metadata = dict(metadata, schema_version=1, status="running", started_at=utc_now(),
                             checkpoint_saved=False, model_saved=False)
        self._files = {}
        self._finished = False
        try:
            _atomic_json(self.run_dir / "run.json", self.metadata)
            self.metadata["source"] = snapshot_source(source_root, self.run_dir)
            _atomic_json(self.run_dir / "run.json", self.metadata)
            for name in _CHANNELS:
                self._files[name] = (self.run_dir / f"{name}.jsonl").open("x", encoding="utf-8")
        except BaseException as error:
            for stream in self._files.values():
                stream.close()
            self.metadata.update(status="failed", finished_at=utc_now(),
                                 error={"type": type(error).__name__, "message": str(error)})
            _atomic_json(self.run_dir / "run.json", self.metadata)
            raise

    def append(self, channel, records):
        """Write one host-side batch, rejecting NaN/Inf instead of invalid JSON."""
        if self._finished:
            raise RuntimeError("experiment log is already finished")
        payload = "".join(_json(record) + "\n" for record in records)
        self._files[channel].write(payload)
        self._files[channel].flush()

    def complete(self, summary):
        self.metadata.update(status="completed", finished_at=utc_now(), summary=summary)
        _atomic_json(self.run_dir / "run.json", self.metadata)
        self._finished = True

    def record_model(self, relative_path, env_step):
        """Publish an already completed weight archive in the local manifest."""
        self.metadata.setdefault("models", []).append({"path": relative_path, "env_step": env_step})
        self.metadata["model_saved"] = True
        _atomic_json(self.run_dir / "run.json", self.metadata)

    def __enter__(self):
        return self

    def __exit__(self, error_type, error, traceback):
        try:
            if not self._finished:
                status = "interrupted" if error_type in (KeyboardInterrupt, SystemExit) or error_type is None else "failed"
                self.metadata.update(status=status, finished_at=utc_now())
                if error is not None:
                    self.metadata["error"] = {"type": error_type.__name__, "message": str(error)}
                _atomic_json(self.run_dir / "run.json", self.metadata)
        finally:
            for stream in self._files.values():
                stream.close()
        return False


class EpisodeTracker:
    """Track episode lengths across logging chunks without consuming RNG."""

    def __init__(self, seeds):
        self.seeds = tuple(seeds)
        self.completed = [0] * len(seeds)
        self.lengths = [0] * len(seeds)

    def records(self, metrics, start_step):
        """Accept host StepMetrics with a contiguous active prefix per seed."""
        records = []
        for index, seed in enumerate(self.seeds):
            for slot, active in enumerate(metrics.active[index]):
                if not active:
                    continue
                self.lengths[index] += 1
                terminated = bool(metrics.terminated[index, slot])
                truncated = bool(metrics.truncated[index, slot])
                if terminated or truncated:
                    self.completed[index] += 1
                    records.append({
                        "seed": seed, "episode": self.completed[index], "env_step": start_step + slot + 1,
                        "length": self.lengths[index], "return": float(metrics.episode_return[index, slot]),
                        "terminated": terminated, "truncated": truncated,
                    })
                    self.lengths[index] = 0
        return records
