"""Run directories: run.json plus strict JSONL logs, written outside compiled code."""

from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys


def _now():
    return datetime.now(timezone.utc).isoformat()


def _dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)


class RunLog:
    """Refuse an existing directory, so repeating a command cannot overwrite a run."""

    CHANNELS = ("evaluations", "evaluation_decisions")

    def __init__(self, directory, metadata):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        versions = {name: importlib.metadata.version(name) for name in ("jax", "jaxlib", "numpy", "optax")}
        self.metadata = {**metadata, "status": "running", "started_at": _now(),
                         "python": sys.version, "versions": versions}
        self._write()
        self._files = {name: (self.directory / f"{name}.jsonl").open("x", encoding="utf-8")
                       for name in self.CHANNELS}

    def _write(self):
        temporary = self.directory / "run.tmp"
        temporary.write_text(_dumps(self.metadata) + "\n", encoding="utf-8")
        temporary.replace(self.directory / "run.json")

    def append(self, channel, records):
        self._files[channel].write("".join(_dumps(record) + "\n" for record in records))
        self._files[channel].flush()

    def __enter__(self):
        return self

    def __exit__(self, error_type, error, traceback):
        for stream in self._files.values():
            stream.close()
        self.metadata.update(status="completed" if error_type is None else "failed", finished_at=_now())
        self._write()
        return False
