"""Binary local bridge to the pinned Python 3.10 Safety-Gymnasium runtime."""

import os
from pathlib import Path
import pickle
import struct
import subprocess


ROOT = Path(__file__).resolve().parents[3]


class ReferenceBridge:
    """Talk to rare.envs.safety.server, started with the rare-safety interpreter."""

    def __init__(self, task, seeds, log_path):
        python = os.environ.get("RARE_SAFETY_PYTHON")
        if not python:
            raise RuntimeError("set RARE_SAFETY_PYTHON to the python of the rare-safety conda environment")
        self.task = task
        self.seeds = tuple(int(seed) for seed in seeds)
        self.log = Path(log_path).open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            [python, "-m", "rare.envs.safety.server"], cwd=ROOT, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self.log, bufsize=0)
        try:
            self.spec = self.call("open", task, self.seeds)
        except BaseException:
            self.close()
            raise

    def call(self, command, *args):
        payload = pickle.dumps((command, *args), protocol=5)
        self.process.stdin.write(struct.pack("!Q", len(payload)))
        self.process.stdin.write(payload)
        self.process.stdin.flush()
        header = self.process.stdout.read(8)
        if len(header) != 8:
            raise RuntimeError(f"Safety bridge exited during {command}; see bridge log")
        size = struct.unpack("!Q", header)[0]
        response = self.process.stdout.read(size)
        if len(response) != size:
            raise RuntimeError(f"incomplete Safety bridge response during {command}")
        ok, result = pickle.loads(response)
        if not ok:
            raise RuntimeError(f"Safety bridge {command} failed:\n{result}")
        return result

    def close(self):
        if self.process.poll() is None:
            try:
                self.call("close")
            except (BrokenPipeError, RuntimeError, OSError):
                pass
            self.process.wait(timeout=30)
        self.log.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
