"""Run the pinned Safety-Gymnasium vector environment in its own Python 3.10 process."""

import pickle
import struct
import sys
import traceback


_OUT = sys.stdout.buffer
sys.stdout = sys.stderr

from rare.envs.safety.vector import make_vector_env


def _read():
    header = sys.stdin.buffer.read(8)
    if len(header) != 8:
        return None
    size = struct.unpack("!Q", header)[0]
    payload = sys.stdin.buffer.read(size)
    if len(payload) != size:
        raise EOFError("incomplete bridge message")
    return pickle.loads(payload)


def _write(value):
    payload = pickle.dumps(value, protocol=5)
    _OUT.write(struct.pack("!Q", len(payload)))
    _OUT.write(payload)
    _OUT.flush()


def main():
    train = evaluate = None
    try:
        while True:
            message = _read()
            if message is None:
                break
            try:
                command = message[0]
                if command == "open":
                    _, task, seeds = message
                    train = make_vector_env(task, seeds, asynchronous=True)
                    evaluate = make_vector_env(task, seeds, asynchronous=True)
                    result = {"observation_shape": train.observation_shape,
                              "action_shape": train.action_shape,
                              "action_low": train.action_low,
                              "action_high": train.action_high}
                elif command == "reset_train":
                    result = train.reset()
                elif command == "step_train":
                    result = train.step(message[1])
                elif command == "reset_eval":
                    result = evaluate.reset(message[1])
                elif command == "step_eval":
                    result = evaluate.step(message[1])
                elif command == "close":
                    _write((True, None))
                    break
                else:
                    raise ValueError(f"unknown bridge command: {command}")
                _write((True, result))
            except BaseException:
                _write((False, traceback.format_exc()))
                break
    finally:
        if train is not None:
            train.close()
        if evaluate is not None:
            evaluate.close()


if __name__ == "__main__":
    main()
