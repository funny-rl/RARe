"""Geometry-based RODR oracle and raw Grid decision serialization."""

from collections import deque
from dataclasses import dataclass

import numpy as np

from rare.envs.grid.core import GridEnv


@dataclass(frozen=True)
class GridRODROracle:
    valid_repeats: dict
    distances: dict
    transitions: np.ndarray
    pits: np.ndarray
    max_steps: int


def build_grid_rodr_oracle(env: GridEnv, max_skip: int):
    """Build best safe repeats and goal distances for each Grid goal.

    Actions reduce shortest-path distance; repeats keep the smallest safe
    endpoint distance. If a repeat reaches the goal, all lengths that reach it
    are valid, since execution terminates there.
    """
    if not isinstance(env, GridEnv) or not env.supports_exact_risk:
        raise TypeError("RODR requires a deterministic GridEnv")
    if type(max_skip) is not int or max_skip < 1:
        raise ValueError("max_skip must be positive")
    transitions = np.asarray(env.params.transitions)
    pits = np.asarray(env.params.pit_mask, dtype=bool)
    oracle = {}
    goal_distances = {}
    for goal in map(int, np.asarray(env.params.goals)):
        predecessors = [[] for _ in range(len(transitions))]
        for cell, successors in enumerate(transitions):
            if pits[cell]:
                continue
            for successor in successors:
                successor = int(successor)
                if successor != cell and not pits[successor]:
                    predecessors[successor].append(cell)
        distances = np.full(len(transitions), np.inf)
        distances[goal] = 0
        queue = deque([goal])
        while queue:
            successor = queue.popleft()
            for cell in predecessors[successor]:
                if np.isinf(distances[cell]):
                    distances[cell] = distances[successor] + 1
                    queue.append(cell)
        goal_distances[goal] = distances

        for cell, successors in enumerate(transitions):
            if pits[cell] or cell == goal or not np.isfinite(distances[cell]):
                continue
            for action, successor in enumerate(successors):
                successor = int(successor)
                if (successor == cell or pits[successor] or
                        not distances[successor] < distances[cell]):
                    continue
                position = cell
                best_distance = np.inf
                valid = set()
                for repeat in range(1, max_skip + 1):
                    position = int(transitions[position, action])
                    if pits[position]:
                        break
                    if position == goal:
                        valid = set(range(repeat, max_skip + 1))
                        break
                    if distances[position] < best_distance:
                        best_distance = distances[position]
                        valid = {repeat}
                if valid:
                    oracle[goal, cell, action] = frozenset(valid)
    return GridRODROracle(oracle, goal_distances, transitions, pits,
                          env.spec.max_steps)


def score_rodr_decisions(decisions, oracle):
    """Credit oracle decisions only when their actual endpoint sets a distance record."""
    count = len(decisions)
    correct = 0
    if not count:
        return {"rodr": 0., "rodr_correct_count": 0, "decision_count": 0}
    first = decisions[0]["observation"]
    goal = int(first["goal_index"])
    distances = oracle.distances[goal]
    best_distance = distances[int(first["agent_index"])]
    for decision in decisions:
        observation = decision["observation"]
        position = int(observation["agent_index"])
        action = int(decision["action"])
        matched = int(decision["skip"]) in oracle.valid_repeats.get(
            (goal, position, action), ())
        step = int(decision["primitive_step"])
        before_endpoint_best = best_distance
        for repeat in range(int(decision["skip"])):
            position = int(oracle.transitions[position, action])
            step += 1
            if oracle.pits[position] or position == goal or step >= oracle.max_steps:
                break
            if repeat + 1 < int(decision["skip"]):
                before_endpoint_best = min(before_endpoint_best, distances[position])
        endpoint_distance = distances[position]
        correct += matched and endpoint_distance < before_endpoint_best
        best_distance = min(before_endpoint_best, endpoint_distance)
    return {"rodr": correct / count,
            "rodr_correct_count": int(correct), "decision_count": count}


def serialize_grid_trace(trace):
    """Record each decision without calculating RODR during training."""
    count = int(trace.count)
    records = []
    for index in range(count):
        position = int(trace.positions[index])
        goal = int(trace.goals[index])
        action = int(trace.actions[index])
        skip = int(trace.skips[index])
        record = {
            "primitive_step": int(trace.steps[index]),
            "observation": {"agent_index": position, "goal_index": goal},
            "action": action, "skip": skip,
            "q_values": np.asarray(trace.q_values[index]).tolist(),
            "skip_values": np.asarray(trace.skip_values[index]).tolist(),
            "skip_selection_scores": np.asarray(trace.skip_selection_scores[index]).tolist(),
        }
        records.append(record)
    return records


def serialize_grid_episodes(traces, *, seed, env_step, cases=None):
    """Store every evaluation episode so offline RODR can be reproduced."""
    count = len(traces.count)
    rows = []
    for episode in range(count):
        trace = type(traces)(*(field[episode] for field in traces))
        row = {"seed": seed, "env_step": env_step, "episode": episode + 1,
               "decision_count": int(trace.count), "decisions": serialize_grid_trace(trace)}
        if cases is not None:
            row.update(case_index=episode, **cases[episode])
        rows.append(row)
    return rows
