"""Pure functions of the table pipeline: statistics, normalization, RODR, shard merging."""

import json

import numpy as np
import pytest

from paper.data import Runs, post_change_auc
from paper.rodr import replay_episode
from paper.tables import column, interval_cell
from rare.envs.grid.core import GridEnv, GridSpec
from rare.metrics.auc import grid_reward_ceiling, normalize_rewards, normalize_to_reference, span_auc
from rare.metrics.rodr import build_grid_rodr_oracle, score_rodr_decisions
from rare.metrics.stats import bootstrap_ci, iqm, paired

RIGHT = 2
CORRIDOR = GridSpec("Corridor", (1, 5), ((0, 0),), ((0, 4),), max_steps=10)


def decision(position, step, skip, action=RIGHT):
    return {"observation": {"agent_index": position, "goal_index": 4}, "primitive_step": step,
            "action": action, "skip": skip}


def test_iqm_is_the_mean_of_the_middle_half():
    values = np.arange(20.)
    assert iqm(values) == pytest.approx(9.5)
    values[-1] = 1e6
    assert iqm(values) == pytest.approx(9.5)


def test_bootstrap_is_deterministic_and_covers_the_iqm():
    values = np.random.default_rng(3).normal(size=20)
    low, high = bootstrap_ci(values)
    assert (low, high) == bootstrap_ci(values)
    assert low < iqm(values) < high
    assert (low, high) != bootstrap_ci(values, seed=1)


def test_paired_resamples_seeds_jointly():
    values = np.random.default_rng(4).normal(size=20)
    assert paired(values, values) == {"difference": 0., "ci": (0., 0.), "win_rate": .5}
    shifted = paired(values + 1., values)
    assert shifted["difference"] == pytest.approx(1.) and shifted["ci"] == pytest.approx((1., 1.))
    assert shifted["win_rate"] == 1.


def test_auc_normalization():
    assert normalize_rewards([-150., -100., -55., -10.], -100., [-10., -10., -10., -10.]) == pytest.approx([0, 0, .5, 1])
    with pytest.raises(ValueError):
        normalize_rewards([-5.], -100., [-10.])
    assert normalize_to_reference([-3., 0., 15.], 0., 10.) == pytest.approx([0., 0., 1.5])  # No upper clip.
    assert span_auc([0, 1, 3], [[1., 1., 1.], [0., 0., 2.]]) == pytest.approx([1., 2. / 3.])


def test_reward_ceiling_is_the_shortest_safe_return():
    spec = {"shape": [2, 3], "starts": [[0, 0]], "goals": [[0, 2]], "pits": [[0, 1]], "actions": [[0, -1], [-1, 0], [0, 1], [1, 0]],
            "max_steps": 10, "step_reward": -1., "goal_reward": -1., "pit_reward": -100.}
    assert grid_reward_ceiling(spec)["cases"][0]["optimal_reward"] == -4.


def test_rodr_credits_maximal_safe_progress_only():
    oracle = build_grid_rodr_oracle(GridEnv(CORRIDOR), 2)
    assert score_rodr_decisions([decision(0, 0, 2), decision(2, 2, 2)], oracle)["rodr"] == 1.
    short = [decision(0, 0, 1), decision(1, 1, 2), decision(3, 3, 1)]  # The first repeat stops early.
    assert score_rodr_decisions(short, oracle) == {"rodr": 2 / 3, "rodr_correct_count": 2, "decision_count": 3}
    backward = [decision(0, 0, 2), decision(2, 2, 2, action=0), decision(0, 4, 2)]  # Only recovers lost ground.
    assert score_rodr_decisions(backward, oracle)["rodr_correct_count"] == 1


def test_rodr_stops_repeats_before_hazards():
    spec = GridSpec("Pit", (1, 6), ((0, 0),), ((0, 5),), pits=((0, 3),), max_steps=10)
    oracle = build_grid_rodr_oracle(GridEnv(spec), 4)
    assert (5, 0, RIGHT) not in oracle.valid_repeats  # The goal is unreachable: no safe path.
    open_spec = GridSpec("Row", (2, 6), ((0, 0),), ((0, 5),), pits=((0, 3),), max_steps=20)
    oracle = build_grid_rodr_oracle(GridEnv(open_spec), 4)
    assert oracle.valid_repeats[5, 6, RIGHT] == frozenset({4})  # The lower row is free.


def test_replay_recovers_the_episode_outcome():
    oracle = build_grid_rodr_oracle(GridEnv(CORRIDOR), 6)
    assert replay_episode([decision(0, 0, 3), decision(3, 3, 6)], CORRIDOR, oracle) == (-4., 4, False, True)
    with pytest.raises(ValueError):
        replay_episode([decision(0, 0, 3)], CORRIDOR, oracle)


def test_shards_merge_by_seed_and_timeouts_score_zero(tmp_path):
    spec = {"shift_step": 20, "max_steps": 10}
    for shard, seeds in (("b", [5, 6, 7, 8, 9]), ("a", [0, 1, 2, 3, 4])):
        path = tmp_path / "Shift1" / "UTE" / shard
        path.mkdir(parents=True)
        (path / "run.json").write_text(json.dumps({
            "status": "completed", "seeds": seeds, "config": {"warmup_steps": 5, "total_steps": 40},
            "environment": {"name": "ignored", "random": False, "spec": spec},
            "evaluation": {"steps": [10, 20, 30, 40], "episodes": 2}}))
        rows = [{"seed": seed, "env_step": step, "length_mean": 10. if seed == 0 else 5.,
                 "success_rate": 0. if seed == 0 else 1., "timeout_rate": 1. if seed == 0 else 0.}
                for seed in seeds for step in (10, 20, 30, 40)]
        (path / "evaluations.jsonl").write_text("\n".join(map(json.dumps, rows)))
    runs = Runs(tmp_path, tmp_path)
    assert runs.cell("Shift1", "UTE").seeds == list(range(10))
    assert post_change_auc(runs, "Shift1", "UTE") == pytest.approx([0.] + [.5] * 9)


def test_cells_format_like_the_paper():
    assert interval_cell(.5224, .4312, .6101, False) == r"$.522_{[.43,\,.61]}$"
    assert interval_cell(.031, .0281, .0342, True) == r"$\boldsymbol{.031}_{[.028,\,.034]}$"
    tied = column({"a": (.1751, .16, .19), "b": (.1749, .16, .19), "c": (.1, .1, .1)})
    assert "boldsymbol" in tied["a"] and "boldsymbol" in tied["b"] and "boldsymbol" not in tied["c"]
