"""Benchmark v2 correctness: Wilson CIs, SPL, map diagnostics and paired maps."""

from __future__ import annotations

import csv
import math

import numpy as np
import pytest

from layanav.envs import GridConfig, GridWorldEnv
from layanav.evaluation.benchmark import (
    benchmark,
    run_episode,
    wilson_success_ci,
)
from layanav.evaluation.difficulty import (
    DIFFICULTY_METRICS,
    map_fingerprint,
    measure_map_difficulty,
)
from layanav.planners import OnlineAStarPolicy


def test_wilson_interval_all_success_or_failure():
    all_success = wilson_success_ci(np.ones(20))
    assert all_success["mean"] == all_success["ci_high"] == 1
    assert 0.83 < all_success["ci_low"] < 0.85
    assert all_success["n"] == 20
    none_success = wilson_success_ci(np.zeros(20))
    assert none_success["mean"] == none_success["ci_low"] == 0
    assert 0.15 < none_success["ci_high"] < 0.17
    half_success = wilson_success_ci(np.array([0, 1] * 10))
    assert half_success["ci_low"] < 0.5 < half_success["ci_high"]
    for values in (np.array([]), np.array([0.5]), np.array([-1, 0])):
        with pytest.raises(ValueError):
            wilson_success_ci(values)
    with pytest.raises(ValueError):
        wilson_success_ci(np.ones(3), confidence=1)


@pytest.mark.parametrize("movement_mode", (4, 8))
def test_exact_known_map_difficulty(movement_mode):
    prior = np.zeros((5, 5), dtype=bool)
    prior[0, :] = prior[-1, :] = True
    prior[:, 0] = prior[:, -1] = True
    hidden = np.zeros_like(prior)
    env = GridWorldEnv(GridConfig(
        size=5, movement_mode=movement_mode,
        observation_radius=1, hidden_obstacle_count=0,
        known_wall_segments=0,
    ))
    obs, _ = env.reset(options={
        "static_map": prior,
        "hidden_obstacles": hidden,
        "start": (1, 1), "goal": (3, 3),
    })
    path = OnlineAStarPolicy._find_path(prior, (1, 1), (3, 3), movement_mode)
    measurements = measure_map_difficulty(
        prior, hidden, (1, 1), (3, 3), path, movement_mode, obs,
    )
    assert measurements["obstacle_density"] == pytest.approx(16 / 25)
    assert measurements["interior_obstacle_density"] == 0.0
    assert measurements["initial_unknown_ratio"] == pytest.approx(6 / 9)
    assert measurements["straight_line_distance"] == pytest.approx(math.sqrt(8))
    assert measurements["minimum_clearance"] == 1.0
    assert measurements["reachable_free_cells"] == 9
    assert measurements["dead_end_count"] == 0
    if movement_mode == 4:
        assert measurements["path_tortuosity"] == pytest.approx(math.sqrt(2))
    else:
        assert measurements["path_tortuosity"] == pytest.approx(1.0)
    assert map_fingerprint(
        prior, hidden, (1, 1), (3, 3), movement_mode,
    ) == map_fingerprint(prior.copy(), hidden.copy(), (1, 1), (3, 3), movement_mode)
    assert map_fingerprint(
        prior, hidden, (1, 1), (3, 3), 4,
    ) != map_fingerprint(prior, hidden, (1, 1), (3, 3), 8)
    env.close()


def test_map_difficulty_rejects_invalid_oracle_path():
    prior = np.zeros((7, 7), dtype=bool)
    hidden = np.zeros_like(prior)
    obs = np.zeros((6, 7, 7), dtype=np.float32)
    with pytest.raises(ValueError, match="Oracle path"):
        measure_map_difficulty(
            prior, hidden, (1, 1), (5, 5),
            [(1, 1), (2, 2), (5, 5)], 8, obs,
        )


def test_benchmark_v2_paired_maps_spl_and_failure_only_stats(tmp_path):
    report = benchmark(
        methods=("astar", "apf"),
        families=("sparse",),
        topologies=("random", "maze", "u_shape"),
        episodes=2, eval_seed=34000, size=12, radius=2,
        sensor="raycast", movement_mode=8, max_steps=1,
        oracle_diagnostic=True, save_failures=False,
        output_dir=tmp_path, bootstrap_replicates=100,
    )
    assert report["schema_version"] == 2
    assert "Wilson" in report["protocol"]["success_confidence_interval"]
    assert report["protocol"]["map_difficulty_is_privileged_diagnostic_only"]
    with (tmp_path / "episodes.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 12
    grouped = {}
    for row in rows:
        group = (row["topology"], row["family"], int(row["seed"]))
        grouped.setdefault(group, []).append(row)
        assert row["success"] == "False"
        assert float(row["spl"]) == 0.0
        assert len(row["map_sha256"]) == 64
        assert float(row["straight_line_distance"]) >= 1
        assert float(row["path_tortuosity"]) >= 1 - 1e-12
        assert float(row["minimum_clearance"]) >= 1
        assert 0 <= float(row["initial_unknown_ratio"]) <= 1
        assert float(row["obstacle_density"]) < 1
        assert int(row["reachable_free_cells"]) >= 2
    for pair in grouped.values():
        assert len(pair) == 2
        assert pair[0]["map_sha256"] == pair[1]["map_sha256"]
        for metric in DIFFICULTY_METRICS:
            assert pair[0][metric] == pair[1][metric]

    for key in ("sparse/astar", "maze/sparse/apf", "u_shape/sparse/apf"):
        summary = report["summary"][key]
        assert summary["success"]["mean"] == 0
        assert summary["success"]["ci_high"] > 0
        assert summary["spl"]["mean"] == 0
        assert summary["spl"]["n"] == 2
        assert summary["successful_steps"] is None
        assert summary["successful_distance_traveled"] is None
        assert summary["straight_line_distance"]["n"] == 2


def test_success_spl_and_success_only_denominators(tmp_path):
    report = benchmark(
        methods=("astar",), families=("sparse",), episodes=3,
        eval_seed=7788, size=9, radius=2, sensor="circle",
        movement_mode=8, max_steps=100, oracle_diagnostic=True,
        save_failures=False, output_dir=tmp_path, bootstrap_replicates=100,
    )
    summary = report["summary"]["sparse/astar"]
    assert summary["success"]["mean"] == 1
    assert summary["success"]["ci_low"] < 1
    assert summary["success"]["n"] == 3
    assert summary["spl"]["n"] == 3
    assert summary["successful_steps"]["n"] == 3
    assert summary["successful_moves"]["n"] == 3
    assert summary["successful_distance_traveled"]["n"] == 3
    assert summary["spl"]["mean"] == pytest.approx(1, abs=1e-10)


def test_no_oracle_still_keeps_truth_private(monkeypatch):
    env = GridWorldEnv(GridConfig(
        size=9, hidden_obstacle_count=0, known_wall_segments=0,
        max_steps=30,
    ))

    def forbidden():
        raise AssertionError("Privileged ground truth used without opt-in")

    monkeypatch.setattr(env, "debug_snapshot", forbidden)
    row = run_episode(env, OnlineAStarPolicy(), seed=42, oracle_diagnostic=False)
    assert row["success"]
    assert row["spl"] is None
    assert row["map_sha256"] is None
    for metric in DIFFICULTY_METRICS:
        assert row[metric] is None
