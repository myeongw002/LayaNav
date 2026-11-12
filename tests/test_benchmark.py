"""Paired-map evaluation and statistical summary regression tests."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from layanav.envs import GridConfig, GridWorldEnv
from layanav.evaluation.benchmark import (
    benchmark,
    bootstrap_mean_ci,
    make_family_config,
    run_episode,
)
from layanav.planners import OnlineAStarPolicy


def test_bootstrap_constant_success_rate_and_single_sample():
    values = np.ones(10)
    result = bootstrap_mean_ci(values, replicates=100, seed=22)
    assert result["mean"] == result["ci_low"] == result["ci_high"] == 1.0
    assert result["n"] == 10
    one = bootstrap_mean_ci(np.array([0.3]), seed=22)
    assert one["mean"] == one["ci_low"] == one["ci_high"] == 0.3
    with pytest.raises(ValueError):
        bootstrap_mean_ci(np.array([]))


def test_observation_privacy_without_oracle(monkeypatch):
    env = GridWorldEnv(
        GridConfig(
            size=9, observation_radius=2, sensor_mode="raycast",
            hidden_obstacle_count=0, known_wall_segments=0, max_steps=30,
        )
    )

    def forbidden():
        raise AssertionError("Policy-evaluation path accessed debug truth")

    monkeypatch.setattr(env, "debug_snapshot", forbidden)
    row = run_episode(env, OnlineAStarPolicy(), seed=28, oracle_diagnostic=False)
    assert row["steps"] <= 30
    assert row["oracle_shortest_moves"] is None
    assert row["oracle_efficiency"] is None


def test_oracle_is_diagnostic_only_and_lower_bound():
    env = GridWorldEnv(
        GridConfig(
            size=9, observation_radius=3, sensor_mode="circle",
            hidden_obstacle_count=0, known_wall_segments=0, max_steps=30,
        )
    )
    row = run_episode(env, OnlineAStarPolicy(), seed=28, oracle_diagnostic=True)
    assert row["success"]
    assert row["oracle_shortest_moves"] is not None
    assert row["oracle_shortest_moves"] <= row["moves"]
    assert 0 < row["oracle_efficiency"] <= 1


def test_benchmark_writes_paired_seed_csv_json(tmp_path):
    report = benchmark(
        methods=("astar",),
        families=("sparse", "standard"),
        episodes=3,
        eval_seed=10001,
        size=9,
        radius=2,
        sensor="raycast",
        max_steps=30,
        output_dir=tmp_path,
        bootstrap_replicates=100,
        oracle_diagnostic=True,
    )
    assert report["protocol"]["paired_map_seeds"]
    assert report["summary"]["sparse/astar"]["success"]["n"] == 3
    assert report["summary"]["standard/astar"]["steps"]["n"] == 3
    with (tmp_path / "episodes.csv").open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 6
    assert {int(row["seed"]) for row in rows} == {10001, 10002, 10003}
    with (tmp_path / "summary.json").open(encoding="utf-8") as file:
        saved = json.load(file)
    assert saved["summary"] == report["summary"]
    assert saved["families"]["sparse"]["size"] == 9


def test_family_presets_and_bad_benchmark_options(tmp_path):
    sparse = make_family_config(
        "sparse", size=32, radius=5, sensor="raycast", max_steps=256,
    )
    dense = make_family_config(
        "dense", size=32, radius=5, sensor="raycast", max_steps=256,
    )
    assert sparse.size == dense.size == 32
    assert sparse.hidden_obstacle_count < dense.hidden_obstacle_count
    with pytest.raises(ValueError, match="family"):
        make_family_config("not-a-family", size=32, radius=5, sensor="circle", max_steps=256)
    with pytest.raises(ValueError, match="checkpoint"):
        benchmark(
            methods=("ppo",), families=("sparse",), episodes=1,
            eval_seed=1, size=9, radius=2, sensor="circle", max_steps=20,
            output_dir=tmp_path,
        )


def test_apf_benchmark_same_seeds_and_configuration(tmp_path):
    from layanav.planners import APFConfig

    report = benchmark(
        methods=("astar", "apf"),
        families=("sparse",),
        episodes=3,
        eval_seed=27001,
        size=9,
        radius=2,
        sensor="raycast",
        max_steps=25,
        output_dir=tmp_path / "apf_run",
        bootstrap_replicates=100,
        apf_config=APFConfig(
            attractive_gain=1.2, repulsive_gain=5.0,
            influence_radius=3.0, unknown_penalty=0.0,
        ),
    )
    assert report["summary"]["sparse/astar"]["success"]["n"] == 3
    assert report["summary"]["sparse/apf"]["success"]["n"] == 3
    assert report["apf_config"]["repulsive_gain"] == 5.0

    with (tmp_path / "apf_run" / "episodes.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 6
    seeds_by_method = {
        method: {int(row["seed"]) for row in rows if row["method"] == method}
        for method in ("astar", "apf")
    }
    assert seeds_by_method["astar"] == seeds_by_method["apf"] == {
        27001, 27002, 27003,
    }


def test_paired_4way_vs_8way_weighted_path_metrics(tmp_path):
    for mode in (4, 8):
        report = benchmark(
            methods=("astar", "apf"), families=("sparse",),
            episodes=2, eval_seed=44441, size=9, radius=3, sensor="circle",
            max_steps=40, movement_mode=mode, oracle_diagnostic=True,
            output_dir=tmp_path / str(mode), bootstrap_replicates=100,
        )
        assert report["protocol"]["movement_mode"] == mode
        with (tmp_path / str(mode) / "episodes.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 4
        assert all(float(row["distance_traveled"]) >= 0 for row in rows)
        for row in rows:
            if row["success"] == "True":
                assert 0 < float(row["oracle_efficiency"]) <= 1.0000001
            if mode == 4:
                assert float(row["distance_traveled"]) == float(row["moves"])
