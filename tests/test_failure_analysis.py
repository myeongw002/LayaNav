"""Failure reason diagnostics, trace capture and cross-topology benchmark."""

from __future__ import annotations

import csv
import json

from layanav.envs import GridConfig, GridWorldEnv
from layanav.evaluation.benchmark import benchmark, run_episode
from layanav.recording import ReplaySession


class FixedStayPolicy:
    last_status = "local_minimum"

    def act(self, observation):
        assert observation.shape[0] == 6
        return 4


def test_failure_trace_roundtrip_contains_diagnostics(tmp_path):
    env = GridWorldEnv(GridConfig(
        size=9, max_steps=6, hidden_obstacle_count=0, known_wall_segments=0,
        topology="rooms", movement_mode=8,
    ))
    trace_file = tmp_path / "failed_episode.json"
    result = run_episode(
        env, FixedStayPolicy(), seed=44,
        failure_trace_path=trace_file, trace_context={
            "topology": "rooms", "family": "sparse", "method": "test_stay",
        },
    )
    assert not result["success"]
    assert result["termination_reason"] == "time_limit"
    assert result["failure_reason"] == "local_minimum"
    assert result["stay_steps"] == result["steps"] == 6
    assert result["max_consecutive_stay"] == 6
    assert result["moves"] == result["revisits"] == 0
    assert result["goal_distance_remaining"] > 0
    assert result["failure_trace"] == "failures/failed_episode.json"
    assert trace_file.exists()

    replay = ReplaySession.load(trace_file)
    assert replay.diagnostics["failure_reason"] == "local_minimum"
    assert replay.diagnostics["topology"] == "rooms"
    assert replay.env.config.topology == "rooms"
    for _ in range(6):
        replay.step()
    assert replay.finished


def test_multi_topology_benchmark_paired_seeds_and_failure_files(tmp_path):
    report = benchmark(
        methods=("astar", "apf"), families=("sparse",),
        topologies=("random", "maze", "u_shape"),
        episodes=2, eval_seed=34000, size=12, radius=2, sensor="raycast",
        movement_mode=8, max_steps=1, output_dir=tmp_path,
        bootstrap_replicates=100,
    )
    assert report["protocol"]["topologies"] == ["random", "maze", "u_shape"]
    assert report["protocol"]["save_failure_traces"]
    with (tmp_path / "episodes.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 12
    assert len({row["failure_trace"] for row in rows}) == 12
    for row in rows:
        assert row["termination_reason"] == "time_limit"
        assert row["failure_reason"]
        path = tmp_path / row["failure_trace"]
        assert path.exists()
        replay = ReplaySession.load(path)
        assert replay.env.config.topology == row["topology"]
        replay.step()
        assert replay.finished
        assert replay.diagnostics["method"] == row["method"]

    assert report["summary"]["sparse/astar"]["success"]["n"] == 2
    assert report["summary"]["maze/sparse/astar"]["success"]["n"] == 2
    assert report["summary"]["u_shape/sparse/apf"]["success"]["n"] == 2
    assert report["families"]["maze/sparse"]["topology"] == "maze"
    saved = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert saved["summary"] == report["summary"]


def test_failure_capture_can_be_disabled(tmp_path):
    report = benchmark(
        methods=("astar",), families=("sparse",),
        episodes=1, eval_seed=121, size=9, radius=2, sensor="circle",
        max_steps=1, save_failures=False, output_dir=tmp_path,
        bootstrap_replicates=100,
    )
    assert not report["protocol"]["save_failure_traces"]
    assert not (tmp_path / "failures").exists()
    assert report["summary"]["sparse/astar"]["failure_traces"] == 0


def test_success_has_no_failure_trace(tmp_path):
    env = GridWorldEnv(GridConfig(
        size=9, hidden_obstacle_count=0, known_wall_segments=0, max_steps=50,
    ))
    from layanav.planners import OnlineAStarPolicy

    result = run_episode(
        env, OnlineAStarPolicy(), seed=11,
        failure_trace_path=tmp_path / "should_not_exist.json",
    )
    assert result["success"]
    assert result["failure_reason"] is None
    assert result["termination_reason"] == "goal_reached"
    assert result["failure_trace"] is None
    assert not (tmp_path / "should_not_exist.json").exists()
