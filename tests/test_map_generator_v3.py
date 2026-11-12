"""Map Generator v3: alternate routes, occluded initial blockers and oracle budgets."""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest

from layanav.envs import GridConfig, GridWorldEnv
from layanav.envs.grid_world import DELTAS, valid_transition
from layanav.envs.map_generation import MAP_GENERATOR_VERSION, _distances
from layanav.evaluation.benchmark import benchmark, run_episode
from layanav.planners import OnlineAStarPolicy
from layanav.recording import EpisodeRecorder, ReplaySession


def _cycle_rank(prior: np.ndarray) -> int:
    free = int((~prior).sum())
    edges = 0
    for y, x in np.argwhere(~prior):
        for dx, dy in (DELTAS[1], DELTAS[3]):
            if valid_transition(prior, int(x), int(y), dx, dy):
                edges += 1
    # All Maze cells are in one cardinally connected component.
    return edges - free + 1


def test_maze_loops_form_cycles_and_zero_disables_them():
    assert MAP_GENERATOR_VERSION == 3
    old = GridWorldEnv(GridConfig(
        size=32, topology="maze", maze_loop_count=0,
        hidden_obstacle_count=0, movement_mode=4,
    ))
    old.reset(seed=31)
    assert _cycle_rank(old.debug_snapshot()["static_map"]) == 0

    looped = GridWorldEnv(GridConfig(
        size=32, topology="maze", maze_loop_count=16,
        hidden_obstacle_count=0, movement_mode=4,
    ))
    looped.reset(seed=31)
    assert _cycle_rank(looped.debug_snapshot()["static_map"]) > 0


@pytest.mark.parametrize("topology", ("maze", "corridor"))
@pytest.mark.parametrize("movement_mode", (4, 8))
def test_required_replanning_has_hidden_initial_blocker_and_alternate_route(
    topology, movement_mode,
):
    cfg = GridConfig(
        size=32, topology=topology, movement_mode=movement_mode,
        hidden_obstacle_count=8, observation_radius=2,
        sensor_mode="raycast", replanning_mode="required",
        oracle_step_reserve=8, max_steps=256,
    )
    env = GridWorldEnv(cfg)
    for seed in (27, 51):
        obs, _ = env.reset(seed=seed)
        snap = env.debug_snapshot()
        prior = snap["static_map"]
        hidden = snap["hidden_obstacles"]
        start, goal = snap["robot"], snap["goal"]
        assert int(hidden.sum()) == 8
        assert not np.any(prior & hidden)
        # Stronger than one solvable route: the entire cardinal component
        # remains connected except for the eight intentionally blocked cells.
        assert (
            len(_distances(prior | hidden, start, 4))
            == len(_distances(prior, start, 4)) - 8
        )
        initial = OnlineAStarPolicy._find_path(
            (obs[0] > 0.5) | (obs[1] > 0.5),
            start, goal, movement_mode,
        )
        unexpected = [
            (x, y) for x, y in initial
            if hidden[y, x] and obs[2, y, x] > 0.5
        ]
        assert unexpected, "Initial plan must traverse an unseen true blocker"
        true_path = OnlineAStarPolicy._find_path(
            prior | hidden, start, goal, movement_mode,
        )
        assert true_path and true_path[0] == start and true_path[-1] == goal
        assert not any(hidden[y, x] for x, y in true_path)
        assert len(true_path) - 1 <= cfg.max_steps - cfg.oracle_step_reserve
        assert sum(math.dist(a, b) for a, b in pairwise(true_path)) > 0


def test_v3_seed_reproducible_and_trace_compatible(tmp_path):
    cfg = GridConfig(
        size=32, topology="maze", maze_loop_count=12,
        hidden_obstacle_count=8, movement_mode=8, observation_radius=2,
        replanning_mode="required", oracle_step_reserve=10,
        max_steps=200,
    )
    env = GridWorldEnv(cfg)
    obs, _ = env.reset(seed=99)
    first = env.debug_snapshot()
    obs_again, _ = env.reset(seed=99)
    next_snapshot = env.debug_snapshot()
    np.testing.assert_array_equal(obs, obs_again)
    for key in ("static_map", "hidden_obstacles"):
        np.testing.assert_array_equal(first[key], next_snapshot[key])
    assert first["robot"] == next_snapshot["robot"]
    assert first["goal"] == next_snapshot["goal"]

    recorder = EpisodeRecorder(env)
    observed, reward, done, truncated, info = env.step(4)
    recorder.record_step(4, observed, reward, done, truncated, info)
    path = tmp_path / "required_replanning.json"
    recorder.save(path)
    replay = ReplaySession.load(path)
    assert replay.env.config.replanning_mode == "required"
    assert replay.env.config.oracle_step_reserve == 10
    assert replay.env.config.maze_loop_count == 12
    replay.step()
    assert replay.finished


def test_oracle_feasibility_and_diagnostics_are_reporting_only(tmp_path):
    cfg = GridConfig(
        size=32, topology="maze", hidden_obstacle_count=8,
        movement_mode=8, sensor_mode="raycast",
        observation_radius=2, max_steps=120, oracle_step_reserve=24,
        replanning_mode="required",
    )
    env = GridWorldEnv(cfg)
    row = run_episode(
        env, OnlineAStarPolicy(8), seed=102,
        oracle_diagnostic=True,
    )
    assert row["oracle_feasible_within_budget"] is True
    assert row["oracle_step_slack"] >= 24
    assert row["oracle_shortest_moves"] <= 96
    assert row["initial_plan_hidden_hits"] >= 1
    assert row["initial_plan_blocked"] is True
    assert row["map_sha256"]
    assert row["prior_layout_sha256"]

    env2 = GridWorldEnv(cfg)
    row_private = run_episode(
        env2, OnlineAStarPolicy(8), seed=102,
        oracle_diagnostic=False,
    )
    assert row_private["initial_plan_hidden_hits"] is None
    assert row_private["initial_plan_blocked"] is None
    assert row_private["oracle_feasible_within_budget"] is None
    assert row_private["oracle_step_slack"] is None
    assert row_private["oracle_shortest_moves"] is None


def test_legacy_random_generation_and_bounds_validation():
    with pytest.raises(ValueError, match="structured"):
        GridConfig(replanning_mode="required", topology="random")
    with pytest.raises(ValueError, match="hidden"):
        GridConfig(replanning_mode="required", topology="maze", hidden_obstacle_count=0)
    with pytest.raises(ValueError, match="replanning_mode"):
        GridConfig(replanning_mode="sometimes")
    with pytest.raises(ValueError, match="maze_loop_count"):
        GridConfig(maze_loop_count=-1)
    with pytest.raises(ValueError, match="oracle_step_reserve"):
        GridConfig(max_steps=32, oracle_step_reserve=32)
    env = GridWorldEnv(GridConfig(size=12, hidden_obstacle_count=2))
    env.reset(seed=12)
    assert env.debug_snapshot()["robot"] != env.debug_snapshot()["goal"]


def test_benchmark_records_replanning_and_feasibility_metrics(tmp_path):
    report = benchmark(
        methods=("astar", "apf"), topologies=("maze",),
        families=("sparse",), episodes=2, eval_seed=201,
        size=32, radius=2, sensor="raycast", movement_mode=8,
        max_steps=140, replanning_mode="required",
        oracle_step_reserve=16, output_dir=tmp_path,
        oracle_diagnostic=True, save_failures=False,
        bootstrap_replicates=100,
    )
    assert report["protocol"]["structured_map_generator_version"] == 3
    assert report["protocol"]["replanning_mode"] == "required"
    assert report["protocol"]["oracle_step_reserve"] == 16
    group = report["summary"]["maze/sparse/astar"]
    assert group["initial_plan_blocked"]["mean"] == 1
    assert group["oracle_feasible_within_budget"]["mean"] == 1
    assert group["initial_plan_hidden_hits"]["n"] == 2
