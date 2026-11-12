"""Connectivity preservation, density nesting, route bounds and generator regression."""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest

from layanav.envs import GridConfig, GridWorldEnv
from layanav.envs.map_generation import _distances
from layanav.evaluation.benchmark import make_family_config
from layanav.evaluation.difficulty import map_fingerprint
from layanav.planners import OnlineAStarPolicy


@pytest.mark.parametrize("topology", ("maze", "corridor", "rooms", "u_shape"))
@pytest.mark.parametrize("mode", (4, 8))
@pytest.mark.parametrize("seed", (42, 300000))
def test_structured_density_nested_and_full_component_preserved(topology, mode, seed):
    instances = []
    for count in (8, 24, 64):
        env = GridWorldEnv(GridConfig(
            size=32, topology=topology, movement_mode=mode,
            hidden_obstacle_count=count, known_wall_segments=5,
        ))
        env.reset(seed=seed)
        snap = env.debug_snapshot()
        prior = snap["static_map"]
        hidden = snap["hidden_obstacles"]
        start, goal = snap["robot"], snap["goal"]
        assert int(hidden.sum()) == count
        assert not np.any(prior & hidden)
        before = _distances(prior, start, 4)
        after = _distances(prior | hidden, start, 4)
        assert len(after) == len(before) - count
        assert goal in after
        route = OnlineAStarPolicy._find_path(prior | hidden, start, goal, mode)
        assert route
        distance = sum(math.dist(a, b) for a, b in pairwise(route))
        instances.append((prior, hidden, start, goal, distance))

    for a, b in pairwise(instances):
        np.testing.assert_array_equal(a[0], b[0])
        assert a[2:4] == b[2:4]
        assert np.all(a[1] <= b[1]), "Hidden sets must be nested, not resampled"
        assert a[4] <= b[4] + 1e-9, "Adding blockers cannot shorten a shortest path"


@pytest.mark.parametrize("topology", ("maze", "corridor", "rooms", "u_shape"))
def test_structured_seed_reproducibility(topology):
    env = GridWorldEnv(GridConfig(size=32, topology=topology, hidden_obstacle_count=64))
    env.reset(seed=73)
    first = env.debug_snapshot()
    env.reset(seed=73)
    second = env.debug_snapshot()
    assert first["robot"] == second["robot"]
    assert first["goal"] == second["goal"]
    np.testing.assert_array_equal(first["static_map"], second["static_map"])
    np.testing.assert_array_equal(first["hidden_obstacles"], second["hidden_obstacles"])


def test_prior_route_lower_bound_is_enforced():
    cfg = GridConfig(
        size=32, topology="maze", movement_mode=8,
        hidden_obstacle_count=24, min_route_steps=24,
    )
    env = GridWorldEnv(cfg)
    env.reset(seed=90)
    snap = env.debug_snapshot()
    assert _distances(
        snap["static_map"], snap["robot"], cfg.movement_mode,
    )[snap["goal"]] >= cfg.min_route_steps
    assert _distances(
        snap["static_map"] | snap["hidden_obstacles"], snap["robot"],
        cfg.movement_mode,
    )[snap["goal"]] >= cfg.min_route_steps


def test_route_interval_and_invalid_constraints():
    for topology in ("random", "rooms", "maze"):
        cfg = GridConfig(
            size=16, topology=topology, hidden_obstacle_count=3,
            min_route_steps=5, max_route_steps=60,
        )
        env = GridWorldEnv(cfg)
        env.reset(seed=7)
        snap = env.debug_snapshot()
        distance = _distances(
            snap["static_map"] | snap["hidden_obstacles"], snap["robot"],
            cfg.movement_mode,
        )[snap["goal"]]
        assert 5 <= distance <= 60
    with pytest.raises(ValueError, match="min_route_steps"):
        GridConfig(min_route_steps=20, max_route_steps=12)
    with pytest.raises(ValueError, match="positive integer"):
        GridConfig(min_route_steps=True)
    with pytest.raises(RuntimeError, match="route"):
        env = GridWorldEnv(GridConfig(
            size=9, topology="maze", hidden_obstacle_count=0,
            max_route_steps=1,
        ))
        env.reset(seed=1)


def test_legacy_random_unchanged_for_known_seed():
    cfg = make_family_config(
        "sparse", size=32, radius=5, sensor="raycast",
        max_steps=256, movement_mode=8,
    )
    env = GridWorldEnv(cfg)
    env.reset(seed=300000)
    snap = env.debug_snapshot()
    # Captured from the user's benchmark_v2.zip before this generator change.
    assert map_fingerprint(
        snap["static_map"], snap["hidden_obstacles"],
        snap["robot"], snap["goal"], cfg.movement_mode,
    ) == "be686be0f6a4c62d213f90708ead8c129f13e36c34f93b69064dad2393245d0b"
