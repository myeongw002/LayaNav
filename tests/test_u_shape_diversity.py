"""U-shape instances must vary while retaining a forced obstacle detour."""

from __future__ import annotations

import math

from layanav.envs import GridConfig, GridWorldEnv
from layanav.planners import OnlineAStarPolicy


def test_u_shape_geometry_varies_across_seeds_and_preserves_detour():
    env = GridWorldEnv(GridConfig(
        size=32, topology="u_shape", hidden_obstacle_count=0,
        known_wall_segments=0, movement_mode=8,
    ))
    positions = set()
    direct_distances = set()
    for seed in range(40):
        env.reset(seed=seed)
        snap = env.debug_snapshot()
        start, goal = snap["robot"], snap["goal"]
        positions.add((start, goal))
        direct_distances.add(round(math.dist(start, goal), 5))
        path = OnlineAStarPolicy._find_path(
            snap["static_map"], start, goal, 8,
        )
        assert path[0] == start and path[-1] == goal
        distance = sum(math.dist(a, b) for a, b in zip(path[:-1], path[1:], strict=True))
        assert distance > math.dist(start, goal) + 2
    assert len(positions) >= 10
    assert len(direct_distances) >= 4


def test_u_shape_seed_remains_deterministic():
    env = GridWorldEnv(GridConfig(size=32, topology="u_shape"))
    env.reset(seed=157)
    initial = env.debug_snapshot()
    env.reset(seed=157)
    repeated = env.debug_snapshot()
    assert initial["robot"] == repeated["robot"]
    assert initial["goal"] == repeated["goal"]
    assert (initial["static_map"] == repeated["static_map"]).all()
    assert (initial["hidden_obstacles"] == repeated["hidden_obstacles"]).all()
