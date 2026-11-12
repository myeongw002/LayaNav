"""Seeded topology generation, connectivity and observation isolation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from layanav.envs import MAP_TOPOLOGIES, GridConfig, GridWorldEnv
from layanav.planners import OnlineAStarPolicy


@pytest.mark.parametrize("topology", MAP_TOPOLOGIES)
@pytest.mark.parametrize("movement_mode", (4, 8))
@pytest.mark.parametrize("seed", (1, 2026))
def test_topologies_seeded_and_reachable(topology, movement_mode, seed):
    config = GridConfig(
        size=16, hidden_obstacle_count=7, known_wall_segments=3,
        movement_mode=movement_mode, topology=topology, sensor_mode="raycast",
    )
    env = GridWorldEnv(config)
    observation, _ = env.reset(seed=seed)
    a = env.debug_snapshot()
    observation_again, _ = env.reset(seed=seed)
    b = env.debug_snapshot()

    np.testing.assert_array_equal(observation, observation_again)
    for key in ("static_map", "hidden_obstacles"):
        np.testing.assert_array_equal(a[key], b[key])
    assert a["robot"] == b["robot"] and a["goal"] == b["goal"]
    assert int(a["hidden_obstacles"].sum()) == config.hidden_obstacle_count
    assert not np.any(a["static_map"] & a["hidden_obstacles"])
    assert GridWorldEnv._reachable(
        a["static_map"] | a["hidden_obstacles"],
        a["robot"], a["goal"], movement_mode,
    )
    assert observation.shape == (6, config.size, config.size)
    unseen_hidden = a["hidden_obstacles"] & (observation[2] > 0.5)
    assert not np.any(observation[1][unseen_hidden])
    env.close()


@pytest.mark.parametrize("topology", MAP_TOPOLOGIES[1:])
def test_structured_small_grid_dense_obstacles(topology):
    config = GridConfig(
        size=9, topology=topology, movement_mode=8,
        hidden_obstacle_count=5, known_wall_segments=3,
    )
    env = GridWorldEnv(config)
    env.reset(seed=17)
    state = env.debug_snapshot()
    assert GridWorldEnv._reachable(
        state["static_map"] | state["hidden_obstacles"],
        state["robot"], state["goal"], movement_mode=8,
    )


def test_u_shape_requires_detour():
    env = GridWorldEnv(GridConfig(
        size=32, hidden_obstacle_count=0, topology="u_shape", movement_mode=8,
    ))
    env.reset(seed=12)
    state = env.debug_snapshot()
    route = OnlineAStarPolicy._find_path(
        state["static_map"], state["robot"], state["goal"], 8,
    )
    assert route
    straight = math.dist(state["robot"], state["goal"])
    actual = sum(math.dist(a, b) for a, b in zip(route[:-1], route[1:], strict=True))
    assert actual > 3 * straight


@pytest.mark.parametrize("topology", ("maze", "corridor", "rooms", "u_shape", "cluttered"))
def test_structured_topology_not_empty_interior(topology):
    env = GridWorldEnv(GridConfig(
        size=32, hidden_obstacle_count=0, known_wall_segments=5,
        topology=topology,
    ))
    env.reset(seed=34)
    state = env.debug_snapshot()
    prior = state["static_map"]
    assert prior[0].all() and prior[-1].all()
    assert prior[:, 0].all() and prior[:, -1].all()
    assert prior[1:-1, 1:-1].any()
    assert (~prior[1:-1, 1:-1]).any()


def test_reject_invalid_topology_and_too_small_u_shape():
    with pytest.raises(ValueError, match="topology"):
        GridConfig(topology="unknown")
    with pytest.raises(ValueError, match="u_shape"):
        GridConfig(size=8, topology="u_shape")
    with pytest.raises(ValueError, match="available free cells"):
        env = GridWorldEnv(GridConfig(
            size=9, topology="maze", hidden_obstacle_count=60,
        ))
        env.reset(seed=3)
