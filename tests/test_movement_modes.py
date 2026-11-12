"""Cardinal/diagonal action contracts, corner safety and weighted planning."""

import math
from itertools import pairwise

import numpy as np
import pytest

from layanav.envs import ACTION_NAMES, GridConfig, GridWorldEnv, action_count
from layanav.envs.grid_world import DELTAS
from layanav.planners import ArtificialPotentialFieldPolicy, OnlineAStarPolicy


def layout(size=7, start=(1, 1), goal=(5, 5), walls=()):
    known = np.zeros((size, size), dtype=bool)
    for x, y in walls:
        known[y, x] = True
    return {
        "static_map": known,
        "hidden_obstacles": np.zeros((size, size), dtype=bool),
        "start": start,
        "goal": goal,
    }


def test_action_ids_unchanged_and_sizes():
    assert ACTION_NAMES[:5] == ("UP", "DOWN", "LEFT", "RIGHT", "STAY")
    assert DELTAS[:5] == ((0, -1), (0, 1), (-1, 0), (1, 0), (0, 0))
    assert DELTAS[5:] == ((-1, -1), (1, -1), (-1, 1), (1, 1))
    assert action_count(4) == 5
    assert action_count(8) == 9
    assert GridWorldEnv(GridConfig()).action_space.n == 5
    assert GridWorldEnv(GridConfig(movement_mode=8)).action_space.n == 9
    for value in (0, 5, 9, True, "8"):
        with pytest.raises(ValueError):
            GridConfig(movement_mode=value)


def test_diagonal_move_cost_and_reset():
    env = GridWorldEnv(GridConfig(size=7, movement_mode=8))
    obs, info = env.reset(options=layout())
    assert obs.shape == (6, 7, 7)
    assert info["distance_traveled"] == 0
    _, _, _, _, info = env.step(8)
    assert info["position"] == (2, 2)
    assert info["distance_traveled"] == pytest.approx(math.sqrt(2))
    _, _, _, _, info = env.step(3)
    assert info["distance_traveled"] == pytest.approx(math.sqrt(2) + 1)
    env.reset(options=layout())
    assert env.debug_snapshot()["distance_traveled"] == 0


@pytest.mark.parametrize("mode", [4, 8])
def test_corner_cutting_forbidden_even_with_hidden_obstacle(mode):
    env = GridWorldEnv(GridConfig(size=7, movement_mode=mode, observation_radius=1))
    game = layout(walls=((2, 1),))
    env.reset(options=game)
    if mode == 4:
        with pytest.raises(ValueError, match="Invalid action"):
            env.step(8)
    else:
        _, _, _, _, info = env.step(8)
        assert info["collision"]
        assert info["position"] == (1, 1)
        assert info["distance_traveled"] == 0
    # A hidden obstacle must also block corner-cutting at the physics layer.
    game = layout()
    game["hidden_obstacles"][1, 2] = True
    env8 = GridWorldEnv(GridConfig(size=7, movement_mode=8, observation_radius=1))
    env8.reset(options=game)
    _, _, _, _, info = env8.step(8)
    assert info["collision"] and info["position"] == (1, 1)


def test_8way_astar_octile_heuristic_and_corner_rule():
    blocked = np.zeros((7, 7), dtype=bool)
    path4 = OnlineAStarPolicy._find_path(blocked, (1, 1), (5, 5), 4)
    path8 = OnlineAStarPolicy._find_path(blocked, (1, 1), (5, 5), 8)
    assert len(path4) == 9
    assert len(path8) == 5
    assert all(
        (x1 - x0, y1 - y0) == (1, 1)
        for (x0, y0), (x1, y1) in pairwise(path8)
    )
    # No squeeze into the diagonal even when its destination is unoccupied.
    blocked[1, 2] = True
    blocked[2, 1] = True
    assert OnlineAStarPolicy._find_path(blocked, (1, 1), (2, 2), 8) == []


def test_apf_8way_scores_diagonal_candidates():
    obs = np.zeros((6, 7, 7), dtype=np.float32)
    obs[3, 1, 1] = 1
    obs[4, 5, 5] = 1
    assert ArtificialPotentialFieldPolicy(movement_mode=8).act(obs) == 8
    assert ArtificialPotentialFieldPolicy(movement_mode=4).act(obs) in range(4)
    obs[0, 1, 2] = 1
    apf = ArtificialPotentialFieldPolicy(movement_mode=8)
    assert apf.act(obs) != 8
    assert 8 not in apf.last_potentials


def test_8way_astar_real_environment_goal():
    env = GridWorldEnv(GridConfig(size=7, movement_mode=8, observation_radius=6))
    observation, _ = env.reset(options=layout(start=(1, 1), goal=(5, 5)))
    agent = OnlineAStarPolicy(movement_mode=8)
    path = []
    for _ in range(10):
        action = agent.act(observation)
        path.append(action)
        observation, _, done, truncated, info = env.step(action)
        if done or truncated:
            break
    assert path == [8, 8, 8, 8]
    assert info["success"]
    assert info["distance_traveled"] == pytest.approx(4 * math.sqrt(2))
