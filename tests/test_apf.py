"""APF baseline: local descent, observed obstacles only, expected local minima."""

from __future__ import annotations

import numpy as np
import pytest

from layanav.envs import GridConfig, GridWorldEnv
from layanav.planners import APFConfig, ArtificialPotentialFieldPolicy, OnlineAStarPolicy


def observation(
    start: tuple[int, int] = (2, 4), goal: tuple[int, int] = (6, 4),
    size: int = 9,
) -> np.ndarray:
    obs = np.zeros((6, size, size), dtype=np.float32)
    obs[3, start[1], start[0]] = 1
    obs[4, goal[1], goal[0]] = 1
    return obs


def test_direct_goal_attraction_on_open_grid():
    apf = ArtificialPotentialFieldPolicy()
    obs = observation()
    assert apf.act(obs) == 3  # RIGHT
    assert apf.last_status == "moving"
    assert apf.last_potentials[3] < apf.last_potentials[0]


@pytest.mark.parametrize("obstacle_channel", [0, 1])
def test_known_and_discovered_obstacles_are_excluded(obstacle_channel):
    obs = observation(start=(3, 4))
    obs[obstacle_channel, 4, 4] = 1
    apf = ArtificialPotentialFieldPolicy()
    assert apf.act(obs) != 3
    assert 3 not in apf.last_potentials  # RIGHT would hit observed obstacle.


def test_undiscovered_obstacle_does_not_leak_into_policy():
    obs = observation(start=(2, 4))
    apf = ArtificialPotentialFieldPolicy()
    assert apf.act(obs) == 3
    # The same position with a newly observed obstacle changes candidate actions.
    obs[1, 4, 3] = 1
    assert apf.act(obs) != 3
    assert 3 not in apf.last_potentials


def test_apf_local_minimum_is_explicit_and_does_not_fall_back_to_astar():
    obs = observation(start=(3, 4))
    obs[0, 2:7, 4] = 1  # Vertical wall; shortest route exists around its top/bottom.
    apf = ArtificialPotentialFieldPolicy()
    assert apf.act(obs) == 4  # STAY at local minimum
    assert apf.last_status == "local_minimum"
    assert OnlineAStarPolicy().act(obs) != 4


def test_goal_and_no_legal_neighbors_return_stay():
    obs = observation(start=(4, 4), goal=(4, 4))
    apf = ArtificialPotentialFieldPolicy()
    assert apf.act(obs) == 4
    assert apf.last_status == "goal"

    obs = observation(start=(4, 4), goal=(6, 6))
    obs[0, 3, 4] = obs[0, 5, 4] = 1
    obs[0, 4, 3] = obs[0, 4, 5] = 1
    assert apf.act(obs) == 4
    assert apf.last_status == "blocked"


def test_invalid_config_or_public_observation_rejected():
    for kwargs in (
        {"attractive_gain": 0},
        {"repulsive_gain": -1},
        {"influence_radius": 0},
        {"unknown_penalty": -1},
        {"descent_tolerance": -1},
        {"repulsive_gain": float("nan")},
    ):
        with pytest.raises(ValueError):
            APFConfig(**kwargs)

    apf = ArtificialPotentialFieldPolicy()
    with pytest.raises(ValueError, match="six-channel"):
        apf.act(np.zeros((9, 9)))
    with pytest.raises(ValueError, match="exactly one"):
        apf.act(np.zeros((6, 9, 9), dtype=np.float32))
    obs = observation()
    obs[1, 3, 2] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        apf.act(obs)


def test_apf_uses_only_observation_in_full_episode(monkeypatch):
    env = GridWorldEnv(
        GridConfig(
            size=9, observation_radius=2, sensor_mode="raycast",
            hidden_obstacle_count=0, known_wall_segments=0, max_steps=30,
        )
    )
    obs, _ = env.reset(seed=19)

    def forbidden():
        raise AssertionError("APF must not access privileged simulator ground truth")

    monkeypatch.setattr(env, "debug_snapshot", forbidden)
    apf = ArtificialPotentialFieldPolicy()
    for _ in range(30):
        action = apf.act(obs)
        assert isinstance(action, int) and 0 <= action <= 4
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    env.close()


def test_unknown_penalty_can_change_preference():
    obs = observation(start=(2, 4), goal=(6, 4))
    obs[2, 4, 3] = 1
    baseline = ArtificialPotentialFieldPolicy(APFConfig(repulsive_gain=0))
    cautious = ArtificialPotentialFieldPolicy(
        APFConfig(repulsive_gain=0, unknown_penalty=100)
    )
    assert baseline.act(obs) == 3
    assert cautious.act(obs) != 3
