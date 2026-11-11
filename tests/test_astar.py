"""A* must navigate using the belief observation, never simulator truth."""

import numpy as np

from layanav.envs import GridConfig, GridWorldEnv
from layanav.planners import OnlineAStarPolicy


def test_replan_around_new_obstacle():
    n = 9
    prior = np.zeros((n, n), dtype=bool)
    prior[0, :] = prior[-1, :] = True
    prior[:, 0] = prior[:, -1] = True
    hidden = np.zeros_like(prior)
    hidden[4, 4] = True
    env = GridWorldEnv(GridConfig(size=n, observation_radius=1, max_steps=60))
    obs, _ = env.reset(options={
        "static_map": prior,
        "hidden_obstacles": hidden,
        "start": (2, 4),
        "goal": (6, 4),
    })
    agent = OnlineAStarPolicy()
    assert agent.act(obs) == 3  # right toward apparently empty shortest route
    obs, _, _, _, _ = env.step(3)
    assert obs[1, 4, 4] == 1.0
    assert agent.act(obs) != 3  # discovered obstacle blocks direct route
    success = False
    for _ in range(60):
        obs, _, done, truncated, info = env.step(agent.act(obs))
        if done or truncated:
            success = info["success"]
            break
    assert success


def test_blocked_goal_returns_stay():
    obs = np.zeros((6, 7, 7), dtype=np.float32)
    obs[0, 3, :] = 1
    obs[3, 1, 1] = 1
    obs[4, 5, 5] = 1
    agent = OnlineAStarPolicy()
    assert agent.act(obs) == 4
    assert agent.last_path == []
