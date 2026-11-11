"""Visibility guarantees for ideal circle and occlusion-aware ray casting."""

import numpy as np
import pytest

from layanav.envs import GridConfig, GridWorldEnv
from layanav.envs.sensor import visibility_mask


def test_raycast_stops_at_first_obstacle_but_circle_does_not():
    blocked = np.zeros((9, 9), dtype=bool)
    blocked[4, 4] = True
    circle = visibility_mask(blocked, (2, 4), 4, "circle")
    ray = visibility_mask(blocked, (2, 4), 4, "raycast")

    assert circle[4, 5]  # Ideal radius sensor ignores occlusion.
    assert ray[4, 4]  # Obstacle surface must be observable.
    assert not ray[4, 5]  # Cell directly behind obstacle hidden.
    assert ray[4, 3] and ray[4, 2]


def test_raycast_known_wall_occludes_hidden_obstacle_until_new_viewpoint():
    n = 9
    prior = np.zeros((n, n), dtype=bool)
    prior[4, 4] = True
    hidden = np.zeros_like(prior)
    hidden[4, 5] = True
    env = GridWorldEnv(
        GridConfig(size=n, observation_radius=4, sensor_mode="raycast")
    )
    obs, _ = env.reset(options={
        "static_map": prior,
        "hidden_obstacles": hidden,
        "start": (2, 4),
        "goal": (7, 7),
    })
    assert obs[0, 4, 4] == 1.0  # Prior map is globally known.
    assert obs[2, 4, 5] == 1.0  # Hidden obstacle not yet observed.
    assert obs[1, 4, 5] == 0.0

    # Step around the wall and look at the hidden obstacle from above.
    for action in (0, 3, 3):
        obs, _, _, _, _ = env.step(action)
    assert env.debug_snapshot()["robot"] == (4, 3)
    assert obs[1, 4, 5] == 1.0

    # Belief persists after leaving the observation radius.
    for action in (2, 2, 2):
        obs, _, _, _, _ = env.step(action)
    assert obs[1, 4, 5] == 1.0


def test_sensor_validation():
    blocked = np.zeros((7, 7), dtype=bool)
    with pytest.raises(ValueError):
        visibility_mask(blocked.astype(np.uint8), (1, 1), 3, "raycast")
    with pytest.raises(ValueError):
        visibility_mask(blocked, (7, 1), 3, "raycast")
    with pytest.raises(ValueError):
        visibility_mask(blocked, (1, 1), 0, "raycast")
    with pytest.raises(ValueError):
        visibility_mask(blocked, (1, 1), 2, "unknown")
    with pytest.raises(ValueError):
        GridConfig(sensor_mode="invalid")


def test_same_random_truth_across_sensor_modes():
    kwargs = dict(size=16, observation_radius=3, hidden_obstacle_count=10)
    circle = GridWorldEnv(GridConfig(**kwargs, sensor_mode="circle"))
    ray = GridWorldEnv(GridConfig(**kwargs, sensor_mode="raycast"))
    circle.reset(seed=123)
    ray.reset(seed=123)
    for key in ("static_map", "hidden_obstacles", "robot", "goal"):
        one = circle.debug_snapshot()[key]
        two = ray.debug_snapshot()[key]
        if isinstance(one, np.ndarray):
            np.testing.assert_array_equal(one, two)
        else:
            assert one == two
