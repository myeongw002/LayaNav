"""Unit tests assert that hidden state does not leak into agent observations."""

import numpy as np
import pytest

from layanav.envs import GridConfig, GridWorldEnv


def scenario(radius=1):
    size = 9
    static = np.zeros((size, size), dtype=bool)
    static[0, :] = static[-1, :] = True
    static[:, 0] = static[:, -1] = True
    hidden = np.zeros_like(static)
    hidden[4, 4] = True
    options = {
        "static_map": static,
        "hidden_obstacles": hidden,
        "start": (2, 4),
        "goal": (6, 4),
    }
    env = GridWorldEnv(GridConfig(size=size, observation_radius=radius, max_steps=50))
    return env, options


def test_unseen_obstacle_is_hidden_then_discovered():
    env, options = scenario(radius=1)
    obs, info = env.reset(options=options)
    assert obs.shape == (6, 9, 9)
    assert obs.dtype == np.float32
    assert obs[1, 4, 4] == 0.0
    assert obs[2, 4, 4] == 1.0
    assert "hidden_obstacles" not in info
    obs, _, _, _, info = env.step(3)
    assert info["position"] == (3, 4)
    assert obs[1, 4, 4] == 1.0
    assert obs[2, 4, 4] == 0.0


def test_collision_no_motion_and_step_ends():
    env, options = scenario(radius=1)
    env.reset(options=options)
    env.step(3)
    _, reward, terminated, truncated, info = env.step(3)
    assert info["collision"] and info["position"] == (3, 4)
    assert reward < -0.5
    assert not terminated and not truncated


def test_hidden_map_is_copied_not_shared():
    env, options = scenario(radius=1)
    env.reset(options=options)
    debug = env.debug_snapshot()
    debug["hidden_obstacles"][:] = False
    assert env.debug_snapshot()["hidden_obstacles"][4, 4]


def test_reproducible_random_map():
    env = GridWorldEnv(GridConfig(size=16, known_wall_segments=4, hidden_obstacle_count=15))
    a, _ = env.reset(seed=91)
    ground_a = env.debug_snapshot()
    b, _ = env.reset(seed=91)
    ground_b = env.debug_snapshot()
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(ground_a["hidden_obstacles"], ground_b["hidden_obstacles"])


def test_reset_invalid_map():
    env, options = scenario(radius=2)
    options["hidden_obstacles"][4, 2] = True
    with pytest.raises(ValueError):
        env.reset(options=options)


def test_timeout_and_no_step_after_finish():
    env, options = scenario(radius=1)
    env.config.max_steps = 1
    env.reset(options=options)
    _, _, terminated, truncated, _ = env.step(4)
    assert truncated and not terminated
    with pytest.raises(RuntimeError):
        env.step(4)


def test_no_truth_in_observation_even_after_reset():
    env, options = scenario(radius=1)
    obs, _ = env.reset(options=options)
    hidden = env.debug_snapshot()["hidden_obstacles"]
    assert hidden[4, 4] and not obs[1, 4, 4]
    assert obs[0, 4, 4] == 0
    assert obs[3].sum() == 1.0 and obs[4].sum() == 1.0
