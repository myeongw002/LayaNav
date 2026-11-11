"""Versioned simulator map serialization; no privileged state in observations."""

import json

import numpy as np
import pytest

from layanav.envs import GridConfig, GridWorldEnv
from layanav.envs.map_io import load_map, save_map, validate_layout


def sample_layout():
    prior = np.zeros((9, 9), dtype=bool)
    prior[2, 3] = True
    hidden = np.zeros_like(prior)
    hidden[4, 5] = True
    return {
        "static_map": prior,
        "hidden_obstacles": hidden,
        "start": (1, 4),
        "goal": (7, 4),
    }


def test_save_load_round_trip_and_env_reset(tmp_path):
    layout = sample_layout()
    path = tmp_path / "map.json"
    save_map(path, layout)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["schema_version"] == 1
    assert raw["size"] == 9
    loaded = load_map(path)
    np.testing.assert_array_equal(loaded["static_map"], layout["static_map"])
    np.testing.assert_array_equal(loaded["hidden_obstacles"], layout["hidden_obstacles"])
    assert loaded["start"] == layout["start"]
    assert loaded["goal"] == layout["goal"]

    env = GridWorldEnv(GridConfig(size=9, observation_radius=1))
    obs, info = env.reset(options=loaded)
    assert info["position"] == layout["start"]
    assert obs[1, 4, 5] == 0.0  # Truth only in simulator, not policy.
    assert obs[2, 4, 5] == 1.0


@pytest.mark.parametrize("change", [
    {"start": (5, 4)},  # Start lies on hidden obstacle.
    {"start": (7, 4)},  # Start and goal coincide.
    {"goal": (9, 4)},  # Out of bounds.
    {"start": (1.0, 4)},  # Strict coordinate type.
    {"start": (True, 4)},
])
def test_invalid_positions_rejected(change):
    layout = sample_layout() | change
    with pytest.raises(ValueError):
        validate_layout(layout)


def test_reject_overlap_and_bad_values():
    layout = sample_layout()
    layout["static_map"][4, 5] = True
    with pytest.raises(ValueError):
        validate_layout(layout)

    layout = sample_layout()
    layout["hidden_obstacles"] = np.zeros((5, 5), dtype=bool)
    with pytest.raises(ValueError):
        validate_layout(layout)

    layout = sample_layout()
    layout["static_map"] = np.full((9, 9), 0.5)
    with pytest.raises(ValueError):
        validate_layout(layout)


def test_reject_bad_version_and_mismatched_size(tmp_path):
    path = tmp_path / "map.json"
    save_map(path, sample_layout())
    data = json.loads(path.read_text(encoding="utf-8"))

    data["schema_version"] = 99
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        load_map(path)

    data["schema_version"] = 1
    data["size"] = 10
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="size"):
        load_map(path)


def test_save_invalid_layout_does_not_overwrite_existing_map(tmp_path):
    path = tmp_path / "map.json"
    save_map(path, sample_layout())
    original = path.read_bytes()
    broken = sample_layout()
    broken["start"] = (5, 4)
    with pytest.raises(ValueError):
        save_map(path, broken)
    assert path.read_bytes() == original
