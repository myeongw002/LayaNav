"""A* demonstration collection uses only policy observations."""

import numpy as np
import pytest

from layanav.envs import GridConfig
from layanav.training.dataset import (
    collect_expert_dataset,
    decode_observations,
    encode_observations,
    load_dataset,
    save_dataset,
)


def test_exact_six_channel_observation_quantization():
    observation = np.zeros((6, 7, 7), dtype=np.float32)
    observation[0, 2, 2] = 1.0
    observation[2, 4, 4] = 1.0
    observation[5, 3, 3] = 0.7
    encoded = encode_observations(observation)
    assert encoded.dtype == np.uint8
    decoded = decode_observations(encoded[None, ...])
    np.testing.assert_allclose(decoded[0], observation, atol=1e-7)


def test_reject_unrepresentable_observations():
    bad = np.zeros((6, 7, 7), dtype=np.float32)
    bad[5, 0, 0] = 0.145
    with pytest.raises(ValueError):
        encode_observations(bad)


def test_collect_save_load_dataset_with_held_out_seeds(tmp_path):
    config = GridConfig(
        size=9, observation_radius=2, sensor_mode="raycast",
        hidden_obstacle_count=0, known_wall_segments=0, max_steps=36
    )
    dataset, stats = collect_expert_dataset(episodes=4, seed_start=50, config=config)
    assert stats["successful_episodes"] == 4
    assert dataset["observations"].shape[1:] == (6, 9, 9)
    assert len(dataset["actions"]) == stats["samples"]
    assert set(np.unique(dataset["episode_seeds"])) == {50, 51, 52, 53}

    path = tmp_path / "teacher.npz"
    save_dataset(path, dataset)
    obs, actions, seeds = load_dataset(path)
    assert obs.shape[0] == len(actions) == len(seeds)
    assert set(np.unique(seeds)) == {50, 51, 52, 53}
    assert np.all((actions >= 0) & (actions < 5))
    assert obs.shape[1:] == (6, 9, 9)


def test_eight_way_expert_dataset_and_legacy_loader(tmp_path):
    from layanav.training.dataset import load_dataset_movement_mode

    config = GridConfig(
        size=9, movement_mode=8, sensor_mode="raycast",
        hidden_obstacle_count=0, known_wall_segments=0, max_steps=32,
    )
    dataset, stats = collect_expert_dataset(episodes=6, seed_start=100, config=config)
    assert stats["successful_episodes"] == 6
    assert int(dataset["movement_mode"]) == 8
    assert np.any(dataset["actions"] >= 5)
    path = tmp_path / "eight.npz"
    save_dataset(path, dataset)
    assert load_dataset_movement_mode(path) == 8
    _, actions, _ = load_dataset(path)
    assert (actions < 9).all()

    # Legacy, before a movement-mode marker, must remain four-direction.
    legacy = dict(dataset)
    legacy.pop("movement_mode")
    legacy["actions"] = np.zeros_like(dataset["actions"])
    save_dataset(tmp_path / "legacy.npz", legacy)
    assert load_dataset_movement_mode(tmp_path / "legacy.npz") == 4
    assert len(load_dataset(tmp_path / "legacy.npz")[1]) == len(actions)
