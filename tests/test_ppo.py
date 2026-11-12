"""Short CPU PPO policy smoke: actual training, checkpointing and evaluation."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
sb3 = pytest.importorskip("stable_baselines3")

from layanav.envs import GridConfig, GridWorldEnv
from layanav.evaluation.benchmark import benchmark
from layanav.training.ppo import (
    GridFeaturesExtractor,
    PPOSettings,
    load_ppo_metadata,
    load_ppo_policy,
    train_ppo,
)


def test_ppo_settings_reject_bad_rollout_and_batch_size():
    with pytest.raises(ValueError, match="batch_size"):
        PPOSettings(n_envs=1, n_steps=8, batch_size=16)
    with pytest.raises(ValueError, match="rollout"):
        PPOSettings(n_envs=1, n_steps=1, batch_size=1)
    with pytest.raises(ValueError, match="positive"):
        PPOSettings(timesteps=0)


def test_ppo_cnn_gradient_shape():
    config = GridConfig(size=9, hidden_obstacle_count=0, known_wall_segments=0)
    env = GridWorldEnv(config)
    extractor = GridFeaturesExtractor(env.observation_space, features_dim=32, channels=8)
    tensor = torch.rand((2, 6, 9, 9), dtype=torch.float32)
    features = extractor(tensor)
    assert features.shape == (2, 32)
    features.sum().backward()
    assert any(p.grad is not None for p in extractor.parameters())


def test_short_ppo_train_save_reload_and_benchmark(tmp_path):
    torch.set_num_threads(1)
    config = GridConfig(
        size=9, observation_radius=2, sensor_mode="raycast",
        hidden_obstacle_count=0, known_wall_segments=0, max_steps=20,
    )
    checkpoint = tmp_path / "ppo_test.zip"
    metadata = train_ppo(
        output=checkpoint,
        env_config=config,
        settings=PPOSettings(
            timesteps=32, n_envs=1, n_steps=8, batch_size=8,
            n_epochs=1, channels=8, device="cpu", seed=23
        ),
    )
    assert checkpoint.exists()
    assert metadata["actual_timesteps"] >= 32
    assert metadata["observation_shape"] == [6, 9, 9]
    assert load_ppo_metadata(checkpoint)["training"]["seed"] == 23

    agent = load_ppo_policy(checkpoint, device="cpu")
    env = GridWorldEnv(config)
    observation, _ = env.reset(seed=44)
    assert agent.act(observation) in range(5)
    env.close()

    report = benchmark(
        methods=("astar", "ppo"), families=("sparse",), episodes=2,
        eval_seed=200000, size=9, radius=2, sensor="raycast", max_steps=20,
        output_dir=tmp_path / "evaluation", ppo_checkpoint=checkpoint,
        device="cpu", bootstrap_replicates=100,
    )
    assert report["summary"]["sparse/ppo"]["success"]["n"] == 2
    assert report["checkpoints"]["ppo"]["training"]["actual_timesteps"] >= 32
    assert len(report["checkpoints"]["ppo"]["sha256"]) == 64
    with pytest.raises(ValueError, match="expects"):
        benchmark(
            methods=("ppo",), families=("sparse",), episodes=1,
            eval_seed=200000, size=12, radius=2, sensor="raycast", max_steps=20,
            output_dir=tmp_path / "wrong-shape", ppo_checkpoint=checkpoint,
        )


def test_eight_way_ppo_action_space(tmp_path):
    config = GridConfig(
        size=9, movement_mode=8, observation_radius=2, sensor_mode="circle",
        hidden_obstacle_count=0, known_wall_segments=0, max_steps=20,
    )
    checkpoint = tmp_path / "ppo_8.zip"
    meta = train_ppo(
        output=checkpoint, env_config=config,
        settings=PPOSettings(
            timesteps=16, n_envs=1, n_steps=8, batch_size=8,
            n_epochs=1, channels=8, device="cpu", seed=32,
        ),
    )
    assert meta["action_count"] == 9
    policy = load_ppo_policy(checkpoint)
    assert policy.model.action_space.n == 9
    observation, _ = GridWorldEnv(config).reset(seed=40)
    assert policy.act(observation) in range(9)
    with pytest.raises(ValueError, match="actions"):
        benchmark(
            methods=("ppo",), families=("sparse",),
            episodes=1, eval_seed=1234, size=9, radius=2, sensor="circle",
            max_steps=20, movement_mode=4,
            output_dir=tmp_path / "mismatch", ppo_checkpoint=checkpoint,
        )
