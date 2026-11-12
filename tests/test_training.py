"""CPU smoke tests of the optional PyTorch training pipeline."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from layanav.envs import GridConfig
from layanav.models.cnn_policy import CNNPolicy
from layanav.training.behavior_cloning import (
    load_policy,
    train_behavior_cloning,
)
from layanav.training.dataset import (
    collect_expert_dataset,
    save_dataset,
)


def test_cnn_forward_backward_and_action():
    model = CNNPolicy(channels=8)
    observation = torch.rand(2, 6, 9, 9)
    logits = model(observation)
    assert logits.shape == (2, 5)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([1, 3]))
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
    action = model.act(np.zeros((6, 9, 9), dtype=np.float32))
    assert 0 <= action < 5


def test_bc_train_checkpoint_and_seed_split(tmp_path):
    config = GridConfig(
        size=9, observation_radius=2, sensor_mode="raycast",
        hidden_obstacle_count=0, known_wall_segments=0, max_steps=36
    )
    train, _ = collect_expert_dataset(episodes=4, seed_start=50, config=config)
    validation, _ = collect_expert_dataset(episodes=2, seed_start=1050, config=config)
    train_path = tmp_path / "train.npz"
    val_path = tmp_path / "val.npz"
    checkpoint = tmp_path / "cnn.pt"
    save_dataset(train_path, train)
    save_dataset(val_path, validation)
    torch.set_num_threads(1)
    history = train_behavior_cloning(
        train_data=train_path, validation_data=val_path,
        output_checkpoint=checkpoint, epochs=1, batch_size=8,
        channels=8, device="cpu", seed=7
    )
    assert len(history) == 1
    assert np.isfinite(history[0]["val_loss"])
    assert checkpoint.exists()
    model = load_policy(checkpoint, device="cpu")
    assert model.act(np.zeros((6, 9, 9), dtype=np.float32)) in range(5)

    with pytest.raises(ValueError, match="disjoint"):
        train_behavior_cloning(
            train_data=train_path, validation_data=train_path,
            output_checkpoint=checkpoint, epochs=1,
            channels=8, device="cpu"
        )
