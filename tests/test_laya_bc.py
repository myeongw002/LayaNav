"""CPU BC training and checkpoint metadata without downloading Laya weights."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn

from layanav.evaluation.benchmark import benchmark
from layanav.models.laya_spatial import SpatialAdapterConfig, SpatialLayaPolicy
from layanav.training.dataset import save_dataset
from layanav.training.laya_bc import load_laya_adapter, train_laya_bc


class DummyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=16)
        self.fc = nn.Linear(16, 16)

    def forward(self, *, inputs_embeds, attention_mask):
        assert attention_mask.shape == inputs_embeds.shape[:2]
        hidden = self.fc(inputs_embeds)
        return SimpleNamespace(
            last_hidden_state=hidden + hidden.mean(dim=1, keepdim=True)
        )


def make_dataset(path, seed_start: int, count: int) -> None:
    encoded = np.zeros((count, 6, 9, 9), dtype=np.uint8)
    encoded[:, 0, 0, :] = 10
    encoded[:, 2] = 10
    encoded[:, 3, 3, 3] = 10
    encoded[:, 4, 6, 6] = 10
    encoded[:, 5, 3, 3] = 1
    dataset = {
        "schema_version": np.asarray(1, dtype=np.int32),
        "observations": encoded,
        "actions": np.asarray([i % 4 for i in range(count)], dtype=np.uint8),
        "episode_seeds": np.arange(seed_start, seed_start + count, dtype=np.int64),
        "grid_size": np.asarray(9, dtype=np.int32),
    }
    save_dataset(path, dataset)


def test_train_adapter_checkpoint_reload_and_benchmark(monkeypatch, tmp_path):
    torch.set_num_threads(1)
    backbone = DummyBackbone()
    reference_backbone = copy.deepcopy(backbone)
    model = SpatialLayaPolicy(
        backbone, SpatialAdapterConfig(feature_channels=8, token_grid=2),
    )
    train_path = tmp_path / "train.npz"
    valid_path = tmp_path / "valid.npz"
    checkpoint = tmp_path / "spatial.pt"
    make_dataset(train_path, 1, 8)
    make_dataset(valid_path, 1001, 4)
    source = {
        "model_id": "test/frozen-backbone",
        "resolved_revision": "deadbeef",
        "requested_revision": "deadbeef",
        "package_version": "fake-test",
        "hidden_size": 16,
    }
    history = train_laya_bc(
        model=model, train_data=train_path, validation_data=valid_path,
        output=checkpoint, source=source,
        epochs=1, batch_size=2, accumulation_steps=2, seed=18,
    )
    assert len(history) == 1
    assert np.isfinite(history[0]["val_loss"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["source"] == source
    assert payload["grid_shape"] == [6, 9, 9]
    assert all(not key.startswith("backbone.") for key in payload["adapter_state"])

    def fake_load_backbone(*, model_id, revision, device):
        assert model_id == "test/frozen-backbone"
        assert revision == "deadbeef"
        assert device == "cpu"
        return copy.deepcopy(reference_backbone), {"resolved_revision": "deadbeef"}

    monkeypatch.setattr(
        "layanav.training.laya_bc.load_laya_backbone", fake_load_backbone,
    )
    restored, saved = load_laya_adapter(checkpoint, device="cpu")
    assert saved["source"] == source
    model.eval()
    x = torch.rand((1, 6, 9, 9))
    torch.testing.assert_close(model(x)[0], restored(x)[0], atol=1e-6, rtol=1e-6)

    report = benchmark(
        methods=("laya",), families=("sparse",), episodes=2, eval_seed=200000,
        size=9, radius=2, sensor="raycast", max_steps=20,
        output_dir=tmp_path / "eval", laya_checkpoint=checkpoint,
        bootstrap_replicates=100,
    )
    assert report["summary"]["sparse/laya"]["steps"]["n"] == 2
    assert report["checkpoints"]["laya"]["training"]["source"] == source

    with pytest.raises(ValueError, match="disjoint"):
        train_laya_bc(
            model=model, train_data=train_path, validation_data=train_path,
            output=checkpoint, source=source, epochs=1, batch_size=2,
        )


def test_nine_action_laya_bc_checkpoint(tmp_path):
    def write_dataset(path, start_seed):
        encoded = np.zeros((4, 6, 9, 9), dtype=np.uint8)
        encoded[:, 3, 1, 1] = 10
        encoded[:, 4, 7, 7] = 10
        save_dataset(path, {
            "schema_version": np.asarray(1, dtype=np.int32),
            "movement_mode": np.asarray(8, dtype=np.int32),
            "observations": encoded,
            "actions": np.asarray([5, 6, 7, 8], dtype=np.uint8),
            "episode_seeds": np.arange(start_seed, start_seed + 4),
        })

    tr, va = tmp_path / "tr9.npz", tmp_path / "va9.npz"
    write_dataset(tr, 10)
    write_dataset(va, 100)
    model = SpatialLayaPolicy(
        DummyBackbone(),
        SpatialAdapterConfig(feature_channels=8, token_grid=2),
        movement_mode=8,
    )
    target = tmp_path / "laya9.pt"
    train_laya_bc(
        model=model, train_data=tr, validation_data=va, output=target,
        source={"model_id": "test/mock"}, epochs=1, batch_size=2,
    )
    payload = torch.load(target, weights_only=True, map_location="cpu")
    assert payload["movement_mode"] == 8
    assert payload["adapter_state"]["actor.1.weight"].shape[0] == 9

    wrong = SpatialLayaPolicy(
        DummyBackbone(),
        SpatialAdapterConfig(feature_channels=8, token_grid=2),
        movement_mode=4,
    )
    with pytest.raises(ValueError, match="movement_mode"):
        train_laya_bc(
            model=wrong, train_data=tr, validation_data=va, output=target,
            source={"model_id": "test/mock"}, epochs=1, batch_size=2,
        )
