"""Small-backbone contract tests; no pretrained checkpoint download in CI."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn

from layanav.models.laya_spatial import (
    SpatialAdapterConfig,
    SpatialLayaPolicy,
    load_laya_backbone,
)


class TinyBackbone(nn.Module):
    def __init__(self, hidden_size: int = 32) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, *, inputs_embeds, attention_mask):
        assert inputs_embeds.ndim == 3
        assert attention_mask.shape == inputs_embeds.shape[:2]
        assert torch.all(attention_mask == 1)
        hidden = self.dropout(self.linear(inputs_embeds))
        hidden = hidden + hidden.mean(dim=1, keepdim=True)
        return SimpleNamespace(last_hidden_state=hidden)


def test_spatial_frozen_encoder_gradients():
    torch.manual_seed(7)
    backbone = TinyBackbone()
    model = SpatialLayaPolicy(
        backbone, SpatialAdapterConfig(feature_channels=8, token_grid=2)
    )
    model.train()
    assert not model.backbone.training
    x = torch.rand(2, 6, 9, 9, dtype=torch.float32)
    logits, values = model(x)
    assert logits.shape == (2, 5)
    assert values.shape == (2,)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([1, 3]))
    loss = loss + values.square().mean()
    loss.backward()
    assert all(p.grad is None for p in backbone.parameters())
    assert any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.tokenizer.parameters()
    )
    assert any(p.grad is not None for p in model.actor.parameters())
    assert any(p.grad is not None for p in model.critic.parameters())
    assert model.act(np.zeros((6, 9, 9), dtype=np.float32)) in range(5)


def test_unfrozen_backbone_gets_gradients():
    backbone = TinyBackbone()
    model = SpatialLayaPolicy(
        backbone, SpatialAdapterConfig(feature_channels=8, token_grid=2, freeze_backbone=False)
    )
    model.train()
    assert model.backbone.training
    logits, value = model(torch.rand((1, 6, 9, 9)))
    (logits.sum() + value.sum()).backward()
    assert any(p.grad is not None for p in backbone.parameters())


def test_adapter_state_roundtrip_without_backbone():
    torch.manual_seed(20)
    backbone = TinyBackbone()
    other = copy.deepcopy(backbone)
    a = SpatialLayaPolicy(
        backbone, SpatialAdapterConfig(feature_channels=8, token_grid=2)
    )
    b = SpatialLayaPolicy(
        other, SpatialAdapterConfig(feature_channels=8, token_grid=2)
    )
    state = a.adapter_state_dict()
    assert state and not any(key.startswith("backbone.") for key in state)
    b.load_adapter_state_dict(state)
    x = torch.rand((1, 6, 9, 9))
    a.eval()
    b.eval()
    actual = a(x)
    restored = b(x)
    torch.testing.assert_close(actual[0], restored[0])
    torch.testing.assert_close(actual[1], restored[1])
    with pytest.raises(ValueError, match="keys"):
        b.load_adapter_state_dict({"invalid": torch.tensor(1)})


def test_invalid_shapes_configs_and_types():
    with pytest.raises(ValueError, match="hidden_size"):
        SpatialLayaPolicy(nn.Linear(3, 3))
    with pytest.raises(ValueError, match="feature_channels"):
        SpatialAdapterConfig(feature_channels=10)
    with pytest.raises(ValueError, match="token_grid"):
        SpatialAdapterConfig(token_grid=0)
    model = SpatialLayaPolicy(
        TinyBackbone(), SpatialAdapterConfig(feature_channels=8, token_grid=2)
    )
    with pytest.raises(ValueError, match="shape"):
        model(torch.rand((2, 4, 9, 9)))
    with pytest.raises(TypeError):
        model(torch.ones((1, 6, 9, 9), dtype=torch.int64))
    with pytest.raises(ValueError, match="public observation"):
        model.act(np.zeros((9, 9), dtype=np.float32))


def test_laya_loader_is_lazy_and_tracks_revision(monkeypatch):
    import sys

    backbone = TinyBackbone()
    arguments = []

    def fake_load(model_id, **kwargs):
        arguments.append((model_id, kwargs))
        return SimpleNamespace(
            model=SimpleNamespace(encoder=backbone),
            revision="0123456789abcdef",
        )

    monkeypatch.setitem(sys.modules, "laya", SimpleNamespace(load=fake_load))
    monkeypatch.setattr(
        "layanav.models.laya_spatial.importlib.metadata.version", lambda x: "0.4.31"
    )
    loaded, source = load_laya_backbone(
        model_id="convaiinnovations/laya", revision="0123456789abcdef", device="cpu"
    )
    assert loaded is backbone
    assert source["resolved_revision"] == "0123456789abcdef"
    assert source["hidden_size"] == 32
    assert arguments[0][1]["backend"] == "eager"
    assert arguments[0][1]["revision"] == "0123456789abcdef"
