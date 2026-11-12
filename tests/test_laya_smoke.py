"""Check real-checkpoint smoke orchestration against a small CPU stand-in.

The CI job is NOT a pretrained model or RTX 5070 test.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from layanav.models.laya_spatial import load_laya_backbone
from layanav.training.laya_smoke import _write_report, probe_device, smoke


class TinyEncoder(nn.Module):
    def __init__(self, hidden_size: int = 24) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = nn.Linear(hidden_size, hidden_size)

    def forward(self, *, inputs_embeds, attention_mask):
        assert attention_mask.shape == inputs_embeds.shape[:2]
        hidden = self.embedding(inputs_embeds)
        return SimpleNamespace(
            last_hidden_state=hidden + hidden.mean(dim=1, keepdim=True)
        )


def test_cpu_preflight_does_not_require_pretrained_laya():
    result = probe_device("cpu")
    assert result["requested_device"] == "cpu"
    assert result["gpu_name"] is None
    assert result["kernel_test"] == "not_applicable"
    assert result["torch_version"]


def test_smoke_one_training_step_and_json_report(monkeypatch, tmp_path):
    torch.set_num_threads(1)

    def fake_loader(*, model_id, revision, device):
        assert model_id == "unit-test/frozen-laya"
        assert revision == "deadbeef"
        assert device == "cpu"
        return TinyEncoder(), {
            "model_id": model_id,
            "resolved_revision": revision,
            "backbone_class": "TinyEncoder",
            "hidden_size": 24,
        }

    monkeypatch.setattr("layanav.training.laya_smoke.load_laya_backbone", fake_loader)
    report = smoke(
        model_id="unit-test/frozen-laya",
        revision="deadbeef",
        device="cpu",
        grid_size=9,
        movement_mode=8,
        batch_size=2,
        token_grid=2,
        feature_channels=8,
        seed=11,
    )
    assert report["status"] == "passed"
    assert report["backbone"]["resolved_revision"] == "deadbeef"
    assert report["actor_logits_shape"] == [2, 9]
    assert report["critic_values_shape"] == [2]
    assert report["spatial_tokens"] == 5
    assert report["optimizer_step_changed_actor"] is True
    assert report["adapter_gradient_l1"] > 0
    assert report["actor_gradient_l1"] > 0
    assert report["global_grad_norm_before_clipping"] > 0
    assert 0 <= report["sample_action_id"] < 9
    assert report["gpu_memory"]["step_peak_allocated_gib"] is None
    assert report["forward_seconds"] >= 0
    assert report["backward_seconds"] >= 0
    assert report["post_update_inference_seconds"] >= 0

    destination = tmp_path / "results" / "smoke.json"
    _write_report(str(destination), report)
    saved = json.loads(destination.read_text(encoding="utf-8"))
    assert saved["status"] == "passed"
    assert saved["backbone"]["resolved_revision"] == "deadbeef"


def test_cuda_cpu_fallback_is_rejected_even_if_laya_load_returns(monkeypatch):
    import sys

    def fake_load(model_id, **kwargs):
        assert kwargs["device"] == "cuda"
        return SimpleNamespace(
            model=SimpleNamespace(encoder=TinyEncoder()),
            revision="same-revision",
        )

    monkeypatch.setitem(sys.modules, "laya", SimpleNamespace(load=fake_load))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(RuntimeError, match="device mismatch"):
        load_laya_backbone(model_id="unit-test/laya", device="cuda")


def test_validation_rejects_invalid_inputs_without_loading(monkeypatch):
    def forbidden_loader(**kwargs):
        raise AssertionError("Invalid input should be rejected before load")

    monkeypatch.setattr("layanav.training.laya_smoke.load_laya_backbone", forbidden_loader)
    with pytest.raises(ValueError, match="batch_size"):
        smoke(
            model_id="test", revision=None, device="cpu",
            grid_size=9, movement_mode=4, batch_size=0,
            token_grid=2, feature_channels=8, seed=42,
        )
    with pytest.raises(ValueError, match="learning_rate"):
        smoke(
            model_id="test", revision=None, device="cpu",
            grid_size=9, movement_mode=4, batch_size=1,
            token_grid=2, feature_channels=8, seed=42,
            learning_rate=float("nan"),
        )
    with pytest.raises(ValueError, match="only CPU or CUDA"):
        probe_device("meta")
