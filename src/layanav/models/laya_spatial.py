"""Experimental 6-channel grid adapter for Laya's pretrained text encoder.

Laya's official DecisionModel.encoder is a Hugging Face ModernBertModel,
which supports `inputs_embeds`. This module uses only that pretrained
backbone; it does NOT reuse Laya's text-specific marker/decision head.

Tokenizing spatial maps is out of distribution for the text-pretrained model.
No navigation transfer or calibration is implied by using pretrained weights.
"""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class SpatialAdapterConfig:
    feature_channels: int = 64
    token_grid: int = 8
    freeze_backbone: bool = True

    def __post_init__(self) -> None:
        if self.feature_channels < 8 or self.feature_channels % 8:
            raise ValueError("feature_channels must be >=8 and divisible by 8")
        if not isinstance(self.token_grid, int) or self.token_grid < 1:
            raise ValueError("token_grid must be a positive integer")


class SpatialTokenizer(nn.Module):
    """(B,6,H,W) float map to (B,1+K,d) summary + row-major spatial tokens."""

    def __init__(self, hidden_size: int, config: SpatialAdapterConfig) -> None:
        super().__init__()
        if hidden_size < 1:
            raise ValueError("Backbone hidden_size must be positive")
        channels = config.feature_channels
        grid = config.token_grid
        self.patch_encoder = nn.Sequential(
            nn.Conv2d(6, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((grid, grid)),
        )
        self.projector = nn.Linear(channels, hidden_size)
        self.summary_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.position_embedding = nn.Parameter(torch.empty(1, grid * grid, hidden_size))
        nn.init.normal_(self.summary_token, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)

    def forward(self, grid: Tensor) -> Tensor:
        if grid.ndim != 4 or grid.shape[1] != 6:
            raise ValueError("Expected grid with shape (B,6,H,W)")
        if not grid.is_floating_point():
            raise TypeError("SpatialTokenizer requires floating-point map input")
        encoded = self.patch_encoder(grid)
        batch, channels, _, _ = encoded.shape
        patches = encoded.flatten(2).transpose(1, 2).contiguous()
        patches = self.projector(patches) + self.position_embedding
        summary = self.summary_token.expand(batch, -1, -1)
        return torch.cat((summary, patches), dim=1)


class SpatialLayaPolicy(nn.Module):
    """Spatial adapter + pretrained bidirectional encoder + actor/critic heads.

    `backbone` must expose a numeric `config.hidden_size` and a
    `forward(inputs_embeds=..., attention_mask=...)` returning
    `.last_hidden_state` or a tuple whose first element is hidden states.

    Freeze sets all backbone parameters to `requires_grad_(False)`, but
    DOES NOT wrap the forward in `no_grad`: spatial projector gradients
    must flow through the frozen transformer. The frozen encoder is kept
    in eval mode during adapter training.
    """

    def __init__(
        self, backbone: nn.Module, config: SpatialAdapterConfig | None = None
    ) -> None:
        super().__init__()
        self.config = config or SpatialAdapterConfig()
        hidden_size = getattr(getattr(backbone, "config", None), "hidden_size", None)
        if not isinstance(hidden_size, int) or hidden_size < 1:
            raise ValueError("Laya encoder must expose config.hidden_size")
        self.backbone = backbone
        self.tokenizer = SpatialTokenizer(hidden_size, self.config)
        self.actor = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 5))
        self.critic = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1))
        if self.config.freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode: bool = True) -> SpatialLayaPolicy:
        super().train(mode)
        if self.config.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        tokens = self.tokenizer(observations)
        try:
            backbone_parameter = next(self.backbone.parameters())
        except StopIteration as exc:
            raise RuntimeError("Backbone has no parameters; cannot establish dtype/device") from exc
        if tokens.device != backbone_parameter.device:
            raise ValueError("Spatial adapter and backbone must be on the same device")
        tokens = tokens.to(dtype=backbone_parameter.dtype)
        attention_mask = torch.ones(
            tokens.shape[:2], device=tokens.device, dtype=torch.long
        )
        outputs = self.backbone(inputs_embeds=tokens, attention_mask=attention_mask)
        hidden = (
            outputs.last_hidden_state
            if hasattr(outputs, "last_hidden_state")
            else outputs[0]
        )
        if hidden.ndim != 3 or hidden.shape != tokens.shape:
            raise ValueError("Backbone returned an unexpected hidden-state shape")
        pooled = hidden[:, 0].float()
        return self.actor(pooled), self.critic(pooled).squeeze(-1)

    @torch.inference_mode()
    def act(self, observation: np.ndarray) -> int:
        if observation.ndim != 3 or observation.shape[0] != 6:
            raise ValueError("Expected public observation with shape (6,H,W)")
        self.eval()
        device = next(self.parameters()).device
        tokens = torch.as_tensor(
            observation, device=device, dtype=torch.float32
        ).unsqueeze(0)
        logits, _ = self(tokens)
        return int(logits.argmax(-1).item())

    def adapter_state_dict(self) -> dict[str, Tensor]:
        """Export ONLY trainable spatial adapter/heads, never 395M backbone weights."""
        if not self.config.freeze_backbone:
            raise ValueError("Adapter-only export requires a frozen backbone")
        return {
            key: value.detach().cpu().clone()
            for key, value in self.state_dict().items()
            if not key.startswith("backbone.")
        }

    def load_adapter_state_dict(self, state: dict[str, Tensor]) -> None:
        """Restore adapter/head tensors while retaining the loaded pretrained backbone."""
        if not self.config.freeze_backbone:
            raise ValueError("Adapter-only restore requires a frozen backbone")
        expected = {
            key for key in self.state_dict()
            if not key.startswith("backbone.")
        }
        if set(state) != expected:
            raise ValueError("Adapter state keys do not match model structure")
        self.load_state_dict(state, strict=False)


def load_laya_backbone(
    *,
    model_id: str = "convaiinnovations/laya",
    device: str = "cpu",
    revision: str | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load the official checkpoint; requires the optional `laya` dependency.

    No downloading occurs until this function is explicitly called.
    The user may provide an exact Hub revision for reproducibility.
    """
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available in this PyTorch environment")
    try:
        import laya
    except ImportError as exc:
        raise ImportError("Install Laya: pip install -e '.[train,laya]'") from exc
    args: dict[str, Any] = {
        "device": device,
        "backend": "eager",
        "compile": False,
        "fast": False,
    }
    if revision:
        args["revision"] = revision
    agent = laya.load(model_id, **args)
    model = getattr(agent, "model", None)
    backbone = getattr(model, "encoder", None)
    if not isinstance(backbone, nn.Module):
        raise RuntimeError("Loaded Laya checkpoint does not expose model.encoder")
    if not hasattr(getattr(backbone, "config", None), "hidden_size"):
        raise RuntimeError("Loaded Laya encoder has no config.hidden_size")
    source = {
        "model_id": model_id,
        "requested_revision": revision,
        "resolved_revision": getattr(agent, "revision", None),
        "package_version": importlib.metadata.version("laya"),
        "backbone_class": type(backbone).__name__,
        "hidden_size": int(backbone.config.hidden_size),
    }
    return backbone, source
