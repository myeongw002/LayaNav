"""Lightweight six-channel spatial policy for navigation baselines."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from layanav.envs import ACTION_NAMES


class CNNPolicy(nn.Module):
    """Convolutional five-action policy; no privileged map access.

    Input: float32 (B, 6, H, W) from the public environment observation.
    Output: unnormalized categorical action logits (B, 5).
    """

    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        if channels < 8 or channels % 8:
            raise ValueError("channels must be a multiple of 8 and >= 8")
        self.channels = channels
        self.encoder = nn.Sequential(
            nn.Conv2d(6, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, channels * 2),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels * 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, channels * 2),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(channels * 2 * 4 * 4, 128),
            nn.GELU(),
            nn.Linear(128, len(ACTION_NAMES)),
        )

    def forward(self, observation: Tensor) -> Tensor:
        if observation.ndim != 4 or observation.shape[1] != 6:
            raise ValueError("Expected input shape (B, 6, H, W)")
        if not observation.is_floating_point():
            raise TypeError("CNNPolicy expects floating-point observations")
        return self.head(self.encoder(observation))

    @torch.inference_mode()
    def act(self, observation: np.ndarray) -> int:
        """Greedy action from public observation; no simulator object supplied."""
        if observation.ndim != 3 or observation.shape[0] != 6:
            raise ValueError("Expected observation (6, H, W)")
        self.eval()
        device = next(self.parameters()).device
        data = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        return int(self(data).argmax(dim=-1).item())
