"""PPO with a spatial CNN encoder on the six-channel public belief state.

Stable-Baselines3 uses a categorical actor and separate critic head. Its
default NatureCNN does not accept this float32 low-resolution map; this
custom feature extractor avoids accidental image normalization.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium
import numpy as np
import stable_baselines3
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
from torch import Tensor, nn

from layanav.envs import GridConfig, GridWorldEnv


class GridFeaturesExtractor(BaseFeaturesExtractor):
    """6-channel float32 map -> fixed-length vector for actor and critic."""

    def __init__(
        self, observation_space: spaces.Box, features_dim: int = 128, channels: int = 32
    ) -> None:
        if len(observation_space.shape) != 3 or observation_space.shape[0] != 6:
            raise ValueError("Expected six-channel (6, H, W) observation space")
        if channels < 8 or channels % 8 != 0 or features_dim < 1:
            raise ValueError("channels must be >=8 and divisible by 8; features_dim >=1")
        super().__init__(observation_space, features_dim=features_dim)
        self.network = nn.Sequential(
            nn.Conv2d(6, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, 2 * channels, 3, stride=2, padding=1),
            nn.GroupNorm(8, 2 * channels),
            nn.GELU(),
            nn.Conv2d(2 * channels, 2 * channels, 3, stride=2, padding=1),
            nn.GroupNorm(8, 2 * channels),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(2 * channels * 4 * 4, features_dim),
            nn.GELU(),
        )

    def forward(self, observations: Tensor) -> Tensor:
        return self.network(observations)


@dataclass(frozen=True)
class PPOSettings:
    timesteps: int = 100_000
    seed: int = 42
    n_envs: int = 4
    n_steps: int = 128
    batch_size: int = 128
    n_epochs: int = 4
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    clip_range: float = 0.2
    channels: int = 32
    device: str = "auto"

    def __post_init__(self) -> None:
        if any(
            type(v) is not int or v <= 0
            for v in (self.timesteps, self.n_envs, self.n_steps, self.batch_size,
                      self.n_epochs, self.channels)
        ):
            raise ValueError("Training counts, batch size and channels must be positive integers")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.n_steps * self.n_envs < 2:
            raise ValueError("PPO rollout buffer requires at least two transitions")
        if self.batch_size > self.n_steps * self.n_envs:
            raise ValueError("batch_size must not exceed n_steps*n_envs")
        if self.channels < 8 or self.channels % 8:
            raise ValueError("channels must be >=8 and divisible by 8")
        for name in ("learning_rate", "gamma", "gae_lambda", "clip_range", "ent_coef"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not (
            self.learning_rate > 0 and 0 < self.gamma <= 1
            and 0 < self.gae_lambda <= 1 and 0 < self.clip_range < 1
            and self.ent_coef >= 0
        ):
            raise ValueError("Invalid PPO optimization parameters")


def _output_path(output: str | Path) -> Path:
    path = Path(output)
    if path.suffix != ".zip":
        raise ValueError("PPO checkpoint filename must end with .zip")
    return path


def sidecar_path(checkpoint: str | Path) -> Path:
    """Metadata alongside the SB3 checkpoint, e.g. ppo.zip -> ppo.meta.json."""
    path = _output_path(checkpoint)
    return path.with_suffix(".meta.json")


def load_ppo_metadata(checkpoint: str | Path) -> dict[str, Any]:
    with sidecar_path(checkpoint).open(encoding="utf-8") as file:
        metadata = json.load(file)
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
        raise ValueError("Unsupported PPO checkpoint metadata")
    return metadata


def train_ppo(
    *,
    output: str | Path,
    env_config: GridConfig,
    settings: PPOSettings,
) -> dict[str, Any]:
    """Train PPO from scratch. No truth-map data enters policy observations."""
    checkpoint = _output_path(output)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    def make_env() -> GridWorldEnv:
        return GridWorldEnv(GridConfig(**asdict(env_config)))

    vec_env = DummyVecEnv([make_env for _ in range(settings.n_envs)])
    try:
        vec_env.seed(settings.seed)
        model = PPO(
            "CnnPolicy",
            vec_env,
            policy_kwargs={
                "features_extractor_class": GridFeaturesExtractor,
                "features_extractor_kwargs": {"channels": settings.channels, "features_dim": 128},
                "normalize_images": False,
                "net_arch": {"pi": [64], "vf": [64]},
            },
            learning_rate=settings.learning_rate,
            n_steps=settings.n_steps,
            batch_size=settings.batch_size,
            n_epochs=settings.n_epochs,
            gamma=settings.gamma,
            gae_lambda=settings.gae_lambda,
            ent_coef=settings.ent_coef,
            clip_range=settings.clip_range,
            seed=settings.seed,
            device=settings.device,
            verbose=0,
        )
        model.learn(total_timesteps=settings.timesteps, progress_bar=False)
        model.save(str(checkpoint))
        metadata = {
            "schema_version": 1,
            "algorithm": "stable-baselines3 PPO CnnPolicy",
            "observation_shape": list(vec_env.observation_space.shape),
            "action_count": int(vec_env.action_space.n),
            "environment": asdict(env_config),
            "training": asdict(settings),
            "actual_timesteps": int(model.num_timesteps),
            "versions": {
                "torch": torch.__version__,
                "gymnasium": gymnasium.__version__,
                "stable_baselines3": stable_baselines3.__version__,
            },
        }
        _atomic_write_json(sidecar_path(checkpoint), metadata)
        return metadata
    finally:
        vec_env.close()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write metadata without leaving a partial file on interruption."""
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix=f".{path.name}.",
        suffix=".tmp", dir=path.parent, delete=False,
    ) as file:
        temp = Path(file.name)
        try:
            json.dump(payload, file, indent=2, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    try:
        temp.replace(path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


class PPOPolicyAdapter:
    """Bridge Stable-Baselines3 `predict` to the existing `act(observation)` API."""

    def __init__(self, model: PPO, *, device: str) -> None:
        self.model = model
        self.device = device

    def synchronize(self) -> None:
        if next(self.model.policy.parameters()).device.type == "cuda":
            torch.cuda.synchronize(next(self.model.policy.parameters()).device)

    def act(self, observation: np.ndarray) -> int:
        action, _ = self.model.predict(observation, deterministic=True)
        return int(np.asarray(action).item())


def load_ppo_policy(checkpoint: str | Path, *, device: str = "cpu") -> PPOPolicyAdapter:
    path = _output_path(checkpoint)
    # Only load checkpoints from trusted authors: SB3 archives may include pickled objects.
    model = PPO.load(str(path), device=device)
    return PPOPolicyAdapter(model, device=device)
