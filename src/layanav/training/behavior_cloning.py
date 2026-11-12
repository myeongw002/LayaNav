"""Behavior cloning for the CNN navigation baseline.

The learner sees only the six-channel belief observation. Train/validation
datasets must come from disjoint map seeds.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from layanav.models.cnn_policy import CNNPolicy
from layanav.training.dataset import load_dataset


def _evaluate_loss(
    model: CNNPolicy,
    batches: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    with torch.inference_mode():
        for features, labels in batches:
            features, labels = features.to(device), labels.to(device)
            logits = model(features)
            loss = criterion(logits, labels)
            loss_sum += float(loss) * len(labels)
            correct += int((logits.argmax(dim=-1) == labels).sum().item())
            total += len(labels)
    return loss_sum / total, correct / total


def train_behavior_cloning(
    *,
    train_data: str | Path,
    validation_data: str | Path,
    output_checkpoint: str | Path,
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    channels: int = 32,
    device: str = "cpu",
    seed: int = 42,
) -> list[dict[str, float]]:
    """Fit a categorical navigation policy and save the best validation checkpoint."""
    if epochs < 1 or batch_size < 1 or learning_rate <= 0 or not np.isfinite(learning_rate):
        raise ValueError("Invalid training hyperparameters")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    train_obs, train_actions, train_seeds = load_dataset(train_data)
    val_obs, val_actions, val_seeds = load_dataset(validation_data)
    if train_obs.shape[1:] != val_obs.shape[1:]:
        raise ValueError("Train and validation observation shapes differ")
    if np.intersect1d(np.unique(train_seeds), np.unique(val_seeds)).size:
        raise ValueError("Train and validation must use disjoint map seeds")

    torch.manual_seed(seed)
    np.random.seed(seed)
    device_obj = torch.device(device)
    model = CNNPolicy(channels=channels).to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    train_set = TensorDataset(
        torch.from_numpy(train_obs), torch.from_numpy(train_actions)
    )
    val_set = TensorDataset(
        torch.from_numpy(val_obs), torch.from_numpy(val_actions)
    )
    train_batches = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed), num_workers=0
    )
    val_batches = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    output = Path(output_checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        samples = 0
        for features, labels in train_batches:
            features, labels = features.to(device_obj), labels.to(device_obj)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(labels)
            samples += len(labels)

        val_loss, val_accuracy = _evaluate_loss(model, val_batches, criterion, device_obj)
        record = {
            "epoch": float(epoch),
            "train_loss": total_loss / samples,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "channels": channels,
                    "seed": seed,
                    "epoch": epoch,
                    "observation_shape": list(train_obs.shape[1:]),
                    "best_val_loss": best_val_loss,
                },
                output,
            )
    output.with_suffix(".history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    return history


def load_policy(checkpoint: str | Path, device: str = "cpu") -> CNNPolicy:
    """Use PyTorch's weights-only deserialization for checkpoints."""
    contents: dict[str, Any] = torch.load(
        checkpoint, map_location="cpu", weights_only=True
    )
    if not isinstance(contents, dict) or "model" not in contents:
        raise ValueError("Malformed CNN checkpoint")
    model = CNNPolicy(channels=int(contents["channels"]))
    model.load_state_dict(contents["model"], strict=True)
    model.to(torch.device(device))
    model.eval()
    return model
