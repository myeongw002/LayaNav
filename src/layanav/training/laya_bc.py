"""Frozen-Laya spatial-adapter behavior cloning without saving backbone weights.

The pretrained Laya encoder is selected by model ID and optional exact Hub
revision. Training updates the spatial tokenizer and actor head only.
The critic is initialized but remains untrained until RL fine-tuning.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from layanav.models.laya_spatial import (
    SpatialAdapterConfig,
    SpatialLayaPolicy,
    load_laya_backbone,
)
from layanav.training.dataset import load_dataset


def _validate_seed_split(
    train_seeds: np.ndarray, validation_seeds: np.ndarray
) -> None:
    if np.intersect1d(np.unique(train_seeds), np.unique(validation_seeds)).size:
        raise ValueError("Train and validation datasets must use disjoint map seeds")


def train_laya_bc(
    *,
    model: SpatialLayaPolicy,
    train_data: str | Path,
    validation_data: str | Path,
    output: str | Path,
    source: dict[str, Any],
    epochs: int = 5,
    batch_size: int = 2,
    accumulation_steps: int = 4,
    learning_rate: float = 1e-4,
    seed: int = 42,
) -> list[dict[str, float]]:
    """Train on expert actions with a frozen encoder; checkpoint only adapter/actor."""
    if not model.config.freeze_backbone:
        raise ValueError("Adapter-only BC requires freeze_backbone=True")
    if any(type(value) is not int or value < 1 for value in (
        epochs, batch_size, accumulation_steps
    )):
        raise ValueError("epochs, batch_size and accumulation_steps must be positive")
    if seed < 0 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Invalid training seed or learning_rate")
    if not source.get("model_id"):
        raise ValueError("source must identify the pretrained Laya backbone")

    train_obs, train_act, train_seeds = load_dataset(train_data)
    val_obs, val_act, val_seeds = load_dataset(validation_data)
    _validate_seed_split(train_seeds, val_seeds)
    if train_obs.shape[1:] != val_obs.shape[1:]:
        raise ValueError("Train and validation grid shapes differ")
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    train_batches = DataLoader(
        TensorDataset(torch.from_numpy(train_obs), torch.from_numpy(train_act)),
        batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed), num_workers=0,
    )
    val_batches = DataLoader(
        TensorDataset(torch.from_numpy(val_obs), torch.from_numpy(val_act)),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )
    parameters = [
        parameter for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("critic.")
    ]
    if not parameters:
        raise ValueError("No trainable spatial adapter or actor parameters")
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    checkpoint = Path(output)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        seen = 0
        number_batches = len(train_batches)
        for index, (obs, actions) in enumerate(train_batches):
            obs, actions = obs.to(device), actions.to(device)
            logits, _ = model(obs)
            loss = criterion(logits, actions)
            group_size = min(
                accumulation_steps,
                number_batches - (index // accumulation_steps) * accumulation_steps,
            )
            (loss / group_size).backward()
            running_loss += float(loss.detach()) * len(actions)
            seen += len(actions)
            if (index + 1) % accumulation_steps == 0 or index + 1 == number_batches:
                torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        model.eval()
        val_loss, val_correct, val_n = 0.0, 0, 0
        with torch.inference_mode():
            for obs, actions in val_batches:
                obs, actions = obs.to(device), actions.to(device)
                logits, _ = model(obs)
                loss = criterion(logits, actions)
                val_loss += float(loss) * len(actions)
                val_correct += int((logits.argmax(dim=-1) == actions).sum().item())
                val_n += len(actions)
        val_loss /= val_n
        row = {
            "epoch": float(epoch),
            "train_loss": running_loss / seen,
            "val_loss": val_loss,
            "val_accuracy": val_correct / val_n,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if val_loss < best_loss:
            best_loss = val_loss
            # The checkpoint does not contain privileged map truth or base
            # encoder weights. The exact upstream revision is tracked instead.
            torch.save(
                {
                    "schema_version": 1,
                    "adapter_state": model.adapter_state_dict(),
                    "adapter_config": asdict(model.config),
                    "source": dict(source),
                    "grid_shape": list(train_obs.shape[1:]),
                    "training_seed": seed,
                    "epoch": epoch,
                    "best_val_loss": best_loss,
                },
                checkpoint,
            )
    checkpoint.with_suffix(".history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    return history


def load_laya_adapter(
    checkpoint: str | Path,
    *,
    device: str = "cpu",
) -> tuple[SpatialLayaPolicy, dict[str, Any]]:
    """Reconstruct from the exact pretrained backbone and adapter-only weights."""
    saved: dict[str, Any] = torch.load(
        checkpoint, map_location="cpu", weights_only=True
    )
    if not isinstance(saved, dict) or saved.get("schema_version") != 1:
        raise ValueError("Unsupported spatial Laya checkpoint")
    source = saved["source"]
    if not isinstance(source, dict) or not source.get("model_id"):
        raise ValueError("Checkpoint does not identify the Laya backbone")
    revision = source.get("resolved_revision") or source.get("requested_revision")
    backbone, loaded_source = load_laya_backbone(
        model_id=source["model_id"], revision=revision, device=device,
    )
    if revision and loaded_source.get("resolved_revision") and (
        loaded_source["resolved_revision"] != revision
    ):
        raise ValueError("The loaded pretrained Laya revision differs from checkpoint")
    config = SpatialAdapterConfig(**saved["adapter_config"])
    if not config.freeze_backbone:
        raise ValueError("Adapter checkpoint requires frozen backbone")
    model = SpatialLayaPolicy(backbone, config).to(device)
    model.load_adapter_state_dict(saved["adapter_state"])
    model.eval()
    return model, saved
