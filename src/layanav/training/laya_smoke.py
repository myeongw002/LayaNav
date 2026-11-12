"""One-step real pretrained-Laya forward/backward and peak-VRAM probe.

Run on the target machine after installing a compatible PyTorch CUDA wheel
and `laya==0.4.31`. Never reports training success or a navigation score.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from torch import nn

from layanav.envs import GridConfig, GridWorldEnv
from layanav.models.laya_spatial import (
    SpatialAdapterConfig,
    SpatialLayaPolicy,
    load_laya_backbone,
)


def smoke(
    *,
    model_id: str,
    revision: str | None,
    device: str,
    grid_size: int,
    movement_mode: int = 4,
    batch_size: int = 1,
    token_grid: int,
    feature_channels: int,
    seed: int,
) -> dict[str, object]:
    """Backpropagate one policy action loss through a frozen real encoder."""
    if batch_size < 1 or seed < 0:
        raise ValueError("batch_size must be positive and seed nonnegative")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; install a compatible GPU PyTorch wheel")
    cuda = device.startswith("cuda")
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    backbone, source = load_laya_backbone(
        model_id=model_id, device=device, revision=revision
    )
    torch.manual_seed(seed)
    model = SpatialLayaPolicy(
        backbone,
        SpatialAdapterConfig(
            feature_channels=feature_channels,
            token_grid=token_grid,
            freeze_backbone=True,
        ),
        movement_mode=movement_mode,
    ).to(device)
    model.train()

    env = GridWorldEnv(GridConfig(
        size=grid_size, observation_radius=5, sensor_mode="raycast",
        movement_mode=movement_mode,
        hidden_obstacle_count=0, known_wall_segments=0
    ))
    try:
        public_observations = [
            env.reset(seed=seed + i)[0] for i in range(batch_size)
        ]
    finally:
        env.close()
    input_tensor = torch.from_numpy(np.stack(public_observations)).to(device)
    targets = torch.zeros(batch_size, dtype=torch.long, device=device)
    if cuda:
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    logits, critic_values = model(input_tensor)
    loss = nn.functional.cross_entropy(logits, targets)
    loss.backward()
    if cuda:
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    adapter_grad_sum = sum(
        float(parameter.grad.detach().abs().sum().cpu())
        for parameter in model.tokenizer.parameters()
        if parameter.grad is not None
    )
    if not np.isfinite(adapter_grad_sum) or adapter_grad_sum <= 0:
        raise RuntimeError("Spatial adapter did not receive finite nonzero gradients")
    if any(parameter.grad is not None for parameter in model.backbone.parameters()):
        raise RuntimeError("Frozen Laya encoder unexpectedly accumulated gradients")
    memory = {
        "allocated_peak_gib": (
            torch.cuda.max_memory_allocated(device) / 1024**3 if cuda else None
        ),
        "reserved_peak_gib": (
            torch.cuda.max_memory_reserved(device) / 1024**3 if cuda else None
        ),
    }
    return {
        "backbone": source,
        "device": device,
        "gpu_name": torch.cuda.get_device_name(device) if cuda else None,
        "batch_size": batch_size,
        "grid_size": grid_size,
        "movement_mode": movement_mode,
        "spatial_tokens": token_grid**2 + 1,
        "actor_logits_shape": list(logits.shape),
        "critic_values_shape": list(critic_values.shape),
        "loss": float(loss.detach().cpu()),
        "adapter_gradient_l1": adapter_grad_sum,
        "forward_backward_seconds": elapsed,
        "gpu_memory": memory,
        "note": (
            "This checks interface/gradients/VRAM only; it is not model fine-tuning "
            "or evidence of navigation accuracy."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrained Laya spatial GPU smoke")
    parser.add_argument("--model-id", default="convaiinnovations/laya")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--movement-mode", type=int, choices=[4, 8], default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--token-grid", type=int, default=8)
    parser.add_argument("--feature-channels", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = smoke(
        model_id=args.model_id,
        revision=args.revision,
        device=args.device,
        grid_size=args.grid_size,
        movement_mode=args.movement_mode,
        batch_size=args.batch_size,
        token_grid=args.token_grid,
        feature_channels=args.feature_channels,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
