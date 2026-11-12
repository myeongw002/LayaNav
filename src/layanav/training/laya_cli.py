"""Frozen-Laya Spatial Adapter behavior cloning.

The first run downloads the pretrained Laya checkpoint (~808 MB English).
On Ubuntu 24.04 / RTX 5070, use a compatible CUDA PyTorch installation.
"""

from __future__ import annotations

import argparse

import torch

from layanav.models.laya_spatial import (
    SpatialAdapterConfig,
    SpatialLayaPolicy,
    load_laya_backbone,
)
from layanav.training.laya_bc import train_laya_bc


def main() -> None:
    parser = argparse.ArgumentParser(description="LayaNav Laya-spatial behavior cloning")
    parser.add_argument("--train", required=True, help="Belief-only teacher train .npz")
    parser.add_argument("--validation", required=True, help="Disjoint validation .npz")
    parser.add_argument("--output", required=True, help="Adapter-only .pt checkpoint")
    parser.add_argument("--model-id", default="convaiinnovations/laya")
    parser.add_argument(
        "--revision", default=None,
        help="Exact Hugging Face checkpoint revision recommended for reproducibility",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--feature-channels", type=int, default=64)
    parser.add_argument("--token-grid", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    backbone, source = load_laya_backbone(
        model_id=args.model_id, device=args.device, revision=args.revision,
    )
    # Seed the trainable spatial/decision heads *before* their initialization.
    torch.manual_seed(args.seed)
    model = SpatialLayaPolicy(
        backbone,
        SpatialAdapterConfig(
            feature_channels=args.feature_channels,
            token_grid=args.token_grid,
            freeze_backbone=True,
        ),
    ).to(args.device)
    train_laya_bc(
        model=model,
        train_data=args.train,
        validation_data=args.validation,
        output=args.output,
        source=source,
        epochs=args.epochs,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
