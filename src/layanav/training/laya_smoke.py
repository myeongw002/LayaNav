"""Real-checkpoint Laya GPU compatibility and one-step BC training probe.

This script is intended to be run on the research machine. In CI we test
the same path with a tiny stand-in encoder; real pretrained weights and
the user's GPU are deliberately not claimed to have been exercised.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from layanav.envs import GridConfig, GridWorldEnv
from layanav.models.laya_spatial import (
    SpatialAdapterConfig,
    SpatialLayaPolicy,
    load_laya_backbone,
)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def probe_device(device: str) -> dict[str, Any]:
    """Test a real CUDA kernel before downloading a large checkpoint.

    CUDA availability alone is insufficient on new architectures: a CPU-
    built or older CUDA wheel can report a visible GPU but fail on kernels.
    """
    requested = torch.device(device)
    if requested.type not in ("cpu", "cuda"):
        raise ValueError("Laya smoke supports only CPU or CUDA")
    result: dict[str, Any] = {
        "requested_device": str(requested),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "transformers_version": _package_version("transformers"),
        "laya_version": _package_version("laya"),
        "cuda_available": bool(torch.cuda.is_available()),
        "kernel_test": "not_applicable",
        "gpu_name": None,
        "compute_capability": None,
        "total_device_gib": None,
        "torch_compiled_cuda_arches": [],
    }
    if requested.type == "cpu":
        return result
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but unavailable. Check the NVIDIA driver, PyTorch "
            "CUDA build and the active Python environment."
        )
    index = requested.index if requested.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    result.update({
        "gpu_name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "total_device_gib": properties.total_memory / 1024**3,
        "torch_compiled_cuda_arches": torch.cuda.get_arch_list(),
    })
    # Actual dispatch detects unsupported SM architectures, missing driver,
    # insufficient runtime version and CPU-only wheels before model download.
    example = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=requested)
    product = example @ example
    torch.cuda.synchronize(requested)
    if not torch.allclose(
        product, torch.tensor([[7.0, 10.0], [15.0, 22.0]], device=requested)
    ):
        raise RuntimeError("CUDA matrix multiplication kernel returned incorrect values")
    result["kernel_test"] = "passed"
    return result


def _sync(device: str) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


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
    learning_rate: float = 1e-4,
) -> dict[str, Any]:
    """Load real weights, do one frozen-encoder BC step, then evaluate logits.

    The loss target is synthetic and does not evaluate navigation ability.
    Actor, tokenizer, input and output correctness are checked independently.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    environment = probe_device(device)
    is_cuda = torch.device(device).type == "cuda"
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    load_start = time.perf_counter()
    backbone, source = load_laya_backbone(
        model_id=model_id, device=device, revision=revision
    )
    _sync(device)
    load_seconds = time.perf_counter() - load_start
    load_memory = (
        torch.cuda.max_memory_allocated(device) / 1024**3 if is_cuda else None
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
    if model.backbone.training or any(
        parameter.requires_grad for parameter in model.backbone.parameters()
    ):
        raise RuntimeError("Laya backbone was not frozen in eval mode")

    env = GridWorldEnv(GridConfig(
        size=grid_size, observation_radius=5, sensor_mode="raycast",
        movement_mode=movement_mode, hidden_obstacle_count=0,
        known_wall_segments=0,
    ))
    try:
        public_observations = [env.reset(seed=seed + i)[0] for i in range(batch_size)]
    finally:
        env.close()

    input_tensor = torch.as_tensor(
        np.stack(public_observations), device=device, dtype=torch.float32
    )
    targets = torch.zeros(batch_size, dtype=torch.long, device=device)
    optimizable = [
        parameter for name, parameter in model.named_parameters()
        if parameter.requires_grad and (
            name.startswith("tokenizer.") or name.startswith("actor.")
        )
    ]
    if not optimizable:
        raise RuntimeError("No trainable spatial adapter/actor parameters")
    optimizer = torch.optim.AdamW(optimizable, lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    start = time.perf_counter()
    logits, critic_values = model(input_tensor)
    if logits.shape != (batch_size, model.action_count):
        raise RuntimeError(f"Unexpected action logits shape: {tuple(logits.shape)}")
    if critic_values.shape != (batch_size,):
        raise RuntimeError(f"Unexpected critic shape: {tuple(critic_values.shape)}")
    if not torch.isfinite(logits).all() or not torch.isfinite(critic_values).all():
        raise RuntimeError("Laya forward returned nonfinite actor/critic outputs")
    loss = loss_fn(logits, targets)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("Laya BC training loss is not finite")
    _sync(device)
    forward_seconds = time.perf_counter() - start

    start = time.perf_counter()
    loss.backward()
    _sync(device)
    backward_seconds = time.perf_counter() - start

    def gradient_l1(module: nn.Module, name: str) -> float:
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        if not gradients or any(not torch.isfinite(g).all() for g in gradients):
            raise RuntimeError(f"{name} gradients are missing or nonfinite")
        total = sum(float(g.detach().abs().sum().cpu()) for g in gradients)
        if not math.isfinite(total) or total <= 0:
            raise RuntimeError(f"{name} gradients are zero or nonfinite")
        return total

    tokenizer_grad = gradient_l1(model.tokenizer, "Tokenizer")
    actor_grad = gradient_l1(model.actor, "Actor")
    if any(p.grad is not None for p in model.backbone.parameters()):
        raise RuntimeError("Frozen Laya encoder accumulated gradients")

    actor_before = model.actor[-1].weight.detach().clone()
    grad_norm = torch.nn.utils.clip_grad_norm_(optimizable, max_norm=1.0)
    if not torch.isfinite(grad_norm):
        raise RuntimeError("Nonfinite global adapter gradient norm")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    _sync(device)
    if torch.equal(actor_before, model.actor[-1].weight.detach()):
        raise RuntimeError("Actor weights did not change after the optimizer step")

    peak_step = (
        {
            "step_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
            "step_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
        } if is_cuda else {
            "step_peak_allocated_gib": None,
            "step_peak_reserved_gib": None,
        }
    )

    model.eval()
    _sync(device)
    start = time.perf_counter()
    with torch.inference_mode():
        updated_logits, updated_values = model(input_tensor)
    _sync(device)
    inference_seconds = time.perf_counter() - start
    if not torch.isfinite(updated_logits).all() or not torch.isfinite(updated_values).all():
        raise RuntimeError("Laya post-update inference produced nonfinite outputs")
    chosen_action = model.act(public_observations[0])
    if not 0 <= chosen_action < model.action_count:
        raise RuntimeError("Laya actor produced an invalid action ID")

    return {
        "status": "passed",
        "backbone": source,
        "runtime": environment,
        "device": device,
        "gpu_name": environment["gpu_name"],
        "batch_size": batch_size,
        "grid_size": grid_size,
        "movement_mode": movement_mode,
        "spatial_tokens": token_grid**2 + 1,
        "actor_logits_shape": list(logits.shape),
        "critic_values_shape": list(critic_values.shape),
        "loss": float(loss.detach().cpu()),
        "adapter_gradient_l1": tokenizer_grad,
        "actor_gradient_l1": actor_grad,
        "global_grad_norm_before_clipping": float(grad_norm.detach().cpu()),
        "optimizer_step_changed_actor": True,
        "sample_action_id": chosen_action,
        "pretrained_load_seconds": load_seconds,
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "forward_backward_seconds": forward_seconds + backward_seconds,
        "post_update_inference_seconds": inference_seconds,
        "gpu_memory": {
            "load_peak_allocated_gib": load_memory,
            **peak_step,
        },
        "note": (
            "Pretrained loader interface, frozen-encoder gradient flow and a "
            "synthetic BC optimizer step only. Not navigation accuracy, "
            "pretraining benefit or a trained RL critic."
        ),
    }


def _write_report(path: str, report: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrained Laya GPU verification")
    parser.add_argument("--model-id", default="convaiinnovations/laya")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--movement-mode", type=int, choices=[4, 8], default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--token-grid", type=int, default=8)
    parser.add_argument("--feature-channels", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--preflight-only", action="store_true",
        help="Check CUDA wheel, GPU and a real CUDA kernel without loading Laya weights",
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional JSON report path, written also when validation raises an exception",
    )
    args = parser.parse_args()
    try:
        if args.preflight_only:
            report: dict[str, Any] = {
                "status": "preflight_passed",
                "runtime": probe_device(args.device),
            }
        else:
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
                learning_rate=args.learning_rate,
            )
    except Exception as exc:
        if args.output:
            _write_report(args.output, {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "requested_device": args.device,
                "torch_version": torch.__version__,
                "torch_cuda_runtime": torch.version.cuda,
            })
        print(
            f"Laya validation failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise
    if args.output:
        _write_report(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
