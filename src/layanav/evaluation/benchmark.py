"""Paired-seed navigation benchmark for A*, APF, CNN/BC and PPO.

Every method receives exactly the same six-channel policy observation,
sensor mode, map family, timeout and collision semantics. Ground truth is
accessed ONLY when explicitly requested as a privileged oracle *diagnostic*.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from itertools import pairwise
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from layanav.envs import GridConfig, GridWorldEnv, action_count
from layanav.planners import APFConfig, ArtificialPotentialFieldPolicy, OnlineAStarPolicy

FAMILIES = ("sparse", "standard", "dense")
FAMILY_LAYOUTS = {
    "sparse": (8, 2),
    "standard": (24, 5),
    "dense": (64, 10),
}
REPORT_FIELDS = (
    "family", "method", "seed", "success", "steps", "moves", "collisions",
    "total_reward", "mean_inference_ms", "distance_traveled",
    "oracle_shortest_moves", "oracle_shortest_distance", "oracle_efficiency",
)


class NavigationPolicy(Protocol):
    def act(self, observation: np.ndarray) -> int: ...


class BCPolicyAdapter:
    def __init__(self, model: Any) -> None:
        self.model = model

    def synchronize(self) -> None:
        import torch

        device = next(self.model.parameters()).device
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def act(self, observation: np.ndarray) -> int:
        return self.model.act(observation)


def make_family_config(
    name: str,
    *,
    size: int,
    radius: int,
    sensor: str,
    max_steps: int,
    movement_mode: int = 4,
) -> GridConfig:
    """Same observation shape; family controls obstacle density and wall segments."""
    if name not in FAMILY_LAYOUTS:
        raise ValueError(f"Unknown map family: {name!r}")
    hidden, walls = FAMILY_LAYOUTS[name]
    scaled_hidden = round(hidden * (size / 32) ** 2)
    scaled_walls = round(walls * size / 32)
    return GridConfig(
        size=size,
        observation_radius=radius,
        sensor_mode=sensor,
        hidden_obstacle_count=scaled_hidden,
        known_wall_segments=scaled_walls,
        max_steps=max_steps,
        movement_mode=movement_mode,
    )


def _synchronize(policy: NavigationPolicy) -> None:
    sync = getattr(policy, "synchronize", None)
    if sync is not None:
        sync()


def run_episode(
    env: GridWorldEnv,
    policy: NavigationPolicy,
    *,
    seed: int,
    oracle_diagnostic: bool = False,
) -> dict[str, Any]:
    """Evaluate one deterministic episode. No privileged data enters policy.act."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    obs, info = env.reset(seed=seed)
    previous = info["position"]
    oracle_moves: int | None = None
    oracle_distance: float | None = None
    if oracle_diagnostic:
        # Intentional privilege boundary: reporting-only lower bound.
        truth = env.debug_snapshot()
        blocked = truth["static_map"] | truth["hidden_obstacles"]
        path = OnlineAStarPolicy._find_path(
            blocked, truth["robot"], truth["goal"], env.config.movement_mode
        )
        oracle_moves = len(path) - 1 if path else None
        oracle_distance = (
            sum(
                math.hypot(x1 - x0, y1 - y0)
                for (x0, y0), (x1, y1) in pairwise(path)
            )
            if path else None
        )

    total_reward = 0.0
    collisions = 0
    moves = 0
    latency_seconds = 0.0
    while True:
        _synchronize(policy)
        start = time.perf_counter()
        action = policy.act(obs)
        _synchronize(policy)
        latency_seconds += time.perf_counter() - start
        if type(action) is not int or not env.action_space.contains(action):
            raise ValueError(
                f"Policy action {action!r} invalid for {env.config.movement_mode}-way motion"
            )
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        collisions += int(info["collision"])
        if info["position"] != previous:
            moves += 1
        previous = info["position"]
        if terminated or truncated:
            break
    success = bool(info["success"])
    efficiency = (
        oracle_distance / info["distance_traveled"]
        if success and oracle_distance is not None and info["distance_traveled"] > 0
        else None
    )
    return {
        "seed": seed,
        "success": success,
        "steps": int(info["steps"]),
        "moves": moves,
        "collisions": collisions,
        "total_reward": total_reward,
        "mean_inference_ms": 1000 * latency_seconds / info["steps"],
        "oracle_shortest_moves": oracle_moves,
        "oracle_shortest_distance": oracle_distance,
        "distance_traveled": info["distance_traveled"],
        "oracle_efficiency": efficiency,
    }


def bootstrap_mean_ci(
    values: np.ndarray, *, confidence: float = 0.95, replicates: int = 2000, seed: int = 42
) -> dict[str, float]:
    """Percentile bootstrap over independent *map seeds* (not timesteps)."""
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if data.size == 0:
        raise ValueError("Cannot bootstrap an empty sample")
    if not 0 < confidence < 1 or replicates < 100:
        raise ValueError("Invalid confidence or replicate count")
    mean = float(data.mean())
    if data.size == 1:
        return {"mean": mean, "ci_low": mean, "ci_high": mean, "n": 1}
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for i in range(replicates):
        estimates[i] = float(rng.choice(data, size=len(data), replace=True).mean())
    half = (1 - confidence) / 2
    low, high = np.quantile(estimates, [half, 1 - half])
    return {
        "mean": mean, "ci_low": float(low), "ci_high": float(high), "n": len(data),
    }


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _create_policy(
    method: str,
    checkpoint: str | Path | None,
    device: str,
    apf_config: APFConfig | None = None,
    movement_mode: int = 4,
) -> NavigationPolicy:
    if method == "astar":
        return OnlineAStarPolicy(movement_mode)
    if method == "apf":
        return ArtificialPotentialFieldPolicy(apf_config, movement_mode)
    if checkpoint is None:
        raise ValueError(f"{method} requires a checkpoint")
    if method == "bc":
        from layanav.training.behavior_cloning import load_policy

        return BCPolicyAdapter(load_policy(checkpoint, device=device))
    if method == "ppo":
        from layanav.training.ppo import load_ppo_policy

        return load_ppo_policy(checkpoint, device=device)
    if method == "laya":
        from layanav.training.laya_bc import load_laya_adapter

        model, _ = load_laya_adapter(checkpoint, device=device)
        return BCPolicyAdapter(model)
    raise ValueError(f"Unknown benchmark method: {method}")


def benchmark(
    *,
    methods: tuple[str, ...],
    families: tuple[str, ...],
    episodes: int,
    eval_seed: int,
    size: int,
    radius: int,
    sensor: str,
    max_steps: int,
    output_dir: str | Path,
    movement_mode: int = 4,
    bc_checkpoint: str | Path | None = None,
    ppo_checkpoint: str | Path | None = None,
    laya_checkpoint: str | Path | None = None,
    apf_config: APFConfig | None = None,
    device: str = "cpu",
    oracle_diagnostic: bool = False,
    bootstrap_seed: int = 2026,
    bootstrap_replicates: int = 2000,
) -> dict[str, Any]:
    """Generate per-episode CSV plus JSON summary over paired held-out map seeds."""
    if episodes < 1 or eval_seed < 0:
        raise ValueError("episodes must be positive and eval_seed nonnegative")
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("methods must be nonempty and unique")
    if not families or len(set(families)) != len(families):
        raise ValueError("families must be nonempty and unique")
    if set(methods) - {"astar", "apf", "bc", "ppo", "laya"}:
        raise ValueError("Unknown benchmark method")
    if set(families) - set(FAMILIES):
        raise ValueError("Unknown map family")
    checkpoints = {"bc": bc_checkpoint, "ppo": ppo_checkpoint, "laya": laya_checkpoint}
    policies = {
        method: _create_policy(
            method, checkpoints.get(method), device, apf_config, movement_mode
        )
        for method in methods
    }
    configurations = {
        family: make_family_config(
            family, size=size, radius=radius, sensor=sensor,
            max_steps=max_steps, movement_mode=movement_mode,
        )
        for family in families
    }
    # Fail on input-shape mismatch before evaluating any episodes.
    for method, policy in policies.items():
        expected_actions = action_count(movement_mode)
        if method in ("bc", "laya") and policy.model.action_count != expected_actions:
            raise ValueError(
                f"{method} checkpoint expects {policy.model.action_count} actions; "
                f"benchmark uses {expected_actions}"
            )
        if method == "ppo" and policy.model.action_space.n != expected_actions:
            raise ValueError(
                f"PPO checkpoint expects {policy.model.action_space.n} actions; "
                f"benchmark uses {expected_actions}"
            )
        if method == "ppo":
            shape = tuple(policy.model.observation_space.shape)
            if shape != (6, size, size):
                raise ValueError(f"PPO model expects {shape}, benchmark uses (6,{size},{size})")

    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for family, config in configurations.items():
        env = GridWorldEnv(config)
        try:
            for method, policy in policies.items():
                episode_rows: list[dict[str, Any]] = []
                for offset in range(episodes):
                    row = run_episode(
                        env, policy, seed=eval_seed + offset,
                        oracle_diagnostic=oracle_diagnostic,
                    )
                    row = {"family": family, "method": method, **row}
                    episode_rows.append(row)
                    rows.append(row)
                summary: dict[str, Any] = {"episodes": episodes}
                for metric in (
                    "success", "steps", "moves", "collisions", "total_reward",
                    "mean_inference_ms", "distance_traveled", "oracle_efficiency",
                ):
                    series = np.asarray([
                        float(row[metric]) if row[metric] is not None else np.nan
                        for row in episode_rows
                    ], dtype=np.float64)
                    valid = series[np.isfinite(series)]
                    summary[metric] = (
                        bootstrap_mean_ci(
                            valid, seed=bootstrap_seed, replicates=bootstrap_replicates
                        )
                        if valid.size else None
                    )
                summaries[f"{family}/{method}"] = summary
        finally:
            env.close()

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "episodes.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    checkpoint_details: dict[str, Any] = {}
    for method in ("bc", "ppo", "laya"):
        checkpoint = checkpoints[method]
        if method in methods and checkpoint is not None:
            checkpoint_details[method] = {
                "path": str(checkpoint),
                "sha256": _sha256(checkpoint),
            }
            if method == "ppo":
                from layanav.training.ppo import load_ppo_metadata

                checkpoint_details[method]["training"] = load_ppo_metadata(checkpoint)
            if method == "laya":
                import torch

                saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
                checkpoint_details[method]["training"] = {
                    "source": saved["source"],
                    "epoch": saved["epoch"],
                    "training_seed": saved["training_seed"],
                    "best_val_loss": saved["best_val_loss"],
                }

    report: dict[str, Any] = {
        "schema_version": 1,
        "protocol": {
            "paired_map_seeds": True,
            "eval_seed_start": eval_seed,
            "eval_seed_end_inclusive": eval_seed + episodes - 1,
            "sensor": sensor,
            "movement_mode": movement_mode,
            "diagonal_cost": "sqrt(2) for diagonal, 1 for cardinal",
            "corner_cutting": "forbidden",
            "oracle_is_privileged_diagnostic_only": oracle_diagnostic,
            "confidence_interval": "95% percentile bootstrap across map seeds",
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": bootstrap_seed,
            "policy_actions": "deterministic; argmax for neural, rule-based for classical",
            "note": (
                "Identical evaluation information does not imply equal training budgets. "
                "BC is imitation-learning; PPO is environment-interaction training."
            ),
        },
        "families": {name: asdict(config) for name, config in configurations.items()},
        "methods": list(methods),
        "apf_config": asdict(apf_config or APFConfig()) if "apf" in methods else None,
        "checkpoints": checkpoint_details,
        "summary": summaries,
    }
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
