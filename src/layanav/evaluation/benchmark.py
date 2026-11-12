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
import time
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path
from statistics import NormalDist
from typing import Any, Protocol

import numpy as np

from layanav.envs import MAP_TOPOLOGIES, GridConfig, GridWorldEnv, action_count
from layanav.envs.map_generation import MAP_GENERATOR_VERSION
from layanav.evaluation.difficulty import (
    DIFFICULTY_METRICS,
    map_fingerprint,
    measure_map_difficulty,
)
from layanav.planners import APFConfig, ArtificialPotentialFieldPolicy, OnlineAStarPolicy
from layanav.recording import EpisodeRecorder

# Descriptive stratification only; these are heuristic bins, not calibrated difficulty.
TORTUOSITY_BINS = ("direct", "moderate_detour", "large_detour")
TORTUOSITY_THRESHOLDS = (1.25, 2.0)


def tortuosity_bin(value: float | None) -> str | None:
    if value is None:
        return None
    if not math.isfinite(value) or value < 1 - 1e-9:
        raise ValueError("path_tortuosity must be finite and >= 1")
    if value < TORTUOSITY_THRESHOLDS[0]:
        return "direct"
    if value < TORTUOSITY_THRESHOLDS[1]:
        return "moderate_detour"
    return "large_detour"


FAMILIES = ("sparse", "standard", "dense")
FAMILY_LAYOUTS = {
    "sparse": (8, 2),
    "standard": (24, 5),
    "dense": (64, 10),
}
REPORT_FIELDS = (
    "topology", "family", "method", "seed", "success", "steps", "moves", "collisions",
    "total_reward", "mean_inference_ms", "distance_traveled",
    "oracle_shortest_moves", "oracle_shortest_distance", "oracle_efficiency",
    "termination_reason", "failure_reason", "last_policy_status",
    "stay_steps", "max_consecutive_stay", "revisits",
    "goal_distance_remaining", "failure_trace", "map_sha256",
    "prior_layout_sha256", "spl",
    "tortuosity_bin",
    *DIFFICULTY_METRICS,
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
    topology: str = "random",
    min_route_steps: int | None = None,
    max_route_steps: int | None = None,
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
        topology=topology,
        min_route_steps=min_route_steps,
        max_route_steps=max_route_steps,
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
    failure_trace_path: str | Path | None = None,
    trace_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate one deterministic episode. No privileged data enters policy.act."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    obs, info = env.reset(seed=seed)
    previous = info["position"]
    goal_y, goal_x = np.unravel_index(int(np.argmax(obs[4])), obs[4].shape)
    goal = (int(goal_x), int(goal_y))
    recorder = EpisodeRecorder(env) if failure_trace_path is not None else None
    visited = {previous}
    revisits = 0
    stay_steps = 0
    consecutive_stay = 0
    max_consecutive_stay = 0
    last_policy_status: str | None = None
    oracle_moves: int | None = None
    oracle_distance: float | None = None
    map_sha256: str | None = None
    prior_layout_sha256: str | None = None
    difficulty: dict[str, float | int | None] = {
        metric: None for metric in DIFFICULTY_METRICS
    }
    if oracle_diagnostic:
        # Intentional privilege boundary: reporting-only lower bound.
        truth = env.debug_snapshot()
        blocked = truth["static_map"] | truth["hidden_obstacles"]
        path = OnlineAStarPolicy._find_path(
            blocked, truth["robot"], truth["goal"], env.config.movement_mode
        )
        oracle_moves = len(path) - 1 if path else None
        map_sha256 = map_fingerprint(
            truth["static_map"], truth["hidden_obstacles"],
            truth["robot"], truth["goal"], env.config.movement_mode,
        )
        prior_layout_sha256 = map_fingerprint(
            truth["static_map"], np.zeros_like(truth["hidden_obstacles"]),
            truth["robot"], truth["goal"], env.config.movement_mode,
        )
        if path:
            difficulty = measure_map_difficulty(
                truth["static_map"], truth["hidden_obstacles"],
                truth["robot"], truth["goal"], path,
                env.config.movement_mode, obs,
            )
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
        last_policy_status = getattr(policy, "last_status", None)
        obs, reward, terminated, truncated, info = env.step(action)
        if recorder is not None:
            recorder.record_step(action, obs, reward, terminated, truncated, info)
        total_reward += float(reward)
        collisions += int(info["collision"])
        if action == 4:
            stay_steps += 1
            consecutive_stay += 1
            max_consecutive_stay = max(max_consecutive_stay, consecutive_stay)
        else:
            consecutive_stay = 0
        if info["position"] != previous:
            moves += 1
            if info["position"] in visited:
                revisits += 1
            visited.add(info["position"])
        previous = info["position"]
        if terminated or truncated:
            break
    success = bool(info["success"])
    efficiency = (
        oracle_distance / info["distance_traveled"]
        if success and oracle_distance is not None and info["distance_traveled"] > 0
        else None
    )
    spl = (
        (oracle_distance / max(oracle_distance, info["distance_traveled"]) if success else 0.0)
        if oracle_distance is not None and oracle_distance > 0 else None
    )
    if success:
        failure_reason = None
    elif last_policy_status in ("local_minimum", "blocked"):
        failure_reason = last_policy_status
    elif (
        action == 4
        and isinstance(policy, OnlineAStarPolicy)
        and not policy.last_path
    ):
        failure_reason = "no_known_path"
    elif consecutive_stay >= 3:
        failure_reason = "persistent_stay"
    else:
        failure_reason = "time_limit"
    trace_relative = (
        f"failures/{Path(failure_trace_path).name}"
        if not success and recorder is not None else None
    )
    result = {
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
        "spl": spl,
        "map_sha256": map_sha256,
        "prior_layout_sha256": prior_layout_sha256,
        **difficulty,
        "tortuosity_bin": tortuosity_bin(difficulty["path_tortuosity"]),
        "termination_reason": "goal_reached" if success else "time_limit",
        "failure_reason": failure_reason,
        "last_policy_status": last_policy_status,
        "stay_steps": stay_steps,
        "max_consecutive_stay": max_consecutive_stay,
        "revisits": revisits,
        "goal_distance_remaining": math.dist(previous, goal),
        "failure_trace": trace_relative,
    }
    if recorder is not None and not success:
        recorder.diagnostics = {**(trace_context or {}), **result}
        destination = Path(failure_trace_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        recorder.save(destination)
    return result


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


def wilson_success_ci(
    values: np.ndarray, *, confidence: float = 0.95,
) -> dict[str, float | int]:
    """Two-sided Wilson score interval for independent binary successes."""
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or not data.size or not np.isin(data, (0.0, 1.0)).all():
        raise ValueError("Success observations must be a nonempty binary vector")
    if not 0 < confidence < 1:
        raise ValueError("Invalid confidence")
    n = len(data)
    p = float(data.mean())
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return {
        "mean": p,
        "ci_low": 0.0 if p == 0.0 else max(0.0, center - half),
        "ci_high": 1.0 if p == 1.0 else min(1.0, center + half),
        "n": n,
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
    save_failures: bool = True,
    topologies: tuple[str, ...] = ("random",),
    min_route_steps: int | None = None,
    max_route_steps: int | None = None,
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
    if not topologies or len(set(topologies)) != len(topologies):
        raise ValueError("topologies must be nonempty and unique")
    if set(topologies) - set(MAP_TOPOLOGIES):
        raise ValueError("Unknown map topology")
    checkpoints = {"bc": bc_checkpoint, "ppo": ppo_checkpoint, "laya": laya_checkpoint}
    policies = {
        method: _create_policy(
            method, checkpoints.get(method), device, apf_config, movement_mode
        )
        for method in methods
    }
    configurations = {
        (topology, family): make_family_config(
            family, size=size, radius=radius, sensor=sensor,
            max_steps=max_steps, movement_mode=movement_mode,
            topology=topology,
            min_route_steps=min_route_steps,
            max_route_steps=max_route_steps,
        )
        for topology in topologies
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
    tortuosity_rows: list[dict[str, Any]] = []
    for (topology, family), config in configurations.items():
        env = GridWorldEnv(config)
        try:
            for method, policy in policies.items():
                episode_rows: list[dict[str, Any]] = []
                for offset in range(episodes):
                    seed = eval_seed + offset
                    trace_path = (
                        Path(output_dir) / "failures"
                        / f"{topology}_{family}_{method}_seed_{seed}.json"
                        if save_failures else None
                    )
                    row = run_episode(
                        env, policy, seed=seed,
                        oracle_diagnostic=oracle_diagnostic,
                        failure_trace_path=trace_path,
                        trace_context={
                            "topology": topology, "family": family, "method": method,
                        },
                    )
                    row = {
                        "topology": topology, "family": family, "method": method, **row,
                    }
                    episode_rows.append(row)
                    rows.append(row)
                summary: dict[str, Any] = {"episodes": episodes}
                for metric in (
                    "success", "steps", "moves", "collisions", "total_reward",
                    "mean_inference_ms", "distance_traveled", "oracle_efficiency",
                    "stay_steps", "max_consecutive_stay", "revisits",
                    "goal_distance_remaining", "spl", *DIFFICULTY_METRICS,
                ):
                    series = np.asarray([
                        float(row[metric]) if row[metric] is not None else np.nan
                        for row in episode_rows
                    ], dtype=np.float64)
                    valid = series[np.isfinite(series)]
                    if not valid.size:
                        summary[metric] = None
                    elif metric == "success":
                        summary[metric] = wilson_success_ci(valid)
                    else:
                        summary[metric] = bootstrap_mean_ci(
                            valid, seed=bootstrap_seed, replicates=bootstrap_replicates
                        )
                for output_name, source_metric in (
                    ("successful_steps", "steps"),
                    ("successful_moves", "moves"),
                    ("successful_distance_traveled", "distance_traveled"),
                ):
                    successful = np.asarray([
                        float(row[source_metric]) for row in episode_rows
                        if row["success"]
                    ], dtype=np.float64)
                    summary[output_name] = (
                        bootstrap_mean_ci(
                            successful, seed=bootstrap_seed, replicates=bootstrap_replicates
                        ) if successful.size else None
                    )
                summary["tortuosity_groups"] = {}
                if oracle_diagnostic:
                    for band in TORTUOSITY_BINS:
                        subset = [
                            row for row in episode_rows if row["tortuosity_bin"] == band
                        ]
                        n = len(subset)
                        success_stats = (
                            wilson_success_ci(np.asarray([
                                float(row["success"]) for row in subset
                            ], dtype=np.float64)) if n else None
                        )
                        spl_stats = (
                            bootstrap_mean_ci(np.asarray([
                                float(row["spl"]) for row in subset
                            ], dtype=np.float64), seed=bootstrap_seed,
                                replicates=bootstrap_replicates) if n else None
                        )
                        summary["tortuosity_groups"][band] = {
                            "episodes": n, "success": success_stats, "spl": spl_stats,
                        }
                        tortuosity_rows.append({
                            "topology": topology, "family": family, "method": method,
                            "tortuosity_bin": band, "episodes": n,
                            "success_rate": success_stats["mean"] if n else None,
                            "success_ci_low": success_stats["ci_low"] if n else None,
                            "success_ci_high": success_stats["ci_high"] if n else None,
                            "spl_mean": spl_stats["mean"] if n else None,
                            "spl_ci_low": spl_stats["ci_low"] if n else None,
                            "spl_ci_high": spl_stats["ci_high"] if n else None,
                        })
                summary["failure_reasons"] = {
                    reason: sum(row["failure_reason"] == reason for row in episode_rows)
                    for reason in sorted({
                        row["failure_reason"] for row in episode_rows
                        if row["failure_reason"] is not None
                    })
                }
                summary["failure_traces"] = sum(
                    row["failure_trace"] is not None for row in episode_rows
                )
                group = (f"{topology}/" if topology != "random" else "")
                summaries[f"{group}{family}/{method}"] = summary
        finally:
            env.close()

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "episodes.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    if oracle_diagnostic:
        with (output / "tortuosity_groups.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            fields = (
                "topology", "family", "method", "tortuosity_bin", "episodes",
                "success_rate", "success_ci_low", "success_ci_high",
                "spl_mean", "spl_ci_low", "spl_ci_high",
            )
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(tortuosity_rows)

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
        "schema_version": 2,
        "protocol": {
            "paired_map_seeds": True,
            "save_failure_traces": save_failures,
            "topologies": list(topologies),
            "structured_map_generator_version": MAP_GENERATOR_VERSION,
            "route_bounds_are_graph_steps": True,
            "tortuosity_bins": {
                "direct": "T < 1.25",
                "moderate_detour": "1.25 <= T < 2.0",
                "large_detour": "T >= 2.0",
            },
            "tortuosity_bins_are_heuristic": True,
            "failure_reason_is_diagnostic": True,
            "eval_seed_start": eval_seed,
            "eval_seed_end_inclusive": eval_seed + episodes - 1,
            "sensor": sensor,
            "movement_mode": movement_mode,
            "diagonal_cost": "sqrt(2) for diagonal, 1 for cardinal",
            "corner_cutting": "forbidden",
            "oracle_is_privileged_diagnostic_only": oracle_diagnostic,
            "confidence_interval": "95% percentile bootstrap for continuous metrics",
            "success_confidence_interval": "95% Wilson score, independent episodes",
            "spl": "mean(S * L / max(P, L)); L=true-map shortest distance",
            "map_difficulty_is_privileged_diagnostic_only": oracle_diagnostic,
            "map_difficulty_definitions": {
                "obstacle_density": "blocked cells / all cells (includes outer boundary)",
                "interior_obstacle_density": "blocked interior / interior cell count",
                "initial_unknown_ratio": "unknown mask / initially non-prior-wall cells",
                "straight_line_distance": "Euclidean start-to-goal distance",
                "path_tortuosity": "shortest geometric route / straight-line distance",
                "minimum_clearance": "minimum path-to-obstacle center distance in cells",
                "mean_path_clearance": "mean path-to-obstacle center distance in cells",
                "dead_end_count": "degree-one cells in start-reachable true-map component",
                "reachable_free_cells": "number of start-reachable true-map free cells",
            },
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": bootstrap_seed,
            "policy_actions": "deterministic; argmax for neural, rule-based for classical",
            "note": (
                "Identical evaluation information does not imply equal training budgets. "
                "BC is imitation-learning; PPO is environment-interaction training."
            ),
        },
        "families": {
            (f"{topology}/" if topology != "random" else "") + family:
                asdict(config)
            for (topology, family), config in configurations.items()
        },
        "methods": list(methods),
        "apf_config": asdict(apf_config or APFConfig()) if "apf" in methods else None,
        "checkpoints": checkpoint_details,
        "summary": summaries,
    }
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
