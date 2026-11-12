"""Evaluate a greedy CNN policy against the same partially observable environment."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from layanav.envs import GridConfig, GridWorldEnv
from layanav.training.behavior_cloning import load_policy


def evaluate_policy(
    checkpoint: str | Path,
    *,
    episodes: int,
    seed_start: int,
    config: GridConfig,
    device: str = "cpu",
) -> dict[str, float | int]:
    if episodes < 1 or seed_start < 0:
        raise ValueError("episodes must be positive and seed_start nonnegative")
    policy = load_policy(checkpoint, device=device)
    env = GridWorldEnv(config)
    if policy.action_count != env.action_space.n:
        raise ValueError("Checkpoint action count does not match environment movement_mode")
    success = 0
    collision_attempts = 0
    total_steps = 0
    total_rewards = 0.0
    inference_time = 0.0
    inference_count = 0
    for index in range(episodes):
        obs, _ = env.reset(seed=seed_start + index)
        while True:
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            action = policy.act(obs)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            inference_time += time.perf_counter() - t0
            inference_count += 1

            obs, reward, terminated, truncated, info = env.step(action)
            collision_attempts += int(info["collision"])
            total_rewards += reward
            if terminated or truncated:
                success += int(info["success"])
                total_steps += info["steps"]
                break
    env.close()
    report = {
        "episodes": episodes,
        "success_rate": success / episodes,
        "mean_steps": total_steps / episodes,
        "collision_attempts_per_episode": collision_attempts / episodes,
        "mean_return": total_rewards / episodes,
        "mean_policy_latency_ms": 1000 * inference_time / max(inference_count, 1),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return report
