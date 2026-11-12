"""Collect belief-only A* demonstrations; store exact 0.1-quantized observations.

The sixth observation channel contains visit count / 10, clipped to [0,1].
All other observation channels are binary. Storing observation * 10 as uint8
preserves the information exactly (up to float representation), while avoiding
unnecessary float32 dataset size. No simulator-private maps are persisted.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from layanav.envs import GridConfig, GridWorldEnv
from layanav.planners import OnlineAStarPolicy

DATASET_SCHEMA_VERSION = 1


def encode_observations(observations: np.ndarray) -> np.ndarray:
    observations = np.asarray(observations, dtype=np.float32)
    if observations.ndim not in (3, 4) or observations.shape[-3] != 6:
        raise ValueError("Expected observation shape (6,H,W) or (N,6,H,W)")
    scaled = np.rint(observations * 10)
    if not np.isfinite(observations).all() or not np.allclose(
        observations * 10, scaled, atol=1e-5
    ) or np.any((scaled < 0) | (scaled > 10)):
        raise ValueError("Observation cannot be encoded losslessly at 0.1 precision")
    return scaled.astype(np.uint8)


def decode_observations(encoded: np.ndarray) -> np.ndarray:
    if encoded.dtype != np.dtype(np.uint8) or encoded.ndim != 4:
        raise ValueError("Expected uint8 shape (N,6,H,W)")
    if encoded.shape[1] != 6 or np.any(encoded > 10):
        raise ValueError("Malformed encoded observations")
    return encoded.astype(np.float32) / 10.0


def collect_expert_dataset(
    *,
    episodes: int,
    seed_start: int,
    config: GridConfig,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Run online A* with exactly the observations available to a learned policy.

    Failed episodes are excluded from demonstrations and reported explicitly.
    """
    if episodes < 1 or seed_start < 0:
        raise ValueError("episodes must be positive and seed_start nonnegative")
    env = GridWorldEnv(config)
    observations: list[np.ndarray] = []
    actions: list[int] = []
    episode_seeds: list[int] = []
    successful = 0
    for index in range(episodes):
        seed = seed_start + index
        observation, _ = env.reset(seed=seed)
        expert = OnlineAStarPolicy()
        obs_in_episode: list[np.ndarray] = []
        actions_in_episode: list[int] = []
        done = False
        succeeded = False
        for _ in range(config.max_steps):
            action = expert.act(observation)
            if action == 4:
                # Known-blocked goal or no route: do not teach repeated no-ops.
                break
            obs_in_episode.append(encode_observations(observation))
            actions_in_episode.append(action)
            observation, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if done:
                succeeded = info["success"]
                break
        if succeeded:
            successful += 1
            observations.extend(obs_in_episode)
            actions.extend(actions_in_episode)
            episode_seeds.extend([seed] * len(actions_in_episode))
    env.close()
    if not observations:
        raise RuntimeError("Teacher produced no successful trajectories; adjust the maps")
    dataset = {
        "schema_version": np.asarray(DATASET_SCHEMA_VERSION, dtype=np.int32),
        "observations": np.stack(observations).astype(np.uint8),
        "actions": np.asarray(actions, dtype=np.uint8),
        "episode_seeds": np.asarray(episode_seeds, dtype=np.int64),
        "grid_size": np.asarray(config.size, dtype=np.int32),
    }
    return dataset, {
        "attempted_episodes": episodes,
        "successful_episodes": successful,
        "samples": len(actions),
    }


def save_dataset(path: str | Path, dataset: dict[str, np.ndarray]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **dataset)


def load_dataset(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return decoded public observations, actions and episode seeds."""
    with np.load(path, allow_pickle=False) as archive:
        if int(archive["schema_version"]) != DATASET_SCHEMA_VERSION:
            raise ValueError("Unsupported dataset schema")
        observations = decode_observations(archive["observations"])
        actions = np.asarray(archive["actions"], dtype=np.int64)
        seeds = np.asarray(archive["episode_seeds"], dtype=np.int64)
    if actions.ndim != 1 or seeds.ndim != 1:
        raise ValueError("Actions and episode seeds must be 1D")
    if not len(actions) or len(actions) != len(observations) or len(actions) != len(seeds):
        raise ValueError("Dataset arrays have incompatible lengths")
    if np.any((actions < 0) | (actions > 4)):
        raise ValueError("Dataset has invalid action labels")
    return observations, actions, seeds
