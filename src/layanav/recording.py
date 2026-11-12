"""Deterministic, inspectable episode traces for the simulator GUI.

Trace files intentionally contain simulator-private obstacle truth so that
an episode can be reconstructed exactly. Never use the trace file contents
as a policy observation or teacher input.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from layanav.envs import GridConfig, GridWorldEnv
from layanav.envs.map_io import validate_layout

TRACE_SCHEMA_VERSION = 1


def _observation_digest(observation: np.ndarray) -> str:
    array = np.ascontiguousarray(observation, dtype=np.float32)
    return hashlib.sha256(array.tobytes()).hexdigest()


class EpisodeRecorder:
    """Collect actions and verifiable observation/reward outcomes after reset."""

    def __init__(self, env: GridWorldEnv) -> None:
        snap = env.debug_snapshot()
        if snap["step"] != 0 or env._ended:
            raise ValueError("Episode recording must start immediately after reset")
        self.layout = validate_layout(
            {
                "static_map": snap["static_map"],
                "hidden_obstacles": snap["hidden_obstacles"],
                "start": snap["robot"],
                "goal": snap["goal"],
            }
        )
        self.config = {
            "sensor_mode": env.config.sensor_mode,
            "observation_radius": env.config.observation_radius,
            "max_steps": env.config.max_steps,
            "movement_mode": env.config.movement_mode,
            "topology": env.config.topology,
            "min_route_steps": env.config.min_route_steps,
            "max_route_steps": env.config.max_route_steps,
        }
        self.action_count = env.action_space.n
        self.initial_digest = _observation_digest(env.observe())
        self.steps: list[dict[str, Any]] = []
        self.diagnostics: dict[str, Any] | None = None

    def record_step(
        self,
        action: int,
        observation: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        if self.steps and (self.steps[-1]["terminated"] or self.steps[-1]["truncated"]):
            raise RuntimeError("Cannot record after episode has ended")
        if type(action) is not int or not 0 <= action < self.action_count:
            raise ValueError("action is outside the configured action space")
        if not np.isfinite(reward) or info.get("steps") != len(self.steps) + 1:
            raise ValueError("Invalid reward or step ordering")
        self.steps.append(
            {
                "action": action,
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "position": list(info["position"]),
                "distance_traveled": info["distance_traveled"],
                "observation_sha256": _observation_digest(observation),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "config": dict(self.config),
            "layout": {
                "static_map": self.layout["static_map"].astype(int).tolist(),
                "hidden_obstacles": self.layout["hidden_obstacles"].astype(int).tolist(),
                "start": list(self.layout["start"]),
                "goal": list(self.layout["goal"]),
            },
            "initial_observation_sha256": self.initial_digest,
            "steps": [dict(step) for step in self.steps],
        }
        if self.diagnostics is not None:
            result["diagnostics"] = dict(self.diagnostics)
        return result

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent,
            prefix=f".{destination.name}.", suffix=".tmp", delete=False
        ) as file:
            temporary = Path(file.name)
            try:
                json.dump(self.to_dict(), file, indent=2)
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        try:
            temporary.replace(destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise


class ReplaySession:
    """Recreate the simulator and verify each action, reward and belief observation."""

    def __init__(self, trace: dict[str, Any]) -> None:
        if not isinstance(trace, dict) or trace.get("schema_version") != TRACE_SCHEMA_VERSION:
            raise ValueError("Unsupported episode trace")
        try:
            layout = validate_layout(trace["layout"])
            config = trace["config"]
            if not isinstance(config, dict):
                raise TypeError("Invalid trace config")
            radius = config["observation_radius"]
            max_steps = config["max_steps"]
            mode = config["sensor_mode"]
            movement_mode = config.get("movement_mode", 4)
            if type(radius) is not int or type(max_steps) is not int:
                raise ValueError("Invalid trace radius/max_steps")
            env_config = GridConfig(
                size=layout["static_map"].shape[0],
                observation_radius=radius,
                max_steps=max_steps,
                sensor_mode=mode,
                movement_mode=movement_mode,
                topology=config.get("topology", "random"),
                min_route_steps=config.get("min_route_steps"),
                max_route_steps=config.get("max_route_steps"),
            )
            steps = trace["steps"]
            if not isinstance(steps, list) or len(steps) > max_steps:
                raise ValueError("Invalid trace step list")
            initial_digest = trace["initial_observation_sha256"]
            if not isinstance(initial_digest, str):
                raise TypeError("Invalid initial observation digest")
        except (KeyError, TypeError) as exc:
            raise ValueError("Malformed episode trace") from exc

        self.env = GridWorldEnv(env_config)
        obs, _ = self.env.reset(options=layout)
        if _observation_digest(obs) != initial_digest:
            raise ValueError("Trace initial observation mismatch")
        self.diagnostics = trace.get("diagnostics")
        if self.diagnostics is not None and not isinstance(self.diagnostics, dict):
            raise ValueError("Malformed trace diagnostics")
        self.steps = steps
        self.index = 0

    @classmethod
    def load(cls, path: str | Path) -> ReplaySession:
        with Path(path).open(encoding="utf-8") as file:
            return cls(json.load(file))

    @property
    def finished(self) -> bool:
        return self.index >= len(self.steps)

    def step(self) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.finished:
            raise RuntimeError("Replay has no further actions")
        item = self.steps[self.index]
        if not isinstance(item, dict):
            raise TypeError("Malformed trace step")
        action = item.get("action")
        if type(action) is not int or not 0 <= action < self.env.action_space.n:
            raise ValueError("Invalid trace action")
        observation, reward, terminated, truncated, info = self.env.step(action)
        expected_position = item.get("position")
        expected_reward = item.get("reward")
        if (
            not isinstance(expected_reward, (int, float))
            or not math.isfinite(expected_reward)
            or not math.isclose(reward, expected_reward, abs_tol=1e-8)
            or terminated != item.get("terminated")
            or truncated != item.get("truncated")
            or list(info["position"]) != expected_position
            or (
                "distance_traveled" in item
                and (
                    not isinstance(item["distance_traveled"], (int, float))
                    or not math.isclose(
                        info["distance_traveled"], item["distance_traveled"], abs_tol=1e-8
                    )
                )
            )
            or _observation_digest(observation) != item.get("observation_sha256")
        ):
            raise ValueError(f"Replay diverged at step {self.index}")
        self.index += 1
        return observation, reward, terminated, truncated, info
