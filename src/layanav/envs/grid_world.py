"""Deterministic GridWorld with simulator-private static unknown obstacles.

Coordinates are (x, y); all numpy maps are indexed [y, x].
A policy must consume the observation, never debug_snapshot().
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from layanav.envs.map_io import validate_layout
from layanav.envs.sensor import SensorMode, visibility_mask

# Preserve all legacy action IDs (especially STAY=4) for old checkpoints.
ACTION_NAMES = (
    "UP", "DOWN", "LEFT", "RIGHT", "STAY",
    "UP_LEFT", "UP_RIGHT", "DOWN_LEFT", "DOWN_RIGHT",
)
DELTAS = (
    (0, -1), (0, 1), (-1, 0), (1, 0), (0, 0),
    (-1, -1), (1, -1), (-1, 1), (1, 1),
)
STAY_ACTION = 4
VALID_MOVEMENT_MODES = (4, 8)


def action_count(movement_mode: int) -> int:
    if type(movement_mode) is not int or movement_mode not in VALID_MOVEMENT_MODES:
        raise ValueError("movement_mode must be 4 or 8")
    return 5 if movement_mode == 4 else 9


def movement_actions(movement_mode: int) -> tuple[int, ...]:
    action_count(movement_mode)
    return (0, 1, 2, 3) if movement_mode == 4 else (0, 1, 2, 3, 5, 6, 7, 8)


def valid_transition(blocked: np.ndarray, x: int, y: int, dx: int, dy: int) -> bool:
    """Shared motion rule: never cut diagonally past either orthogonal blocker."""
    h, w = blocked.shape
    nx, ny = x + dx, y + dy
    if not (0 <= nx < w and 0 <= ny < h) or blocked[ny, nx]:
        return False
    if dx and dy and (blocked[y, nx] or blocked[ny, x]):
        return False
    return True


def step_distance(dx: int, dy: int) -> float:
    return math.hypot(dx, dy)


@dataclass
class GridConfig:
    size: int = 32
    observation_radius: int = 5
    sensor_mode: SensorMode = "circle"
    movement_mode: int = 4
    hidden_obstacle_count: int = 24
    max_steps: int = 256
    known_wall_segments: int = 5

    def __post_init__(self) -> None:
        if self.size < 5:
            raise ValueError("size must be >= 5")
        if self.observation_radius < 1:
            raise ValueError("observation_radius must be >= 1")
        if self.sensor_mode not in ("circle", "raycast"):
            raise ValueError("sensor_mode must be 'circle' or 'raycast'")
        action_count(self.movement_mode)
        if self.hidden_obstacle_count < 0 or self.known_wall_segments < 0:
            raise ValueError("obstacle counts must be nonnegative")
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")


class GridWorldEnv(gym.Env[np.ndarray, int]):
    """Grid navigation; ideal circular sensor without occlusion (v0.1).

    The prior static walls are globally known. Newly discovered obstacles
    remain in the belief map because all unknown obstacles are stationary.
    `reset(options=...)` accepts `static_map`, `hidden_obstacles` (boolean
    arrays), `start` and `goal` (x, y) for deterministic hand-built scenarios.
    """

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, config: GridConfig | None = None):
        super().__init__()
        self.config = config if config is not None else GridConfig()
        n = self.config.size
        self.action_space = spaces.Discrete(action_count(self.config.movement_mode))
        self.observation_space = spaces.Box(0.0, 1.0, shape=(6, n, n), dtype=np.float32)
        self._prior = np.zeros((n, n), dtype=bool)
        self._hidden = np.zeros((n, n), dtype=bool)
        self._known_hidden = np.zeros((n, n), dtype=bool)
        self._unknown = np.ones((n, n), dtype=bool)
        self._visits = np.zeros((n, n), dtype=np.int32)
        self._robot = (0, 0)
        self._goal = (0, 0)
        self._steps = 0
        self._distance_traveled = 0.0
        self._ended = True

    @staticmethod
    def _reachable(
        blocked: np.ndarray,
        start: tuple[int, int],
        goal: tuple[int, int],
        movement_mode: int = 4,
    ) -> bool:
        queue = deque([start])
        seen = {start}
        while queue:
            x, y = queue.popleft()
            if (x, y) == goal:
                return True
            for action in movement_actions(movement_mode):
                dx, dy = DELTAS[action]
                xx, yy = x + dx, y + dy
                if valid_transition(blocked, x, y, dx, dy) and (xx, yy) not in seen:
                    seen.add((xx, yy))
                    queue.append((xx, yy))
        return False

    def _random_layout(self) -> tuple[np.ndarray, np.ndarray, tuple[int, int], tuple[int, int]]:
        n = self.config.size
        for _ in range(400):
            prior = np.zeros((n, n), dtype=bool)
            prior[0, :] = prior[-1, :] = True
            prior[:, 0] = prior[:, -1] = True
            for _ in range(self.config.known_wall_segments):
                horizontal = bool(self.np_random.integers(0, 2))
                x = int(self.np_random.integers(1, n - 1))
                y = int(self.np_random.integers(1, n - 1))
                length = int(self.np_random.integers(1, max(2, n // 3)))
                if horizontal:
                    prior[y, x : min(n - 1, x + length)] = True
                else:
                    prior[y : min(n - 1, y + length), x] = True
            free = np.argwhere(~prior)
            if len(free) < 2:
                continue
            indexes = self.np_random.choice(len(free), size=2, replace=False)
            sy, sx = map(int, free[indexes[0]])
            gy, gx = map(int, free[indexes[1]])
            start, goal = (sx, sy), (gx, gy)
            if abs(sx - gx) + abs(sy - gy) < n // 2:
                continue
            available = free[
                ~((free[:, 0] == sy) & (free[:, 1] == sx))
                & ~((free[:, 0] == gy) & (free[:, 1] == gx))
            ]
            if self.config.hidden_obstacle_count > len(available):
                raise ValueError("hidden_obstacle_count exceeds free grid cells")
            hidden = np.zeros_like(prior)
            k = self.config.hidden_obstacle_count
            if k:
                selected = self.np_random.choice(len(available), size=k, replace=False)
                locations = available[selected]
                hidden[locations[:, 0], locations[:, 1]] = True
            if self._reachable(prior | hidden, start, goal, self.config.movement_mode):
                return prior, hidden, start, goal
        raise RuntimeError("Unable to generate a solvable map; reduce obstacle density")

    def _provided_layout(
        self, options: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, int], tuple[int, int]]:
        clean = validate_layout(options)
        if clean["static_map"].shape != (self.config.size, self.config.size):
            raise ValueError("Layout dimensions must match GridConfig.size")
        return (
            clean["static_map"],
            clean["hidden_obstacles"],
            clean["start"],
            clean["goal"],
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if options is not None and "static_map" in options:
            self._prior, self._hidden, self._robot, self._goal = self._provided_layout(options)
        else:
            self._prior, self._hidden, self._robot, self._goal = self._random_layout()
        self._known_hidden = np.zeros_like(self._prior)
        self._unknown = ~self._prior.copy()
        self._visits = np.zeros_like(self._visits)
        self._steps = 0
        self._distance_traveled = 0.0
        self._ended = False
        x, y = self._robot
        self._visits[y, x] = 1
        self._observe()
        return self._observation(), self._info(False, False)

    def _observe(self) -> None:
        visible = visibility_mask(
            self._prior | self._hidden,
            self._robot,
            self.config.observation_radius,
            self.config.sensor_mode,
        )
        self._unknown[visible] = False
        self._known_hidden[visible] |= self._hidden[visible]

    def _observation(self) -> np.ndarray:
        n = self.config.size
        obs = np.zeros((6, n, n), dtype=np.float32)
        obs[0] = self._prior
        obs[1] = self._known_hidden
        obs[2] = self._unknown
        x, y = self._robot
        gx, gy = self._goal
        obs[3, y, x] = 1.0
        obs[4, gy, gx] = 1.0
        obs[5] = np.minimum(self._visits, 10) / 10.0
        return obs

    def observe(self) -> np.ndarray:
        """Return a fresh copy of the current policy-safe observation."""
        if self._ended and self._steps == 0:
            raise RuntimeError("Call reset() before observe()")
        return self._observation()

    def _info(self, collision: bool, success: bool) -> dict[str, Any]:
        # No hidden truth or map-generation seed leaks through info.
        return {
            "position": self._robot,
            "steps": self._steps,
            "distance_traveled": self._distance_traveled,
            "collision": collision,
            "success": success,
        }

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._ended:
            raise RuntimeError("Episode ended; call reset() before step()")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action!r}")
        dx, dy = DELTAS[int(action)]
        x, y = self._robot
        xx, yy = x + dx, y + dy
        blocked = self._prior | self._hidden
        collision = not valid_transition(blocked, x, y, dx, dy)
        if not collision:
            self._robot = (xx, yy)
            self._distance_traveled += step_distance(dx, dy)
        self._steps += 1
        x, y = self._robot
        self._visits[y, x] += 1
        self._observe()
        terminated = self._robot == self._goal
        truncated = self._steps >= self.config.max_steps and not terminated
        self._ended = terminated or truncated
        reward = (10.0 if terminated else -0.01) - (1.0 if collision else 0.0)
        return self._observation(), reward, terminated, truncated, self._info(collision, terminated)

    def debug_snapshot(self) -> dict[str, Any]:
        """Privileged copy exclusively for GUI/debugging, NEVER policy input."""
        return {
            "static_map": self._prior.copy(),
            "hidden_obstacles": self._hidden.copy(),
            "discovered_obstacles": self._known_hidden.copy(),
            "unknown": self._unknown.copy(),
            "visits": self._visits.copy(),
            "robot": self._robot,
            "goal": self._goal,
            "step": self._steps,
            "distance_traveled": self._distance_traveled,
        }

    def render(self) -> np.ndarray:
        """Simple RGB-array rendering; GUI uses a separate Qt painter."""
        state = self.debug_snapshot()
        n = self.config.size
        image = np.full((n, n, 3), (234, 238, 241), dtype=np.uint8)
        image[state["unknown"]] = (187, 193, 200)
        image[state["static_map"]] = (82, 91, 108)
        image[state["discovered_obstacles"]] = (215, 99, 70)
        x, y = state["goal"]
        image[y, x] = (29, 166, 112)
        x, y = state["robot"]
        image[y, x] = (36, 133, 219)
        return np.repeat(np.repeat(image, 12, axis=0), 12, axis=1)
