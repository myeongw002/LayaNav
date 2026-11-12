"""Discrete Artificial Potential Field (APF) using only the public belief map.

This is intentionally a *classical, strictly descending local controller*:
- attractive quadratic potential toward the known goal;
- inverse-distance repulsion from observed obstacles and prior walls;
- choose the neighboring free cell with the lowest potential, but move only
  when it strictly decreases potential;
- STAY when trapped at a local minimum or with no known-free neighbor.

There is no A* escape fallback, global map access, or use of simulator truth.
As a result, U-shaped barriers and narrow passages can trap the policy.
That is an expected and reportable limitation, not an implementation error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from layanav.envs.grid_world import DELTAS


@dataclass(frozen=True)
class APFConfig:
    """All distances are in grid-cell units.

    The attractive term is 0.5 * attractive_gain * squared_distance(goal).
    Within influence_radius of the *nearest observed obstacle*, the
    repulsive term is 0.5 * repulsive_gain * (1 / d - 1 / radius)^2.
    unknown_penalty optionally adds a heuristic cost for unobserved cells;
    leave it at zero for an unmodified classical APF comparison.
    """

    attractive_gain: float = 1.0
    repulsive_gain: float = 8.0
    influence_radius: float = 3.0
    unknown_penalty: float = 0.0
    descent_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        for name in (
            "attractive_gain", "repulsive_gain", "influence_radius",
            "unknown_penalty", "descent_tolerance",
        ):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, float)
            ) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite numeric value")
        if self.attractive_gain <= 0:
            raise ValueError("attractive_gain must be positive")
        if self.repulsive_gain < 0:
            raise ValueError("repulsive_gain must be nonnegative")
        if self.influence_radius <= 0:
            raise ValueError("influence_radius must be positive")
        if self.unknown_penalty < 0 or self.descent_tolerance < 0:
            raise ValueError("unknown_penalty and descent_tolerance must be nonnegative")


class ArtificialPotentialFieldPolicy:
    """Stateless-on-environment 4-connected APF navigation policy.

    The only public input to act() is the same (6,H,W) observation as A*
    and learned policies. `last_status` is diagnostic state for the GUI.
    """

    def __init__(self, config: APFConfig | None = None) -> None:
        self.config = config if config is not None else APFConfig()
        self.last_status = "ready"
        self.last_potentials: dict[int, float] = {}

    @staticmethod
    def _position(channel: np.ndarray, label: str) -> tuple[int, int]:
        locations = np.argwhere(channel > 0.5)
        if locations.shape != (1, 2):
            raise ValueError(f"{label} must identify exactly one cell")
        y, x = locations[0]
        return int(x), int(y)

    def _potential(
        self,
        x: int,
        y: int,
        goal: tuple[int, int],
        obstacles: np.ndarray,
        unknown: np.ndarray,
    ) -> float:
        gx, gy = goal
        c = self.config
        potential = 0.5 * c.attractive_gain * ((x - gx) ** 2 + (y - gy) ** 2)
        if obstacles.size and c.repulsive_gain > 0:
            # obstacles are N×2 (y, x) indices; use distance to the nearest
            # *known* blocker, never undiscovered ground-truth obstacles.
            squared = (obstacles[:, 0] - y) ** 2 + (obstacles[:, 1] - x) ** 2
            nearest = math.sqrt(float(np.min(squared)))
            if nearest < c.influence_radius:
                # Blocked cells themselves are never scored, so nearest > 0.
                if nearest <= 0:
                    return math.inf
                potential += 0.5 * c.repulsive_gain * (
                    1.0 / nearest - 1.0 / c.influence_radius
                ) ** 2
        if unknown[y, x]:
            potential += c.unknown_penalty
        return potential

    def act(self, observation: np.ndarray) -> int:
        """Choose UP/DOWN/LEFT/RIGHT/STAY; never read simulator internals."""
        if (
            not isinstance(observation, np.ndarray)
            or observation.ndim != 3
            or observation.shape[0] != 6
        ):
            raise ValueError("Expected six-channel (6,H,W) numpy observation")
        _, height, width = observation.shape
        if not np.isfinite(observation).all():
            raise ValueError("Observation contains NaN or infinity")
        start = self._position(observation[3], "robot")
        goal = self._position(observation[4], "goal")
        blocked = (observation[0] > 0.5) | (observation[1] > 0.5)
        unknown = observation[2] > 0.5
        x, y = start
        if blocked[y, x] or blocked[goal[1], goal[0]]:
            raise ValueError("Robot and goal must be unblocked in the belief map")
        self.last_potentials = {}
        if start == goal:
            self.last_status = "goal"
            return 4

        obstacles = np.argwhere(blocked)
        current = self._potential(x, y, goal, obstacles, unknown)
        best = current
        best_action = 4
        valid_neighbors = 0
        for action, (dx, dy) in enumerate(DELTAS[:4]):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < width and 0 <= ny < height) or blocked[ny, nx]:
                continue
            valid_neighbors += 1
            potential = self._potential(nx, ny, goal, obstacles, unknown)
            self.last_potentials[action] = potential
            # Strict descent prohibits two-cell oscillations but cannot
            # escape local minima. Stable DELTAS order breaks equal-cost ties.
            if potential < best - self.config.descent_tolerance:
                best = potential
                best_action = action
        self.last_status = (
            "moving" if best_action != 4
            else "blocked" if valid_neighbors == 0
            else "local_minimum"
        )
        return best_action
