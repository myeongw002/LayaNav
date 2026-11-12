"""Privileged, reporting-only map complexity diagnostics.

Every metric is derived from the initial *ground-truth* layout and the
shortest geometric A* path. Callers must never forward these values to a
navigation policy, BC dataset, or PPO reward/observation.
"""

from __future__ import annotations

import hashlib
import math
from collections import deque
from collections.abc import Sequence
from itertools import pairwise

import numpy as np

from layanav.envs.grid_world import DELTAS, movement_actions, valid_transition

DIFFICULTY_METRICS = (
    "obstacle_density",
    "interior_obstacle_density",
    "initial_unknown_ratio",
    "straight_line_distance",
    "path_tortuosity",
    "minimum_clearance",
    "mean_path_clearance",
    "dead_end_count",
    "reachable_free_cells",
)


def map_fingerprint(
    prior: np.ndarray,
    hidden: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    movement_mode: int,
) -> str:
    """Stable hash of map geometry, known/hidden split and movement contract."""
    digest = hashlib.sha256()
    for layer in (prior, hidden):
        arr = np.ascontiguousarray(layer, dtype=np.uint8)
        digest.update(np.asarray(arr.shape, dtype="<i8").tobytes())
        digest.update(arr.tobytes())
    digest.update(np.asarray((*start, *goal, movement_mode), dtype="<i8").tobytes())
    return digest.hexdigest()


def _reachable_graph(
    blocked: np.ndarray, start: tuple[int, int], movement_mode: int,
) -> tuple[int, int]:
    """Return reachable component size and undirected degree-one cell count."""
    actions = tuple(DELTAS[a] for a in movement_actions(movement_mode))
    queue = deque([start])
    visited = {start}
    dead_ends = 0
    while queue:
        x, y = queue.popleft()
        degree = 0
        for dx, dy in actions:
            if not valid_transition(blocked, x, y, dx, dy):
                continue
            degree += 1
            neighbor = (x + dx, y + dy)
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
        if degree == 1:
            dead_ends += 1
    return len(visited), dead_ends


def measure_map_difficulty(
    prior: np.ndarray,
    hidden: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    path: Sequence[tuple[int, int]],
    movement_mode: int,
    initial_observation: np.ndarray,
) -> dict[str, float | int | None]:
    """Measure *descriptive* complexity; not a calibrated difficulty score.

    - Occupancy is counted over all cells (and separately over interior).
    - Unknown fraction uses initial belief mask / cells not known as walls.
    - Tortuosity is optimal geometric path length / Euclidean displacement.
    - Clearance is min center-to-center distance from shortest path cells to
      any blocked cell; it does NOT account for robot geometry/footprint.
    - Dead ends count degree-one cells in the start-reachable ground-truth
      movement graph under the active 4/8-way corner-cutting rules.
    """
    if (
        prior.ndim != 2 or prior.shape[0] != prior.shape[1]
        or hidden.shape != prior.shape
        or prior.dtype != np.dtype(bool) or hidden.dtype != np.dtype(bool)
    ):
        raise ValueError("Expected equally shaped square boolean map layers")
    n = prior.shape[0]
    if initial_observation.shape != (6, n, n):
        raise ValueError("Expected (6,H,W) initial observation")
    if not path or path[0] != start or path[-1] != goal:
        raise ValueError("Oracle path must connect start and goal")
    blocked = prior | hidden
    if any(
        not (0 <= x < n and 0 <= y < n) or blocked[y, x] for x, y in path
    ):
        raise ValueError("Oracle path contains an invalid cell")
    allowed = {DELTAS[action] for action in movement_actions(movement_mode)}
    if any(
        (x1 - x0, y1 - y0) not in allowed
        or not valid_transition(blocked, x0, y0, x1 - x0, y1 - y0)
        for (x0, y0), (x1, y1) in pairwise(path)
    ):
        raise ValueError("Oracle path violates motion constraints")
    direct = math.dist(start, goal)
    if direct <= 0:
        raise ValueError("Start and goal must differ")
    optimal = sum(math.dist(a, b) for a, b in pairwise(path))
    unknown = initial_observation[2] > 0.5
    initial_free = ~prior
    denominator = int(initial_free.sum())
    if denominator <= 0:
        raise ValueError("No non-prior-wall cells")
    obstacle_y, obstacle_x = np.nonzero(blocked)
    clearance: float | None = None
    mean_clearance: float | None = None
    if obstacle_x.size:
        path_clearances = [
            math.sqrt(float(np.min((obstacle_x - x) ** 2 + (obstacle_y - y) ** 2)))
            for x, y in path
        ]
        clearance = min(path_clearances)
        mean_clearance = float(np.mean(path_clearances))
    reachable_count, dead_ends = _reachable_graph(blocked, start, movement_mode)
    return {
        "obstacle_density": float(blocked.mean()),
        "interior_obstacle_density": float(blocked[1:-1, 1:-1].mean()),
        "initial_unknown_ratio": float(np.count_nonzero(unknown & initial_free))
        / denominator,
        "straight_line_distance": direct,
        "path_tortuosity": optimal / direct,
        "minimum_clearance": clearance,
        "mean_path_clearance": mean_clearance,
        "dead_end_count": dead_ends,
        "reachable_free_cells": reachable_count,
    }
