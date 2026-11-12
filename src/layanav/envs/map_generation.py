"""Seeded static topology generators. Their output is privileged map *truth*.

Map structures are generated before initial observations. These layouts must
never be passed directly to a navigation policy. The legacy 'random' mode is
handled by GridWorldEnv to preserve its original seed-to-map behavior.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from layanav.envs.grid_world import GridConfig

MAP_TOPOLOGIES = (
    "random", "maze", "corridor", "rooms", "u_shape", "cluttered",
)


def _empty_walls(n: int) -> np.ndarray:
    wall = np.zeros((n, n), dtype=bool)
    wall[[0, -1], :] = True
    wall[:, [0, -1]] = True
    return wall


def _maze(n: int, rng: np.random.Generator) -> np.ndarray:
    """Randomized depth-first carving, one-cell-wide corridors."""
    wall = np.ones((n, n), dtype=bool)
    wall[1, 1] = False
    stack = [(1, 1)]
    while stack:
        x, y = stack[-1]
        available = []
        for dx, dy in ((2, 0), (-2, 0), (0, 2), (0, -2)):
            xx, yy = x + dx, y + dy
            if 1 <= xx < n - 1 and 1 <= yy < n - 1 and wall[yy, xx]:
                available.append((xx, yy))
        if not available:
            stack.pop()
            continue
        nx, ny = available[int(rng.integers(len(available)))]
        wall[(y + ny) // 2, (x + nx) // 2] = False
        wall[ny, nx] = False
        stack.append((nx, ny))
    return wall


def _corridor(n: int, rng: np.random.Generator) -> np.ndarray:
    """A network of two horizontal and two vertical narrow corridors."""
    wall = np.ones((n, n), dtype=bool)
    width = max(2, n // 12)
    offset = max(1, n // 16)
    for center in (n // 3, 2 * n // 3):
        shift = int(rng.integers(-offset, offset + 1))
        c = max(2, min(n - 3, center + shift))
        low = max(1, c - width // 2)
        high = min(n - 1, low + width)
        wall[low:high, 1:-1] = False
        wall[1:-1, low:high] = False
    return wall


def _rooms(n: int, rng: np.random.Generator) -> np.ndarray:
    """Four connected rooms separated by walls with randomized doorways."""
    wall = _empty_walls(n)
    middle = n // 2
    wall[middle, 1:-1] = True
    wall[1:-1, middle] = True
    for a, b in ((1, middle), (middle + 1, n - 1)):
        if b > a:
            x = int(rng.integers(a, b))
            y = int(rng.integers(a, b))
            wall[middle, x] = False
            wall[y, middle] = False
    return wall


def _rotated_coordinate(
    x: int, y: int, n: int, quarters: int,
) -> tuple[int, int]:
    for _ in range(quarters):
        x, y = y, n - 1 - x
    return x, y


def _u_shape(
    n: int, rng: np.random.Generator,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    """Concave obstacle, with start inside and goal across its closed end."""
    wall = _empty_walls(n)
    left, right = max(2, n // 3), min(n - 3, 2 * n // 3)
    top, bottom = max(2, n // 4), min(n - 3, 3 * n // 4)
    wall[top:bottom + 1, left] = True
    wall[top:bottom + 1, right] = True
    wall[bottom, left:right + 1] = True
    middle_x = (left + right) // 2
    start, goal = (middle_x, bottom - 1), (middle_x, bottom + 1)
    quarters = int(rng.integers(4))
    return (
        np.rot90(wall, quarters).copy(),
        _rotated_coordinate(*start, n, quarters),
        _rotated_coordinate(*goal, n, quarters),
    )


def _cluttered(
    n: int, rng: np.random.Generator, wall_segments: int,
) -> np.ndarray:
    """Multi-cell rectangular obstacle clusters rather than isolated pixels."""
    wall = _empty_walls(n)
    max_width = max(3, n // 6)
    for _ in range(max(3, wall_segments)):
        x = int(rng.integers(1, n - 1))
        y = int(rng.integers(1, n - 1))
        width = int(rng.integers(2, max_width + 1))
        height = int(rng.integers(2, max_width + 1))
        wall[y:min(n - 1, y + height), x:min(n - 1, x + width)] = True
    return wall


def _distances(
    blocked: np.ndarray, start: tuple[int, int], movement_mode: int,
) -> dict[tuple[int, int], int]:
    # Imported lazily to keep the core GridWorld module's imports acyclic.
    from layanav.envs.grid_world import DELTAS, movement_actions, valid_transition

    queue = deque([start])
    distances = {start: 0}
    while queue:
        x, y = queue.popleft()
        for action in movement_actions(movement_mode):
            dx, dy = DELTAS[action]
            next_pos = (x + dx, y + dy)
            if next_pos not in distances and valid_transition(blocked, x, y, dx, dy):
                distances[next_pos] = distances[x, y] + 1
                queue.append(next_pos)
    return distances


def _choose_start_goal(
    blocked: np.ndarray, movement_mode: int, rng: np.random.Generator,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Pick a path-separated pair; shorter Euclidean separation is allowed in mazes."""
    free = np.argwhere(~blocked)
    if len(free) < 2:
        return None
    min_route_steps = max(3, blocked.shape[0] // 2)
    for _ in range(min(32, len(free))):
        y, x = free[int(rng.integers(len(free)))]
        start = (int(x), int(y))
        distances = _distances(blocked, start, movement_mode)
        furthest = max(distances.values())
        if furthest < min_route_steps:
            continue
        candidates = [point for point, distance in distances.items()
                      if distance >= min_route_steps]
        goal = candidates[int(rng.integers(len(candidates)))]
        return start, goal
    return None


def generate_structured_layout(
    config: GridConfig, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], tuple[int, int]]:
    """Generate a connected instance or reject it; never silently reduce density."""
    if config.topology not in MAP_TOPOLOGIES or config.topology == "random":
        raise ValueError("Expected a non-random structured topology")
    n = config.size
    for _ in range(400):
        fixed: tuple[tuple[int, int], tuple[int, int]] | None = None
        if config.topology == "maze":
            prior = _maze(n, rng)
        elif config.topology == "corridor":
            prior = _corridor(n, rng)
        elif config.topology == "rooms":
            prior = _rooms(n, rng)
        elif config.topology == "u_shape":
            prior, start, goal = _u_shape(n, rng)
            fixed = (start, goal)
        else:
            prior = _cluttered(n, rng, config.known_wall_segments)

        eligible = np.argwhere(~prior)
        if fixed is not None:
            (sx, sy), (gx, gy) = fixed
            eligible = eligible[
                ~((eligible[:, 0] == sy) & (eligible[:, 1] == sx))
                & ~((eligible[:, 0] == gy) & (eligible[:, 1] == gx))
            ]
        if config.hidden_obstacle_count > len(eligible) - (0 if fixed else 2):
            raise ValueError(
                "hidden_obstacle_count exceeds available free cells for topology"
            )
        hidden = np.zeros_like(prior)
        if config.hidden_obstacle_count:
            selected = rng.choice(
                len(eligible), size=config.hidden_obstacle_count, replace=False
            )
            indices = eligible[selected]
            hidden[indices[:, 0], indices[:, 1]] = True

        blocked = prior | hidden
        if fixed is None:
            pair = _choose_start_goal(blocked, config.movement_mode, rng)
        else:
            pair = fixed if goal in _distances(blocked, start, config.movement_mode) else None
        if pair is not None:
            return prior, hidden, pair[0], pair[1]
    raise RuntimeError(
        "Unable to generate connected structured map; reduce obstacle density"
    )
