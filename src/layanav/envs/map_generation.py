"""Seeded static topology generators. Their output is privileged map *truth*.

Map structures are generated before initial observations. These layouts must
never be passed directly to a navigation policy. The legacy 'random' mode is
handled by GridWorldEnv to preserve its original seed-to-map behavior.
"""

from __future__ import annotations

from collections import deque
from heapq import heappop, heappush
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from layanav.envs.grid_world import GridConfig

MAP_TOPOLOGIES = (
    "random", "maze", "corridor", "rooms", "u_shape", "cluttered",
)
MAP_GENERATOR_VERSION = 2  # Structured layouts change; legacy random is unchanged.


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
    """Concave barrier with varied in-cavity start and beyond-wall goal."""
    wall = _empty_walls(n)
    left, right = max(2, n // 3), min(n - 3, 2 * n // 3)
    top, bottom = max(2, n // 4), min(n - 3, 3 * n // 4)
    wall[top:bottom + 1, left] = True
    wall[top:bottom + 1, right] = True
    wall[bottom, left:right + 1] = True
    # Sample positions inside the concavity and beyond the closed wall.
    # Both remain on opposite sides of the barrier, but direct distances
    # and required detours now vary across seeds. In small grids the range
    # naturally shrinks, while retaining at least one valid cell.
    start = (
        int(rng.integers(left + 1, right)),
        int(rng.integers(top + 1, bottom)),
    )
    goal = (
        int(rng.integers(left + 1, right)),
        int(rng.integers(bottom + 1, min(n - 1, bottom + 4))),
    )
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
    prior: np.ndarray,
    movement_mode: int,
    rng: np.random.Generator,
    *,
    min_route_steps: int | None = None,
    max_route_steps: int | None = None,
    minimum_component_cells: int = 2,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Choose the pair on the *prior* layout, before adding hidden blockers.

    This fixes the Maze/Corridor pathology in which dense hidden blockers
    create small connected components and the start/goal shrink with them.
    Route length bounds count actions (not weighted geometric distance).
    """
    free = np.argwhere(~prior)
    if len(free) < minimum_component_cells:
        return None
    lower = min_route_steps if min_route_steps is not None else max(3, prior.shape[0] // 2)
    for _ in range(min(64, len(free))):
        y, x = free[int(rng.integers(len(free)))]
        start = (int(x), int(y))
        distances = _distances(prior, start, movement_mode)
        if len(distances) < minimum_component_cells:
            continue
        candidates = [
            point for point, distance in distances.items()
            if distance >= lower and (max_route_steps is None or distance <= max_route_steps)
        ]
        if candidates:
            return start, candidates[int(rng.integers(len(candidates)))]
    return None


def _connected_hidden_obstacles(
    prior: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Remove leaves from a cardinal-edge spanning tree, never its endpoints.

    Tree-leaf removal preserves the whole start-reachable component except
    for the removed cells, not just one start-to-goal path. We deliberately
    construct a *4-way* tree even for 8-way motion: deleting a cell can
    invalidate diagonal edges under the no-corner-cutting rule, whereas
    cardinal tree edges remain valid. Each successive removal preserves
    4-connectivity, hence 8-connectivity as well.

    One seeded leaf-removal order is used for every hidden-obstacle count.
    For equal prior geometry and start/goal, sparse obstacles are a subset
    of standard, which are a subset of dense (nested map realizations).
    """
    if count < 0:
        raise ValueError("hidden obstacle count must be nonnegative")
    hidden = np.zeros_like(prior)
    if count == 0:
        return hidden

    from layanav.envs.grid_world import DELTAS, movement_actions, valid_transition

    # Breadth-first spanning tree, rooted at the chosen start.
    queue = deque([start])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    child_counts = {start: 0}
    while queue:
        x, y = queue.popleft()
        for action in movement_actions(4):
            dx, dy = DELTAS[action]
            neighbor = (x + dx, y + dy)
            if neighbor not in parent and valid_transition(prior, x, y, dx, dy):
                parent[neighbor] = (x, y)
                child_counts[neighbor] = 0
                child_counts[(x, y)] += 1
                queue.append(neighbor)
    if goal not in parent or count > len(parent) - 2:
        raise ValueError(
            "Hidden obstacle count exceeds removable cells in start component"
        )
    # Rank all removable cells once, independently of target obstacle count.
    # Priority is deterministic for a given seed and prior topology.
    removable = sorted(point for point in parent if point not in (start, goal))
    priority = rng.permutation(len(removable))
    rank = {point: int(priority[i]) for i, point in enumerate(removable)}
    available: list[tuple[int, tuple[int, int]]] = []
    for point in removable:
        if child_counts[point] == 0:
            heappush(available, (rank[point], point))

    for _ in range(count):
        if not available:
            # A tree with >2 nodes always has an unprotected leaf unless
            # it is exactly the preserved start-goal path.
            raise RuntimeError("Not enough safely removable leaves")
        _, point = heappop(available)
        x, y = point
        hidden[y, x] = True
        ancestor = parent[point]
        assert ancestor is not None
        child_counts[ancestor] -= 1
        if ancestor not in (start, goal) and child_counts[ancestor] == 0:
            heappush(available, (rank[ancestor], ancestor))
    return hidden


def generate_structured_layout(
    config: GridConfig, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], tuple[int, int]]:
    """Seeded topology with controlled start-goal route and connected hidden cells.

    The initial prior geometry and start/goal are sampled *before* hidden
    blockers. The latter are removed in a nested, connectivity-preserving
    order. Impossible route constraints fail explicitly rather than
    silently changing hidden obstacle counts or relaxing bounds.
    """
    if config.topology not in MAP_TOPOLOGIES or config.topology == "random":
        raise ValueError("Expected a non-random structured topology")
    n = config.size
    lower = config.min_route_steps
    upper = config.max_route_steps
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

        if fixed is None:
            pair = _choose_start_goal(
                prior, config.movement_mode, rng,
                min_route_steps=lower, max_route_steps=upper,
                minimum_component_cells=config.hidden_obstacle_count + 2,
            )
        else:
            pair = fixed
            distance = _distances(prior, pair[0], config.movement_mode).get(pair[1])
            required = lower if lower is not None else max(3, n // 2)
            if distance is None or distance < required or (
                upper is not None and distance > upper
            ):
                pair = None

        if pair is None:
            continue
        start, goal = pair
        if config.hidden_obstacle_count > len(_distances(prior, start, 4)) - 2:
            # Only cells that preserve cardinal connectivity are eligible.
            continue
        hidden = _connected_hidden_obstacles(
            prior, start, goal, config.hidden_obstacle_count, rng,
        )
        return prior, hidden, start, goal
    raise RuntimeError(
        "Unable to meet structured map route/size constraints "
        "without disconnecting the map; reduce hidden obstacles "
        "or loosen route-step bounds"
    )
