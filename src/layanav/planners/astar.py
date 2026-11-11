"""Online A* using ONLY the public six-channel policy observation."""

from __future__ import annotations

import heapq

import numpy as np

from layanav.envs.grid_world import DELTAS


class OnlineAStarPolicy:
    def __init__(self) -> None:
        self.last_path: list[tuple[int, int]] = []

    @staticmethod
    def _position(channel: np.ndarray) -> tuple[int, int]:
        y, x = np.unravel_index(int(np.argmax(channel)), channel.shape)
        return int(x), int(y)

    def act(self, observation: np.ndarray) -> int:
        """Return an action index; never accesses the environment or truth map."""
        if observation.ndim != 3 or observation.shape[0] != 6:
            raise ValueError("Expected six-channel (6, H, W) observation")
        prior = observation[0] > 0.5
        discovered = observation[1] > 0.5
        blocked = prior | discovered
        start = self._position(observation[3])
        goal = self._position(observation[4])
        self.last_path = self._find_path(blocked, start, goal)
        if len(self.last_path) < 2:
            return 4  # stay if at goal or known path does not exist
        nx, ny = self.last_path[1]
        x, y = start
        delta = (nx - x, ny - y)
        return DELTAS.index(delta)

    @staticmethod
    def _find_path(
        blocked: np.ndarray, start: tuple[int, int], goal: tuple[int, int]
    ) -> list[tuple[int, int]]:
        h, w = blocked.shape
        if blocked[start[1], start[0]] or blocked[goal[1], goal[0]]:
            return []
        frontier: list[tuple[int, int, tuple[int, int]]] = []
        heapq.heappush(frontier, (0, 0, start))
        parents: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        costs = {start: 0}
        order = 0
        while frontier:
            _, _, node = heapq.heappop(frontier)
            if node == goal:
                path = [node]
                while parents[node] is not None:
                    node = parents[node]
                    path.append(node)
                return list(reversed(path))
            x, y = node
            for dx, dy in DELTAS[:4]:
                nx, ny = x + dx, y + dy
                other = (nx, ny)
                if not (0 <= nx < w and 0 <= ny < h) or blocked[ny, nx]:
                    continue
                cost = costs[node] + 1
                if cost >= costs.get(other, 10**9):
                    continue
                costs[other] = cost
                parents[other] = node
                heuristic = abs(nx - goal[0]) + abs(ny - goal[1])
                order += 1
                heapq.heappush(frontier, (cost + heuristic, order, other))
        return []
