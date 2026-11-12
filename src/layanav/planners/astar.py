"""Online A* using ONLY the public six-channel policy observation."""

from __future__ import annotations

import heapq
import math

import numpy as np

from layanav.envs.grid_world import (
    DELTAS,
    STAY_ACTION,
    action_count,
    movement_actions,
    step_distance,
    valid_transition,
)


class OnlineAStarPolicy:
    def __init__(self, movement_mode: int = 4) -> None:
        action_count(movement_mode)
        self.movement_mode = movement_mode
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
        self.last_path = self._find_path(blocked, start, goal, self.movement_mode)
        if len(self.last_path) < 2:
            return STAY_ACTION  # goal or no known path
        nx, ny = self.last_path[1]
        x, y = start
        delta = (nx - x, ny - y)
        return DELTAS.index(delta)

    @staticmethod
    def _find_path(
        blocked: np.ndarray, start: tuple[int, int], goal: tuple[int, int],
        movement_mode: int = 4,
    ) -> list[tuple[int, int]]:
        action_count(movement_mode)
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
            for action in movement_actions(movement_mode):
                dx, dy = DELTAS[action]
                if not valid_transition(blocked, x, y, dx, dy):
                    continue
                nx, ny = x + dx, y + dy
                other = (nx, ny)
                cost = costs[node] + step_distance(dx, dy)
                if cost >= costs.get(other, 10**9):
                    continue
                costs[other] = cost
                parents[other] = node
                difference_x = abs(nx - goal[0])
                difference_y = abs(ny - goal[1])
                heuristic = (
                    difference_x + difference_y
                    if movement_mode == 4
                    else max(difference_x, difference_y)
                    + (math.sqrt(2.0) - 1.0) * min(difference_x, difference_y)
                )
                order += 1
                heapq.heappush(frontier, (cost + heuristic, order, other))
        return []
