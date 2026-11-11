"""Grid-cell visibility with an ideal radius sensor or occlusion-aware ray casting.

Coordinates are (x, y); array indexing is [y, x].
The line-of-sight model casts center-to-center Bresenham rays. The obstacle
cell at the end of a ray is observable; cells behind an intervening obstacle
are not. Corner-grazing rays are a known discretization approximation.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

SensorMode = Literal["circle", "raycast"]


def _line_cells(
    start: tuple[int, int], end: tuple[int, int]
) -> list[tuple[int, int]]:
    """Integer Bresenham line, including both endpoints."""
    x0, y0 = start
    x1, y1 = end
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    cells: list[tuple[int, int]] = []
    while True:
        cells.append((x0, y0))
        if (x0, y0) == (x1, y1):
            return cells
        twice_error = 2 * err
        if twice_error > -dy:
            err -= dy
            x0 += sx
        if twice_error < dx:
            err += dx
            y0 += sy


def visibility_mask(
    blocked: np.ndarray,
    position: tuple[int, int],
    radius: int,
    mode: SensorMode = "circle",
) -> np.ndarray:
    """Return observable cells without revealing their contents.

    `blocked` is the simulator's combined static+hidden obstacle mask;
    it must not be shared with the policy. Only this visibility mask is
    used to reveal truth about cells actually observed.
    """
    if blocked.ndim != 2 or blocked.dtype != np.dtype(bool):
        raise ValueError("blocked must be a 2D boolean array")
    height, width = blocked.shape
    x0, y0 = position
    if not (0 <= x0 < width and 0 <= y0 < height):
        raise ValueError("position outside grid")
    if not isinstance(radius, int) or radius < 1:
        raise ValueError("radius must be a positive integer")
    if mode not in ("circle", "raycast"):
        raise ValueError(f"Unsupported sensor mode: {mode}")

    yy, xx = np.ogrid[:height, :width]
    within_radius = (xx - x0) ** 2 + (yy - y0) ** 2 <= radius**2
    if mode == "circle":
        return within_radius

    visible = np.zeros_like(blocked)
    ys, xs = np.nonzero(within_radius)
    for y, x in zip(ys, xs, strict=True):
        # The target itself may be occupied; this reveals the obstacle surface.
        # Only blockers strictly between the robot and target cause occlusion.
        intermediate = _line_cells((x0, y0), (int(x), int(y)))[1:-1]
        if all(not blocked[iy, ix] for ix, iy in intermediate):
            visible[y, x] = True
    return visible
