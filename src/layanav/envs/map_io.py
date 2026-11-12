"""Versioned, validated JSON layout format for simulator maps.

A saved layout contains privileged obstacle truth and is intended ONLY for
environment creation and GUI editing, never for a navigation policy.
"""

from __future__ import annotations

from collections.abc import Mapping

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

MAP_SCHEMA_VERSION = 1
MAX_MAP_SIZE = 512


def _binary_grid(value: Any, name: str) -> np.ndarray:
    try:
        arr = np.asarray(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be a rectangular binary grid") from exc
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square 2D grid")
    n = arr.shape[0]
    if not 5 <= n <= MAX_MAP_SIZE:
        raise ValueError(f"{name} size must be between 5 and {MAX_MAP_SIZE}")
    if arr.dtype.kind not in ("b", "i", "u") or not np.isin(arr, (0, 1)).all():
        raise ValueError(f"{name} must contain only boolean or 0/1 cells")
    return arr.astype(bool, copy=True)


def _coordinate(value: Any, name: str, size: int) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be (x, y)")
    x, y = value
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
           for v in (x, y)):
        raise ValueError(f"{name} coordinates must be integers")
    if not (0 <= x < size and 0 <= y < size):
        raise ValueError(f"{name} is outside the map")
    return int(x), int(y)


def validate_layout(layout: Mapping[str, Any]) -> dict[str, Any]:
    """Validate layout, return independent boolean arrays and integer coordinates."""
    if not isinstance(layout, Mapping):
        raise TypeError("Map layout must be a mapping")
    try:
        prior = _binary_grid(layout["static_map"], "static_map")
        hidden = _binary_grid(layout["hidden_obstacles"], "hidden_obstacles")
        if prior.shape != hidden.shape:
            raise ValueError("Map layers must have equal shape")
        n = prior.shape[0]
        start = _coordinate(layout["start"], "start", n)
        goal = _coordinate(layout["goal"], "goal", n)
    except KeyError as exc:
        raise ValueError(f"Missing map field: {exc.args[0]}") from exc
    if np.any(prior & hidden):
        raise ValueError("Unknown obstacles cannot overlap known walls")
    if start == goal:
        raise ValueError("Start and goal must be different")
    for name, (x, y) in (("start", start), ("goal", goal)):
        if prior[y, x] or hidden[y, x]:
            raise ValueError(f"{name} must be on a free cell")
    return {
        "static_map": prior,
        "hidden_obstacles": hidden,
        "start": start,
        "goal": goal,
    }


def save_map(path: str | Path, layout: Mapping[str, Any]) -> None:
    """Atomically write a validated layout as human-readable JSON."""
    clean = validate_layout(layout)
    size = clean["static_map"].shape[0]
    data = {
        "schema_version": MAP_SCHEMA_VERSION,
        "size": size,
        "static_map": clean["static_map"].astype(int).tolist(),
        "hidden_obstacles": clean["hidden_obstacles"].astype(int).tolist(),
        "start": list(clean["start"]),
        "goal": list(clean["goal"]),
    }
    target = Path(path)
    # Do not create an unexpected directory tree for mistyped paths.
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=target.parent, prefix=f".{target.name}.",
        suffix=".tmp", delete=False
    ) as handle:
        temp_path = Path(handle.name)
        try:
            json.dump(data, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise
    try:
        temp_path.replace(target)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def load_map(path: str | Path) -> dict[str, Any]:
    """Read and validate schema v1; ready for GridWorldEnv.reset(options=...)."""
    with Path(path).open(encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise TypeError("Map JSON root must be an object")
    if type(document.get("schema_version")) is not int or (
        document["schema_version"] != MAP_SCHEMA_VERSION
    ):
        raise ValueError("Unsupported map schema_version")
    size = document.get("size")
    if type(size) is not int or not 5 <= size <= MAX_MAP_SIZE:
        raise ValueError("Invalid map size")
    clean = validate_layout(document)
    if clean["static_map"].shape != (size, size):
        raise ValueError("Map size does not match layer dimensions")
    return clean
