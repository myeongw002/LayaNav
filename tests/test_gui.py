"""Smoke-test the real PySide6 UI without a display server."""

from __future__ import annotations

import numpy as np
import pytest

QtWidgets = pytest.importorskip("PySide6.QtWidgets")


# Keep the Qt application alive for the whole Python interpreter lifetime.
# Qt widgets may otherwise outlive a locally-scoped QApplication during pytest teardown.
_APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_gui_start_edit_step_and_reset(tmp_path):
    from layanav.gui.app import MainWindow

    app = _APP
    window = MainWindow()
    window.show()
    app.processEvents()
    assert window.env.config.size == 32
    assert window.start == window.env.debug_snapshot()["robot"]
    step0 = window.env.debug_snapshot()["step"]
    window.mode.setCurrentIndex(1)
    window._apply_action(4)
    assert window.env.debug_snapshot()["step"] == step0 + 1

    snap = window.env.debug_snapshot()
    candidates = np.argwhere(
        ~snap["static_map"] & ~snap["hidden_obstacles"]
    )
    for y, x in candidates:
        if (int(x), int(y)) not in (window.start, snap["goal"]):
            edit_x, edit_y = int(x), int(y)
            break
    window.edit_mode.setCurrentText("Hidden obstacle")
    window._edit_map(edit_x, edit_y)
    assert window.env.debug_snapshot()["hidden_obstacles"][edit_y, edit_x]
    assert window.env.debug_snapshot()["step"] == 0
    window.radius.setValue(3)
    assert window.env.config.observation_radius == 3

    window.sensor_mode.setCurrentIndex(1)
    assert window.env.config.sensor_mode == "raycast"

    saved = tmp_path / "gui_map.json"
    window._save_map(str(saved))
    before = window.env.debug_snapshot()
    window._new_map()
    window._load_map(str(saved))
    after = window.env.debug_snapshot()
    np.testing.assert_array_equal(before["static_map"], after["static_map"])
    np.testing.assert_array_equal(before["hidden_obstacles"], after["hidden_obstacles"])
    assert before["goal"] == after["goal"]
    assert window.start == before["robot"]
    assert window.env.config.sensor_mode == "raycast"
    window._reset_same_map()
    assert window.env.debug_snapshot()["robot"] == window.start
    # Record a policy step, restore its initial map and replay exactly.
    window.mode.setCurrentIndex(0)
    window._tick()
    assert window.recorder is not None
    assert len(window.recorder.steps) == 1
    assert len(window.trail) == 2
    assert window.belief_canvas.path

    episode_path = tmp_path / "episode.json"
    window._save_episode(str(episode_path))
    expected_position = window.env.debug_snapshot()["robot"]
    window._load_episode(str(episode_path))
    assert window.mode.currentIndex() == 2
    assert window.replay is not None
    window._tick()
    assert window.env.debug_snapshot()["robot"] == expected_position
    assert window.replay.finished

    window._reset_same_map()
    assert window.mode.currentIndex() == 0
    assert window.replay is None
    window.close()
    window.deleteLater()
    app.processEvents()
