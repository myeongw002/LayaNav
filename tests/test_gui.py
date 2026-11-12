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

    # APF must be selectable without a model checkpoint or privileged map.
    apf_index = window.mode.findText("APF")
    assert apf_index >= 0
    window.mode.setCurrentIndex(apf_index)
    window._tick()
    assert window.env.debug_snapshot()["step"] == 1
    assert window.belief_canvas.path == []
    assert window.recorder is not None
    assert len(window.recorder.steps) == 1

    window._reset_same_map()
    window.mode.setCurrentIndex(0)
    # Move diagonally in 8-way manual mode; replay preserves the action space.
    window.movement_mode.setCurrentIndex(window.movement_mode.findData(8))
    assert window.env.action_space.n == 9
    assert window.env.config.movement_mode == 8
    window.mode.setCurrentIndex(1)
    window._apply_action(8)
    assert window.recorder is not None
    assert window.recorder.steps[0]["action"] == 8
    eight_path = tmp_path / "eight_way.json"
    window._save_episode(str(eight_path))
    recorded = window.env.debug_snapshot()
    window._load_episode(str(eight_path))
    assert window.movement_mode.currentData() == 8
    assert window.env.action_space.n == 9
    window._tick()
    assert window.env.debug_snapshot()["robot"] == recorded["robot"]
    assert window.replay.finished
    window.movement_mode.setCurrentIndex(window.movement_mode.findData(4))
    assert window.env.action_space.n == 5
    assert window.replay is None

    # A different generator updates the map and survives trace/replay roundtrips.
    window.topology.setCurrentIndex(window.topology.findData("u_shape"))
    assert window.env.config.topology == "u_shape"
    assert window.env.debug_snapshot()["step"] == 0
    before_new_map = window.env.debug_snapshot()
    window._new_map()
    after_new_map = window.env.debug_snapshot()
    assert window.env.config.topology == "u_shape"
    assert window.env.action_space.n == 5
    assert before_new_map["robot"] != after_new_map["robot"] or (
        not np.array_equal(
            before_new_map["static_map"], after_new_map["static_map"]
        )
    )
    window.mode.setCurrentIndex(1)
    window._apply_action(4)
    topology_path = tmp_path / "u_shape.json"
    window._save_episode(str(topology_path))
    window._load_episode(str(topology_path))
    assert window.topology.currentData() == "u_shape"
    window._tick()
    assert window.replay.finished

    window.close()
    window.deleteLater()
    app.processEvents()
