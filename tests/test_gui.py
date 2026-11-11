"""Smoke-test the real PySide6 UI without a display server."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from layanav.gui.app import MainWindow


# Keep the Qt application alive for the whole Python interpreter lifetime.
# Qt widgets may otherwise outlive a locally-scoped QApplication during pytest teardown.
_APP = QApplication.instance() or QApplication([])


def test_gui_start_edit_step_and_reset():
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

    window._reset_same_map()
    assert window.env.debug_snapshot()["robot"] == window.start
    window.close()
    window.deleteLater()
    app.processEvents()
