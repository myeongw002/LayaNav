"""Interactive PySide6 simulator/editor; no Qt dependency in the environment core."""

from __future__ import annotations

import sys
from collections.abc import Callable

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from layanav.envs import GridConfig, GridWorldEnv
from layanav.planners import OnlineAStarPolicy


class MapCanvas(QWidget):
    def __init__(
        self,
        env: GridWorldEnv,
        truth: bool,
        on_edit: Callable[[int, int], None] | None = None,
    ) -> None:
        super().__init__()
        self.env, self.truth, self.on_edit = env, truth, on_edit
        self.setMinimumSize(330, 330)

    def _geometry(self) -> tuple[float, float, float]:
        n = self.env.config.size
        width = float(min(self.width(), self.height()) - 16)
        size = width / n
        return (self.width() - width) / 2, (self.height() - width) / 2, size

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt API)
        state = self.env.debug_snapshot()
        prior, hidden = state["static_map"], state["hidden_obstacles"]
        discovered, unknown = state["discovered_obstacles"], state["unknown"]
        left, top, cell = self._geometry()
        n = self.env.config.size
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#17212e"))
        for y in range(n):
            for x in range(n):
                if prior[y, x]:
                    fill = "#65748b"
                elif self.truth and hidden[y, x]:
                    fill = "#df765b"
                elif not self.truth and discovered[y, x]:
                    fill = "#df765b"
                elif not self.truth and unknown[y, x]:
                    fill = "#48525e"
                else:
                    fill = "#d4dce1"
                painter.fillRect(
                    int(left + x * cell),
                    int(top + y * cell),
                    max(1, int(cell - 1)),
                    max(1, int(cell - 1)),
                    QColor(fill),
                )
        x, y = state["goal"]
        painter.setBrush(QColor("#20ac77"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(
            int(left + (x + 0.16) * cell),
            int(top + (y + 0.16) * cell),
            max(3, int(cell * 0.68)),
            max(3, int(cell * 0.68)),
        )
        x, y = state["robot"]
        painter.setPen(QPen(QColor("#358fe0"), 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        r = self.env.config.observation_radius * cell
        painter.drawEllipse(
            int(left + (x + 0.5) * cell - r),
            int(top + (y + 0.5) * cell - r),
            int(2 * r),
            int(2 * r),
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#2787e3"))
        painter.drawEllipse(
            int(left + (x + 0.1) * cell),
            int(top + (y + 0.1) * cell),
            max(3, int(cell * 0.8)),
            max(3, int(cell * 0.8)),
        )
        painter.end()

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt API)
        if not self.truth or self.on_edit is None:
            return
        left, top, cell = self._geometry()
        x = int((event.position().x() - left) // cell)
        y = int((event.position().y() - top) // cell)
        if 0 <= x < self.env.config.size and 0 <= y < self.env.config.size:
            self.on_edit(x, y)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("LayaNav Studio — v0.1")
        self.resize(1040, 650)
        self.env = GridWorldEnv(GridConfig())
        self.env.reset(seed=42)
        self.policy = OnlineAStarPolicy()
        self.total_reward = 0.0
        self.playing = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)

        container = QWidget()
        self.setCentralWidget(container)
        root = QVBoxLayout(container)
        controls = QHBoxLayout()
        root.addLayout(controls)

        self.mode = QComboBox()
        self.mode.addItems(["Online A*", "Manual (arrow keys)"])
        controls.addWidget(QLabel("Policy:"))
        controls.addWidget(self.mode)

        self.edit_mode = QComboBox()
        self.edit_mode.addItems(["Hidden obstacle", "Known wall", "Start", "Goal", "Erase"])
        controls.addWidget(QLabel("Edit:"))
        controls.addWidget(self.edit_mode)

        self.radius = QSpinBox()
        self.radius.setRange(1, 16)
        self.radius.setValue(5)
        self.radius.valueChanged.connect(self._change_radius)
        controls.addWidget(QLabel("Sensor radius:"))
        controls.addWidget(self.radius)

        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self._toggle_play)
        controls.addWidget(self.play_button)
        step_button = QPushButton("Step")
        step_button.clicked.connect(self._tick)
        controls.addWidget(step_button)
        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(self._reset_same_map)
        controls.addWidget(reset_button)
        new_button = QPushButton("New map")
        new_button.clicked.connect(self._new_map)
        controls.addWidget(new_button)

        maps = QHBoxLayout()
        root.addLayout(maps, 1)
        for truth, title in ((True, "Ground Truth (debug / editing)"), (False, "Robot Belief (policy view)")):
            column = QVBoxLayout()
            column.addWidget(QLabel(title))
            canvas = MapCanvas(self.env, truth, self._edit_map if truth else None)
            column.addWidget(canvas, 1)
            maps.addLayout(column, 1)
            if truth:
                self.truth_canvas = canvas
            else:
                self.belief_canvas = canvas

        self.status_label = QLabel("")
        root.addWidget(self.status_label)
        root.addWidget(
            QLabel("Blue: robot   Green: goal   Orange: unknown obstacle   "
                   "Grey: prior wall   Dark grey: not observed")
        )
        root.addWidget(
            QLabel("Edit by clicking Ground Truth. Use Play for A*, or switch to Manual "
                   "and use arrow keys. Editing resets the episode.")
        )
        self._update_view()

    def _stop(self) -> None:
        self.timer.stop()
        self.playing = False
        self.play_button.setText("Play")

    def _change_radius(self, value: int) -> None:
        self.env.config.observation_radius = value
        self._reset_same_map()

    def _reset_same_map(self) -> None:
        self._stop()
        snap = self.env.debug_snapshot()
        self.env.reset(
            options={
                "static_map": snap["static_map"],
                "hidden_obstacles": snap["hidden_obstacles"],
                "start": snap["robot"] if False else self._initial_start(),
                "goal": snap["goal"],
            }
        )
        self.total_reward = 0.0
        self.policy = OnlineAStarPolicy()
        self._update_view()

    def _initial_start(self) -> tuple[int, int]:
        # Original start is held separately from the robot's current position.
        return self.start

    def _new_map(self) -> None:
        self._stop()
        self.env.reset(seed=int(np.random.default_rng().integers(0, 2**31)))
        self.start = self.env.debug_snapshot()["robot"]
        self.total_reward = 0.0
        self.policy = OnlineAStarPolicy()
        self._update_view()

    def _edit_map(self, x: int, y: int) -> None:
        self._stop()
        snap = self.env.debug_snapshot()
        prior, hidden = snap["static_map"], snap["hidden_obstacles"]
        start, goal = self.start, snap["goal"]
        mode = self.edit_mode.currentText()
        if mode == "Start":
            if prior[y, x] or hidden[y, x] or (x, y) == goal:
                return
            start = (x, y)
        elif mode == "Goal":
            if prior[y, x] or hidden[y, x] or (x, y) == start:
                return
            goal = (x, y)
        elif (x, y) in (start, goal):
            return
        elif mode == "Hidden obstacle":
            prior[y, x] = False
            hidden[y, x] = not hidden[y, x]
        elif mode == "Known wall":
            hidden[y, x] = False
            prior[y, x] = not prior[y, x]
        else:
            prior[y, x], hidden[y, x] = False, False
        try:
            self.env.reset(options={
                "static_map": prior,
                "hidden_obstacles": hidden,
                "start": start,
                "goal": goal,
            })
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid map", str(exc))
            return
        self.start = start
        self.total_reward = 0.0
        self.policy = OnlineAStarPolicy()
        self._update_view()

    def _toggle_play(self) -> None:
        if self.playing:
            self._stop()
            return
        if self.mode.currentIndex() != 0:
            return
        self.playing = True
        self.play_button.setText("Pause")
        self.timer.start(90)

    def _tick(self) -> None:
        if self.env._ended:
            self._stop()
            return
        action = self.policy.act(self.env._observation()) if self.mode.currentIndex() == 0 else 4
        self._apply_action(action)

    def _apply_action(self, action: int) -> None:
        if self.env._ended:
            return
        _, reward, terminated, truncated, info = self.env.step(action)
        self.total_reward += reward
        self._update_view(info)
        if terminated or truncated:
            self._stop()

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt API)
        if self.mode.currentIndex() == 1:
            key_to_action = {
                Qt.Key.Key_Up: 0, Qt.Key.Key_Down: 1,
                Qt.Key.Key_Left: 2, Qt.Key.Key_Right: 3,
            }
            action = key_to_action.get(event.key())
            if action is not None:
                self._apply_action(action)
                return
        super().keyPressEvent(event)

    def _update_view(self, info: dict | None = None) -> None:
        if not hasattr(self, "start"):
            self.start = self.env.debug_snapshot()["robot"]
        self.truth_canvas.update()
        self.belief_canvas.update()
        snap = self.env.debug_snapshot()
        suffix = ""
        if info:
            suffix = " | collision" if info["collision"] else ""
            if info["success"]:
                suffix = " | GOAL REACHED"
        self.status_label.setText(
            f"Step {snap['step']}/{self.env.config.max_steps} | "
            f"position {snap['robot']} | goal {snap['goal']} | "
            f"reward {self.total_reward:.2f} | "
            f"discovered {int(snap['discovered_obstacles'].sum())}"
            + suffix
        )


def launch() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()
