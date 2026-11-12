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
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from layanav.envs import GridConfig, GridWorldEnv
from layanav.envs.map_io import load_map, save_map
from layanav.planners import OnlineAStarPolicy
from layanav.recording import EpisodeRecorder, ReplaySession


class MapCanvas(QWidget):
    def __init__(
        self,
        env: GridWorldEnv,
        truth: bool,
        on_edit: Callable[[int, int], None] | None = None,
    ) -> None:
        super().__init__()
        self.env, self.truth, self.on_edit = env, truth, on_edit
        self.trail: list[tuple[int, int]] = []
        self.path: list[tuple[int, int]] = []
        self.setMinimumSize(330, 330)

    def _geometry(self) -> tuple[float, float, float]:
        n = self.env.config.size
        width = float(min(self.width(), self.height()) - 16)
        size = width / n
        return (self.width() - width) / 2, (self.height() - width) / 2, size

    def paintEvent(self, event) -> None:
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
                elif (self.truth and hidden[y, x]) or (
                    not self.truth and discovered[y, x]
                ):
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
        # Both panels show the robot trajectory; planned path is policy-derived.
        if len(self.trail) > 1:
            painter.setPen(QPen(QColor("#8654c7"), max(1, round(cell * 0.2))))
            for (x0, y0), (x1, y1) in zip(self.trail[:-1], self.trail[1:], strict=True):
                painter.drawLine(
                    round(left + (x0 + 0.5) * cell),
                    round(top + (y0 + 0.5) * cell),
                    round(left + (x1 + 0.5) * cell),
                    round(top + (y1 + 0.5) * cell),
                )
        if not self.truth and len(self.path) > 1:
            painter.setPen(QPen(QColor("#f5ad38"), max(1, round(cell * 0.2))))
            for (x0, y0), (x1, y1) in zip(self.path[:-1], self.path[1:], strict=True):
                painter.drawLine(
                    round(left + (x0 + 0.5) * cell),
                    round(top + (y0 + 0.5) * cell),
                    round(left + (x1 + 0.5) * cell),
                    round(top + (y1 + 0.5) * cell),
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

    def mousePressEvent(self, event) -> None:
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
        self.setWindowTitle("LayaNav Studio — v0.2-dev")
        self.resize(1040, 650)
        self.env = GridWorldEnv(GridConfig())
        self.env.reset(seed=42)
        self.policy = OnlineAStarPolicy()
        self.recorder: EpisodeRecorder | None = None
        self.replay: ReplaySession | None = None
        self.trail: list[tuple[int, int]] = []
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
        self.mode.addItems(["Online A*", "Manual (arrow keys)", "Replay"])
        self.mode.currentIndexChanged.connect(self._mode_changed)
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

        self.sensor_mode = QComboBox()
        self.sensor_mode.addItem("Circle (ideal)", "circle")
        self.sensor_mode.addItem("Raycast (occluded)", "raycast")
        self.sensor_mode.currentIndexChanged.connect(self._change_sensor_mode)
        controls.addWidget(QLabel("Sensor:"))
        controls.addWidget(self.sensor_mode)

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
        save_button = QPushButton("Save map")
        save_button.clicked.connect(self._save_map_dialog)
        controls.addWidget(save_button)
        load_button = QPushButton("Load map")
        load_button.clicked.connect(self._load_map_dialog)
        controls.addWidget(load_button)

        playback_controls = QHBoxLayout()
        root.addLayout(playback_controls)
        playback_controls.addWidget(QLabel("Speed (steps/s):"))
        self.speed = QSpinBox()
        self.speed.setRange(1, 30)
        self.speed.setValue(10)
        self.speed.valueChanged.connect(self._change_speed)
        playback_controls.addWidget(self.speed)
        save_episode_button = QPushButton("Save episode")
        save_episode_button.clicked.connect(self._save_episode_dialog)
        playback_controls.addWidget(save_episode_button)
        load_episode_button = QPushButton("Load replay")
        load_episode_button.clicked.connect(self._load_episode_dialog)
        playback_controls.addWidget(load_episode_button)
        playback_controls.addStretch(1)

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
        self._start_episode()
        self._update_view()

    def _start_episode(self) -> None:
        self.replay = None
        self.recorder = EpisodeRecorder(self.env)
        self.trail = [self.env.debug_snapshot()["robot"]]
        self.truth_canvas.trail = self.trail
        self.belief_canvas.trail = self.trail
        self.truth_canvas.path = []
        self.belief_canvas.path = []

    def _mode_changed(self, index: int) -> None:
        if self.replay is not None and index != 2:
            self._reset_same_map()

    def _change_speed(self, speed: int) -> None:
        if self.playing:
            self.timer.start(max(20, round(1000 / speed)))

    def _save_episode(self, path: str) -> None:
        if self.recorder is None or self.replay is not None:
            raise ValueError("No live episode available to save")
        self.recorder.save(path)

    def _save_episode_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save recorded episode", "", "LayaNav episodes (*.json)"
        )
        if path:
            try:
                self._save_episode(path)
            except (OSError, TypeError, ValueError) as exc:
                QMessageBox.warning(self, "Episode save failed", str(exc))

    def _load_episode(self, path: str) -> None:
        session = ReplaySession.load(path)
        self._stop()
        self.replay = session
        self.recorder = None
        self.env = session.env
        self.truth_canvas.env = self.env
        self.belief_canvas.env = self.env
        self.start = self.env.debug_snapshot()["robot"]
        self.trail = [self.start]
        self.truth_canvas.trail = self.trail
        self.belief_canvas.trail = self.trail
        self.truth_canvas.path = []
        self.belief_canvas.path = []
        self.radius.blockSignals(True)
        self.radius.setValue(self.env.config.observation_radius)
        self.radius.blockSignals(False)
        self.sensor_mode.blockSignals(True)
        sensor_index = self.sensor_mode.findData(self.env.config.sensor_mode)
        self.sensor_mode.setCurrentIndex(sensor_index)
        self.sensor_mode.blockSignals(False)
        self.mode.setCurrentIndex(2)
        self.total_reward = 0.0
        self._update_view()

    def _load_episode_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open recorded episode", "", "LayaNav episodes (*.json)"
        )
        if path:
            try:
                self._load_episode(path)
            except (OSError, TypeError, ValueError) as exc:
                QMessageBox.warning(self, "Episode replay failed", str(exc))

    def _stop(self) -> None:
        self.timer.stop()
        self.playing = False
        self.play_button.setText("Play")

    def _change_radius(self, value: int) -> None:
        self.env.config.observation_radius = value
        self._reset_same_map()

    def _change_sensor_mode(self, index: int) -> None:
        self.env.config.sensor_mode = self.sensor_mode.itemData(index)
        self._reset_same_map()

    def _save_map(self, path: str) -> None:
        snap = self.env.debug_snapshot()
        save_map(
            path,
            {
                "static_map": snap["static_map"],
                "hidden_obstacles": snap["hidden_obstacles"],
                "start": self.start,
                "goal": snap["goal"],
            },
        )

    def _save_map_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save simulator map", "", "LayaNav maps (*.json)"
        )
        if path:
            try:
                self._save_map(path)
            except (OSError, TypeError, ValueError) as exc:
                QMessageBox.warning(self, "Map save failed", str(exc))

    def _load_map(self, path: str) -> None:
        layout = load_map(path)
        size = layout["static_map"].shape[0]
        config = GridConfig(
            size=size,
            observation_radius=self.radius.value(),
            sensor_mode=self.sensor_mode.currentData(),
            hidden_obstacle_count=min(24, (size - 2) ** 2 // 8),
            known_wall_segments=min(5, size // 2),
            max_steps=self.env.config.max_steps,
        )
        candidate = GridWorldEnv(config)
        candidate.reset(options=layout)
        self._stop()
        self.env = candidate
        self.truth_canvas.env = candidate
        self.belief_canvas.env = candidate
        self.start = layout["start"]
        self.policy = OnlineAStarPolicy()
        self.total_reward = 0.0
        self._start_episode()
        self._update_view()

    def _load_map_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open simulator map", "", "LayaNav maps (*.json)"
        )
        if path:
            try:
                self._load_map(path)
            except (OSError, TypeError, ValueError) as exc:
                QMessageBox.warning(self, "Map load failed", str(exc))

    def _reset_same_map(self) -> None:
        self._stop()
        snap = self.env.debug_snapshot()
        self.env.reset(
            options={
                "static_map": snap["static_map"],
                "hidden_obstacles": snap["hidden_obstacles"],
                "start": self.start,
                "goal": snap["goal"],
            }
        )
        self.total_reward = 0.0
        self.policy = OnlineAStarPolicy()
        self._start_episode()
        self._update_view()

    def _new_map(self) -> None:
        self._stop()
        self.env.reset(seed=int(np.random.default_rng().integers(0, 2**31)))
        self.start = self.env.debug_snapshot()["robot"]
        self.total_reward = 0.0
        self.policy = OnlineAStarPolicy()
        self._start_episode()
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
        self._start_episode()
        self._update_view()

    def _toggle_play(self) -> None:
        if self.playing:
            self._stop()
            return
        if self.mode.currentIndex() == 1:
            return
        if self.mode.currentIndex() == 2 and (
            self.replay is None or self.replay.finished
        ):
            return
        self.playing = True
        self.play_button.setText("Pause")
        self.timer.start(max(20, round(1000 / self.speed.value())))

    def _tick(self) -> None:
        if self.mode.currentIndex() == 2:
            self._replay_step()
            return
        if self.env._ended or self.mode.currentIndex() != 0:
            self._stop()
            return
        action = self.policy.act(self.env.observe())
        self.belief_canvas.path = list(self.policy.last_path)
        self._apply_action(action)

    def _replay_step(self) -> None:
        if self.replay is None or self.replay.finished:
            self._stop()
            return
        try:
            _, reward, terminated, truncated, info = self.replay.step()
        except (RuntimeError, ValueError) as exc:
            self._stop()
            QMessageBox.warning(self, "Replay diverged", str(exc))
            return
        self.total_reward += reward
        self.trail.append(info["position"])
        self._update_view(info)
        if terminated or truncated or self.replay.finished:
            self._stop()

    def _apply_action(self, action: int) -> None:
        if self.env._ended or self.replay is not None:
            return
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.recorder is not None:
            self.recorder.record_step(action, obs, reward, terminated, truncated, info)
        self.trail.append(info["position"])
        self.total_reward += reward
        self._update_view(info)
        if terminated or truncated:
            self._stop()

    def keyPressEvent(self, event) -> None:
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
            f"discovered {int(snap['discovered_obstacles'].sum())} | "
            f"sensor {self.env.config.sensor_mode}"
            + suffix
        )


def launch() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()
