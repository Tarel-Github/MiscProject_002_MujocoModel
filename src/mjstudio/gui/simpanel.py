"""Simulation controls, actuator controls, sensor and contact read-outs."""
from __future__ import annotations

import mujoco
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QHBoxLayout, QHeaderView, QLabel, QPushButton, QSlider,
                               QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from ..core import mathx as mx
from ..core.editor import Editor
from ..core.sim import Simulator
from .theme import C, no_buttons

OBJ = mujoco.mjtObj
RES = 1000


def _ro(text: str) -> QTableWidgetItem:
    it = QTableWidgetItem(text)
    it.setFlags(it.flags() & ~Qt.ItemIsEditable)
    return it


class SimPanel(QWidget):
    def __init__(self, ed: Editor, sim: Simulator, parent=None):
        super().__init__(parent)
        self.ed, self.sim = ed, sim
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        bar = QHBoxLayout()
        self.play = QPushButton("▶ Play")
        self.play.setCheckable(True)
        self.play.toggled.connect(self._toggle)
        bar.addWidget(self.play)
        b = QPushButton("Step")
        b.clicked.connect(lambda: self.sim.step(1))
        bar.addWidget(b)
        b = QPushButton("Reset")
        b.clicked.connect(self._reset)
        bar.addWidget(b)
        bar.addWidget(QLabel("Speed"))
        self.speed = QComboBox()
        for s in ("0.1x", "0.25x", "0.5x", "1x", "2x"):
            self.speed.addItem(s)
        self.speed.setCurrentText("1x")
        self.speed.currentTextChanged.connect(lambda t: setattr(self.sim, "speed", float(t[:-1])))
        bar.addWidget(self.speed)
        bar.addWidget(QLabel("Reset to"))
        self.key = QComboBox()
        bar.addWidget(self.key)
        self.status = QLabel()
        self.status.setStyleSheet(f"color:{C['dim']}")
        bar.addWidget(self.status, 1)
        lay.addLayout(bar)
        self.tabs = QTabWidget()
        lay.addWidget(self.tabs)
        # actuators
        self.act = QTableWidget(0, 3)
        self.act.setHorizontalHeaderLabels(["Actuator", "Control", "ctrl"])
        self.act.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.act.setColumnWidth(0, 170)
        self.act.verticalHeader().setVisible(False)
        self.tabs.addTab(self.act, "Actuators")
        # sensors
        self.sens = QTableWidget(0, 3)
        self.sens.setHorizontalHeaderLabels(["Sensor", "Type", "Value"])
        self.sens.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.sens.setColumnWidth(0, 170)
        self.sens.verticalHeader().setVisible(False)
        self.tabs.addTab(self.sens, "Sensors")
        # contacts
        self.con = QTableWidget(0, 4)
        self.con.setHorizontalHeaderLabels(["Geom 1", "Geom 2", "Normal force (N)", "Position"])
        self.con.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.con.verticalHeader().setVisible(False)
        self.tabs.addTab(self.con, "Contacts")
        self._sliders: list[tuple[int, QSlider, QDoubleSpinBox]] = []
        self.timer = QTimer(self)
        self.timer.setInterval(150)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()
        ed.compiled.connect(lambda r: r.ok and self.rebuild())
        self.rebuild()

    def _toggle(self, on: bool) -> None:
        self.sim.play() if on else self.sim.pause()
        self.play.setText("❚❚ Pause" if on else "▶ Play")

    def sync_play(self) -> None:
        self.play.blockSignals(True)
        self.play.setChecked(self.sim.running)
        self.play.setText("❚❚ Pause" if self.sim.running else "▶ Play")
        self.play.blockSignals(False)

    def _reset(self) -> None:
        i = self.key.currentIndex() - 1
        self.sim.reset(i if i >= 0 else None)

    def rebuild(self) -> None:
        m = self.ed.model
        self.act.setRowCount(0)
        self._sliders = []
        self.key.clear()
        self.key.addItem("qpos0")
        if m is None:
            return
        for k in range(m.nkey):
            self.key.addItem(mujoco.mj_id2name(m, OBJ.mjOBJ_KEY, k) or f"key {k}")
        for i in range(m.nu):
            e = self.ed.idmap.elem(int(OBJ.mjOBJ_ACTUATOR), i)
            name = (e.get("name") if e is not None else None) or f"actuator {i}"
            self.act.insertRow(i)
            self.act.setItem(i, 0, _ro(name))
            lo, hi = (m.actuator_ctrlrange[i] if m.actuator_ctrllimited[i] else (-1.0, 1.0))
            s = QSlider(Qt.Horizontal)
            s.setRange(0, RES)
            sp = QDoubleSpinBox()
            sp.setDecimals(3)
            sp.setRange(float(lo), float(hi))
            sp.setKeyboardTracking(False)
            no_buttons(sp)
            s.valueChanged.connect(lambda v, i=i, lo=lo, hi=hi, sp=sp: self._ctrl(i, lo + (hi - lo) * v / RES, sp))
            sp.valueChanged.connect(lambda v, i=i: self._ctrl(i, v, None))
            self.act.setCellWidget(i, 1, s)
            self.act.setCellWidget(i, 2, sp)
            self._sliders.append((i, s, sp))
        self.refresh()

    def _ctrl(self, i: int, v: float, sp) -> None:
        self.ed.data.ctrl[i] = v
        if sp is not None:
            sp.blockSignals(True)
            sp.setValue(v)
            sp.blockSignals(False)

    def refresh(self) -> None:
        m, d = self.ed.model, self.ed.data
        if m is None or not self.isVisible():
            return
        iss = self.sim.last_issue
        self.status.setText(f"t = {d.time:.3f} s   contacts: {d.ncon}" + (f"   ⚠ {iss}" if iss else ""))
        self.status.setStyleSheet(f"color:{C['warn'] if iss else C['dim']}")
        for i, s, sp in self._sliders:
            if not s.isSliderDown() and not sp.hasFocus():
                lo, hi = sp.minimum(), sp.maximum()
                s.blockSignals(True)
                sp.blockSignals(True)
                s.setValue(int(round((d.ctrl[i] - lo) / max(hi - lo, 1e-9) * RES)))
                sp.setValue(float(d.ctrl[i]))
                s.blockSignals(False)
                sp.blockSignals(False)
        if self.tabs.currentWidget() is self.sens:
            rows = self.sim.sensors()
            self.sens.setRowCount(len(rows))
            for r, (name, typ, val) in enumerate(rows):
                self.sens.setItem(r, 0, _ro(name))
                self.sens.setItem(r, 1, _ro(typ))
                self.sens.setItem(r, 2, _ro(mx.fmt_vec(np.round(val, 5), 6)))
        elif self.tabs.currentWidget() is self.con:
            cs = self.sim.contacts()
            self.con.setRowCount(len(cs))
            gname = lambda g: mujoco.mj_id2name(m, OBJ.mjOBJ_GEOM, g) or f"geom {g}"  # noqa: E731
            for r, c in enumerate(cs):
                n1, n2 = gname(c.geom1), gname(c.geom2)
                self.con.setItem(r, 0, _ro(n1 if not n1.startswith("__mms_") else f"geom {c.geom1}"))
                self.con.setItem(r, 1, _ro(n2 if not n2.startswith("__mms_") else f"geom {c.geom2}"))
                self.con.setItem(r, 2, _ro(f"{c.normal_force:.3f}"))
                self.con.setItem(r, 3, _ro(mx.fmt_vec(np.round(c.pos, 4), 5)))
