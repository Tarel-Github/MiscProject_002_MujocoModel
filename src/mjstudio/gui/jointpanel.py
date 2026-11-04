"""Joint control: sliders for every joint, range editing for the selected joint."""
from __future__ import annotations

import logging
import math

import mujoco
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QHeaderView, QLabel,
                               QLineEdit, QPushButton, QSlider, QTableWidget, QVBoxLayout, QWidget)

from ..core import mathx as mx
from ..core import ops
from ..core.editor import Editor
from .theme import C, no_buttons

log = logging.getLogger("mjstudio")
JT = mujoco.mjtJoint
RES = 2000


class JointRow:
    def __init__(self, panel: "JointPanel", info: dict, comp: int = -1):
        self.panel, self.info, self.comp = panel, info, comp
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, RES)
        self.spin = QDoubleSpinBox()
        self.spin.setKeyboardTracking(False)
        self.spin.setDecimals(2 if info["type"] in (JT.mjJNT_HINGE, JT.mjJNT_BALL) else 4)
        no_buttons(self.spin)
        self.spin.setFixedWidth(78)
        self.slider.valueChanged.connect(self._slid)
        self.spin.valueChanged.connect(self._spun)
        self._block = False
        self.update_range()

    # displayed units: degrees for hinge/ball, metres for slide
    def bounds(self) -> tuple[float, float]:
        t = self.info["type"]
        if t == JT.mjJNT_HINGE:
            if self.info["limited"]:
                return tuple(math.degrees(x) for x in self.info["range"])  # type: ignore[return-value]
            return -180.0, 180.0
        if t == JT.mjJNT_SLIDE:
            if self.info["limited"]:
                return float(self.info["range"][0]), float(self.info["range"][1])
            return -1.0, 1.0
        if t == JT.mjJNT_BALL:
            lim = math.degrees(self.info["range"][1]) if self.info["limited"] and self.info["range"][1] > 0 else 180
            return -lim, lim
        return -1.0, 1.0

    def update_range(self) -> None:
        lo, hi = self.bounds()
        if hi <= lo:
            hi = lo + 1e-6
        self.spin.blockSignals(True)
        self.spin.setRange(lo, hi)
        self.spin.setSuffix("°" if self.info["type"] != JT.mjJNT_SLIDE else " m")
        self.spin.blockSignals(False)

    def value(self) -> float:
        d = self.panel.ed.data
        a = self.info["qposadr"]
        t = self.info["type"]
        if t == JT.mjJNT_HINGE:
            return math.degrees(d.qpos[a])
        if t == JT.mjJNT_SLIDE:
            return float(d.qpos[a])
        if t == JT.mjJNT_BALL:
            e = mx.q2euler(d.qpos[a:a + 4], "XYZ")
            return math.degrees(e[self.comp]) if e is not None else 0.0
        return 0.0

    def refresh(self) -> None:
        if self.spin.hasFocus() or self.slider.isSliderDown():
            return
        v = self.value()
        lo, hi = self.bounds()
        self._block = True
        self.spin.setValue(v)
        self.slider.setValue(int(round((v - lo) / max(hi - lo, 1e-9) * RES)))
        self._block = False

    def _slid(self, i: int) -> None:
        if self._block:
            return
        lo, hi = self.bounds()
        self._set(lo + (hi - lo) * i / RES)

    def _spun(self, v: float) -> None:
        if self._block:
            return
        self._set(v)

    def _set(self, v: float) -> None:
        ed = self.panel.ed
        t = self.info["type"]
        a = self.info["qposadr"]
        if t == JT.mjJNT_HINGE:
            ed.data.qpos[a] = math.radians(v)
        elif t == JT.mjJNT_SLIDE:
            ed.data.qpos[a] = v
        elif t == JT.mjJNT_BALL:
            e = mx.q2euler(ed.data.qpos[a:a + 4], "XYZ")
            e = np.zeros(3) if e is None else e
            e[self.comp] = math.radians(v)
            ed.data.qpos[a:a + 4] = mx.euler2q(e, "XYZ")
        self.panel.from_ui = True
        ed.forward()
        self.panel.from_ui = False
        lo, hi = self.bounds()
        self._block = True
        self.spin.setValue(v)
        self.slider.setValue(int(round((v - lo) / max(hi - lo, 1e-9) * RES)))
        self._block = False


class JointPanel(QWidget):
    def __init__(self, ed: Editor, parent=None):
        super().__init__(parent)
        self.ed = ed
        self.rows: list[JointRow] = []
        self.from_ui = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        bar = QHBoxLayout()
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter joints…")
        self.filter.setClearButtonEnabled(True)
        self.filter.textChanged.connect(self._filter)
        bar.addWidget(self.filter, 1)
        self.only_sel = QCheckBox("Selected body subtree")
        self.only_sel.toggled.connect(lambda *_: self._filter(self.filter.text()))
        bar.addWidget(self.only_sel)
        b = QPushButton("Reset pose")
        b.setToolTip("Reset all joints to qpos0 (model reference pose)")
        b.clicked.connect(lambda: ed.reset_pose())
        bar.addWidget(b)
        self.keys = QComboBox()
        self.keys.setMinimumWidth(110)
        bar.addWidget(self.keys)
        b = QPushButton("Load key")
        b.clicked.connect(self._load_key)
        bar.addWidget(b)
        b = QPushButton("Save pose as key…")
        b.clicked.connect(self._save_key)
        bar.addWidget(b)
        lay.addLayout(bar)
        # selected joint range editor
        rbar = QHBoxLayout()
        self.sel_label = QLabel("Selected joint: -")
        self.sel_label.setStyleSheet(f"color:{C['dim']}")
        rbar.addWidget(self.sel_label, 1)
        self.lim = QCheckBox("limited")
        self.lo = QDoubleSpinBox()
        self.hi = QDoubleSpinBox()
        for s in (self.lo, self.hi):
            s.setRange(-1e4, 1e4)
            s.setDecimals(3)
            s.setKeyboardTracking(False)
            s.setFixedWidth(90)
        rbar.addWidget(self.lim)
        rbar.addWidget(QLabel("min"))
        rbar.addWidget(self.lo)
        rbar.addWidget(QLabel("max"))
        rbar.addWidget(self.hi)
        self.lo.valueChanged.connect(self._range_edit)
        self.hi.valueChanged.connect(self._range_edit)
        self.lim.toggled.connect(self._limit_edit)
        lay.addLayout(rbar)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Joint", "Type", "Position", "Value"])
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Interactive)
        hh.setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.setColumnWidth(0, 170)
        self.table.setColumnWidth(1, 60)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(24)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.cellClicked.connect(self._row_clicked)
        lay.addWidget(self.table)
        self._sel_joint = None
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self.refresh_values)
        self._timer.start()
        ed.compiled.connect(self._on_compiled)
        ed.selection_changed.connect(self._on_sel)
        self._model = None
        self.rebuild()

    def _on_compiled(self, res) -> None:
        if res.ok:
            self.rebuild()

    def rebuild(self) -> None:
        self.table.setRowCount(0)
        self.rows = []
        infos = self.ed.joints()
        r = 0
        names = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
        for info in infos:
            comps = [0, 1, 2] if info["type"] == JT.mjJNT_BALL else ([-1] if info["type"] != JT.mjJNT_FREE else [])
            if info["type"] == JT.mjJNT_FREE:
                self.table.insertRow(r)
                self._cells(r, info, names[0], None)
                r += 1
                continue
            for c in comps:
                row = JointRow(self, info, c)
                self.table.insertRow(r)
                suffix = f" [{'xyz'[c]}]" if c >= 0 else ""
                self._cells(r, info, names[info["type"]] + suffix, row)
                self.rows.append(row)
                r += 1
        keys = []
        m = self.ed.model
        if m is not None:
            for k in range(m.nkey):
                keys.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_KEY, k) or f"key {k}")
        self.keys.clear()
        self.keys.addItems(keys)
        self.refresh_values()
        self._on_sel(self.ed.selection)
        self._filter(self.filter.text())

    def _cells(self, r: int, info: dict, tname: str, row: JointRow | None) -> None:
        from PySide6.QtWidgets import QTableWidgetItem
        it = QTableWidgetItem(info["label"])
        it.setData(Qt.UserRole, info["elem"])
        it.setFlags(it.flags() & ~Qt.ItemIsEditable)
        self.table.setItem(r, 0, it)
        t = QTableWidgetItem(tname)
        t.setFlags(t.flags() & ~Qt.ItemIsEditable)
        self.table.setItem(r, 1, t)
        if row is not None:
            self.table.setCellWidget(r, 2, row.slider)
            self.table.setCellWidget(r, 3, row.spin)

    def refresh_values(self) -> None:
        if not self.isVisible() or self.ed.model is None:
            return
        for row in self.rows:
            row.refresh()

    def _filter(self, text: str) -> None:
        text = text.strip().lower()
        sub = None
        if self.only_sel.isChecked() and self.ed.selection:
            b = self.ed.selection[-1]
            while b is not None and b.tag != "body":
                b = b.getparent()
            if b is not None:
                sub = set(b.iter())
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 0)
            e = it.data(Qt.UserRole)
            vis = (not text or text in it.text().lower()) and (sub is None or e in sub)
            self.table.setRowHidden(r, not vis)

    def _row_clicked(self, r: int, _c: int) -> None:
        e = self.table.item(r, 0).data(Qt.UserRole)
        if e is not None:
            self.ed.select([e])

    # ------------------------------------------------------------- selected joint
    def _on_sel(self, sel) -> None:
        j = next((e for e in reversed(sel) if e.tag == "joint"), None)
        self._sel_joint = j
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 0)
            if it is not None:
                f = it.font()
                f.setBold(it.data(Qt.UserRole) is j and j is not None)
                it.setFont(f)
        en = j is not None and (self.ed.doc.get(j, "type") or "hinge") in ("hinge", "slide", "ball")
        for w in (self.lo, self.hi, self.lim):
            w.setEnabled(en)
        if not en:
            self.sel_label.setText("Selected joint: - (select a joint to edit its range)")
            for w in (self.lo, self.hi):
                w.blockSignals(True)
                w.setValue(0.0)
                w.blockSignals(False)
            if self.only_sel.isChecked():
                self._filter(self.filter.text())
            return
        jt = self.ed.doc.get(j, "type") or "hinge"
        unit = "m" if jt == "slide" else ("deg" if self.ed.doc.angle_unit == "degree" else "rad")
        self.sel_label.setText(f"Selected joint: <b>{j.get('name') or '(unnamed)'}</b> — range in {unit}")
        self.sel_label.setTextFormat(Qt.RichText)
        r = mx.parse_vec(self.ed.doc.get(j, "range") or "0 0", 2)
        lim = self.ed.doc.get(j, "limited") or "auto"
        auto = self.ed.doc.compiler_attr("autolimits", "true") == "true"
        limited = lim == "true" or (lim == "auto" and auto and r is not None and r[0] < r[1])
        for w, v in ((self.lo, r[0] if r is not None else 0), (self.hi, r[1] if r is not None else 0)):
            w.blockSignals(True)
            w.setValue(float(v))
            w.blockSignals(False)
        self.lim.blockSignals(True)
        self.lim.setChecked(bool(limited))
        self.lim.blockSignals(False)
        if self.only_sel.isChecked():
            self._filter(self.filter.text())

    def _range_edit(self, *_):
        j = self._sel_joint
        if j is None:
            return
        lo, hi = self.lo.value(), self.hi.value()
        if hi < lo:
            return
        self.ed.set_attr(j, "range", mx.fmt_vec([lo, hi]), "Edit joint range", merge_key=f"range:{id(j)}")

    def _limit_edit(self, on: bool) -> None:
        j = self._sel_joint
        if j is None:
            return
        self.ed.set_attr(j, "limited", "true" if on else "false", "Toggle joint limit")

    def _load_key(self) -> None:
        i = self.keys.currentIndex()
        if i >= 0:
            self.ed.reset_pose(i)

    def _save_key(self) -> None:
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "Save keyframe", "Keyframe name:",
                                        text=self.ed.doc.unique_name("key", "pose"))
        if ok and name.strip():
            try:
                ops.add_keyframe(self.ed, name.strip())
            except Exception as ex:  # noqa: BLE001
                log.error("Cannot add keyframe: %s", ex)
