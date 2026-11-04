"""Property editor for the selected element (schema driven, never lossy)."""
from __future__ import annotations

import logging
import math

import mujoco
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QColorDialog, QComboBox, QDoubleSpinBox, QFormLayout, QFrame, QGridLayout,
                               QGroupBox, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QMessageBox,
                               QPushButton, QScrollArea, QToolButton, QVBoxLayout, QWidget)

from ..core import mathx as mx
from ..core.doc import parent_tag
from ..core.editor import Editor
from ..core.schema import (BOOL, ENUM, INT, NAME, ORIENT_KEYS, REAL, REF, RGBA, VEC, Attr, attrs_for)
from .theme import C, no_buttons, icon_for

log = logging.getLogger("mjstudio")
OBJ = mujoco.mjtObj
POSE_TAGS = {"body", "geom", "site", "camera", "light", "inertial", "frame"}
GROUP_ORDER = ["General", "Transform", "Transmission", "Dynamics", "Physics", "Limits", "Contact", "Solver",
               "Visual", "User"]


def _valid(spec: Attr, text: str) -> bool:
    if text == "":
        return True
    if spec.kind == INT:
        try:
            int(text)
            return True
        except ValueError:
            return False
    if spec.kind == REAL:
        return mx.parse_vec(text, 1) is not None
    if spec.kind in (VEC, RGBA):
        v = mx.parse_vec(text)
        if v is None:
            return False
        if spec.n is not None and v.size != spec.n:
            return False
        if spec.nmax is not None and v.size > spec.nmax:
            return False
        return True
    return True


class Spin(QDoubleSpinBox):
    def __init__(self, lo=-1e4, hi=1e4, dec=4, step=0.01, suffix=""):
        super().__init__()
        self.setRange(lo, hi)
        self.setDecimals(dec)
        self.setSingleStep(step)
        self.setKeyboardTracking(False)
        self.setSuffix(suffix)
        no_buttons(self)
        self.setMinimumWidth(60)

    def wheelEvent(self, ev):  # avoid accidental edits while scrolling the panel
        if self.hasFocus():
            super().wheelEvent(ev)
        else:
            ev.ignore()


def _drop_layout(lay) -> None:
    while lay.count():
        it = lay.takeAt(0)
        w = it.widget()
        if w is not None:
            w.hide()
            w.setParent(None)
            w.deleteLater()
        elif it.layout() is not None:
            _drop_layout(it.layout())


class AttrRow:
    """Label + editor + reset button for one attribute."""

    def __init__(self, panel: "PropsPanel", spec: Attr):
        self.panel = panel
        self.spec = spec
        self.label = QLabel(spec.name)
        tip = spec.tip or ""
        self.label.setToolTip(tip)
        self.reset = QToolButton()
        self.reset.setText("×")
        self.reset.setToolTip("Remove explicit value (inherit default)")
        self.reset.setFixedWidth(20)
        self.reset.clicked.connect(lambda: self.commit(None))
        self.color_btn = None
        if spec.kind in (ENUM, BOOL):
            self.w = QComboBox()
            self.choices = list(spec.choices) if spec.kind == ENUM else ["false", "true"]
            self.w.addItem("(inherit)")
            self.w.addItems(self.choices)
            self.w.activated.connect(self._combo)
        elif spec.kind == REF:
            self.w = QComboBox()
            self.w.setEditable(True)
            self.w.setInsertPolicy(QComboBox.NoInsert)
            self.w.lineEdit().editingFinished.connect(self._ref_done)
            self.w.activated.connect(lambda _i: self._ref_done())
        else:
            self.w = QLineEdit()
            self.w.editingFinished.connect(self._line_done)
            self.w.textEdited.connect(self._validate)
            if spec.kind == RGBA:
                self.color_btn = QPushButton()
                self.color_btn.setFixedWidth(26)
                self.color_btn.clicked.connect(self._pick_color)
        self.w.setToolTip(tip)

    def widget(self) -> QWidget:
        box = QWidget()
        h = QHBoxLayout(box)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(3)
        h.addWidget(self.w, 1)
        if self.color_btn is not None:
            h.addWidget(self.color_btn)
        h.addWidget(self.reset)
        return box

    # -- values
    def refresh(self) -> None:
        e = self.panel.elem
        r = self.panel.resolved.get(self.spec.name)
        explicit = self.spec.name in e.attrib
        val = e.get(self.spec.name)
        inh = None
        if r is not None and not explicit:
            inh = r
        elif explicit:
            inh = self.panel.ed.doc.inherited(e, self.spec.name)
        f = self.label.font()
        f.setBold(explicit)
        self.label.setFont(f)
        self.label.setStyleSheet(f"color: {C['text'] if explicit else C['dim']};")
        src = f"inherited from {inh[1]}: {inh[0]}" if inh else "no default"
        self.label.setToolTip((self.spec.tip + "\n" if self.spec.tip else "") + src)
        self.reset.setEnabled(explicit)
        self.w.blockSignals(True)
        if isinstance(self.w, QComboBox) and not self.w.isEditable():
            self.w.setItemText(0, f"(inherit: {inh[0]})" if inh else "(unset)")
            self.w.setCurrentIndex(self.choices.index(val) + 1 if val in self.choices else 0)
        elif isinstance(self.w, QComboBox):
            names = self.panel.ref_names(self.spec)
            cur = val or ""
            self.w.clear()
            self.w.addItems([""] + names)
            self.w.setCurrentText(cur)
            self.w.lineEdit().setPlaceholderText(f"{inh[0]}" if inh else "")
        else:
            self.w.setText(val or "")
            self.w.setPlaceholderText(inh[0] if inh else "")
            self.w.setProperty("invalid", False)
            self.w.style().polish(self.w)
            if self.color_btn is not None:
                rgba = mx.parse_vec(val or (inh[0] if inh else "0.5 0.5 0.5 1"), 4)
                if rgba is not None:
                    c = QColor.fromRgbF(*[float(np.clip(x, 0, 1)) for x in rgba[:3]])
                    self.color_btn.setStyleSheet(f"background: {c.name()}; border: 1px solid {C['border']};")
        self.w.blockSignals(False)

    def _validate(self, text: str) -> None:
        self.w.setProperty("invalid", not _valid(self.spec, text.strip()))
        self.w.style().polish(self.w)

    def _line_done(self) -> None:
        text = self.w.text().strip()
        if not _valid(self.spec, text):
            return
        self.commit(text or None)

    def _combo(self, idx: int) -> None:
        self.commit(None if idx == 0 else self.choices[idx - 1])

    def _ref_done(self) -> None:
        self.commit(self.w.currentText().strip() or None)

    def _pick_color(self) -> None:
        e = self.panel.elem
        cur = mx.parse_vec(e.get(self.spec.name) or self.w.placeholderText() or "0.5 0.5 0.5 1", 4)
        if cur is None:
            cur = np.array([0.5, 0.5, 0.5, 1.0])
        c = QColorDialog.getColor(QColor.fromRgbF(*[float(np.clip(x, 0, 1)) for x in cur]), self.panel,
                                  "Colour", QColorDialog.ShowAlphaChannel)
        if c.isValid():
            self.commit(mx.fmt_vec([c.redF(), c.greenF(), c.blueF(), c.alphaF()], 4))

    def commit(self, value: str | None) -> None:
        e = self.panel.elem
        if e is None or e.get(self.spec.name) == value:
            return
        self.panel.apply(lambda: self.panel.ed.set_attr(e, self.spec.name, value))


class PropsPanel(QScrollArea):
    def __init__(self, ed: Editor, parent=None):
        super().__init__(parent)
        self.ed = ed
        self.elem = None
        self.rows: list[AttrRow] = []
        self.resolved: dict = {}
        self._self_edit = False
        self.setWidgetResizable(True)
        self.body = QWidget()
        self.setWidget(self.body)
        self.lay = QVBoxLayout(self.body)
        self.lay.setContentsMargins(6, 6, 6, 6)
        self._live = QTimer(self)
        self._live.setInterval(200)
        self._live.timeout.connect(self._update_info)
        ed.selection_changed.connect(self._on_sel)
        ed.doc_changed.connect(self._on_doc)
        ed.compiled.connect(lambda *_: self._update_info())
        ed.doc_loaded.connect(lambda: self.set_elem(None))
        self.set_elem(None)

    # ------------------------------------------------------------- plumbing
    def apply(self, fn) -> None:
        self._self_edit = True
        try:
            fn()
        except Exception as ex:  # noqa: BLE001
            QMessageBox.warning(self, "Edit failed", str(ex))
            log.error("Edit failed: %s", ex)
        finally:
            self._self_edit = False
        self.refresh()

    def ref_names(self, spec: Attr) -> list[str]:
        kind = spec.ref or ""
        if kind.startswith("@") and self.elem is not None:
            t = self.elem.get(kind[1:]) or ""
            kind = "body" if t in ("body", "xbody") else t
        return self.ed.doc.names(kind) if kind else []

    def _on_sel(self, sel) -> None:
        self.set_elem(sel[-1] if sel else None)

    def _on_doc(self, cmd, action) -> None:
        if self.elem is None:
            return
        root = self.ed.doc.root
        if self.elem is not root and root not in list(self.elem.iterancestors()):
            self.set_elem(None)
            return
        if self._self_edit and action == "do":
            return  # refresh happens in apply()
        self.refresh()

    # ------------------------------------------------------------- building
    def _clear(self) -> None:
        while self.lay.count():
            it = self.lay.takeAt(0)
            w = it.widget()
            if w is not None:
                w.hide()              # remove from view now; deleteLater only runs after the event loop
                w.setParent(None)
                w.deleteLater()
            elif it.layout() is not None:
                _drop_layout(it.layout())
        self.rows = []
        self.tf = None

    def set_elem(self, e) -> None:
        self._clear()
        self.elem = e
        if e is None:
            lbl = QLabel("Nothing selected.\n\nSelect an element in the hierarchy or click in the viewport.")
            lbl.setStyleSheet(f"color: {C['dim']};")
            lbl.setWordWrap(True)
            self.lay.addWidget(lbl)
            self.lay.addStretch(1)
            self._live.stop()
            return
        doc = self.ed.doc
        self.resolved = doc.resolved(e)
        ptag = parent_tag(e)
        # header
        head = QHBoxLayout()
        ic = QLabel()
        ic.setPixmap(icon_for(e.tag, ptag).pixmap(18, 18))
        head.addWidget(ic)
        title = QLabel(f"<b>{e.tag}</b>" + (f"  <span style='color:{C['dim']}'>in {ptag}</span>" if ptag else ""))
        head.addWidget(title, 1)
        self.lay.addLayout(head)
        specs = attrs_for(e.tag, ptag)
        # name editor
        if any(s.kind == NAME for s in specs) or e.tag == "default":
            key = "class" if e.tag == "default" else "name"
            row = QHBoxLayout()
            row.addWidget(QLabel("Name" if key == "name" else "Class"))
            self.name_edit = QLineEdit(e.get(key) or "")
            self.name_edit.setPlaceholderText("(unnamed)")
            self.name_edit.editingFinished.connect(self._rename)
            row.addWidget(self.name_edit, 1)
            self.lay.addLayout(row)
        # transform
        if e.tag in POSE_TAGS or e.tag == "joint":
            self.tf = TransformBox(self, e)
            self.lay.addWidget(self.tf)
        # schema groups
        known = set()
        groups: dict[str, list[Attr]] = {}
        for s in specs:
            known.add(s.name)
            if s.kind == NAME or s.name in ("pos",) + ORIENT_KEYS + ("fromto", "axis", "dir"):
                if not (s.name in ("pos", "axis", "dir") and self.tf is None):
                    continue
            groups.setdefault(s.group, []).append(s)
        for g in sorted(groups, key=lambda x: GROUP_ORDER.index(x) if x in GROUP_ORDER else 99):
            box = QGroupBox(g)
            form = QGridLayout(box)
            form.setContentsMargins(6, 4, 6, 4)
            form.setVerticalSpacing(3)
            for r, s in enumerate(groups[g]):
                row = AttrRow(self, s)
                if s.angle:
                    row.label.setText(f"{s.name} ({'deg' if doc.angle_unit == 'degree' else 'rad'})")
                form.addWidget(row.label, r, 0)
                form.addWidget(row.widget(), r, 1)
                self.rows.append(row)
            self.lay.addWidget(box)
        # orientation / fromto raw specifiers (advanced)
        if e.tag in POSE_TAGS:
            adv = [Attr(k, VEC, None, group="Transform") for k in ("fromto",) + ORIENT_KEYS
                   if any(s.name == k for s in specs)]
            if adv:
                box = QGroupBox("Raw transform attributes")
                form = QGridLayout(box)
                form.setContentsMargins(6, 4, 6, 4)
                for r, s in enumerate(adv):
                    spec = next(x for x in specs if x.name == s.name)
                    row = AttrRow(self, spec)
                    form.addWidget(row.label, r, 0)
                    form.addWidget(row.widget(), r, 1)
                    self.rows.append(row)
                self.lay.addWidget(box)
        # unknown attributes (kept verbatim)
        extra = [k for k in e.attrib if k not in known and k not in ("name",) and not (e.tag == "default" and k == "class")]
        box = QGroupBox("Other attributes")
        v = QVBoxLayout(box)
        grid = QGridLayout()
        for r, k in enumerate(extra):
            row = AttrRow(self, Attr(k))
            grid.addWidget(row.label, r, 0)
            grid.addWidget(row.widget(), r, 1)
            self.rows.append(row)
        v.addLayout(grid)
        add = QPushButton("+ Add attribute…")
        add.clicked.connect(self._add_attr)
        v.addWidget(add)
        self.lay.addWidget(box)
        # compiled info
        self.info = QLabel()
        self.info.setTextFormat(Qt.RichText)
        self.info.setWordWrap(True)
        self.info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        ibox = QGroupBox("Compiled (read-only)")
        il = QVBoxLayout(ibox)
        il.addWidget(self.info)
        self.lay.addWidget(ibox)
        self.lay.addStretch(1)
        self.refresh()
        self._live.start()

    def refresh(self) -> None:
        e = self.elem
        if e is None:
            return
        self.resolved = self.ed.doc.resolved(e)
        for r in self.rows:
            r.refresh()
        if self.tf is not None and not self._self_edit:
            self.tf.refresh()
        if hasattr(self, "name_edit"):
            key = "class" if e.tag == "default" else "name"
            if not self.name_edit.hasFocus():
                self.name_edit.setText(e.get(key) or "")
        self._update_info()

    def _rename(self) -> None:
        e = self.elem
        if e is None:
            return
        new = self.name_edit.text().strip()
        key = "class" if e.tag == "default" else "name"
        if new == (e.get(key) or ""):
            return
        self.apply(lambda: self.ed.rename(e, new))
        self.name_edit.setText(e.get(key) or "")

    def _add_attr(self) -> None:
        e = self.elem
        k, ok = QInputDialog.getText(self, "Add attribute", "Attribute name:")
        if not ok or not k.strip():
            return
        v, ok = QInputDialog.getText(self, "Add attribute", f"Value for '{k.strip()}':")
        if not ok:
            return
        self.apply(lambda: self.ed.set_attr(e, k.strip(), v))
        self.set_elem(e)

    # ------------------------------------------------------------- compiled info
    def _update_info(self) -> None:
        e = self.elem
        if e is None or not hasattr(self, "info"):
            return
        try:
            self.info.setText(self._info_html(e))
        except RuntimeError:
            pass

    def _info_html(self, e) -> str:
        m, d, im = self.ed.model, self.ed.data, self.ed.idmap
        if m is None or im is None:
            return "<i>no compiled model</i>"
        o = im.obj(e)
        if o is None:
            return "<i>not a compiled object (or not mapped)</i>"
        ot, i = o
        fv = lambda v, p=4: mx.fmt_vec(np.round(v, p), 6)  # noqa: E731
        rows = [("id", str(i))]
        if ot == int(OBJ.mjOBJ_BODY):
            rows += [("mass", f"{m.body_mass[i]:.6g} kg"), ("subtree mass", f"{m.body_subtreemass[i]:.6g} kg"),
                     ("inertia (diag)", fv(m.body_inertia[i], 6)), ("world pos", fv(d.xpos[i])),
                     ("COM (world)", fv(d.xipos[i])), ("dofs", str(int(m.body_dofnum[i])))]
        elif ot == int(OBJ.mjOBJ_JOINT):
            jt = int(m.jnt_type[i])
            a = m.jnt_qposadr[i]
            names = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
            if jt == 3:
                q = f"{math.degrees(d.qpos[a]):.3f}°  ({d.qpos[a]:.5f} rad)"
                rng = f"{math.degrees(m.jnt_range[i][0]):.2f}° … {math.degrees(m.jnt_range[i][1]):.2f}°"
            elif jt == 2:
                q, rng = f"{d.qpos[a]:.5f} m", f"{m.jnt_range[i][0]:.4f} … {m.jnt_range[i][1]:.4f} m"
            else:
                n = 7 if jt == 0 else 4
                q, rng = fv(d.qpos[a:a + n]), (f"{math.degrees(m.jnt_range[i][1]):.1f}°" if jt == 1 else "-")
            rows += [("type", names[jt]), ("qpos", q), ("range", rng + ("" if m.jnt_limited[i] else " (unlimited)")),
                     ("qposadr / dofadr", f"{a} / {m.jnt_dofadr[i]}"), ("anchor (world)", fv(d.xanchor[i])),
                     ("axis (world)", fv(d.xaxis[i]))]
        elif ot == int(OBJ.mjOBJ_GEOM):
            coll = bool(m.geom_contype[i] or m.geom_conaffinity[i])
            rows += [("type", mujoco.mjtGeom(m.geom_type[i]).name[7:].lower()), ("size", fv(m.geom_size[i])),
                     ("world pos", fv(d.geom_xpos[i])), ("collision", "yes" if coll else "no (visual only)"),
                     ("body", mujoco.mj_id2name(m, OBJ.mjOBJ_BODY, m.geom_bodyid[i]) or str(m.geom_bodyid[i])),
                     ("bounding radius", f"{m.geom_rbound[i]:.4g}")]
        elif ot == int(OBJ.mjOBJ_SITE):
            rows += [("world pos", fv(d.site_xpos[i])), ("size", fv(m.site_size[i]))]
        elif ot == int(OBJ.mjOBJ_ACTUATOR):
            rows += [("ctrl", f"{d.ctrl[i]:.4g}"), ("ctrlrange", fv(m.actuator_ctrlrange[i])),
                     ("force", f"{d.actuator_force[i]:.4g}"), ("gear", fv(m.actuator_gear[i][:3]))]
        elif ot == int(OBJ.mjOBJ_SENSOR):
            a, n = m.sensor_adr[i], m.sensor_dim[i]
            rows += [("dim", str(n)), ("value", fv(d.sensordata[a:a + n], 5))]
        elif ot == int(OBJ.mjOBJ_CAMERA):
            rows += [("world pos", fv(d.cam_xpos[i])), ("fovy", f"{m.cam_fovy[i]:.1f}°")]
        return "<table>" + "".join(
            f"<tr><td style='color:{C['dim']};padding-right:8px'>{k}</td><td>{v}</td></tr>" for k, v in rows) + "</table>"


class TransformBox(QGroupBox):
    """Position + rotation (Euler, degrees) editor; joints get position + axis."""

    def __init__(self, panel: PropsPanel, e):
        super().__init__("Transform (parent frame)")
        self.panel, self.e = panel, e
        doc = panel.ed.doc
        g = QGridLayout(self)
        g.setContentsMargins(6, 4, 6, 4)
        self.pos = [Spin() for _ in range(3)]
        self.rot = [Spin(-360, 360, 3, 1.0) for _ in range(3)]
        seq = doc.eulerseq if (doc.eulerseq.islower() or doc.eulerseq.isupper()) else "xyz"
        self.seq = seq
        g.addWidget(QLabel("Position"), 0, 0)
        for k, s in enumerate(self.pos):
            g.addWidget(s, 0, k + 1)
            s.valueChanged.connect(self._changed)
        if e.tag == "joint":
            g.addWidget(QLabel("Axis"), 1, 0)
        elif e.tag == "light":
            g.addWidget(QLabel("Direction"), 1, 0)
        else:
            g.addWidget(QLabel(f"Rotation°\n(euler {seq})"), 1, 0)
        for k, s in enumerate(self.rot):
            if e.tag in ("joint", "light"):
                s.setRange(-1e3, 1e3)
                s.setDecimals(4)
                s.setSingleStep(0.1)
            g.addWidget(s, 1, k + 1)
            s.valueChanged.connect(self._changed)
        if doc.fromto_active(e) is not None:
            note = QLabel("frame defined by 'fromto' (rotation about the segment is ignored)")
            note.setStyleSheet(f"color:{C['dim']}")
            note.setWordWrap(True)
            g.addWidget(note, 2, 0, 1, 4)
        self._block = False
        self.refresh()

    def refresh(self) -> None:
        doc = self.panel.ed.doc
        e = self.e
        self._block = True
        try:
            if e.tag == "joint":
                p = mx.parse_vec(doc.get(e, "pos") or "0 0 0", 3)
                a = mx.parse_vec(doc.get(e, "axis") or "0 0 1", 3)
                vals = (p if p is not None else np.zeros(3), a if a is not None else np.array([0, 0, 1.0]))
                for s, v in zip(self.pos, vals[0]):
                    s.setValue(float(v))
                for s, v in zip(self.rot, vals[1]):
                    s.setValue(float(v))
            elif e.tag == "light":
                p = mx.parse_vec(doc.get(e, "pos") or "0 0 0", 3)
                dv = mx.parse_vec(doc.get(e, "dir") or "0 0 -1", 3)
                for s, v in zip(self.pos, p if p is not None else np.zeros(3)):
                    s.setValue(float(v))
                for s, v in zip(self.rot, dv if dv is not None else [0, 0, -1]):
                    s.setValue(float(v))
            else:
                p, q = doc.local_pose(e)
                eu = mx.q2euler(q, self.seq)
                for s, v in zip(self.pos, p):
                    s.setValue(float(v))
                for s, v in zip(self.rot, eu if eu is not None else np.zeros(3)):
                    s.setValue(math.degrees(float(v)))
        finally:
            self._block = False

    def _changed(self, *_):
        if self._block:
            return
        ed, e = self.panel.ed, self.e
        p = [s.value() for s in self.pos]
        r = [s.value() for s in self.rot]
        key = f"tf:{id(e)}"
        if e.tag == "joint":
            self.panel.apply(lambda: ed.set_attrs(e, {"pos": mx.fmt_vec(p), "axis": mx.fmt_vec(r)},
                                                  "Edit joint frame", merge_key=key))
        elif e.tag == "light":
            self.panel.apply(lambda: ed.set_attrs(e, {"pos": mx.fmt_vec(p), "dir": mx.fmt_vec(r)},
                                                  "Edit light", merge_key=key))
        else:
            q = mx.euler2q([math.radians(x) for x in r], self.seq)
            self.panel.apply(lambda: ed.set_pose(e, p, q, "Edit transform", merge_key=key))
