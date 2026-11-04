"""Main window: menus, toolbars, docks and file handling."""
from __future__ import annotations

import logging
import os
from pathlib import Path

import mujoco
from lxml import etree
from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QGuiApplication, QKeySequence
from PySide6.QtWidgets import (QApplication, QDockWidget, QFileDialog, QInputDialog, QLabel, QMainWindow, QMenu,
                               QMessageBox, QToolBar)

from .. import __version__
from ..core import ops
from ..core.compiler import compile_doc, validate_steps
from ..core.doc import MjcfError
from ..core.editor import Editor
from ..core.sim import Simulator
from .jointpanel import JointPanel
from .logpanel import LogPanel
from .props import PropsPanel
from .simpanel import SimPanel
from .theme import C
from .tree import TreePanel
from .viewport import Viewport

log = logging.getLogger("mjstudio")
VF = mujoco.mjtVisFlag
RF = mujoco.mjtRndFlag
MJCF_FILTER = "MuJoCo MJCF (*.xml *.mjcf);;All files (*)"


class MainWindow(QMainWindow):
    def __init__(self, ed: Editor | None = None):
        super().__init__()
        self.settings = QSettings("mjstudio", "MuJoCoModelStudio")
        self.log_panel = LogPanel()  # first, so it receives early log records
        self.ed = ed or Editor()
        self.sim = Simulator(self.ed)
        self.protected = False  # True while the open file is an original that was never saved by us
        self.view = Viewport(self.ed)
        self.view.sim = self.sim
        self.setCentralWidget(self.view)
        self.tree = TreePanel(self.ed)
        self.props = PropsPanel(self.ed)
        self.joints = JointPanel(self.ed)
        self.simp = SimPanel(self.ed, self.sim)
        dt = self._dock("Hierarchy", self.tree, Qt.LeftDockWidgetArea)
        dp = self._dock("Properties", self.props, Qt.RightDockWidgetArea)
        self.resizeDocks([dt, dp], [330, 340], Qt.Horizontal)
        dj = self._dock("Joints", self.joints, Qt.BottomDockWidgetArea)
        ds = self._dock("Simulation", self.simp, Qt.BottomDockWidgetArea)
        dl = self._dock("Log", self.log_panel, Qt.BottomDockWidgetArea)
        self.tabifyDockWidget(dj, ds)
        self.tabifyDockWidget(ds, dl)
        dj.raise_()
        self.status_compile = QLabel()
        self.statusBar().addPermanentWidget(self.status_compile)
        self._build_actions()
        self.view.picked.connect(self._picked)
        self.view.status.connect(lambda t: self.statusBar().showMessage(t, 3000))
        self.tree.contextRequested.connect(self._context_menu)
        self.ed.compiled.connect(self._on_compiled)
        self.ed.doc_changed.connect(lambda *_: self._update_title())
        self.ed.doc_loaded.connect(self._update_title)
        self.ed.undo.changed.connect(lambda *_: self._update_undo())
        self.sim.on_issue = lambda msg: self.statusBar().showMessage(f"Physics instability: {msg}", 8000)
        self.timer = QTimer(self)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self._tick)
        self.timer.start()
        self.resize(1600, 980)
        geo = self.settings.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        if self.ed.result is not None:
            self._on_compiled(self.ed.result)
        self._update_title()
        self._update_undo()
        log.info("MuJoCo Model Studio %s — MuJoCo %s", __version__, mujoco.__version__)

    def _dock(self, title, w, area) -> QDockWidget:
        d = QDockWidget(title, self)
        d.setObjectName(title)
        d.setWidget(w)
        self.addDockWidget(area, d)
        return d

    # ================================================================ actions
    def _act(self, menu, text, slot=None, key=None, checkable=False, checked=False, tip=None) -> QAction:
        a = QAction(text, self)
        if key:
            a.setShortcut(QKeySequence(key))
        if checkable:
            a.setCheckable(True)
            a.setChecked(checked)
            if slot:
                a.toggled.connect(slot)
        elif slot:
            a.triggered.connect(lambda *_: slot())
        if tip:
            a.setToolTip(tip)
            a.setStatusTip(tip)
        if menu is not None:
            menu.addAction(a)
        self.addAction(a)
        return a

    def _build_actions(self) -> None:
        mb = self.menuBar()
        # ---- File
        f = mb.addMenu("&File")
        self._act(f, "&New model", self.file_new, "Ctrl+N")
        self._act(f, "&Open…", self.file_open, "Ctrl+O")
        self.recent_menu = f.addMenu("Open &recent")
        self._fill_recent()
        f.addSeparator()
        self._act(f, "&Save", self.file_save, "Ctrl+S")
        self._act(f, "Save &As…", self.file_save_as, "Ctrl+Shift+S")
        self._act(f, "Export compiled MJCF (as MuJoCo sees it)…", self.export_compiled)
        f.addSeparator()
        self._act(f, "&Validate model (compile + 200 steps)", self.validate, "F7")
        f.addSeparator()
        self._act(f, "&Quit", self.close, "Ctrl+Q")
        # ---- Edit
        e = mb.addMenu("&Edit")
        self.a_undo = self._act(e, "&Undo", self.ed.undo_step, QKeySequence.Undo)
        self.a_redo = self._act(e, "&Redo", self.ed.redo_step, QKeySequence.Redo)
        e.addSeparator()
        self._act(e, "&Delete", self.delete_sel, QKeySequence.Delete)
        self._act(e, "D&uplicate", self.duplicate_sel, "Ctrl+D")
        self._act(e, "&Mirror duplicate (left ↔ right)", self.mirror_sel, "Ctrl+Shift+M")
        self._act(e, "Re&name…", self.rename_sel, "F2")
        self._act(e, "Select &parent", self.select_parent, "Alt+Up")
        # ---- Model
        mm = mb.addMenu("&Model")
        self._act(mm, "Add &body", lambda: self._add(lambda p: ops.add_body(self.ed, p)), "Ctrl+B")
        g = mm.addMenu("Add &geom")
        for t in ("sphere", "box", "capsule", "cylinder", "ellipsoid", "plane"):
            self._act(g, t, lambda t=t: self._add(lambda p: ops.add_geom(self.ed, p, t)))
        self._act(mm, "Add &site", lambda: self._add(lambda p: ops.add_site(self.ed, p)))
        self._act(mm, "Add &camera", lambda: self._add(lambda p: ops.add_camera(self.ed, p)))
        self._act(mm, "Add &light", lambda: self._add(lambda p: ops.add_light(self.ed, p)))
        mm.addSeparator()
        self._act(mm, "&Recompile", self.ed.recompile, "F5")
        # ---- Joint
        j = mb.addMenu("&Joint")
        for t in ("hinge", "slide", "ball", "free"):
            self._act(j, f"Add {t} joint", lambda t=t: self._add_joint(t))
        j.addSeparator()
        self._act(j, "Add position actuator to joint", lambda: self._add_actuator("position"))
        self._act(j, "Add motor to joint", lambda: self._add_actuator("motor"))
        j.addSeparator()
        self._act(j, "Reset pose (qpos0)", self.ed.reset_pose, "Ctrl+R")
        self._act(j, "Save pose as keyframe…", self.joints._save_key)
        # ---- Sensor
        s = mb.addMenu("Se&nsor")
        for t in ("jointpos", "jointvel"):
            self._act(s, f"Add {t} (selected joint)", lambda t=t: self._add_sensor(t, "joint"))
        s.addSeparator()
        for t in ("touch", "accelerometer", "gyro", "velocimeter", "force", "torque", "framepos", "framequat"):
            self._act(s, f"Add {t} (selected site)", lambda t=t: self._add_sensor(t, "site"))
        # ---- Simulation
        sm = mb.addMenu("&Simulation")
        self.a_play = self._act(sm, "&Play / Pause", self.toggle_sim, "Space")
        self._act(sm, "&Step", lambda: self.sim.step(1), ".")
        self._act(sm, "&Reset", lambda: self.sim.reset(), "Backspace")
        # ---- View
        v = mb.addMenu("&View")
        self._act(v, "Show visual geoms", lambda on: self._setv("show_visual", on), checkable=True, checked=True)
        self._act(v, "Show collision geoms", lambda on: self._setv("show_collision", on), checkable=True, checked=True)
        v.addSeparator()
        for name, flag, on in (("Joints", VF.mjVIS_JOINT, False), ("Contact points", VF.mjVIS_CONTACTPOINT, False),
                               ("Contact forces", VF.mjVIS_CONTACTFORCE, False), ("Center of mass", VF.mjVIS_COM, False),
                               ("Inertia boxes", VF.mjVIS_INERTIA, False), ("Convex hulls", VF.mjVIS_CONVEXHULL, False),
                               ("Transparent", VF.mjVIS_TRANSPARENT, False), ("Actuators", VF.mjVIS_ACTUATOR, False),
                               ("Tendons", VF.mjVIS_TENDON, True), ("Skins", VF.mjVIS_SKIN, True)):
            self._act(v, name, lambda c, flag=flag: self._visflag(flag, c), checkable=True, checked=on)
        v.addSeparator()
        for name, flag, on in (("Wireframe", RF.mjRND_WIREFRAME, False), ("Shadows", RF.mjRND_SHADOW, True),
                               ("Reflections", RF.mjRND_REFLECTION, True), ("Skybox", RF.mjRND_SKYBOX, True)):
            self._act(v, name, lambda c, flag=flag: self._rndflag(flag, c), checkable=True, checked=on)
        gg = v.addMenu("Geom groups")
        for i in range(6):
            self._act(gg, f"Group {i}", lambda c, i=i: self._group("geomgroup", i, c), checkable=True,
                      checked=bool(self.view.opt.geomgroup[i]))
        sg = v.addMenu("Site groups")
        for i in range(6):
            self._act(sg, f"Group {i}", lambda c, i=i: self._group("sitegroup", i, c), checkable=True,
                      checked=bool(self.view.opt.sitegroup[i]))
        v.addSeparator()
        self._act(v, "Frame selection", self.view.frame_selection, "F")
        self._act(v, "Reset camera", self.view.reset_camera, "Home")
        # ---- tools (gizmo) + toolbar
        tb = QToolBar("Main")
        tb.setObjectName("maintb")
        self.addToolBar(tb)
        for txt, slot, tip in (("New", self.file_new, "New model"), ("Open", self.file_open, "Open MJCF"),
                               ("Save", self.file_save, "Save")):
            a = QAction(txt, self)
            a.setToolTip(tip)
            a.triggered.connect(lambda *_, s=slot: s())
            tb.addAction(a)
        tb.addSeparator()
        tb.addAction(self.a_undo)
        tb.addAction(self.a_redo)
        tb.addSeparator()
        grp = QActionGroup(self)
        self.gizmo_acts = {}
        for txt, mode, key in (("Select", "none", "Q"), ("Move", "translate", "W"), ("Rotate", "rotate", "E")):
            a = self._act(None, txt, lambda c, mode=mode: c and self._gizmo(mode), key=key, checkable=True,
                          checked=(mode == "translate"))
            a.setToolTip(f"{txt} tool ({key})")
            grp.addAction(a)
            tb.addAction(a)
            self.gizmo_acts[mode] = a
        self.a_local = self._act(None, "Local", lambda c: self._space(c), "L", checkable=True,
                                 tip="Gizmo in local (element) axes instead of world axes (L)")
        tb.addAction(self.a_local)
        pick = QActionGroup(self)
        for txt, mode in (("Pick body", "body"), ("Pick geom", "geom")):
            a = self._act(None, txt, lambda c, mode=mode: c and setattr(self.view, "pick_mode", mode),
                          checkable=True, checked=(mode == "body"))
            pick.addAction(a)
            tb.addAction(a)
        tb.addSeparator()
        a = QAction("Visual", self, checkable=True, checked=True)
        a.toggled.connect(lambda on: self._setv("show_visual", on))
        tb.addAction(a)
        a = QAction("Collision", self, checkable=True, checked=True)
        a.toggled.connect(lambda on: self._setv("show_collision", on))
        tb.addAction(a)
        tb.addSeparator()
        self.tb_play = QAction("▶ Play", self)
        self.tb_play.triggered.connect(self.toggle_sim)
        tb.addAction(self.tb_play)
        a = QAction("Reset", self)
        a.triggered.connect(lambda: self.sim.reset())
        tb.addAction(a)
        tb.addAction(self._act(None, "Validate", self.validate))
        # ---- Help
        h = mb.addMenu("&Help")
        self._act(h, "Viewport controls", self.show_controls, "F1")
        self._act(h, "About", self.about)

    # ================================================================ helpers
    def _setv(self, attr, on):
        setattr(self.view, attr, on)
        self.view.update()

    def _visflag(self, flag, on):
        self.view.opt.flags[flag] = 1 if on else 0
        self.view.update()

    def _rndflag(self, flag, on):
        if self.view.scn is not None:
            self.view.scn.flags[flag] = 1 if on else 0
        self.view.update()

    def _group(self, field, i, on):
        getattr(self.view.opt, field)[i] = 1 if on else 0
        self.view.update()

    def _gizmo(self, mode):
        self.view.gizmo.mode = mode
        self.view.update()

    def _space(self, local: bool):
        self.view.gizmo.space = "local" if local else "world"
        self.view.update()

    def _picked(self, e, add):
        self.ed.select(e if e is not None else [], add=add)

    def _target(self):
        return self.ed.selection[-1] if self.ed.selection else None

    def _guard(self, fn):
        try:
            return fn()
        except (ValueError, KeyError, MjcfError) as ex:
            QMessageBox.warning(self, "Operation failed", str(ex))
            log.error("%s", ex)
            return None

    def _add(self, fn):
        e = self._guard(lambda: fn(self._target()))
        if e is not None:
            self.ed.select([e])

    def _add_joint(self, jtype):
        t = self._target()
        body = t
        while body is not None and body.tag != "body":
            body = body.getparent()
        if body is None:
            QMessageBox.information(self, "Add joint", "Select a body (or an element inside a body) first.")
            return
        e = self._guard(lambda: ops.add_joint(self.ed, body, jtype))
        if e is not None:
            self.ed.select([e])

    def _add_actuator(self, kind):
        t = self._target()
        if t is None or t.tag != "joint":
            QMessageBox.information(self, "Add actuator", "Select a named joint first.")
            return
        e = self._guard(lambda: ops.add_actuator(self.ed, t, kind))
        if e is not None:
            self.ed.select([e])

    def _add_sensor(self, kind, need):
        t = self._target()
        if t is None or t.tag != need or not t.get("name"):
            QMessageBox.information(self, "Add sensor", f"Select a named {need} first.")
            return
        kw = {need: t.get("name")}
        if kind.startswith("frame"):
            kw = {"objtype": "site", "objname": t.get("name")}
        e = self._guard(lambda: ops.add_sensor(self.ed, kind, **kw))
        if e is not None:
            self.ed.select([e])

    def delete_sel(self):
        sel = [e for e in self.ed.selection if e.getparent() is not None and e.tag not in ("worldbody",)]
        if not sel:
            return
        n = sum(1 for e in sel for _ in e.iter())
        if n > 20:
            r = QMessageBox.warning(
                self, "Delete elements",
                f"This deletes {len(sel)} element(s) containing {n} XML nodes in total, plus actuators/sensors\n"
                "that reference them. You can undo this with Ctrl+Z.\n\nContinue?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if r != QMessageBox.Yes:
                return
        self._guard(lambda: self.ed.delete(sel))

    def duplicate_sel(self):
        sel = [e for e in self.ed.selection if e.getparent() is not None]
        out = self._guard(lambda: ops.duplicate(self.ed, sel)) if sel else None
        if out:
            self.ed.select([o for o in out if o.getparent() is not None and o.getparent().tag in ("body", "worldbody")][:1] or out[:1])

    def mirror_sel(self):
        t = self._target()
        if t is None or t.tag != "body":
            QMessageBox.information(self, "Mirror", "Select a body to mirror across its parent's XZ plane (y → -y).")
            return
        out = self._guard(lambda: ops.mirror_duplicate(self.ed, t))
        if out:
            self.ed.select([out[0]])

    def rename_sel(self):
        t = self._target()
        if t is None:
            return
        key = "class" if t.tag == "default" else "name"
        name, ok = QInputDialog.getText(self, "Rename", f"New name for {t.tag}:", text=t.get(key) or "")
        if ok:
            self._guard(lambda: self.ed.rename(t, name))

    def select_parent(self):
        t = self._target()
        if t is not None and t.getparent() is not None and t.getparent().tag not in ("worldbody", "mujoco"):
            self.ed.select([t.getparent()])

    def toggle_sim(self):
        self.sim.toggle()
        self.simp.sync_play()
        self.tb_play.setText("❚❚ Pause" if self.sim.running else "▶ Play")

    def _tick(self):
        if self.sim.running:
            self.sim.tick()
            if not self.sim.running:  # paused because of an instability
                self.simp.sync_play()
                self.tb_play.setText("▶ Play")

    # ================================================================ context menu
    def _context_menu(self, e, gpos):
        menu = QMenu(self)
        if e is not None and e.tag not in ("worldbody",):
            self.ed.select([e]) if e not in self.ed.selection else None
        tgt = e
        add = menu.addMenu("Add")
        add.addAction("Body", lambda: self._add(lambda p: ops.add_body(self.ed, tgt)))
        gm = add.addMenu("Geom")
        for t in ("sphere", "box", "capsule", "cylinder", "ellipsoid"):
            gm.addAction(t, lambda t=t: self._add(lambda p: ops.add_geom(self.ed, tgt, t)))
        jm = add.addMenu("Joint")
        for t in ("hinge", "slide", "ball", "free"):
            jm.addAction(t, lambda t=t: self._add_joint(t))
        add.addAction("Site", lambda: self._add(lambda p: ops.add_site(self.ed, tgt)))
        add.addAction("Camera", lambda: self._add(lambda p: ops.add_camera(self.ed, tgt)))
        add.addAction("Light", lambda: self._add(lambda p: ops.add_light(self.ed, tgt)))
        if e is not None and e.getparent() is not None and e.tag != "worldbody":
            menu.addSeparator()
            menu.addAction("Duplicate", self.duplicate_sel)
            if e.tag == "body":
                menu.addAction("Mirror duplicate (L↔R)", self.mirror_sel)
            menu.addAction("Rename…", self.rename_sel)
            menu.addAction("Copy XML", lambda: QGuiApplication.clipboard().setText(
                etree.tostring(e, pretty_print=True, encoding="unicode")))
            menu.addAction("Frame in viewport", self.view.frame_selection)
            menu.addSeparator()
            menu.addAction("Delete", self.delete_sel)
        menu.exec(gpos)

    # ================================================================ files
    def _confirm_discard(self) -> bool:
        if not self.ed.dirty:
            return True
        r = QMessageBox.question(self, "Unsaved changes", "The model has unsaved changes. Save them first?",
                                 QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Save)
        if r == QMessageBox.Cancel:
            return False
        if r == QMessageBox.Save:
            return self.file_save()
        return True

    def file_new(self):
        if not self._confirm_discard():
            return
        self.sim.pause()
        self.ed.new("new_model")
        self.protected = False
        self.view.reset_camera()

    def file_open(self):
        if not self._confirm_discard():
            return
        start = str(self.ed.doc.base_dir) if self.ed.doc.path else os.getcwd()
        path, _ = QFileDialog.getOpenFileName(self, "Open MJCF model", start, MJCF_FILTER)
        if path:
            self.open_path(path)

    def open_path(self, path) -> bool:
        self.sim.pause()
        try:
            self.ed.open(path)
        except (MjcfError, OSError) as ex:
            QMessageBox.critical(self, "Cannot open", str(ex))
            log.error("Cannot open %s: %s", path, ex)
            return False
        self.protected = True
        self._push_recent(str(Path(path).resolve()))
        self.view.reset_camera()
        if self.ed.result is not None and not self.ed.result.ok:
            QMessageBox.warning(self, "Model has errors",
                                "The file was loaded but MuJoCo reports an error:\n\n" +
                                self.ed.result.pretty_error(self.ed.doc))
        return True

    def file_save(self) -> bool:
        if self.ed.doc.path is None:
            return self.file_save_as()
        if self.protected:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Warning)
            box.setWindowTitle("Protect original file")
            box.setText(f"Overwrite the original file?\n\n{self.ed.doc.path}")
            box.setInformativeText("Saving under a new name keeps the original untouched.")
            sa = box.addButton("Save As… (recommended)", QMessageBox.AcceptRole)
            ow = box.addButton("Overwrite original", QMessageBox.DestructiveRole)
            box.addButton(QMessageBox.Cancel)
            box.setDefaultButton(sa)
            box.exec()
            if box.clickedButton() is sa:
                return self.file_save_as()
            if box.clickedButton() is not ow:
                return False
        return self._save_to(self.ed.doc.path)

    def file_save_as(self) -> bool:
        start = str(self.ed.doc.path.with_name(self.ed.doc.path.stem + "_edited.xml")) if self.ed.doc.path \
            else os.path.join(os.getcwd(), "model.xml")
        path, _ = QFileDialog.getSaveFileName(self, "Save MJCF as", start, MJCF_FILTER)
        if not path:
            return False
        return self._save_to(path)

    def _save_to(self, path) -> bool:
        try:
            notes = self.ed.save(path)
        except OSError as ex:
            QMessageBox.critical(self, "Save failed", str(ex))
            return False
        self.protected = False
        self._push_recent(str(Path(path).resolve()))
        self.statusBar().showMessage(f"Saved {path}" + (f" ({len(notes)} asset paths rewritten)" if notes else ""), 5000)
        self._update_title()
        return True

    def export_compiled(self):
        res = compile_doc(self.ed.doc)
        if not res.ok:
            QMessageBox.warning(self, "Export", "Model does not compile:\n" + res.pretty_error(self.ed.doc))
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export MuJoCo-saved XML", "compiled.xml", MJCF_FILTER)
        if path:
            mujoco.mj_saveLastXML(path, res.model)
            log.info("Exported MuJoCo-generated XML to %s", path)

    def validate(self):
        res = compile_doc(self.ed.doc)
        if not res.ok:
            msg = res.pretty_error(self.ed.doc)
            log.error("Validation: compile error: %s", msg)
            QMessageBox.warning(self, "Validation", "Compile error:\n\n" + msg)
            return
        issues = validate_steps(res.model, 200)
        m = res.model
        summary = (f"Compiled OK in {res.elapsed_ms:.1f} ms — bodies {m.nbody}, joints {m.njnt}, dofs {m.nv}, "
                   f"geoms {m.ngeom}, actuators {m.nu}, sensors {m.nsensor}")
        log.info("Validation: %s", summary)
        if issues:
            for i in issues:
                log.warning("Validation: %s", i)
            QMessageBox.warning(self, "Validation", summary + "\n\nSimulation issues:\n- " + "\n- ".join(issues))
        else:
            log.info("Validation: 200 simulation steps without numerical problems")
            QMessageBox.information(self, "Validation", summary + "\n\n200 simulation steps: no problems detected.")

    def _push_recent(self, path):
        rec = [p for p in (self.settings.value("recent") or []) if p != path]
        rec.insert(0, path)
        self.settings.setValue("recent", rec[:10])
        self._fill_recent()

    def _fill_recent(self):
        self.recent_menu.clear()
        for p in (self.settings.value("recent") or []):
            self.recent_menu.addAction(p, lambda p=p: self._confirm_discard() and self.open_path(p))

    # ================================================================ status
    def _on_compiled(self, res):
        if res.ok:
            m = res.model
            self.status_compile.setText(f"<span style='color:{C['ok']}'>● compiled</span> {res.elapsed_ms:.0f} ms · "
                                        f"{m.nbody - 1} bodies · {m.njnt} joints · {m.ngeom} geoms · {m.nu} act · {m.nsensor} sens")
        else:
            self.status_compile.setText(f"<span style='color:{C['err']}'>● compile error</span> (see Log)")

    def _update_title(self):
        p = self.ed.doc.path
        name = p.name if p else "untitled"
        self.setWindowTitle(f"{name}{' *' if self.ed.dirty else ''} — MuJoCo Model Studio")

    def _update_undo(self):
        self.a_undo.setEnabled(self.ed.undo.can_undo())
        self.a_redo.setEnabled(self.ed.undo.can_redo())
        self.a_undo.setText(f"&Undo {self.ed.undo.undo_text()}".strip())
        self.a_redo.setText(f"&Redo {self.ed.undo.redo_text()}".strip())
        self._update_title()

    def show_controls(self):
        QMessageBox.information(self, "Viewport controls", (
            "Left drag: orbit   ·   Right drag: pan   ·   Middle drag / wheel: zoom\n"
            "Shift + drag: horizontal variants\n"
            "Click: select (Shift/Ctrl: add)   ·   Double click: center view on point\n"
            "Q / W / E: select / move / rotate tool   ·   L: local/world gizmo axes\n"
            "Ctrl while dragging gizmo: snap (1 cm / 15°)   ·   Esc: cancel drag\n"
            "F: frame selection   ·   Home: reset camera   ·   Space: play/pause"))

    def about(self):
        QMessageBox.about(self, "About", f"MuJoCo Model Studio {__version__}\nMuJoCo {mujoco.__version__}\n"
                                         "Visual MJCF editor and assembler.")

    def closeEvent(self, ev):
        if not self._confirm_discard():
            ev.ignore()
            return
        self.settings.setValue("geometry", self.saveGeometry())
        self.timer.stop()
        self.log_panel.detach()
        ev.accept()
