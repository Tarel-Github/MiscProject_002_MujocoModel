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
from ..core import parts
from ..core.compiler import compile_doc, validate_steps
from ..core.project import Project, missing_resources
from ..library import anatomy
from ..core.doc import MjcfDoc, MjcfError
from ..core.editor import Editor
from ..core.sim import Simulator
from ..core import meshops
from ..core import meshes as M
from .dialogs import JointGroupDialog, LibraryDialog, MeshImportDialog, PickBodyDialog
from .facepanel import FacePanel
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
        self.project: Project | None = None
        self.view = Viewport(self.ed)
        self.view.sim = self.sim
        self.setCentralWidget(self.view)
        self.tree = TreePanel(self.ed)
        self.props = PropsPanel(self.ed)
        self.joints = JointPanel(self.ed, groups=lambda: self.project.groups if self.project else [])
        self.simp = SimPanel(self.ed, self.sim)
        self.face = FacePanel(self.ed, self.sim)
        dt = self._dock("Hierarchy", self.tree, Qt.LeftDockWidgetArea)
        dp = self._dock("Properties", self.props, Qt.RightDockWidgetArea)
        self.resizeDocks([dt, dp], [330, 340], Qt.Horizontal)
        dj = self._dock("Joints", self.joints, Qt.BottomDockWidgetArea)
        ds = self._dock("Simulation", self.simp, Qt.BottomDockWidgetArea)
        dl = self._dock("Log", self.log_panel, Qt.BottomDockWidgetArea)
        df = self._dock("Face", self.face, Qt.BottomDockWidgetArea)
        self.tabifyDockWidget(dj, ds)
        self.tabifyDockWidget(ds, df)
        self.tabifyDockWidget(df, dl)
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
        self._act(f, "Open &project…", self.project_open, "Ctrl+Shift+O")
        self.recent_menu = f.addMenu("Open &recent")
        self._fill_recent()
        f.addSeparator()
        self._act(f, "&Save", self.file_save, "Ctrl+S")
        self._act(f, "Save &As…", self.file_save_as, "Ctrl+Shift+S")
        self._act(f, "Save as pro&ject (folder)…", self.project_save_as)
        self._act(f, "Export compiled MJCF (as MuJoCo sees it)…", self.export_compiled)
        f.addSeparator()
        self._act(f, "&Import part from MJCF…", self.import_part_file, "Ctrl+I")
        self._act(f, "E&xport selected body as part…", self.export_part)
        self._act(f, "Check missing resources", self.check_resources)
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
        lib = mm.addMenu("Insert &library part")
        self._act(lib, "Hand (5 fingers, 22 joints)…", lambda: self.insert_library("hand"))
        self._act(lib, "Foot (ankle/hindfoot/midfoot/forefoot/toes)…", lambda: self.insert_library("foot"))
        self._act(lib, "Head (eyes + jaw + neck)…", lambda: self.insert_library("head"))
        self._act(mm, "Replace selected body with part…", self.replace_with_part)
        mm.addSeparator()
        self._act(mm, "&Recompile", self.ed.recompile, "F5")
        # ---- Mesh
        me = mb.addMenu("Me&sh")
        self._act(me, "&Import mesh (OBJ / STL / GLB)…", self.import_mesh_file, "Ctrl+Shift+I")
        pm = me.addMenu("Add mesh &primitive")
        for t in M.PRIMITIVES:
            self._act(pm, t, lambda t=t: self._add(lambda p: meshops.create_primitive_mesh(self.ed, p, t)["geoms"][0]))
        me.addSeparator()
        self.a_meshedit = self._act(me, "&Edit vertices of selected mesh geom (toggle)", self.toggle_mesh_edit, "Tab")
        me.addAction("(in edit mode: click vertex, Shift+click add, W move, Del delete, F face from 3, P add vertex"
                     " in clicked face, Esc exit)").setEnabled(False)
        me.addSeparator()
        self._act(me, "E&xport mesh of selected geom…", self.export_mesh)
        self._act(me, "&Duplicate mesh asset of selected geom", self.duplicate_mesh)
        # ---- Joint
        j = mb.addMenu("&Joint")
        for t in ("hinge", "slide", "ball", "free"):
            self._act(j, f"Add {t} joint", lambda t=t: self._add_joint(t))
        j.addSeparator()
        self._act(j, "Add position actuator to joint", lambda: self._add_actuator("position"))
        self._act(j, "Add motor to joint", lambda: self._add_actuator("motor"))
        j.addSeparator()
        self._act(j, "Couple two selected joints (equality)…", self.couple_selected)
        self._act(j, "Create joint group from selected joints…", self.group_selected)
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
        self._act(v, "Frame selection (mesh edit: face from 3 vertices)", self._key_f, "F")
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
        if self.view.mesh_mode is not None:      # Delete acts on vertices/faces in mesh edit mode
            self.view.mesh_mode.delete()
            self.view.update()
            return
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

    # ================================================================ parts / library / project
    def _attach_parent(self):
        t = self._target()
        while t is not None and t.tag not in ("body", "worldbody"):
            t = t.getparent()
        return t if t is not None else self.ed.doc.worldbody

    def import_part_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import part from MJCF", os.getcwd(), MJCF_FILTER)
        if not path:
            return
        try:
            src = MjcfDoc.load(path)
        except (MjcfError, OSError) as ex:
            QMessageBox.critical(self, "Import part", str(ex))
            return
        dlg = PickBodyDialog(src, self)
        if dlg.exec() != PickBodyDialog.Accepted or dlg.body() is None:
            return
        part = self._guard(lambda: parts.extract_part(src, dlg.body()))
        if part is None:
            return
        out = self._guard(lambda: parts.import_part(self.ed, part, self._attach_parent(),
                                                    prefix=dlg.prefix.text().strip() or None))
        if out:
            self.ed.select([out[0]])
            log.info("Imported part '%s' from %s", out[0].get("name"), path)

    def export_part(self):
        t = self._target()
        if t is None or t.tag != "body":
            QMessageBox.information(self, "Export part", "Select a body first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save part as MJCF", f"{t.get('name') or 'part'}.xml", MJCF_FILTER)
        if not path:
            return
        part = self._guard(lambda: parts.extract_part(self.ed.doc, t))
        if part is not None:
            part.save(path)
            log.info("Saved part %s (%s)", path, "; ".join(part.meta["part"]["notes"]) or "self-contained")

    def replace_with_part(self):
        t = self._target()
        if t is None or t.tag != "body":
            QMessageBox.information(self, "Replace", "Select the body to replace (e.g. a head or a hand).")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Replacement part (MJCF)", os.getcwd(), MJCF_FILTER)
        if not path:
            return
        try:
            src = MjcfDoc.load(path)
        except (MjcfError, OSError) as ex:
            QMessageBox.critical(self, "Replace", str(ex))
            return
        dlg = PickBodyDialog(src, self, "Replacement part")
        if dlg.exec() != PickBodyDialog.Accepted or dlg.body() is None:
            return
        part = self._guard(lambda: parts.extract_part(src, dlg.body()))
        if part is not None:
            out = self._guard(lambda: parts.replace_body(self.ed, t, part, prefix=dlg.prefix.text().strip() or None))
            if out:
                self.ed.select([out[0]])

    def insert_library(self, kind):
        """Build the part in its own document and assemble it (names are made unique)."""
        dlg = LibraryDialog(kind, self)
        if dlg.exec() != LibraryDialog.Accepted:
            return
        side, sc = dlg.side.currentText(), dlg.scale.value()
        if kind == "hand":
            part = anatomy.make_part(anatomy.build_hand, f"hand_{side}", side=side, scale=sc)
        elif kind == "foot":
            part = anatomy.make_part(anatomy.build_foot, f"foot_{side}", side=side, scale=sc,
                                     toes=dlg.toes.currentText())
        else:
            part = anatomy.make_part(anatomy.build_head, "head", scale=sc)
        out = self._guard(lambda: parts.import_part(self.ed, part, self._attach_parent(), prefix=f"{kind}2_"))
        if out:
            self.ed.select([out[0]])
            log.info("Inserted library part %s as '%s'", kind, out[0].get("name"))

    def couple_selected(self):
        js = [e for e in self.ed.selection if e.tag == "joint" and e.get("name")]
        if len(js) != 2:
            QMessageBox.information(self, "Couple joints", "Select exactly two named joints (Ctrl+click).")
            return
        ratio, ok = QInputDialog.getDouble(self, "Couple joints",
                                           f"{js[0].get('name')} = ratio × {js[1].get('name')}", 1.0, -100, 100, 4)
        if ok:
            self._guard(lambda: ops.couple_joints(self.ed, js[0], js[1], ratio))

    def group_selected(self):
        js = [e.get("name") for e in self.ed.selection if e.tag == "joint" and e.get("name")]
        if not js:
            QMessageBox.information(self, "Joint group", "Select one or more named joints.")
            return
        dlg = JointGroupDialog(js, parent=self)
        if dlg.exec() != JointGroupDialog.Accepted:
            return
        if self.project is None:
            QMessageBox.information(self, "Joint group", "Joint groups are stored in the project file. "
                                    "The model will be saved as a project folder next.")
            if not self.project_save_as():
                return
        self.project.set_group(dlg.result_group())
        self.project.save(self.ed.doc, package=False, rigs=self.ed.rigs)
        self.joints.rebuild()

    def project_open(self):
        if not self._confirm_discard():
            return
        d = QFileDialog.getExistingDirectory(self, "Open project folder", os.getcwd())
        if not d:
            return
        try:
            prj, doc = Project.load(d)
        except (OSError, ValueError, MjcfError) as ex:
            QMessageBox.critical(self, "Open project", str(ex))
            return
        self.sim.pause()
        self.project = prj
        self.view.set_mesh_mode(None)
        self.ed.set_doc(doc, prj.load_rigs())
        self.protected = False
        self.view.reset_camera()
        self.joints.rebuild()
        log.info("Opened project %s", prj.root)

    def project_save_as(self) -> bool:
        d = QFileDialog.getExistingDirectory(self, "Choose (empty) project folder", os.getcwd())
        if not d:
            return False
        prj = self.project if self.project is not None else Project(d)
        prj.root = Path(d).resolve()
        try:
            notes = prj.save(self.ed.doc, package=True, rigs=self.ed.rigs)
        except OSError as ex:
            QMessageBox.critical(self, "Save project", str(ex))
            return False
        for n in notes:
            log.info("%s", n)
        self.project = prj
        self.protected = False
        self.ed.undo.set_clean()
        self._update_title()
        return True

    def check_resources(self):
        miss = missing_resources(self.ed.doc)
        if not miss:
            QMessageBox.information(self, "Resources", "All referenced asset files exist.")
            log.info("Resource check: all asset files found")
            return
        for lbl, k, p in miss:
            log.warning("missing resource: %s/%s -> %s", lbl, k, p)
        QMessageBox.warning(self, "Resources", f"{len(miss)} missing file(s):\n" +
                            "\n".join(f"{lbl} ({k}): {p}" for lbl, k, p in miss[:20]))

    def _key_f(self):
        if self.view.mesh_mode is not None:
            self.view.mesh_mode.make_face()
            self.view.update()
        else:
            self.view.frame_selection()

    # ================================================================ meshes
    def _sel_mesh_geom(self):
        t = self._target()
        if t is not None and t.tag == "geom" and meshops.geom_mesh_elem(self.ed, t) is not None:
            return t
        return None

    def import_mesh_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import mesh", os.getcwd(),
                                              "Meshes (*.obj *.stl *.glb *.gltf);;All files (*)")
        if not path:
            return
        t = self._target()
        dlg = MeshImportDialog(path, t is not None and t.tag != "worldbody", self)
        if dlg.exec() != MeshImportDialog.Accepted:
            return
        new_body = bool(dlg.mode.currentData())
        out = self._guard(lambda: meshops.import_mesh(
            self.ed, path, parent=self._attach_parent(), new_body=new_body, name=dlg.name.text().strip() or None,
            units=dlg.units.currentText(), axes=dlg.axes.currentData(), collision=dlg.collision.currentText()))
        if out:
            self.ed.select([out["body"] if new_body else out["geoms"][0]])

    def toggle_mesh_edit(self):
        if self.view.mesh_mode is not None:
            self.view.set_mesh_mode(None)
            self.statusBar().showMessage("Mesh edit mode off", 2000)
            return
        g = self._sel_mesh_geom()
        if g is None:
            QMessageBox.information(self, "Edit mesh", "Select a mesh geom first (Pick geom mode helps).")
            return
        from .meshedit import MeshEditMode
        mode = self._guard(lambda: MeshEditMode(self.view, g))
        if mode is None:
            return
        self.view.set_mesh_mode(mode)
        self.view.setFocus()
        self.statusBar().showMessage("Mesh edit mode: click vertices (Shift adds), W/E gizmo, Del, F, P, Esc", 6000)

    def export_mesh(self):
        g = self._sel_mesh_geom()
        if g is None:
            QMessageBox.information(self, "Export mesh", "Select a mesh geom first.")
            return
        me = meshops.geom_mesh_elem(self.ed, g)
        path, _ = QFileDialog.getSaveFileName(self, "Export mesh", f"{self.ed.doc.obj_name(me)}.obj",
                                              "OBJ (*.obj);;STL (*.stl)")
        if path:
            p = self._guard(lambda: meshops.export_mesh(self.ed, me, path))
            if p:
                log.info("Exported mesh to %s", p)

    def duplicate_mesh(self):
        g = self._sel_mesh_geom()
        if g is None:
            QMessageBox.information(self, "Duplicate mesh", "Select a mesh geom first.")
            return
        me = self._guard(lambda: meshops.duplicate_mesh_asset(self.ed, meshops.geom_mesh_elem(self.ed, g)))
        if me is not None:
            self._guard(lambda: self.ed.set_attr(g, "mesh", me.get("name"), "Use duplicated mesh"))

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
        self.view.set_mesh_mode(None)
        try:
            self.ed.open(path)
        except (MjcfError, OSError) as ex:
            QMessageBox.critical(self, "Cannot open", str(ex))
            log.error("Cannot open %s: %s", path, ex)
            return False
        self.protected = True
        self.project = None
        self._push_recent(str(Path(path).resolve()))
        self.view.reset_camera()
        if self.ed.result is not None and not self.ed.result.ok:
            QMessageBox.warning(self, "Model has errors",
                                "The file was loaded but MuJoCo reports an error:\n\n" +
                                self.ed.result.pretty_error(self.ed.doc))
        return True

    def file_save(self) -> bool:
        if self.project is not None:
            try:
                for n in self.project.save(self.ed.doc, package=True, rigs=self.ed.rigs):
                    log.info("%s", n)
            except OSError as ex:
                QMessageBox.critical(self, "Save project", str(ex))
                return False
            self.ed.undo.set_clean()
            self._update_title()
            self.statusBar().showMessage(f"Saved project {self.project.root}", 4000)
            return True
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
