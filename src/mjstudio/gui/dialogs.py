"""Dialogs for part import and joint groups."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QLabel, QLineEdit,
                               QTableWidget, QTableWidgetItem, QTreeWidget, QTreeWidgetItem, QVBoxLayout)

from ..core.doc import MjcfDoc, children
from .theme import C, icon_for


class PickBodyDialog(QDialog):
    """Choose a body (subtree) of an external MJCF file to import as a part."""

    def __init__(self, doc: MjcfDoc, parent=None, title="Import part"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(460, 560)
        self.doc = doc
        lay = QVBoxLayout(self)
        info = QLabel(f"<b>{doc.path.name if doc.path else 'model'}</b> — choose the body whose subtree "
                      "becomes the part. Required assets, default classes and actuators/sensors that only "
                      "reference the subtree are taken along.")
        info.setWordWrap(True)
        lay.addWidget(info)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        lay.addWidget(self.tree)
        wb = doc.section("worldbody")

        def add(parent_item, e):
            for c in children(e, "body"):
                it = QTreeWidgetItem([doc.label(c)])
                it.setIcon(0, icon_for("body"))
                it.setData(0, Qt.UserRole, c)
                (parent_item.addChild(it) if parent_item is not None else self.tree.addTopLevelItem(it))
                add(it, c)

        if wb is not None:
            add(None, wb)
        self.tree.expandToDepth(2)
        form = QFormLayout()
        self.prefix = QLineEdit()
        self.prefix.setPlaceholderText("prefix used only for colliding names")
        form.addRow("Name prefix", self.prefix)
        lay.addLayout(form)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        if self.tree.topLevelItemCount():
            self.tree.setCurrentItem(self.tree.topLevelItem(0))

    def body(self):
        it = self.tree.currentItem()
        return it.data(0, Qt.UserRole) if it is not None else None


class JointGroupDialog(QDialog):
    """Create/edit an editor-level joint group (weights per joint)."""

    def __init__(self, joints: list[str], name: str = "group", weights: dict | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Joint group")
        self.resize(420, 480)
        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.name = QLineEdit(name)
        form.addRow("Group name", self.name)
        lay.addLayout(form)
        note = QLabel("value v drives each joint as  q = q0 + weight · v  (stored in project.json)")
        note.setStyleSheet(f"color:{C['dim']}")
        lay.addWidget(note)
        self.table = QTableWidget(len(joints), 2)
        self.table.setHorizontalHeaderLabels(["Joint", "Weight"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.spins = []
        for r, j in enumerate(joints):
            it = QTableWidgetItem(j)
            it.setFlags(it.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(r, 0, it)
            s = QDoubleSpinBox()
            s.setRange(-10, 10)
            s.setDecimals(3)
            s.setValue((weights or {}).get(j, 1.0))
            self.table.setCellWidget(r, 1, s)
            self.spins.append((j, s))
        lay.addWidget(self.table)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def result_group(self):
        from ..core.project import JointGroup
        return JointGroup(self.name.text().strip() or "group", {j: s.value() for j, s in self.spins})


class LibraryDialog(QDialog):
    """Options for inserting a library part."""

    def __init__(self, kind: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Insert {kind}")
        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.side = QComboBox()
        self.side.addItems(["left", "right"])
        self.toes = QComboBox()
        self.toes.addItems(["single", "detailed"])
        self.scale = QDoubleSpinBox()
        self.scale.setRange(0.2, 5)
        self.scale.setValue(1.0)
        self.scale.setSingleStep(0.05)
        if kind in ("hand", "foot"):
            form.addRow("Side", self.side)
        if kind == "foot":
            form.addRow("Toes", self.toes)
        form.addRow("Scale", self.scale)
        lay.addLayout(form)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)


class MeshImportDialog(QDialog):
    """Options for importing an OBJ/STL/GLB mesh."""

    def __init__(self, path: str, has_body: bool, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import mesh")
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(f"<b>{path}</b>"))
        form = QFormLayout()
        self.units = QComboBox()
        self.units.addItems(["m", "cm", "mm", "in", "ft"])
        self.axes = QComboBox()
        self.axes.addItem("MuJoCo (Z up, +X front)", "mujoco")
        self.axes.addItem("Blender scene (Z up, -Y front)", "blender")
        self.axes.addItem("Y up (+Z front)", "yup")
        self.mode = QComboBox()
        self.mode.addItem("new body", True)
        if has_body:
            self.mode.addItem("attach to selected body", False)
        self.collision = QComboBox()
        self.collision.addItems(["hull", "box", "capsule", "sphere", "cylinder", "ellipsoid", "none"])
        self.name = QLineEdit()
        self.name.setPlaceholderText("from file name")
        if path.lower().endswith((".glb", ".gltf")):
            self.axes.setEnabled(False)
            self.axes.setToolTip("glTF defines its axes (Y up, +Z front); converted automatically")
        form.addRow("Units", self.units)
        form.addRow("Axes", self.axes)
        form.addRow("Place as", self.mode)
        form.addRow("Collision shape", self.collision)
        form.addRow("Name", self.name)
        lay.addLayout(form)
        note = QLabel("Visual geom: group 2, no contacts. Collision geom: group 3 (MuJoCo convex hull or a "
                      "primitive fitted to the mesh).")
        note.setWordWrap(True)
        note.setStyleSheet(f"color:{C['dim']}")
        lay.addWidget(note)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
