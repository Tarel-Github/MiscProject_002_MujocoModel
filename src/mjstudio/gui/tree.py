"""Model hierarchy tree (every element of the document is reachable here)."""
from __future__ import annotations

import logging

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import (QAbstractItemView, QHBoxLayout, QLineEdit, QMenu, QTreeWidget, QTreeWidgetItem,
                               QVBoxLayout, QWidget)

from ..core.doc import children, is_elem, parent_tag
from ..core.editor import Editor
from .theme import C, icon_for, shape_icon

log = logging.getLogger("mjstudio")
ROLE = Qt.UserRole

GROUPS = [  # (title, section tags)
    ("World", ("worldbody",)),
    ("Assets", ("asset",)),
    ("Defaults", ("default",)),
    ("Actuators", ("actuator",)),
    ("Sensors", ("sensor",)),
    ("Tendons", ("tendon",)),
    ("Equality", ("equality",)),
    ("Contact", ("contact",)),
    ("Keyframes", ("keyframe",)),
    ("Deformable", ("deformable",)),
    ("Settings", ("compiler", "option", "size", "visual", "statistic", "custom", "extension")),
]
TREE_MOVABLE = {"body", "geom", "joint", "freejoint", "site", "camera", "light", "inertial", "frame"}


class ModelTree(QTreeWidget):
    elementsDropped = Signal(list, object, int)  # elems, new parent element, index

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderLabels(["Element", "Type"])
        from PySide6.QtWidgets import QHeaderView
        self.header().setStretchLastSection(False)
        self.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setDragDropMode(QAbstractItemView.InternalMove)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setAlternatingRowColors(True)
        self.setUniformRowHeights(True)
        self.setExpandsOnDoubleClick(False)

    def dropEvent(self, ev) -> None:
        target = self.itemAt(ev.position().toPoint())
        elems = [it.data(0, ROLE) for it in self.selectedItems() if it.data(0, ROLE) is not None]
        elems = [e for e in elems if e.tag in TREE_MOVABLE]
        ev.ignore()  # the document is the source of truth; the tree is rebuilt
        if target is None or not elems:
            return
        tel = target.data(0, ROLE)
        if tel is None:
            return
        pos = self.dropIndicatorPosition()
        if pos == QAbstractItemView.OnItem:
            parent, index = tel, None
        else:
            parent = tel.getparent()
            if parent is None:
                return
            index = parent.index(tel) + (1 if pos == QAbstractItemView.BelowItem else 0)
        if parent.tag not in ("body", "worldbody", "frame"):
            return
        self.elementsDropped.emit(elems, parent, -1 if index is None else index)


class TreePanel(QWidget):
    contextRequested = Signal(object, object)  # element, global pos

    def __init__(self, ed: Editor, parent=None):
        super().__init__(parent)
        self.ed = ed
        self.items: dict = {}
        self._sync = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        top = QHBoxLayout()
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter by name…")
        self.filter.setClearButtonEnabled(True)
        self.filter.textChanged.connect(self._apply_filter)
        top.addWidget(self.filter)
        lay.addLayout(top)
        self.tree = ModelTree()
        lay.addWidget(self.tree)
        self.tree.itemSelectionChanged.connect(self._on_tree_sel)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._ctx)
        self.tree.elementsDropped.connect(self._dropped)
        self.tree.itemDoubleClicked.connect(lambda it, _c: it.setExpanded(not it.isExpanded()))
        ed.doc_loaded.connect(self.rebuild)
        ed.doc_changed.connect(self._on_changed)
        ed.selection_changed.connect(self._on_sel)
        ed.compiled.connect(self._on_compiled)
        self.rebuild()

    # ------------------------------------------------------------- building
    def rebuild(self) -> None:
        expanded = {e for e, it in self.items.items() if it.isExpanded()}
        first = not self.items
        bar = self.tree.verticalScrollBar().value()
        self.tree.blockSignals(True)
        self.tree.clear()
        self.items = {}
        doc = self.ed.doc
        for title, tags in GROUPS:
            secs = [s for t in tags for s in doc.sections(t)]
            if not secs and title not in ("World",):
                continue
            g = QTreeWidgetItem([title, ""])
            g.setIcon(0, shape_icon("square", C["dim"]))
            f = QFont()
            f.setBold(True)
            g.setFont(0, f)
            g.setFlags(g.flags() & ~Qt.ItemIsDragEnabled)
            self.tree.addTopLevelItem(g)
            if title == "Settings":
                for s in secs:
                    self._add(g, s, recurse=True)
            else:
                if secs:
                    g.setData(0, ROLE, secs[0])
                    self.items[secs[0]] = g
                for s in secs:
                    for c in children(s):
                        self._add(g, c, recurse=True)
            if first:
                g.setExpanded(title == "World")
        for e, it in self.items.items():
            if e in expanded:
                it.setExpanded(True)
        if first:
            wb = doc.section("worldbody")
            for c in (children(wb) if wb is not None else []):
                if c.tag == "body" and c in self.items:
                    self.items[c].setExpanded(True)
        self.tree.blockSignals(False)
        self._on_sel(self.ed.selection)
        self._mark_error()
        self._apply_filter(self.filter.text())
        self.tree.verticalScrollBar().setValue(bar)

    def _add(self, parent_item, e, recurse=True) -> None:
        if not is_elem(e):
            return
        it = QTreeWidgetItem([self.ed.doc.label(e), self._type_text(e)])
        it.setData(0, ROLE, e)
        it.setIcon(0, icon_for(e.tag, parent_tag(e)))
        it.setForeground(1, QBrush(QColor(C["dim"])))
        if e.tag not in TREE_MOVABLE:
            it.setFlags(it.flags() & ~Qt.ItemIsDragEnabled)
        if e.tag not in ("body", "worldbody", "frame"):
            it.setFlags(it.flags() & ~Qt.ItemIsDropEnabled)
        parent_item.addChild(it)
        self.items[e] = it
        if recurse:
            for c in children(e):
                self._add(it, c)

    def _type_text(self, e) -> str:
        cls = e.get("class") or e.get("childclass")
        t = e.tag
        return f"{t} · {cls}" if cls else t

    def _refresh_labels(self, elems) -> None:
        for e in elems:
            it = self.items.get(e)
            if it is not None and is_elem(e):
                it.setText(0, self.ed.doc.label(e))
                it.setText(1, self._type_text(e))

    # ------------------------------------------------------------- events
    def _on_changed(self, cmd, action) -> None:
        if cmd.struct:
            self.rebuild()
        else:
            els = cmd.elems()
            self._refresh_labels(els)
            # a label may depend on attributes of other elements (e.g. mesh names)
            if any(e.get("name") is not None for e in els):
                self._refresh_labels(list(self.items.keys()))

    def _on_compiled(self, res) -> None:
        self._mark_error()

    def _mark_error(self) -> None:
        res = self.ed.result
        for it in self.items.values():
            it.setForeground(0, QBrush())
            it.setToolTip(0, "")
        if res is not None and not res.ok and res.error_elem is not None:
            it = self.items.get(res.error_elem)
            if it is not None:
                it.setForeground(0, QBrush(QColor(C["err"])))
                it.setToolTip(0, res.pretty_error(self.ed.doc))
                p = it.parent()
                while p is not None:
                    p.setExpanded(True)
                    p = p.parent()

    def _on_tree_sel(self) -> None:
        if self._sync:
            return
        elems = [it.data(0, ROLE) for it in self.tree.selectedItems()]
        elems = [e for e in elems if e is not None and e.tag not in ("worldbody",)]
        self._sync = True
        try:
            self.ed.select(elems)
        finally:
            self._sync = False

    def _on_sel(self, sel) -> None:
        if self._sync:
            return
        self._sync = True
        try:
            self.tree.clearSelection()
            last = None
            for e in sel:
                it = self.items.get(e)
                if it is not None:
                    it.setSelected(True)
                    p = it.parent()
                    while p is not None:
                        p.setExpanded(True)
                        p = p.parent()
                    last = it
            if last is not None:
                self.tree.scrollToItem(last)
        finally:
            self._sync = False

    def _apply_filter(self, text: str) -> None:
        text = text.strip().lower()

        def walk(it) -> bool:
            match = not text or text in it.text(0).lower()
            child_match = False
            for i in range(it.childCount()):
                child_match |= walk(it.child(i))
            vis = match or child_match
            it.setHidden(not vis)
            if text and child_match:
                it.setExpanded(True)
            return vis

        for i in range(self.tree.topLevelItemCount()):
            walk(self.tree.topLevelItem(i))

    def _ctx(self, pos) -> None:
        it = self.tree.itemAt(pos)
        e = it.data(0, ROLE) if it is not None else None
        self.contextRequested.emit(e, self.tree.viewport().mapToGlobal(pos))

    def _dropped(self, elems, parent, index) -> None:
        try:
            with self.ed.batch(f"Move {len(elems)} element(s)"):
                for k, e in enumerate(elems):
                    if parent is e or parent in list(e.iterdescendants()):
                        raise ValueError("cannot move an element into its own subtree")
                    self.ed.move(e, parent, None if index < 0 else index + k)
        except Exception as ex:  # noqa: BLE001
            log.error("Move failed: %s", ex)
