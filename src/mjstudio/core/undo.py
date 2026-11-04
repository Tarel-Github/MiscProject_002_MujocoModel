"""Command based undo/redo operating directly on the lxml tree.

Element objects are kept alive by the commands, so identities (and therefore
selections, id maps, tree items) survive undo/redo.
"""
from __future__ import annotations

import time
from typing import Callable

from .doc import Element


class Signal:
    """Minimal observer (no Qt dependency in the core)."""

    def __init__(self) -> None:
        self._subs: list[Callable] = []

    def connect(self, fn: Callable) -> None:
        if fn not in self._subs:
            self._subs.append(fn)

    def disconnect(self, fn: Callable) -> None:
        if fn in self._subs:
            self._subs.remove(fn)

    def emit(self, *a, **kw) -> None:
        for fn in list(self._subs):
            fn(*a, **kw)


class Cmd:
    text = ""
    struct = False  # True when the element structure (not only attributes) changes

    def redo(self) -> None:
        raise NotImplementedError

    def undo(self) -> None:
        raise NotImplementedError

    def elems(self) -> list[Element]:
        return []

    def can_merge(self, other: "Cmd") -> bool:
        return False

    def merge(self, other: "Cmd") -> None:
        pass


class SetAttrs(Cmd):
    """Set (value) or delete (None) several attributes of one element."""

    def __init__(self, elem: Element, changes: dict[str, str | None], text: str = "", merge_key: str | None = None):
        self.elem = elem
        self.changes = dict(changes)
        self.text = text or f"Edit {elem.tag}"
        self.merge_key = merge_key
        self._old = list(elem.attrib.items())
        self.t = time.monotonic()

    def redo(self) -> None:
        for k, v in self.changes.items():
            if v is None:
                if k in self.elem.attrib:
                    del self.elem.attrib[k]
            else:
                self.elem.set(k, v)

    def undo(self) -> None:
        self.elem.attrib.clear()
        for k, v in self._old:
            self.elem.set(k, v)

    def elems(self) -> list[Element]:
        return [self.elem]

    def can_merge(self, other: Cmd) -> bool:
        return (isinstance(other, SetAttrs) and self.merge_key is not None
                and other.merge_key == self.merge_key and other.elem is self.elem
                and other.t - self.t < self.MERGE_WINDOW)

    MERGE_WINDOW = 1.5  # seconds between edits that still merge into one undo step

    def merge(self, other: Cmd) -> None:
        assert isinstance(other, SetAttrs)
        self.changes.update(other.changes)
        self.t = other.t


class Insert(Cmd):
    struct = True

    def __init__(self, parent: Element, index: int, elem: Element, text: str = ""):
        self.parent, self.index, self.elem = parent, index, elem
        self.text = text or f"Add {elem.tag}"

    def redo(self) -> None:
        idx = min(max(self.index, 0), len(self.parent))
        self.parent.insert(idx, self.elem)

    def undo(self) -> None:
        self.parent.remove(self.elem)

    def elems(self) -> list[Element]:
        return [self.elem, self.parent]


class Remove(Cmd):
    struct = True

    def __init__(self, elem: Element, text: str = ""):
        self.elem = elem
        self.parent = elem.getparent()
        if self.parent is None:
            raise ValueError("최상위 요소는 삭제할 수 없습니다")
        self.index = self.parent.index(elem)
        self.text = text or f"Delete {elem.tag}"

    def redo(self) -> None:
        self.index = self.parent.index(self.elem)
        tail = self.elem.tail
        self.parent.remove(self.elem)
        self.elem.tail = tail

    def undo(self) -> None:
        self.parent.insert(self.index, self.elem)

    def elems(self) -> list[Element]:
        return [self.parent]


class Move(Cmd):
    struct = True

    def __init__(self, elem: Element, new_parent: Element, new_index: int | None = None, text: str = ""):
        self.elem = elem
        self.new_parent = new_parent
        self.new_index = new_index
        self.old_parent = elem.getparent()
        self.old_index = self.old_parent.index(elem)
        self.text = text or f"Move {elem.tag}"

    def redo(self) -> None:
        self.old_parent = self.elem.getparent()
        self.old_index = self.old_parent.index(self.elem)
        self.old_parent.remove(self.elem)
        if self.new_index is None:
            idx = len(self.new_parent)
        else:
            idx = self.new_index
            if self.new_parent is self.old_parent and self.old_index < idx:
                idx -= 1  # removal shifted the following siblings
            idx = min(max(idx, 0), len(self.new_parent))
        self.new_parent.insert(idx, self.elem)

    def undo(self) -> None:
        self.new_parent.remove(self.elem)
        self.old_parent.insert(self.old_index, self.elem)

    def elems(self) -> list[Element]:
        return [self.elem, self.old_parent, self.new_parent]


class Macro(Cmd):
    def __init__(self, text: str, cmds: list[Cmd] | None = None):
        self.text = text
        self.cmds: list[Cmd] = list(cmds or [])

    @property
    def struct(self) -> bool:  # type: ignore[override]
        return any(c.struct for c in self.cmds)

    def redo(self) -> None:
        for c in self.cmds:
            c.redo()

    def undo(self) -> None:
        for c in reversed(self.cmds):
            c.undo()

    def elems(self) -> list[Element]:
        out: list[Element] = []
        for c in self.cmds:
            out += c.elems()
        return out


class UndoStack:
    def __init__(self, limit: int = 1000) -> None:
        self.limit = limit
        self._done: list[Cmd] = []
        self._undone: list[Cmd] = []
        self._macros: list[Macro] = []
        self._clean: Cmd | None = None
        self._clean_valid = True
        self.changed = Signal()  # (cmd, action) action in {'do', 'undo', 'redo'}

    # -- macro grouping
    def begin(self, text: str) -> None:
        self._macros.append(Macro(text))

    def end(self) -> None:
        m = self._macros.pop()
        if not m.cmds:
            return
        if self._macros:
            self._macros[-1].cmds.append(m)
            return
        self._record(m)
        self.changed.emit(m, "do")

    def abort(self) -> None:
        """Roll back the open macro (used when an operation fails midway)."""
        m = self._macros.pop()
        m.undo()

    @property
    def in_macro(self) -> bool:
        return bool(self._macros)

    def push(self, cmd: Cmd, merge: bool = False) -> None:
        cmd.redo()
        if self._macros:
            self._macros[-1].cmds.append(cmd)
            return
        if merge and self._done and self._done[-1].can_merge(cmd) and self._done[-1] is not self._clean:
            self._done[-1].merge(cmd)
            self._undone.clear()
        else:
            self._record(cmd)
        self.changed.emit(cmd, "do")

    def _record(self, cmd: Cmd) -> None:
        self._done.append(cmd)
        self._undone.clear()
        if len(self._done) > self.limit:
            dropped = self._done.pop(0)
            if dropped is self._clean:
                self._clean_valid = False

    def undo(self) -> Cmd | None:
        if not self._done or self._macros:
            return None
        cmd = self._done.pop()
        cmd.undo()
        self._undone.append(cmd)
        self.changed.emit(cmd, "undo")
        return cmd

    def redo(self) -> Cmd | None:
        if not self._undone or self._macros:
            return None
        cmd = self._undone.pop()
        cmd.redo()
        self._done.append(cmd)
        self.changed.emit(cmd, "redo")
        return cmd

    def can_undo(self) -> bool:
        return bool(self._done)

    def can_redo(self) -> bool:
        return bool(self._undone)

    def undo_text(self) -> str:
        return self._done[-1].text if self._done else ""

    def redo_text(self) -> str:
        return self._undone[-1].text if self._undone else ""

    def set_clean(self) -> None:
        self._clean = self._done[-1] if self._done else None
        self._clean_valid = True

    def is_clean(self) -> bool:
        if not self._clean_valid:
            return False
        top = self._done[-1] if self._done else None
        return top is self._clean

    def clear(self) -> None:
        self._done.clear()
        self._undone.clear()
        self._clean = None
        self._clean_valid = True
