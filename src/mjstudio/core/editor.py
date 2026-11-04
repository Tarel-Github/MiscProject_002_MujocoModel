"""Editing session: document + undo + compilation + selection + kinematic state.

This is the GUI-independent API of MuJoCo Model Studio. The GUI only calls
these methods and listens to the signals; scripts can use it headless.
"""
from __future__ import annotations

import logging
import math
from contextlib import contextmanager
from pathlib import Path

import mujoco
import numpy as np

from . import mathx as mx
from .compiler import CompileResult, compile_doc
from .doc import Element, MjcfDoc, children, is_elem, parent_tag
from .schema import NAMED_KINDS, reference_attrs
from .undo import Cmd, Insert, Move, Remove, SetAttrs, Signal, UndoStack

log = logging.getLogger("mjstudio")
OBJ = mujoco.mjtObj
JT = mujoco.mjtJoint

KIND_OF_TAG: dict[str, str] = {}
for _k, _tags in NAMED_KINDS.items():
    for _t in _tags:
        KIND_OF_TAG.setdefault(_t, _k)


def ref_kind(e: Element) -> str | None:
    """Reference namespace of an element (body, joint, geom, ...)."""
    pt = parent_tag(e)
    if pt in ("fixed", "spatial") or (pt == "equality"):
        return "equality" if pt == "equality" else None
    if e.tag in ("joint", "freejoint"):
        return "joint"
    return KIND_OF_TAG.get(e.tag)


class Editor:
    def __init__(self, doc: MjcfDoc | None = None, auto_compile: bool = True):
        self.doc = doc or MjcfDoc.new()
        self.undo = UndoStack()
        self.undo.changed.connect(self._on_undo)
        self.auto_compile = auto_compile
        self.result: CompileResult | None = None
        self.valid: CompileResult | None = None
        self.data: mujoco.MjData | None = None
        self.selection: list[Element] = []
        self._batch = 0
        self._batch_struct = False
        self._batch_dirty = False
        # signals
        self.doc_loaded = Signal()        # ()
        self.doc_changed = Signal()       # (cmd, action)
        self.compiled = Signal()          # (CompileResult)
        self.selection_changed = Signal() # (list[Element])
        self.state_changed = Signal()     # ()  data/pose changed
        if auto_compile:
            self.recompile()

    # ================================================================ files
    @property
    def model(self) -> mujoco.MjModel | None:
        return self.valid.model if self.valid else None

    @property
    def idmap(self):
        return self.valid.idmap if self.valid else None

    def set_doc(self, doc: MjcfDoc) -> None:
        self.doc = doc
        self.undo.clear()
        self.selection = []
        self.valid = None
        self.data = None
        self.result = None
        if doc.meta.get("flattened_includes"):
            log.warning("<include> files were inlined: %s (saving writes a single file)",
                        ", ".join(doc.meta["flattened_includes"]))
        self.doc_loaded.emit()
        self.selection_changed.emit(self.selection)
        self.recompile()

    def new(self, model: str = "model") -> None:
        self.set_doc(MjcfDoc.new(model))

    def open(self, path: str | Path) -> None:
        doc = MjcfDoc.load(path)
        log.info("Loaded %s", doc.path)
        self.set_doc(doc)

    def save(self, path: str | Path | None = None) -> list[str]:
        if path is None:
            if self.doc.path is None:
                raise ValueError("document has no path; use save(path)")
            path = self.doc.path
        notes = self.doc.save(path)
        for n in notes:
            log.info("path rewritten: %s", n)
        self.undo.set_clean()
        log.info("Saved %s", self.doc.path)
        return notes

    @property
    def dirty(self) -> bool:
        return not self.undo.is_clean()

    # ================================================================ compile
    def recompile(self) -> CompileResult:
        res = compile_doc(self.doc)
        self.result = res
        if res.ok:
            old_valid, old_data = self.valid, self.data
            self.valid = res
            self.data = mujoco.MjData(res.model)
            if old_valid is not None and old_data is not None:
                self._transfer_state(old_valid, old_data)
            mujoco.mj_forward(res.model, self.data)
            log.debug("compiled in %.1f ms", res.elapsed_ms)
        else:
            log.error("Compile error: %s", res.pretty_error(self.doc))
        self.compiled.emit(res)
        self.state_changed.emit()
        return res

    def _transfer_state(self, old: CompileResult, od: mujoco.MjData) -> None:
        """Carry joint positions/velocities, ctrl and time over to the new model by element identity."""
        nm, nd = self.valid.model, self.data
        om = old.model
        for e, (ot, oid) in old.idmap.by_elem.items():
            if ot == int(OBJ.mjOBJ_JOINT):
                nid = self.valid.idmap.id(e)
                if nid < 0 or nm.jnt_type[nid] != om.jnt_type[oid]:
                    continue
                n = {0: 7, 1: 4, 2: 1, 3: 1}[int(om.jnt_type[oid])]
                nv = {0: 6, 1: 3, 2: 1, 3: 1}[int(om.jnt_type[oid])]
                nd.qpos[nm.jnt_qposadr[nid]:nm.jnt_qposadr[nid] + n] = od.qpos[om.jnt_qposadr[oid]:om.jnt_qposadr[oid] + n]
                nd.qvel[nm.jnt_dofadr[nid]:nm.jnt_dofadr[nid] + nv] = od.qvel[om.jnt_dofadr[oid]:om.jnt_dofadr[oid] + nv]
            elif ot == int(OBJ.mjOBJ_ACTUATOR):
                nid = self.valid.idmap.id(e)
                if nid >= 0:
                    nd.ctrl[nid] = od.ctrl[oid]
        nd.time = od.time

    @contextmanager
    def batch(self, text: str):
        """Group edits into one undo step and compile once at the end."""
        self.undo.begin(text)
        self._batch += 1
        ok = False
        try:
            yield
            ok = True
        finally:
            self._batch -= 1
            if ok:
                self.undo.end()
            else:
                self.undo.abort()
            if self._batch == 0:
                if self._batch_dirty or not ok:
                    self._batch_dirty = False
                    if self.auto_compile:
                        self.recompile()

    def _on_undo(self, cmd: Cmd, action: str) -> None:
        if action in ("undo", "redo"):
            self._prune_selection()
        self.doc_changed.emit(cmd, action)
        if self._batch:
            self._batch_dirty = True
            return
        if self.auto_compile:
            self.recompile()

    def push(self, cmd: Cmd, merge: bool = False) -> None:
        if self._batch:
            self._batch_dirty = True
        self.undo.push(cmd, merge)

    def undo_step(self) -> None:
        self.undo.undo()

    def redo_step(self) -> None:
        self.undo.redo()

    # ================================================================ attribute edits
    def set_attrs(self, e: Element, changes: dict[str, str | None], text: str = "",
                  merge_key: str | None = None) -> None:
        ch = {k: v for k, v in changes.items() if e.get(k) != v}
        if not ch:
            return
        if "name" in ch:
            # route renames through rename() to keep references valid
            newname = ch.pop("name")
            with self.batch(text or "Rename"):
                self.rename(e, newname or "")
                if ch:
                    self.push(SetAttrs(e, ch, text))
            return
        self.push(SetAttrs(e, ch, text or f"Edit {self.doc.label(e)}", merge_key), merge=merge_key is not None)

    def set_attr(self, e: Element, key: str, value: str | None, text: str = "", merge_key: str | None = None) -> None:
        self.set_attrs(e, {key: value}, text or f"Set {key}", merge_key)

    def set_pose(self, e: Element, pos, quat, text: str = "Transform", merge_key: str | None = None) -> None:
        self.set_attrs(e, self.doc.pose_changes(e, pos, quat), text, merge_key)

    # ================================================================ naming / references
    def references_to(self, kind: str, name: str) -> list[tuple[Element, str]]:
        """(element, attribute) pairs that reference object *name* of reference *kind*."""
        out: list[tuple[Element, str]] = []
        if not name:
            return out
        sites = [r for r in reference_attrs() if r.kind == kind]
        for e in self.doc.root.iter():
            if not is_elem(e):
                continue
            pt = parent_tag(e)
            for r in sites:
                if r.tag != e.tag:
                    continue
                if r.parent_tag is not None and r.parent_tag != pt:
                    continue
                if r.parent_tag is None and pt in ("equality", "fixed", "spatial", "skin") and e.tag in (
                        "joint", "geom", "site", "connect", "weld", "tendon", "bone"):
                    continue
                if e.get(r.attr) == name:
                    out.append((e, r.attr))
            # frame sensors: objname/refname with objtype/reftype
            if pt == "sensor":
                for a_name, a_type in (("objname", "objtype"), ("refname", "reftype")):
                    if e.get(a_name) == name:
                        t = e.get(a_type)
                        tk = "body" if t in ("body", "xbody") else t
                        if tk == kind:
                            out.append((e, a_name))
        if kind == "class":
            for e in self.doc.root.iter():
                if is_elem(e):
                    for a in ("class", "childclass"):
                        if e.get(a) == name and e.tag != "default":
                            out.append((e, a))
        return out

    def rename(self, e: Element, new: str) -> None:
        old = e.get("class" if e.tag == "default" else "name") or ""
        key = "class" if e.tag == "default" else "name"
        new = new.strip()
        if new == old:
            return
        kind = "class" if e.tag == "default" else ref_kind(e)
        if new and kind and kind not in ("equality",) and new in self.doc.names(kind):
            raise ValueError(f"a {kind} named '{new}' already exists")
        if new.startswith("__mms_"):
            raise ValueError("names starting with '__mms_' are reserved")
        with self.batch(f"Rename {old or e.tag} -> {new}"):
            if old and kind:
                for ref, attr in self.references_to(kind, old):
                    if ref is e:
                        continue
                    if not new:
                        raise ValueError(f"'{old}' is referenced by {ref.tag}/{attr}; cannot clear its name")
                    self.push(SetAttrs(ref, {attr: new}, "update reference"))
            self.push(SetAttrs(e, {key: new or None}, "rename"))

    # ================================================================ structure edits
    def insert(self, parent: Element, elem: Element, index: int | None = None, text: str = "") -> Element:
        idx = len(parent) if index is None else index
        self.push(Insert(parent, idx, elem, text))
        return elem

    def dependents(self, elems: list[Element]) -> tuple[list[Element], list[tuple[Element, dict]]]:
        """What else must change when *elems* are deleted.

        Returns (extra elements to remove, [(element, attribute changes)]).
        Removal is transitive (a deleted joint removes its tendon, which removes
        the actuator driving that tendon). Soft references (camera/light target,
        material, texture, class) are cleared instead of deleting the referrer."""
        removed: set = set()
        for e in elems:
            removed.update(e.iter())
        extra: list[Element] = []
        clears: list[tuple[Element, dict]] = []
        frontier = list(elems)
        while frontier:
            nxt: list[Element] = []
            for top in frontier:
                for d in top.iter():
                    if not is_elem(d):
                        continue
                    kind = "class" if d.tag == "default" else ref_kind(d)
                    nm = d.get("class") if d.tag == "default" else self.doc.obj_name(d)
                    if not kind or not nm or kind == "equality":
                        continue
                    for ref, attr in self.references_to(kind, nm):
                        if ref in removed:
                            continue
                        if (ref.tag in ("camera", "light") and attr == "target") or \
                                attr in ("material", "texture", "class", "childclass"):
                            ch = {attr: None}
                            if attr == "target":
                                ch["mode"] = "fixed"
                            clears.append((ref, ch))
                            continue
                        tgt = ref
                        if parent_tag(ref) in ("fixed", "spatial", "skin"):
                            tgt = ref.getparent()
                        if tgt in removed:
                            continue
                        removed.update(tgt.iter())
                        extra.append(tgt)
                        nxt.append(tgt)
            frontier = nxt
        return extra, clears

    @contextmanager
    def structural(self, text: str):
        """Batch for edits that may change the joint/actuator/mocap layout:
        keyframes are remapped by element identity at the end."""
        old = self.valid
        with self.batch(text):
            yield
            self._remap_keyframes(old)

    def _remap_keyframes(self, old: CompileResult | None) -> None:
        keys = [k for kf in self.doc.sections("keyframe") for k in children(kf, "key")]
        if not keys or old is None:
            return
        new = compile_doc(self.doc, drop_tags=("keyframe",))
        if not new.ok:
            return
        om, nm = old.model, new.model
        J, A, B = int(OBJ.mjOBJ_JOINT), int(OBJ.mjOBJ_ACTUATOR), int(OBJ.mjOBJ_BODY)
        nq = {0: 7, 1: 4, 2: 1, 3: 1}
        nv = {0: 6, 1: 3, 2: 1, 3: 1}

        def remap(old_vec, new_vec, kind):
            for e, (ot, oid) in old.idmap.by_elem.items():
                nid = new.idmap.id(e)
                if nid < 0:
                    continue
                if kind in ("qpos", "qvel") and ot == J and nm.jnt_type[nid] == om.jnt_type[oid]:
                    t = int(om.jnt_type[oid])
                    if kind == "qpos":
                        n, oa, na = nq[t], om.jnt_qposadr[oid], nm.jnt_qposadr[nid]
                    else:
                        n, oa, na = nv[t], om.jnt_dofadr[oid], nm.jnt_dofadr[nid]
                    new_vec[na:na + n] = old_vec[oa:oa + n]
                elif kind == "ctrl" and ot == A:
                    new_vec[nid] = old_vec[oid]
                elif kind in ("mpos", "mquat") and ot == B:
                    om_id, nm_id = om.body_mocapid[oid], nm.body_mocapid[nid]
                    if om_id >= 0 and nm_id >= 0:
                        n = 3 if kind == "mpos" else 4
                        new_vec[nm_id * n:nm_id * n + n] = old_vec[om_id * n:om_id * n + n]
            return new_vec

        sizes = {"qpos": (om.nq, nm.nq, nm.qpos0), "qvel": (om.nv, nm.nv, np.zeros(nm.nv)),
                 "ctrl": (om.nu, nm.nu, np.zeros(nm.nu)),
                 "mpos": (3 * om.nmocap, 3 * nm.nmocap, None), "mquat": (4 * om.nmocap, 4 * nm.nmocap, None)}
        for k in keys:
            ch: dict[str, str | None] = {}
            for attr, (on, nn, init) in sizes.items():
                v = mx.parse_vec(k.get(attr))
                if v is None or (on == nn and v.size == nn and self._same_layout(old, new)):
                    continue
                if v.size != on:
                    continue
                if init is None:
                    init = np.zeros(nn)
                    if attr == "mquat":
                        init[0::4] = 1.0
                ch[attr] = mx.fmt_vec(remap(v, np.array(init, float).copy(), attr)) if nn else None
            if k.get("act") is not None and om.na != nm.na:
                ch["act"] = None
                log.warning("keyframe '%s': act vector dropped (actuator state layout changed)", k.get("name", ""))
            if ch:
                self.push(SetAttrs(k, ch, "remap keyframe"))

    @staticmethod
    def _same_layout(old: CompileResult, new: CompileResult) -> bool:
        om, nm = old.model, new.model
        if om.njnt != nm.njnt or om.nu != nm.nu:
            return False
        for e, (ot, oid) in old.idmap.by_elem.items():
            if ot in (int(OBJ.mjOBJ_JOINT), int(OBJ.mjOBJ_ACTUATOR)) and new.idmap.id(e) != oid:
                return False
        return True

    def delete(self, elems: list[Element], cascade: bool = True) -> list[Element]:
        """Delete elements (and, with cascade, actuators/sensors/... that reference them)."""
        elems = [e for e in elems if e.getparent() is not None]
        # drop elements nested in other deleted elements
        top = [e for e in elems if not any(a in elems for a in e.iterancestors())]
        if not top:
            return []
        deps, clears = self.dependents(top) if cascade else ([], [])
        with self.structural(f"Delete {len(top)} element(s)"):
            for ref, ch in clears:
                log.info("clearing reference %s on %s", ",".join(ch), self.doc.label(ref))
                self.push(SetAttrs(ref, ch, "clear reference"))
            for d in deps:
                if d.getparent() is not None:
                    log.info("also removing dependent %s %s", d.tag, self.doc.label(d))
                    self.push(Remove(d))
            for e in top:
                if e.getparent() is not None:
                    self.push(Remove(e))
        self._prune_selection()
        return top + deps

    def move(self, e: Element, new_parent: Element, index: int | None = None, keep_world: bool = True,
             text: str = "") -> None:
        """Re-parent an element; with keep_world the world pose is preserved."""
        if new_parent is e or new_parent in list(e.iterdescendants()):
            raise ValueError("cannot move an element into itself")
        with self.structural(text or f"Move {self.doc.label(e)}"):
            if keep_world and e.tag in ("body", "geom", "site", "camera", "light", "joint", "frame") and self.valid:
                wp, wq = self.world_pose(e)
                pp, pq = self.world_frame_of(new_parent)
                ip, iq = mx.pose_inv(pp, pq)
                lp, lq = mx.pose_mul(ip, iq, wp, wq)
                self.push(Move(e, new_parent, index))
                self.push(SetAttrs(e, self.doc.pose_changes(e, lp, lq), "keep world pose"))
            else:
                self.push(Move(e, new_parent, index))

    # ================================================================ selection
    def select(self, elems: list[Element] | Element | None, add: bool = False) -> None:
        if elems is None:
            elems = []
        if is_elem(elems):  # type: ignore[arg-type]
            elems = [elems]  # type: ignore[list-item]
        new = list(self.selection) if add else []
        for e in elems:  # type: ignore[union-attr]
            if e in new:
                if add:
                    new.remove(e)
            else:
                new.append(e)
        if new != self.selection:
            self.selection = new
            self.selection_changed.emit(self.selection)

    def _prune_selection(self) -> None:
        alive = [e for e in self.selection if _attached(self.doc.root, e)]
        if alive != self.selection:
            self.selection = alive
            self.selection_changed.emit(self.selection)

    # ================================================================ kinematics
    def world_frame_of(self, e: Element) -> tuple[np.ndarray, np.ndarray]:
        """World pose of the frame that children of *e* are expressed in."""
        if e.tag == "worldbody" or e.tag == "mujoco":
            return np.zeros(3), mx.QID.copy()
        if e.tag == "body":
            return self.world_pose(e)
        if e.tag == "frame":
            return self.world_pose(e)
        p = e.getparent()
        return self.world_frame_of(p) if p is not None else (np.zeros(3), mx.QID.copy())

    def world_pose(self, e: Element) -> tuple[np.ndarray, np.ndarray]:
        """World pose of an element's own frame (bodies from MuJoCo data when available)."""
        if e.tag == "body" and self.valid is not None and self.data is not None:
            bid = self.valid.idmap.id(e)
            if bid >= 0:
                return self.data.xpos[bid].copy(), self.data.xquat[bid].copy()
        parent = e.getparent()
        pp, pq = self.world_frame_of(parent) if parent is not None else (np.zeros(3), mx.QID.copy())
        lp, lq = self.doc.local_pose(e)
        if e.tag in ("geom", "site") and parent is not None and parent.tag == "body":
            pass
        return mx.pose_mul(pp, pq, lp, lq)

    def joints(self) -> list[dict]:
        """Joint table of the compiled model (for sliders and APIs)."""
        m = self.model
        if m is None:
            return []
        out = []
        for j in range(m.njnt):
            e = self.valid.idmap.elem(int(OBJ.mjOBJ_JOINT), j)
            out.append({
                "id": j, "elem": e, "name": mujoco.mj_id2name(m, OBJ.mjOBJ_JOINT, j) or "",
                "label": (e.get("name") if e is not None and e.get("name") else f"joint {j}"),
                "type": int(m.jnt_type[j]), "qposadr": int(m.jnt_qposadr[j]),
                "dofadr": int(m.jnt_dofadr[j]), "limited": bool(m.jnt_limited[j]),
                "range": m.jnt_range[j].copy(), "body": int(m.jnt_bodyid[j]),
            })
        return out

    def joint_id(self, ref: Element | str | int) -> int:
        m = self.model
        if m is None:
            return -1
        if isinstance(ref, int):
            return ref
        if isinstance(ref, str):
            e = self.doc.find_named("joint", ref)
            if e is None:
                return mujoco.mj_name2id(m, OBJ.mjOBJ_JOINT, ref)
            ref = e
        return self.valid.idmap.id(ref)

    def get_qpos(self, joint: Element | str | int) -> np.ndarray:
        j = self.joint_id(joint)
        m = self.model
        n = {0: 7, 1: 4, 2: 1, 3: 1}[int(m.jnt_type[j])]
        a = m.jnt_qposadr[j]
        return self.data.qpos[a:a + n].copy()

    def set_qpos(self, joint: Element | str | int, value, forward: bool = True) -> None:
        j = self.joint_id(joint)
        if j < 0:
            raise KeyError(f"unknown joint {joint!r}")
        m = self.model
        n = {0: 7, 1: 4, 2: 1, 3: 1}[int(m.jnt_type[j])]
        a = m.jnt_qposadr[j]
        v = np.atleast_1d(np.asarray(value, float))
        if v.size != n:
            raise ValueError(f"joint expects {n} values")
        self.data.qpos[a:a + n] = v
        if forward:
            self.forward()

    def forward(self) -> None:
        if self.model is not None:
            mujoco.mj_forward(self.model, self.data)
            self.state_changed.emit()

    def reset_pose(self, keyframe: int | None = None) -> None:
        if self.model is None:
            return
        if keyframe is not None and 0 <= keyframe < self.model.nkey:
            mujoco.mj_resetDataKeyframe(self.model, self.data, keyframe)
        else:
            mujoco.mj_resetData(self.model, self.data)
        self.forward()


def _attached(root: Element, e: Element) -> bool:
    if e is root:
        return True
    for a in e.iterancestors():
        if a is root:
            return True
    return False


def angle_display(doc: MjcfDoc, rad: float) -> float:
    return math.degrees(rad)
