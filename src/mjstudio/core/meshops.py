"""Mesh assets in the document: import, primitives, visual/collision geoms, editing.

Generated or edited meshes live in ``doc.mem_assets`` (OBJ bytes keyed by the
<mesh file> value) until the model/project is saved. Changing them goes through
the undo stack (SetMemAsset). Original mesh files are never overwritten: the first
edit of a file-based mesh switches it to a new '<stem>_edit.obj' asset.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

from . import mathx as mx
from . import meshes as M
from .doc import Element, children, parent_tag
from .editor import Editor
from .ops import _index, _q, _section_index, body_target, make
from .undo import Cmd, Insert, SetAttrs

log = logging.getLogger("mjstudio")
VISUAL = {"contype": "0", "conaffinity": "0", "group": "2"}
COLLISION_RGBA = "0.35 0.65 1 0.35"
COLLISION_TYPES = ("hull", "box", "sphere", "capsule", "cylinder", "ellipsoid", "none")


class SetMemAsset(Cmd):
    """Set/replace (or delete with None) an in-memory asset file."""

    def __init__(self, ed: Editor, key: str, data: bytes | None, text: str = "Edit mesh data"):
        self.ed, self.key, self.new, self.text = ed, key, data, text
        doc = ed.doc
        if key in doc.mem_assets:
            self.old = doc.mem_assets[key]
        else:
            p = doc.asset_dir("mesh") / key
            self.old = p.read_bytes() if p.exists() else None

    def _set(self, data):
        if data is None:
            self.ed.doc.mem_assets.pop(self.key, None)
        else:
            self.ed.doc.mem_assets[self.key] = data

    def redo(self) -> None:
        self._set(self.new)

    def undo(self) -> None:
        self._set(self.old)

    def elems(self) -> list[Element]:
        return [e for e in self.ed.doc.iter_tags("mesh") if e.get("file") == self.key]


# ---------------------------------------------------------------- helpers
def _unique_key(ed: Editor, stem: str, ext: str = "obj") -> str:
    base = ed.doc.asset_dir("mesh")
    k, i = f"{stem}.{ext}", 1
    while k in ed.doc.mem_assets or (base / k).exists() or any(
            e.get("file") == k for e in ed.doc.iter_tags("mesh", "hfield", "skin")):
        k = f"{stem}_{i}.{ext}"
        i += 1
    return k


def _asset_section(ed: Editor) -> Element:
    sec = ed.doc.section("asset")
    if sec is None:
        sec = make("asset")
        ed.push(Insert(ed.doc.root, _section_index(ed, "asset"), sec, "Add asset section"))
    return sec


def mesh_scale(ed: Editor, mesh_elem: Element) -> np.ndarray:
    s = mx.parse_vec(ed.doc.get(mesh_elem, "scale") or "1 1 1", 3)
    return s if s is not None else np.ones(3)


# ---------------------------------------------------------------- creation
def add_mesh_asset(ed: Editor, mesh: M.TriMesh, name: str | None = None, ext: str = "obj",
                   scale=None) -> Element:
    issues = mesh.validate()
    if issues:
        raise ValueError("invalid mesh: " + "; ".join(issues))
    nm = ed.doc.unique_name("mesh", name or mesh.name)
    key = _unique_key(ed, nm, ext)
    e = make("mesh", name=nm, file=key, scale=None if scale is None else mx.fmt_vec(np.broadcast_to(scale, 3)))
    from .compiler import supports
    if mesh.is_planar() and supports("mesh_inertia"):
        e.set("inertia", "shell")   # newer MuJoCo computes mesh volume even for visual-only use
    with ed.batch(f"Add mesh {nm}"):
        ed.push(SetMemAsset(ed, key, M.encode(mesh, ext), f"Add mesh data {nm}"))
        sec = _asset_section(ed)
        ed.push(Insert(sec, len(sec), e, f"Add mesh {nm}"))
    return e


def mesh_geoms(ed: Editor, body: Element, mesh_name: str, visual: bool = True, collision: str = "hull",
               pos=None, quat=None, rgba: str | None = None, planar: bool = False,
               name: str | None = None) -> list[Element]:
    """Visual geom (group 2, no contacts) and/or collision geom (group 3) for a mesh asset.
    collision: 'hull' (MuJoCo convex hull), a primitive type fitted by MuJoCo, or 'none'."""
    if collision not in COLLISION_TYPES:
        raise ValueError(f"collision must be one of {COLLISION_TYPES}")
    if planar and collision != "none":
        log.warning("mesh '%s' is flat: no collision geom created (a convex hull needs volume)", mesh_name)
        collision = "none"
    q = _q(quat)
    out: list[Element] = []
    with ed.batch(f"Add geoms for mesh {mesh_name}"):
        if visual:
            g = make("geom", name=name, type="mesh", mesh=mesh_name, pos=pos, quat=q, rgba=rgba, **VISUAL)
            if collision != "none" or planar:
                g.set("mass", "0")  # the collision geom (or nothing, for flat meshes) carries the mass
            ed.push(Insert(body, _index(body, "geom"), g, "Add visual mesh geom"))
            out.append(g)
        if collision != "none":
            ctype = "mesh" if collision == "hull" else collision
            g = make("geom", name=(f"{name}_collision" if name else None), type=ctype, mesh=mesh_name, pos=pos,
                     quat=q, group="3", rgba=COLLISION_RGBA)
            ed.push(Insert(body, _index(body, "geom"), g, "Add collision geom"))
            out.append(g)
    return out


def create_primitive_mesh(ed: Editor, parent: Element | None, kind: str, collision: str = "hull",
                          name: str | None = None, pos=None, **params) -> dict:
    if kind not in M.PRIMITIVES:
        raise ValueError(f"unknown primitive {kind}")
    mesh = M.PRIMITIVES[kind](**params)
    body = body_target(ed, parent)
    with ed.batch(f"Add {kind} mesh"):
        me = add_mesh_asset(ed, mesh, name or kind)
        geoms = mesh_geoms(ed, body, me.get("name"), True, collision, pos=pos, planar=mesh.is_planar())
    return {"mesh": me, "body": body, "geoms": geoms}


def import_mesh(ed: Editor, path: str | os.PathLike, parent: Element | None = None, new_body: bool = True,
                name: str | None = None, units: str = "m", axes: str = "mujoco", collision: str = "hull",
                pos=(0, 0, 0)) -> dict:
    """Reference an OBJ/STL file as a mesh asset (units -> <mesh scale>, axis preset ->
    geom orientation) and attach visual/collision geoms to a new or existing body."""
    p = Path(path).resolve()
    if units not in M.UNITS:
        raise ValueError(f"units must be one of {list(M.UNITS)}")
    if p.suffix.lower() in (".glb", ".gltf"):
        return _import_gltf_mesh(ed, p, parent, new_body, name, units, collision, pos)
    data = M.read_mesh_file(p)           # validates readability, gives planarity/stats
    if axes not in M.AXES:
        raise ValueError(f"axes must be one of {list(M.AXES)}")
    nm = ed.doc.unique_name("mesh", name or p.stem)
    try:
        rel = Path(os.path.relpath(p, ed.doc.asset_dir("mesh"))).as_posix()
    except ValueError:
        rel = p.as_posix()
    s = M.UNITS[units]
    me = make("mesh", name=nm, file=rel, scale=None if s == 1.0 else mx.fmt_vec([s, s, s]))
    with ed.structural(f"Import mesh {p.name}"):
        sec = _asset_section(ed)
        ed.push(Insert(sec, len(sec), me, f"Import mesh {nm}"))
        if new_body:
            from .ops import add_body
            body = add_body(ed, parent, ed.doc.unique_name("body", nm), pos=pos)
        else:
            body = body_target(ed, parent)
        geoms = mesh_geoms(ed, body, nm, True, collision, quat=M.AXES[axes], planar=data.is_planar())
    log.info("Imported %s: %d vertices, %d faces (units %s, axes %s)", p.name, data.nv, data.nf, units, axes)
    return {"mesh": me, "body": body, "geoms": geoms, "stats": (data.nv, data.nf)}


# ---------------------------------------------------------------- data access / editing
def get_mesh_data(ed: Editor, mesh_elem: Element) -> M.TriMesh:
    """Mesh data in its file frame (before <mesh scale>)."""
    doc = ed.doc
    key = mesh_elem.get("file")
    name = doc.obj_name(mesh_elem) or "mesh"
    if key and key in doc.mem_assets:
        ext = Path(key).suffix.lower()
        raw = doc.mem_assets[key]
        return M.read_obj(raw, name) if ext == ".obj" else M.read_stl(raw, name)
    if key:
        p = doc.resolve_file(mesh_elem)
        if p is not None and Path(p).exists() and Path(p).suffix.lower() in (".obj", ".stl"):
            return M.read_mesh_file(p)
    v = mx.parse_vec(mesh_elem.get("vertex"))
    if v is not None:
        f = mx.parse_vec(mesh_elem.get("face"))
        if f is None:
            raise ValueError("inline mesh without faces cannot be edited")
        return M.TriMesh(v.reshape(-1, 3), f.astype(int).reshape(-1, 3), name)
    if ed.model is not None and ed.idmap is not None and ed.idmap.id(mesh_elem) >= 0:
        return M.from_compiled(ed.model, ed.idmap.id(mesh_elem), mesh_scale(ed, mesh_elem))
    raise ValueError(f"no editable data for mesh '{name}'")


def set_mesh_data(ed: Editor, mesh_elem: Element, mesh: M.TriMesh, text: str = "Edit mesh") -> None:
    issues = mesh.validate()
    if issues:
        raise ValueError("invalid mesh: " + "; ".join(issues))
    doc = ed.doc
    with ed.batch(text):
        if mesh_elem.get("vertex") is not None and not mesh_elem.get("file"):
            ed.push(SetAttrs(mesh_elem, {"vertex": mx.fmt_vec(mesh.V.ravel()),
                                         "face": " ".join(str(int(i)) for i in mesh.F.ravel())}, text))
            return
        key = mesh_elem.get("file")
        if not key or key not in doc.mem_assets:
            # never overwrite a file on disk: branch to a new in-memory asset
            stem = Path(key).stem if key else (doc.obj_name(mesh_elem) or "mesh")
            new_key = _unique_key(ed, f"{stem}_edit", "obj")
            if not mesh_elem.get("name"):
                ed.push(SetAttrs(mesh_elem, {"name": doc.obj_name(mesh_elem)}, "keep mesh name"))
            ed.push(SetMemAsset(ed, new_key, M.encode(mesh, "obj"), text))
            ed.push(SetAttrs(mesh_elem, {"file": new_key, "vertex": None, "face": None}, text))
        else:
            ed.push(SetMemAsset(ed, key, M.encode(mesh, Path(key).suffix), text))


def export_mesh(ed: Editor, mesh_elem: Element, path: str | os.PathLike, apply_scale: bool = True) -> Path:
    m = get_mesh_data(ed, mesh_elem)
    if apply_scale:
        m = m.transformed(scale=mesh_scale(ed, mesh_elem))
    p = Path(path)
    p.write_bytes(M.encode(m, p.suffix))
    return p


def duplicate_mesh_asset(ed: Editor, mesh_elem: Element, name: str | None = None) -> Element:
    m = get_mesh_data(ed, mesh_elem)
    sc = mesh_elem.get("scale")
    return add_mesh_asset(ed, m, name or f"{ed.doc.obj_name(mesh_elem)}_copy",
                          scale=None if sc is None else mx.parse_vec(sc, 3))


def geom_mesh_elem(ed: Editor, geom: Element) -> Element | None:
    nm = ed.doc.get(geom, "mesh")
    return ed.doc.find_named("mesh", nm) if nm else None


def mesh_users(ed: Editor, mesh_elem: Element) -> list[Element]:
    nm = ed.doc.obj_name(mesh_elem)
    return [g for g in ed.doc.iter_tags("geom") if ed.doc.get(g, "mesh") == nm]


def _import_gltf_mesh(ed: Editor, p: Path, parent, new_body, name, units, collision, pos) -> dict:
    """glTF has a fixed axis convention (Y up, +Z front): node transforms and the axis change
    are baked into the vertices, which are stored as an in-memory OBJ asset."""
    from . import gltf
    sc = gltf.load(p)
    if not sc.meshes:
        raise ValueError(f"no mesh in {p}")
    Vs, Fs, off = [], [], 0
    for gm in sc.meshes:
        Vs.append(gltf.to_mujoco(gm.V, gm.world))
        Fs.append(gm.F + off)
        off += len(gm.V)
    m = M.TriMesh(np.vstack(Vs) * M.UNITS[units], np.vstack(Fs), name or p.stem)
    m.weld(1e-9)
    with ed.structural(f"Import mesh {p.name}"):
        me = add_mesh_asset(ed, m, name or p.stem)
        if new_body:
            from .ops import add_body
            body = add_body(ed, parent, ed.doc.unique_name("body", me.get("name")), pos=pos)
        else:
            body = body_target(ed, parent)
        geoms = mesh_geoms(ed, body, me.get("name"), True, collision, planar=m.is_planar())
    log.info("Imported %s: %d vertices, %d faces from %d glTF mesh(es)", p.name, m.nv, m.nf, len(sc.meshes))
    return {"mesh": me, "body": body, "geoms": geoms, "stats": (m.nv, m.nf)}
