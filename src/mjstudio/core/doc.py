"""MJCF document model.

The lxml element tree *is* the document: everything that is not understood by
the editor (unknown elements/attributes, comments, plugins, ...) is preserved
verbatim. Editor features read and write attributes on this tree; compiling to
MuJoCo is done on a copy (see compiler.py).
"""
from __future__ import annotations

import copy
import math
import os
from pathlib import Path
from typing import Iterator

import numpy as np
from lxml import etree

from . import mathx as mx
from .schema import ACTUATORS, ORIENT_KEYS, attr_spec, NAMED_KINDS, SECTIONS

Element = etree._Element

_PARSER = etree.XMLParser(remove_blank_text=True, remove_comments=False, resolve_entities=False,
                          huge_tree=True)

# canonical order used when a missing section must be created
_SECTION_ORDER = {t: i for i, t in enumerate(SECTIONS)}

# file-valued attributes and which asset directory they are relative to
FILE_ATTRS = {
    "mesh": (("file",), "mesh"),
    "hfield": (("file",), "mesh"),
    "skin": (("file",), "mesh"),
    "texture": (("file", "fileright", "fileleft", "fileup", "filedown", "filefront", "fileback"), "texture"),
}

FROMTO_TYPES = {"capsule", "cylinder", "box", "ellipsoid"}

DEFAULT_TAG_FOR = {  # element tag -> tag used inside <default>
    "geom": "geom", "joint": "joint", "site": "site", "camera": "camera", "light": "light",
    "mesh": "mesh", "material": "material", "pair": "pair", "spatial": "tendon", "fixed": "tendon",
}


class MjcfError(Exception):
    pass


def is_elem(e) -> bool:
    return isinstance(e, etree._Element) and isinstance(e.tag, str)


def children(e: Element, tag: str | None = None) -> list[Element]:
    return [c for c in e if is_elem(c) and (tag is None or c.tag == tag)]


def parent_tag(e: Element) -> str | None:
    p = e.getparent()
    return p.tag if p is not None and is_elem(p) else None


class MjcfDoc:
    """An editable MJCF model."""

    def __init__(self, root: Element, path: str | os.PathLike | None = None,
                 base_dir: str | os.PathLike | None = None):
        if root.tag != "mujoco":
            raise MjcfError(f"root element must be <mujoco>, got <{root.tag}>")
        self.root = root
        self.path: Path | None = Path(path).resolve() if path else None
        if base_dir is not None:
            self.base_dir = Path(base_dir).resolve()
        elif self.path is not None:
            self.base_dir = self.path.parent
        else:
            self.base_dir = Path.cwd()
        self.meta: dict = {"flattened_includes": []}
        # in-memory asset files (relative path -> bytes), e.g. generated meshes not yet saved
        self.mem_assets: dict[str, bytes] = {}

    # ------------------------------------------------------------ creation
    @classmethod
    def load(cls, path: str | os.PathLike) -> "MjcfDoc":
        path = Path(path).resolve()
        try:
            tree = etree.parse(str(path), _PARSER)
        except etree.XMLSyntaxError as ex:
            raise MjcfError(f"XML syntax error in {path}: {ex}") from ex
        doc = cls(tree.getroot(), path=path)
        doc._flatten_includes()
        return doc

    @classmethod
    def from_string(cls, text: str, base_dir: str | os.PathLike | None = None,
                    path: str | os.PathLike | None = None) -> "MjcfDoc":
        try:
            root = etree.fromstring(text.encode("utf-8"), _PARSER)
        except etree.XMLSyntaxError as ex:
            raise MjcfError(f"XML syntax error: {ex}") from ex
        doc = cls(root, path=path, base_dir=base_dir)
        doc._flatten_includes()
        return doc

    @classmethod
    def new(cls, model: str = "model") -> "MjcfDoc":
        xml = f"""<mujoco model="{model}">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <visual><global offwidth="1920" offheight="1080"/></visual>
  <asset>
    <texture name="skybox" type="skybox" builtin="gradient" rgb1="0.25 0.3 0.38" rgb2="0.05 0.06 0.08" width="64" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.16 0.18 0.21" rgb2="0.22 0.24 0.28" width="512" height="512" mark="edge" markrgb="0.3 0.32 0.36"/>
    <material name="grid" texture="grid" texrepeat="8 8" texuniform="true" reflectance="0.1"/>
  </asset>
  <worldbody>
    <light name="sun" pos="0 0 4" dir="0 0 -1" directional="true" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="grid"/>
  </worldbody>
</mujoco>"""
        return cls.from_string(xml)

    def clone(self) -> "MjcfDoc":
        d = MjcfDoc(copy.deepcopy(self.root), path=self.path, base_dir=self.base_dir)
        d.meta = copy.deepcopy(self.meta)
        d.mem_assets = dict(self.mem_assets)
        return d

    # ------------------------------------------------------------ includes
    def _flatten_includes(self, depth: int = 0) -> None:
        """Inline <include file=...> (MuJoCo resolves paths relative to the main file)."""
        if depth > 20:
            raise MjcfError("include nesting too deep (cycle?)")
        incs = [e for e in self.root.iter("include")]
        if not incs:
            return
        for inc in incs:
            f = inc.get("file")
            if not f:
                raise MjcfError("<include> without file attribute")
            fp = (self.base_dir / f) if not os.path.isabs(f) else Path(f)
            try:
                sub = etree.parse(str(fp), _PARSER).getroot()
            except (OSError, etree.XMLSyntaxError) as ex:
                raise MjcfError(f"cannot include {fp}: {ex}") from ex
            if sub.tag != "mujoco":
                raise MjcfError(f"included file {fp} must have a <mujoco> root")
            parent = inc.getparent()
            idx = parent.index(inc)
            parent.remove(inc)
            parent.insert(idx, etree.Comment(f" mjstudio: begin include {f} "))
            idx += 1
            for c in list(sub):
                parent.insert(idx, c)
                idx += 1
            parent.insert(idx, etree.Comment(f" mjstudio: end include {f} "))
            self.meta["flattened_includes"].append(str(f))
        self._flatten_includes(depth + 1)

    # ------------------------------------------------------------ serialisation
    def to_string(self) -> str:
        root = self.root
        etree.indent(root, space="  ")
        # serialise the whole document so comments before/after <mujoco> survive
        return etree.tostring(root.getroottree(), pretty_print=True, encoding="unicode")

    def save(self, path: str | os.PathLike, relocate_assets: bool = True) -> list[str]:
        """Write MJCF to *path*. If the directory changes, relative asset paths are
        rewritten so they keep pointing to the same files. Returns notes."""
        path = Path(path).resolve()
        notes: list[str] = []
        if relocate_assets and path.parent != self.base_dir:
            notes = self.relocate(path.parent)
        path.parent.mkdir(parents=True, exist_ok=True)
        notes += self._write_mem_assets(path.parent)
        text = self.to_string()
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
        self.path = path
        self.base_dir = path.parent
        return notes

    def _write_mem_assets(self, model_dir: Path) -> list[str]:
        """Write generated/edited asset files into <model_dir>/meshes and point the model at them."""
        notes: list[str] = []
        if not self.mem_assets:
            return notes
        model_dir = Path(model_dir).resolve()
        old_base = self.base_dir
        self.base_dir = model_dir           # asset_dir() must be evaluated for the new location
        try:
            out = model_dir / "meshes"
            mesh_dir = self.asset_dir("mesh")
            for key in list(self.mem_assets):
                data = self.mem_assets[key]
                target = out / Path(key).name
                mine = (mesh_dir / key).resolve() == target.resolve()
                i = 1
                while target.exists() and not mine and target.read_bytes() != data:
                    target = out / f"{Path(key).stem}_{i}{Path(key).suffix}"
                    i += 1
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                try:
                    rel = Path(os.path.relpath(target, mesh_dir)).as_posix()
                except ValueError:
                    rel = target.as_posix()
                for tag in ("mesh", "hfield", "skin"):
                    for e in self.root.iter(tag):
                        if e.get("file") == key:
                            e.set("file", rel)
                self.mem_assets[rel] = self.mem_assets.pop(key)
                notes.append(f"wrote {target}")
        finally:
            self.base_dir = old_base
        return notes

    def relocate(self, new_base: Path) -> list[str]:
        """Rewrite relative asset paths for a new base directory (in place)."""
        new_base = Path(new_base).resolve()
        notes: list[str] = []

        def rel(abs_path: Path) -> str:
            try:
                return Path(os.path.relpath(abs_path, new_base)).as_posix()
            except ValueError:  # different drive on Windows
                return abs_path.as_posix()

        dirs_set = {"mesh": False, "texture": False}
        for comp in self.sections("compiler"):
            for key in ("meshdir", "texturedir", "assetdir"):
                v = comp.get(key)
                if v is None:
                    continue
                if key in ("meshdir", "assetdir"):
                    dirs_set["mesh"] = True
                if key in ("texturedir", "assetdir"):
                    dirs_set["texture"] = True
                if not os.path.isabs(v):
                    nv = rel(self.base_dir / v)
                    if nv != v:
                        comp.set(key, nv)
                        notes.append(f"compiler/{key}: {v} -> {nv}")
        for tag, (keys, kind) in FILE_ATTRS.items():
            if dirs_set[kind]:
                continue
            for e in self.root.iter(tag):
                if self.in_defaults(e):
                    continue
                for k in keys:
                    v = e.get(k)
                    if v is None or os.path.isabs(v) or v in self.mem_assets:
                        continue
                    nv = rel(self.base_dir / v)
                    if nv != v:
                        e.set(k, nv)
                        notes.append(f"{tag} {e.get('name', '')}/{k}: {v} -> {nv}")
        self.base_dir = new_base
        return notes

    # ------------------------------------------------------------ sections
    def sections(self, tag: str) -> list[Element]:
        return children(self.root, tag)

    def section(self, tag: str, create: bool = False) -> Element | None:
        s = self.sections(tag)
        if s:
            return s[0]
        if not create:
            return None
        e = etree.Element(tag)
        order = _SECTION_ORDER.get(tag, len(_SECTION_ORDER))
        idx = len(self.root)
        for i, c in enumerate(self.root):
            if is_elem(c) and _SECTION_ORDER.get(c.tag, 999) > order:
                idx = i
                break
        self.root.insert(idx, e)
        return e

    @property
    def worldbody(self) -> Element:
        return self.section("worldbody", create=True)

    @property
    def model_name(self) -> str:
        return self.root.get("model", "")

    # ------------------------------------------------------------ compiler settings
    def compiler_attr(self, key: str, default: str | None = None) -> str | None:
        val = default
        for c in self.sections("compiler"):
            if c.get(key) is not None:
                val = c.get(key)
        return val

    @property
    def angle_unit(self) -> str:
        return self.compiler_attr("angle", "degree")

    @property
    def eulerseq(self) -> str:
        return self.compiler_attr("eulerseq", "xyz")

    def to_rad(self, v: float) -> float:
        return math.radians(v) if self.angle_unit == "degree" else v

    def from_rad(self, v: float) -> float:
        return math.degrees(v) if self.angle_unit == "degree" else v

    def asset_dir(self, kind: str) -> Path:
        """Absolute directory for 'mesh' or 'texture' files."""
        key = "meshdir" if kind == "mesh" else "texturedir"
        d = self.compiler_attr(key) or self.compiler_attr("assetdir")
        if not d:
            return self.base_dir
        p = Path(d)
        return p if p.is_absolute() else (self.base_dir / p)

    def resolve_file(self, elem: Element, key: str = "file") -> Path | None:
        v = elem.get(key)
        if not v:
            return None
        kind = FILE_ATTRS.get(elem.tag, ((), "mesh"))[1]
        if self.compiler_attr("strippath", "false") == "true":
            v = os.path.basename(v)
        p = Path(v)
        return p if p.is_absolute() else (self.asset_dir(kind) / p)

    # ------------------------------------------------------------ queries
    def in_defaults(self, e: Element) -> bool:
        for a in e.iterancestors():
            if a.tag == "default":
                return True
        return e.tag == "default"

    def iter_tags(self, *tags: str, include_defaults: bool = False) -> Iterator[Element]:
        for e in self.root.iter(*tags):
            if include_defaults or not self.in_defaults(e):
                yield e

    def obj_name(self, e: Element) -> str | None:
        """Name used to reference *e* (file stem for unnamed file assets, as MuJoCo does)."""
        n = e.get("name")
        if n:
            return n
        if e.tag in ("mesh", "hfield", "skin", "texture") and e.get("file"):
            return Path(e.get("file").replace("\\", "/")).stem
        return None

    def names(self, kind: str) -> list[str]:
        """Names of elements of a reference kind (body, joint, mesh, class, ...)."""
        if kind == "class":
            return list(self.default_classes().keys())
        if kind == "body":
            out = ["world"]
        else:
            out = []
        tags = NAMED_KINDS.get(kind, (kind,))
        for e in self.iter_tags(*tags):
            if kind == "joint" and parent_tag(e) in ("fixed", "equality"):
                continue
            if kind == "geom" and parent_tag(e) == "spatial":
                continue
            if kind == "site" and parent_tag(e) == "spatial":
                continue
            n = self.obj_name(e)
            if n:
                out.append(n)
        return out

    def find_named(self, kind: str, name: str) -> Element | None:
        tags = NAMED_KINDS.get(kind, (kind,))
        for e in self.iter_tags(*tags):
            if self.obj_name(e) == name and not (parent_tag(e) in ("fixed", "spatial", "equality")
                                              and e.tag in ("joint", "geom", "site")):
                return e
        return None

    def unique_name(self, kind: str, base: str) -> str:
        used = set(self.names(kind))
        if base not in used:
            return base
        stem = base.rstrip("0123456789").rstrip("_") or base
        i = 1
        while f"{stem}_{i}" in used:
            i += 1
        return f"{stem}_{i}"

    # ------------------------------------------------------------ defaults
    def default_classes(self) -> dict[str, Element]:
        out: dict[str, Element] = {}

        def walk(d: Element, top: bool):
            name = d.get("class") or ("main" if top else None)
            if name and name not in out:
                out[name] = d
            for c in children(d, "default"):
                walk(c, False)

        for d in self.sections("default"):
            walk(d, True)
        return out

    def class_parent(self, d: Element) -> Element | None:
        p = d.getparent()
        return p if p is not None and p.tag == "default" else None

    def elem_class(self, e: Element) -> str:
        c = e.get("class")
        if c:
            return c
        if e.tag in ("geom", "joint", "site", "camera", "light", "frame"):
            for a in e.iterancestors():
                if a.tag in ("body", "frame") and a.get("childclass"):
                    return a.get("childclass")
                if a.tag == "worldbody":
                    break
        return "main"

    def default_chain(self, cls: str) -> list[Element]:
        classes = self.default_classes()
        d = classes.get(cls)
        chain: list[Element] = []
        while d is not None:
            chain.append(d)
            d = self.class_parent(d)
        return chain[::-1]

    def _default_tags(self, e: Element) -> tuple[str, ...]:
        if parent_tag(e) == "equality":
            return ("equality",)
        if e.tag in ACTUATORS:
            return tuple(ACTUATORS.keys())
        t = DEFAULT_TAG_FOR.get(e.tag)
        return (t,) if t else ()

    def resolved(self, e: Element) -> dict[str, tuple[str, str]]:
        """attr -> (value, source) with source 'explicit' | 'class:<name>' | 'builtin'."""
        out: dict[str, tuple[str, str]] = {}
        ptag = parent_tag(e)
        from .schema import attrs_for
        for a in attrs_for(e.tag, ptag):
            if a.default is not None:
                out[a.name] = (a.default, "builtin")
        if not self.in_defaults(e) and e.tag != "freejoint":
            dtags = self._default_tags(e)
            if dtags:
                for d in self.default_chain(self.elem_class(e)):
                    for c in children(d):
                        if c.tag in dtags:
                            for k, v in c.attrib.items():
                                out[k] = (v, f"class:{d.get('class') or 'main'}")
        for k, v in e.attrib.items():
            out[k] = (v, "explicit")
        return out

    def get(self, e: Element, key: str, builtin: bool = True) -> str | None:
        r = self.resolved(e).get(key)
        if r is None:
            return None
        if not builtin and r[1] == "builtin":
            return None
        return r[0]

    def inherited(self, e: Element, key: str) -> tuple[str, str] | None:
        """Value the element would have without its explicit attribute."""
        r = self.resolved(copy_without(e, key)) if key in e.attrib else self.resolved(e)
        return r.get(key)

    # ------------------------------------------------------------ transforms
    def _orient_from(self, attrs: dict[str, str]) -> np.ndarray | None:
        for key in ORIENT_KEYS:
            v = attrs.get(key)
            if v is None:
                continue
            arr = mx.parse_vec(v)
            if arr is None:
                return None
            if key == "quat" and arr.size == 4:
                return mx.qnorm(arr)
            if key == "axisangle" and arr.size == 4:
                return mx.axang2q(arr[:3], self.to_rad(arr[3]))
            if key == "euler" and arr.size == 3:
                return mx.euler2q([self.to_rad(x) for x in arr], self.eulerseq)
            if key == "xyaxes" and arr.size == 6:
                return mx.xyaxes2q(arr)
            if key == "zaxis" and arr.size == 3:
                return mx.zaxis2q(arr)
        return None

    def fromto_active(self, e: Element) -> np.ndarray | None:
        if e.tag not in ("geom", "site"):
            return None
        r = self.resolved(e)
        ft = r.get("fromto")
        typ = r.get("type", ("sphere", ""))[0]
        if ft is None or typ not in FROMTO_TYPES:
            return None
        return mx.parse_vec(ft[0], 6)

    def local_pose(self, e: Element) -> tuple[np.ndarray, np.ndarray]:
        """(pos, quat) of an element's frame in its parent frame."""
        if e.tag == "joint" or e.tag == "freejoint":
            p = mx.parse_vec(self.get(e, "pos") or "0 0 0", 3)
            ax = mx.parse_vec(self.get(e, "axis") or "0 0 1", 3)
            return (p if p is not None else np.zeros(3)), mx.zaxis2q(ax if ax is not None else [0, 0, 1])
        ft = self.fromto_active(e)
        if ft is not None:
            # MuJoCo (verified on 3.0.1) builds the frame with z along (from - to)
            a, b = ft[:3], ft[3:]
            return (a + b) / 2, mx.zaxis2q(a - b)
        explicit = {k: v for k, v in e.attrib.items() if k in ORIENT_KEYS}
        q = self._orient_from(explicit)
        if q is None and not explicit:
            r = self.resolved(e)
            q = self._orient_from({k: r[k][0] for k in ORIENT_KEYS if k in r and r[k][1] != "builtin"})
        if q is None:
            q = mx.QID.copy()
        if e.tag == "light":
            d = mx.parse_vec(self.get(e, "dir") or "0 0 -1", 3)
            q = mx.zaxis2q(-d if d is not None else [0, 0, 1])
        p = mx.parse_vec(self.get(e, "pos") or "0 0 0", 3)
        return (p if p is not None else np.zeros(3)), q

    def pose_changes(self, e: Element, pos, quat) -> dict[str, str | None]:
        """Attribute changes that give element *e* the local pose (pos, quat).
        Keeps the element's orientation specifier style where possible."""
        pos = np.asarray(pos, float)
        quat = mx.qcanon(quat)
        ch: dict[str, str | None] = {}
        R = mx.q2mat(quat)
        if e.tag in ("joint",):
            ch["pos"] = mx.fmt_vec(pos)
            ch["axis"] = mx.fmt_vec(R[:, 2])
            return ch
        if e.tag == "light":
            ch["pos"] = mx.fmt_vec(pos)
            ch["dir"] = mx.fmt_vec(-R[:, 2])
            return ch
        ft = self.fromto_active(e)
        if ft is not None:
            half = float(np.linalg.norm(ft[3:] - ft[:3])) / 2
            z = R[:, 2]
            ch["fromto"] = mx.fmt_vec(np.concatenate([pos + z * half, pos - z * half]))
            for k in ("pos",) + ORIENT_KEYS:
                if k in e.attrib:
                    ch[k] = None
            return ch
        inh = self.resolved(copy_without(e, "pos"))
        default_pos = inh.get("pos", ("0 0 0", "builtin"))
        if np.allclose(pos, 0, atol=1e-12) and default_pos[1] == "builtin":
            if "pos" in e.attrib:
                ch["pos"] = None
        else:
            ch["pos"] = mx.fmt_vec(pos)
        style = next((k for k in ORIENT_KEYS if k in e.attrib), "quat")
        val: str | None = None
        if style == "euler":
            eu = mx.q2euler(quat, self.eulerseq)
            if eu is not None:
                val = mx.fmt_vec([self.from_rad(x) for x in eu])
            else:
                style = "quat"
        elif style == "axisangle":
            ax, ang = mx.q2axang(quat)
            val = mx.fmt_vec(list(ax) + [self.from_rad(ang)])
        elif style == "xyaxes":
            val = mx.fmt_vec(np.concatenate([R[:, 0], R[:, 1]]))
        elif style == "zaxis":
            if mx.qangle(mx.zaxis2q(R[:, 2]), quat) < 1e-9:
                val = mx.fmt_vec(R[:, 2])
            else:
                style = "quat"
        if style == "quat":
            inh_or = [k for k in ORIENT_KEYS if inh.get(k, (None, "builtin"))[1] != "builtin"]
            if mx.qangle(quat, mx.QID) < 1e-12 and not inh_or:
                val = None
            else:
                val = mx.fmt_vec(quat)
        for k in ORIENT_KEYS:
            if k != style and k in e.attrib:
                ch[k] = None
        if val is None:
            if style in e.attrib:
                ch[style] = None
        else:
            ch[style] = val
        return ch

    # ------------------------------------------------------------ labels
    def label(self, e: Element) -> str:
        if not is_elem(e):
            return ""
        n = e.get("name")
        t = e.tag
        if t == "body":
            return n or "(body)"
        if t in ("geom", "site"):
            typ = self.get(e, "type") or "sphere"
            if t == "geom" and typ == "mesh":
                typ = f"mesh:{self.get(e, 'mesh') or '?'}"
            return f"{n} [{typ}]" if n else f"({t} {typ})"
        if t == "joint":
            typ = self.get(e, "type") or "hinge"
            return f"{n} [{typ}]" if n else f"(joint {typ})"
        if t == "freejoint":
            return f"{n} [free]" if n else "(freejoint)"
        if t == "default":
            return f"class {e.get('class') or 'main'}"
        if n:
            return f"{n}"
        if t in ACTUATORS or parent_tag(e) in ("sensor",):
            tgt = e.get("joint") or e.get("site") or e.get("tendon") or e.get("actuator") or e.get("body") \
                or e.get("objname") or ""
            return f"({t} {tgt})" if tgt else f"({t})"
        return f"({t})"


def copy_without(e: Element, key: str) -> Element:
    """Shallow stand-in of *e* without attribute *key*, still placed in the tree
    (needed for class resolution) - implemented by temporarily removing the key."""
    return _Without(e, key)  # type: ignore[return-value]


class _Without:
    """Proxy giving MjcfDoc.resolved() an element view without one attribute."""

    def __init__(self, e: Element, key: str):
        self._e = e
        self._key = key
        self.tag = e.tag
        self.attrib = {k: v for k, v in e.attrib.items() if k != key}

    def get(self, k, default=None):
        return self.attrib.get(k, default)

    def getparent(self):
        return self._e.getparent()

    def iterancestors(self, *a):
        return self._e.iterancestors(*a)


def fmt_attr(value) -> str:
    """Format a python value for an attribute."""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, np.ndarray)):
        return mx.fmt_vec(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return mx.fmt_num(float(value))


__all__ = ["MjcfDoc", "MjcfError", "Element", "is_elem", "children", "parent_tag", "attr_spec",
           "copy_without", "fmt_attr", "FILE_ATTRS"]
