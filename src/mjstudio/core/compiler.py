"""Compile an MjcfDoc with MuJoCo and map compiled object ids <-> XML elements.

The document itself is never modified. Compilation happens on a deep copy in
which (a) unnamed tree objects receive reserved internal names so that ids can
be mapped back, and (b) asset file paths are made absolute (or redirected to
in-memory assets) so that compilation does not depend on the working dir.
"""
from __future__ import annotations

import copy
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
from lxml import etree

from .doc import FILE_ATTRS, Element, MjcfDoc, is_elem, parent_tag
from .schema import ACTUATORS, OBJ_TAGS, SENSORS

PREFIX = "__mms_"
OBJ = mujoco.mjtObj

_ASSET_ORDER = (("mesh", OBJ.mjOBJ_MESH, "nmesh"), ("texture", OBJ.mjOBJ_TEXTURE, "ntex"),
                ("material", OBJ.mjOBJ_MATERIAL, "nmat"), ("hfield", OBJ.mjOBJ_HFIELD, "nhfield"),
                ("skin", OBJ.mjOBJ_SKIN, "nskin"), ("key", OBJ.mjOBJ_KEY, "nkey"))


def obj_type(e: Element) -> int | None:
    """mjtObj code for an element that compiles to a named object (None otherwise)."""
    t = e.tag
    pt = parent_tag(e)
    if pt == "equality":
        return int(OBJ.mjOBJ_EQUALITY)
    if t in ("joint", "geom", "site") and pt in ("fixed", "spatial"):
        return None
    if t in SENSORS and pt != "sensor":
        return None
    if t in ACTUATORS and pt != "actuator":
        return None
    name = OBJ_TAGS.get(t)
    if name is None:
        return None
    return int(getattr(OBJ, name))


INJECT_OK = {"body", "joint", "freejoint", "geom", "site", "camera", "light", "spatial", "fixed",
             *ACTUATORS.keys(), *SENSORS.keys(), "connect", "weld"}


class IdMap:
    """Bidirectional map between XML elements and (mjtObj, id)."""

    def __init__(self) -> None:
        self.by_elem: dict[Element, tuple[int, int]] = {}
        self.by_obj: dict[tuple[int, int], Element] = {}

    def add(self, e: Element, objtype: int, oid: int) -> None:
        self.by_elem[e] = (int(objtype), int(oid))
        self.by_obj[(int(objtype), int(oid))] = e

    def elem(self, objtype: int, oid: int) -> Element | None:
        return self.by_obj.get((int(objtype), int(oid)))

    def id(self, e: Element) -> int:
        r = self.by_elem.get(e)
        return r[1] if r else -1

    def obj(self, e: Element) -> tuple[int, int] | None:
        return self.by_elem.get(e)

    def __len__(self) -> int:
        return len(self.by_elem)


@dataclass
class CompileResult:
    ok: bool
    model: mujoco.MjModel | None = None
    idmap: IdMap = field(default_factory=IdMap)
    error: str = ""
    error_elem: Element | None = None
    xml: str = ""
    elapsed_ms: float = 0.0
    names: dict[str, Element] = field(default_factory=dict)  # injected name -> element

    def pretty_error(self, doc: MjcfDoc | None = None) -> str:
        msg = self.error
        for nm, e in self.names.items():
            if nm in msg:
                lbl = doc.label(e) if doc is not None else f"({e.tag})"
                msg = msg.replace(nm, lbl)
        return msg


def _prepare_assets(doc: MjcfDoc, cp: Element, cps: list, origs: list) -> dict[str, bytes]:
    assets: dict[str, bytes] = {}
    strip = doc.compiler_attr("strippath", "false") == "true"
    for c in cp.iter("compiler"):
        for k in ("meshdir", "texturedir", "assetdir"):
            if k in c.attrib:
                del c.attrib[k]
    dirs = {"mesh": doc.asset_dir("mesh"), "texture": doc.asset_dir("texture")}
    in_def = _default_set(cp)
    n = 0
    for tag, (keys, kind) in FILE_ATTRS.items():
        for e in cp.iter(tag):
            if e in in_def:
                continue
            for k in keys:
                v = e.get(k)
                if not v:
                    continue
                if v in doc.mem_assets:
                    n += 1
                    key = f"mms_mem_{n}_{os.path.basename(v)}"
                    assets[key] = doc.mem_assets[v]
                    e.set(k, key)
                    continue
                if strip:
                    v = os.path.basename(v)
                p = Path(v)
                if not p.is_absolute():
                    p = dirs[kind] / p
                e.set(k, str(p))
    return assets


def _default_set(root: Element) -> set:
    s = set()
    for d in root.iter("default"):
        for e in d.iter():
            s.add(e)
    return s


_LINE_RE = re.compile(r"line\s*=?\s*(\d+)")
_NAME_RE = re.compile(r"(?:Element name|Object name)\s*=?\s*'?([^',\s]+)'?")


def compile_doc(doc: MjcfDoc, drop_tags: tuple[str, ...] = ()) -> CompileResult:
    """Compile *doc*. Sections listed in drop_tags (e.g. 'keyframe') are left out."""
    t0 = time.perf_counter()
    cp = copy.deepcopy(doc.root)
    origs = list(doc.root.iter())
    cps = list(cp.iter())
    if len(origs) != len(cps):  # pragma: no cover - deepcopy keeps structure
        raise RuntimeError("tree copy mismatch")
    in_def = _default_set(doc.root)
    want: dict[tuple[int, str], Element] = {}
    injected: dict[str, Element] = {}
    k = 0
    for o, c in zip(origs, cps):
        if not is_elem(o) or o in in_def:
            continue
        ot = obj_type(o)
        if ot is None:
            continue
        nm = o.get("name")
        if not nm:
            if o.tag not in INJECT_OK and parent_tag(o) != "equality":
                continue
            nm = f"{PREFIX}{o.tag}_{k}"
            k += 1
            c.set("name", nm)
            injected[nm] = o
        want[(ot, nm)] = o
    assets = _prepare_assets(doc, cp, cps, origs)
    for t in drop_tags:
        for sec in [c for c in cp if is_elem(c) and c.tag == t]:
            cp.remove(sec)
    etree.indent(cp, space=" ")
    xml = etree.tostring(cp, pretty_print=True, encoding="unicode")
    try:
        model = mujoco.MjModel.from_xml_string(xml, assets) if assets else mujoco.MjModel.from_xml_string(xml)
    except Exception as ex:  # MuJoCo raises ValueError with the compiler message
        msg = str(ex).strip()
        res = CompileResult(ok=False, error=msg, xml=xml, names=injected,
                            elapsed_ms=(time.perf_counter() - t0) * 1000)
        res.error_elem = _locate(msg, xml, origs, want, injected)
        return res
    idmap = IdMap()
    for (ot, nm), o in want.items():
        oid = mujoco.mj_name2id(model, ot, nm)
        if oid >= 0:
            idmap.add(o, ot, oid)
    for tag, ot, cnt in _ASSET_ORDER:
        els = [e for e in doc.root.iter(tag) if e not in in_def and parent_tag(e) in ("asset", "deformable", "keyframe")]
        if len(els) == getattr(model, cnt):
            for i, e in enumerate(els):
                idmap.add(e, int(ot), i)
    return CompileResult(ok=True, model=model, idmap=idmap, xml=xml, names=injected,
                         elapsed_ms=(time.perf_counter() - t0) * 1000)


def _locate(msg: str, xml: str, origs: list, want: dict, injected: dict) -> Element | None:
    m = _LINE_RE.search(msg)
    if m:
        line = int(m.group(1))
        try:
            reparsed = list(etree.fromstring(xml.encode("utf-8")).iter())
            if len(reparsed) == len(origs):
                best = None
                for r, o in zip(reparsed, origs):
                    if is_elem(r) and r.sourceline == line:
                        best = o
                        break
                if best is not None:
                    return best
        except etree.XMLSyntaxError:
            pass
    m = _NAME_RE.search(msg)
    if m:
        nm = m.group(1)
        if nm in injected:
            return injected[nm]
        for (_ot, n), o in want.items():
            if n == nm:
                return o
    return None


def validate_steps(model: mujoco.MjModel, nsteps: int = 200) -> list[str]:
    """Run a short simulation from qpos0 and report numerical problems."""
    data = mujoco.MjData(model)
    issues: list[str] = []
    import numpy as np
    for i in range(nsteps):
        mujoco.mj_step(model, data)
        if not (np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))):
            issues.append(f"non-finite state at step {i} (t={data.time:.4f}s)")
            break
    for w in range(len(data.warning)):
        cnt = int(data.warning[w].number)
        if cnt:
            issues.append(f"{mujoco.mjtWarning(w).name}: {cnt} time(s)")
    vmax = float(np.max(np.abs(data.qvel))) if model.nv else 0.0
    if vmax > 1e3:
        issues.append(f"very large joint velocity after {nsteps} steps: {vmax:.3g}")
    return issues


_FEATURES: dict[str, bool] = {}


def supports(feature: str) -> bool:
    """Feature detection against the installed MuJoCo (cached): compile a tiny model."""
    if feature not in _FEATURES:
        probes = {
            # <mesh inertia="shell"> (needed for flat meshes by newer MuJoCo, unknown to 3.0.x)
            "mesh_inertia": '<mujoco><asset><mesh name="m" inertia="shell" '
                            'vertex="0 0 0 1 0 0 0 1 0 1 1 0" face="0 1 2 1 3 2"/></asset></mujoco>',
            "position_kv": '<mujoco><worldbody><body><joint name="j"/><geom size=".1"/></body></worldbody>'
                           '<actuator><position joint="j" kv="1"/></actuator></mujoco>',
        }
        try:
            mujoco.MjModel.from_xml_string(probes[feature])
            _FEATURES[feature] = True
        except Exception:  # noqa: BLE001
            _FEATURES[feature] = False
    return _FEATURES[feature]
