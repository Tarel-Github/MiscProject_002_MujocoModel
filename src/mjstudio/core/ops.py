"""High-level, undoable model building operations.

Every function takes the Editor and goes through its undo stack, so GUI
actions and scripts share exactly the same behaviour.
"""
from __future__ import annotations

import copy
import math
from typing import Callable

import numpy as np
from lxml import etree

from . import mathx as mx
from .doc import Element, children, fmt_attr, is_elem, parent_tag
from .editor import Editor, ref_kind
from .schema import ACTUATORS, ORIENT_KEYS, SENSORS, reference_attrs
from .undo import Insert, SetAttrs

GEOM_SIZE = {"sphere": "0.05", "box": "0.05 0.05 0.05", "capsule": "0.03 0.1", "cylinder": "0.03 0.1",
             "ellipsoid": "0.06 0.04 0.03", "plane": "1 1 0.05"}
_ORDER = {"inertial": 0, "freejoint": 1, "joint": 1, "geom": 2, "site": 3, "camera": 4, "light": 4, "body": 9}


def _q(quat):
    """Accept a quaternion as sequence or MJCF string; returns normalised array or None."""
    if quat is None:
        return None
    if isinstance(quat, str):
        quat = mx.parse_vec(quat, 4)
    return mx.qnorm(quat)


def make(tag: str, **attrs) -> Element:
    e = etree.Element(tag)
    for k, v in attrs.items():
        if v is None:
            continue
        e.set(k.rstrip("_"), fmt_attr(v))
    return e


def body_target(ed: Editor, elem: Element | None) -> Element:
    """Body (or worldbody) that new tree elements should go into."""
    if elem is None:
        return ed.doc.worldbody
    e = elem
    while e is not None and e.tag not in ("body", "worldbody", "frame"):
        e = e.getparent()
    return e if e is not None else ed.doc.worldbody


def _index(body: Element, tag: str) -> int:
    k = _ORDER.get(tag, 5)
    for i, c in enumerate(body):
        if is_elem(c) and _ORDER.get(c.tag, 5) > k:
            return i
    return len(body)


def _name(ed: Editor, kind: str, name: str | None, base: str) -> str:
    if name:
        if name in ed.doc.names(kind):
            raise ValueError(f"이름이 '{name}'인 {kind}이(가) 이미 있습니다")
        return name
    return ed.doc.unique_name(kind, base)


# ----------------------------------------------------------------- tree elements
def add_body(ed: Editor, parent: Element | None = None, name: str | None = None, pos=(0, 0, 0), quat=None,
             geom: str | None = None, **attrs) -> Element:
    parent = body_target(ed, parent)
    nm = _name(ed, "body", name, "body")
    quat = _q(quat)
    b = make("body", name=nm, pos=pos, quat=quat, **attrs)
    if quat is not None and mx.qangle(quat, mx.QID) < 1e-12:
        del b.attrib["quat"]
    with ed.batch(f"바디 추가 {nm}"):
        ed.push(Insert(parent, _index(parent, "body"), b, f"바디 추가 {nm}"))
        if geom:
            add_geom(ed, b, geom)
    return b


def add_geom(ed: Editor, parent: Element | None, gtype: str = "sphere", name: str | None = None, size=None,
             pos=None, quat=None, **attrs) -> Element:
    parent = body_target(ed, parent)
    if name:
        _name(ed, "geom", name, "geom")
    g = make("geom", name=name, type=gtype, size=size if size is not None else GEOM_SIZE.get(gtype),
             pos=pos, quat=_q(quat), **attrs)
    ed.push(Insert(parent, _index(parent, "geom"), g, f"{gtype} 형상 추가"))
    return g


def add_joint(ed: Editor, body: Element, jtype: str = "hinge", name: str | None = None, axis=(0, 0, 1),
              pos=None, range_rad=None, **attrs) -> Element:
    if body is None or body.tag != "body":
        raise ValueError("관절은 바디에만 추가할 수 있습니다")
    nm = _name(ed, "joint", name, f"{body.get('name') or 'body'}_{jtype}")
    if jtype == "free":
        if body.getparent().tag != "worldbody":
            raise ValueError("자유 관절은 월드 바로 아래 바디에만 둘 수 있습니다")
        j = make("freejoint", name=nm)
    else:
        rng = None
        if range_rad is not None:
            lo, hi = range_rad
            if jtype in ("hinge", "ball"):
                lo, hi = ed.doc.from_rad(lo), ed.doc.from_rad(hi)
            rng = (lo, hi)
        j = make("joint", name=nm, type=jtype, pos=pos,
                 axis=None if jtype == "ball" else mx.fmt_vec(np.asarray(axis, float) / np.linalg.norm(axis)),
                 range=rng, **attrs)
        if rng is not None and ed.doc.compiler_attr("autolimits", "true") != "true":
            j.set("limited", "true")
        if rng is None:
            # a default class may force limited="true": then a range is mandatory
            body.append(j)
            r = ed.doc.resolved(j)
            body.remove(j)
            rv = mx.parse_vec(r.get("range", ("0 0", ""))[0], 2)
            if r.get("limited", ("auto", ""))[0] == "true" and (rv is None or rv[0] >= rv[1]):
                lo, hi = {"hinge": (-math.pi / 2, math.pi / 2), "slide": (-0.1, 0.1),
                          "ball": (0.0, math.pi / 2)}[jtype]
                if jtype in ("hinge", "ball"):
                    lo, hi = ed.doc.from_rad(lo), ed.doc.from_rad(hi)
                j.set("range", mx.fmt_vec([lo, hi]))
    with ed.structural(f"{jtype} 관절 추가 {nm}"):  # qpos layout changes -> remap keyframes
        ed.push(Insert(body, _index(body, "joint"), j, f"{jtype} 관절 추가 {nm}"))
    return j


def add_site(ed: Editor, parent: Element | None, name: str | None = None, pos=(0, 0, 0), size=0.01,
             stype: str = "sphere", **attrs) -> Element:
    parent = body_target(ed, parent)
    nm = _name(ed, "site", name, "site")
    s = make("site", name=nm, type=None if stype == "sphere" else stype, pos=pos, size=size, **attrs)
    ed.push(Insert(parent, _index(parent, "site"), s, f"사이트 추가 {nm}"))
    return s


def add_camera(ed: Editor, parent: Element | None, name: str | None = None, pos=(0, -1, 1), quat=None,
               **attrs) -> Element:
    parent = body_target(ed, parent)
    nm = _name(ed, "camera", name, "camera")
    if quat is None:
        quat = mx.xyaxes2q([1, 0, 0, 0, 0.7, 0.7])
    c = make("camera", name=nm, pos=pos, quat=_q(quat), **attrs)
    ed.push(Insert(parent, _index(parent, "camera"), c, f"카메라 추가 {nm}"))
    return c


def add_light(ed: Editor, parent: Element | None, name: str | None = None, pos=(0, 0, 2), dir=(0, 0, -1),
              **attrs) -> Element:
    parent = body_target(ed, parent)
    nm = _name(ed, "light", name, "light")
    li = make("light", name=nm, pos=pos, dir=dir, **attrs)
    ed.push(Insert(parent, _index(parent, "light"), li, f"조명 추가 {nm}"))
    return li


def add_inertial(ed: Editor, body: Element, mass: float, diaginertia=None, pos=(0, 0, 0)) -> Element:
    old = children(body, "inertial")
    if old:
        raise ValueError("바디에 이미 <inertial> 요소가 있습니다")
    if diaginertia is None:
        r = 0.05
        diaginertia = [0.4 * mass * r * r] * 3
    e = make("inertial", pos=pos, mass=mass, diaginertia=diaginertia)
    ed.push(Insert(body, 0, e, "관성 추가"))
    return e


# ----------------------------------------------------------------- actuators / sensors
def add_actuator(ed: Editor, joint: Element | str, kind: str = "position", name: str | None = None,
                 **attrs) -> Element:
    if isinstance(joint, str):
        jn = joint
        joint = ed.doc.find_named("joint", jn)
    jn = joint.get("name") if joint is not None else None
    if not jn:
        raise ValueError("액추에이터에는 이름이 있는 관절이 필요합니다")
    if kind not in ACTUATORS:
        raise ValueError(f"알 수 없는 액추에이터 종류 {kind}")
    nm = _name(ed, "actuator", name, f"{kind[:3]}_{jn}")
    if kind == "position" and "ctrlrange" not in attrs:
        rng = ed.doc.get(joint, "range")
        r = mx.parse_vec(rng, 2) if rng else None
        if r is not None and r[0] < r[1]:
            attrs["ctrlrange"] = mx.fmt_vec(r)
            if ed.doc.compiler_attr("autolimits", "true") != "true":
                attrs["ctrllimited"] = "true"
    a = make(kind, name=nm, joint=jn, **attrs)
    sec = ed.doc.section("actuator")
    with ed.structural(f"액추에이터 추가 {nm}"):
        if sec is None:
            sec = make("actuator")
            ed.push(Insert(ed.doc.root, _section_index(ed, "actuator"), sec, "actuator 섹션 추가"))
        ed.push(Insert(sec, len(sec), a, f"액추에이터 추가 {nm}"))
    return a


def add_sensor(ed: Editor, kind: str, name: str | None = None, **attrs) -> Element:
    if kind not in SENSORS:
        raise ValueError(f"알 수 없는 센서 종류 {kind}")
    target = attrs.get("joint") or attrs.get("site") or attrs.get("body") or attrs.get("objname") or kind
    nm = _name(ed, "sensor", name, f"{kind}_{target}")
    s = make(kind, name=nm, **attrs)
    sec = ed.doc.section("sensor")
    with ed.batch(f"센서 추가 {nm}"):
        if sec is None:
            sec = make("sensor")
            ed.push(Insert(ed.doc.root, _section_index(ed, "sensor"), sec, "sensor 섹션 추가"))
        ed.push(Insert(sec, len(sec), s, f"센서 추가 {nm}"))
    return s


def add_keyframe(ed: Editor, name: str | None = None, from_state: bool = True) -> Element:
    m, d = ed.model, ed.data
    nm = name or ed.doc.unique_name("key", "pose")
    k = make("key", name=nm)
    if from_state and m is not None:
        k.set("qpos", mx.fmt_vec(d.qpos))
        if m.nu:
            k.set("ctrl", mx.fmt_vec(d.ctrl))
        if m.nmocap:
            k.set("mpos", mx.fmt_vec(d.mocap_pos.ravel()))
            k.set("mquat", mx.fmt_vec(d.mocap_quat.ravel()))
    sec = ed.doc.section("keyframe")
    with ed.batch(f"키프레임 추가 {nm}"):
        if sec is None:
            sec = make("keyframe")
            ed.push(Insert(ed.doc.root, _section_index(ed, "keyframe"), sec, "keyframe 섹션 추가"))
        ed.push(Insert(sec, len(sec), k, f"키프레임 추가 {nm}"))
    return k


def _section_index(ed: Editor, tag: str) -> int:
    from .schema import SECTIONS
    order = {t: i for i, t in enumerate(SECTIONS)}
    k = order.get(tag, 99)
    for i, c in enumerate(ed.doc.root):
        if is_elem(c) and order.get(c.tag, 99) > k:
            return i
    return len(ed.doc.root)


# ----------------------------------------------------------------- duplicate / mirror
def _kind(e: Element) -> str | None:
    if e.tag == "default":
        return "class"
    return ref_kind(e)


def _remap_refs(e: Element, mapping: dict[tuple[str, str], str]) -> bool:
    """Rewrite reference attributes inside subtree e. Returns True if anything changed."""
    sites = reference_attrs()
    changed = False
    for d in e.iter():
        if not is_elem(d):
            continue
        pt = parent_tag(d)
        for r in sites:
            if r.tag != d.tag or (r.parent_tag is not None and r.parent_tag != pt):
                continue
            v = d.get(r.attr)
            if v is not None and (r.kind, v) in mapping:
                d.set(r.attr, mapping[(r.kind, v)])
                changed = True
        for a_name, a_type in (("objname", "objtype"), ("refname", "reftype")):
            v = d.get(a_name)
            t = d.get(a_type)
            if v is not None and t:
                k = "body" if t in ("body", "xbody") else t
                if (k, v) in mapping:
                    d.set(a_name, mapping[(k, v)])
                    changed = True
    return changed


def duplicate(ed: Editor, elems: list[Element], rename: Callable[[str], str] | None = None,
              with_dependents: bool = True, transform: Callable[[Element], None] | None = None,
              text: str = "복제") -> list[Element]:
    """Copy elements (with unique names) and, optionally, the actuators/sensors/
    tendons/equalities/contacts that reference them, with references remapped."""
    top = [e for e in elems if e.getparent() is not None and not any(a in elems for a in e.iterancestors())]
    out: list[Element] = []
    used: dict[str, set] = {}

    def uniq(kind: str, base: str) -> str:
        taken = used.setdefault(kind, set(ed.doc.names(kind)))
        n = base
        if n in taken:
            stem = base
            i = 1
            while f"{stem}_{i}" in taken:
                i += 1
            n = f"{stem}_{i}"
        taken.add(n)
        return n

    with ed.structural(text):
        mapping: dict[tuple[str, str], str] = {}
        copies = []
        for e in top:
            cp = copy.deepcopy(e)
            for d in cp.iter():
                if not is_elem(d):
                    continue
                k = _kind(d)
                old = d.get("name")
                if k and old:
                    new = uniq(k, rename(old) if rename else f"{old}_copy")
                    mapping[(k, old)] = new
                    d.set("name", new)
            copies.append((e, cp))
        for e, cp in copies:
            _remap_refs(cp, mapping)
            if transform is not None:
                transform(cp)
            parent = e.getparent()
            ed.push(Insert(parent, parent.index(e) + 1, cp, text))
            out.append(cp)
        if with_dependents and mapping:
            inside = set()
            for e in top:
                inside.update(e.iter())
            seen = set()
            for (k, old), _new in list(mapping.items()):
                for ref, _attr in ed.references_to(k, old):
                    tgt = ref
                    while tgt.getparent() is not None and tgt.getparent().tag not in (
                            "actuator", "sensor", "tendon", "equality", "contact", "deformable"):
                        tgt = tgt.getparent()
                    if tgt in inside or tgt in seen or tgt.getparent() is None or tgt.tag in (
                            "actuator", "sensor", "tendon", "equality", "contact", "deformable", "worldbody", "mujoco"):
                        continue
                    seen.add(tgt)
                    cp = copy.deepcopy(tgt)
                    if not _remap_refs(cp, mapping):
                        continue
                    k2 = _kind(cp)
                    if cp.get("name") and k2:
                        nn = uniq(k2, rename(cp.get("name")) if rename else f"{cp.get('name')}_copy")
                        cp.set("name", nn)
                    sec = tgt.getparent()
                    ed.push(Insert(sec, sec.index(tgt) + 1, cp, text))
                    out.append(cp)
    return out


def mirror_subtree(ed: Editor, e: Element, axis: int = 1) -> None:
    """In-place reflection of a (copied) subtree across the plane normal to *axis* through the
    parent frame origin. Proper rotations are kept (R' = M R M) and hinge/slide axes are
    transformed as axial/polar vectors so the same joint angle gives mirrored motion."""
    from .doc import MjcfDoc  # noqa: F401
    doc = ed.doc
    M = np.eye(3)
    M[axis, axis] = -1

    def mpos(v):
        return M @ np.asarray(v, float)

    for d in e.iter():
        if not is_elem(d):
            continue
        tag = d.tag
        if tag in ("body", "geom", "site", "camera", "inertial", "frame") and parent_tag(d) not in ("spatial",):
            ft = mx.parse_vec(d.get("fromto"), 6) if d.get("fromto") else None
            if ft is not None:
                d.set("fromto", mx.fmt_vec(np.concatenate([mpos(ft[:3]), mpos(ft[3:])])))
                continue
            p = mx.parse_vec(d.get("pos")) if d.get("pos") else None
            if p is not None and p.size == 3:
                d.set("pos", mx.fmt_vec(mpos(p)))
            if any(k in d.attrib for k in ORIENT_KEYS):
                _, q = doc.local_pose(d) if d.getparent() is not None else (None, mx.QID)
                R = M @ mx.q2mat(q) @ M
                for k in ORIENT_KEYS:
                    d.attrib.pop(k, None)
                q2 = mx.mat2q(R)
                if mx.qangle(q2, mx.QID) > 1e-12:
                    d.set("quat", mx.fmt_vec(q2))
        elif tag == "joint" and parent_tag(d) not in ("fixed", "equality"):
            p = mx.parse_vec(d.get("pos")) if d.get("pos") else None
            if p is not None and p.size == 3:
                d.set("pos", mx.fmt_vec(mpos(p)))
            jt = doc.get(d, "type") or "hinge"
            a = mx.parse_vec(doc.get(d, "axis") or "0 0 1", 3)
            if a is not None and jt in ("hinge", "slide"):
                a2 = -mpos(a) if jt == "hinge" else mpos(a)  # axial vs polar vector
                d.set("axis", mx.fmt_vec(a2))
                if jt == "slide":
                    pass
        elif tag == "light":
            p = mx.parse_vec(d.get("pos")) if d.get("pos") else None
            if p is not None:
                d.set("pos", mx.fmt_vec(mpos(p)))
            dv = mx.parse_vec(d.get("dir")) if d.get("dir") else None
            if dv is not None:
                d.set("dir", mx.fmt_vec(mpos(dv)))


def mirror_duplicate(ed: Editor, e: Element, axis: int = 1, rename: Callable[[str], str] | None = None,
                     with_dependents: bool = True) -> list[Element]:
    """Duplicate *e* mirrored across the parent's plane normal to *axis* (default: XZ plane, y -> -y)."""
    if rename is None:
        rename = swap_side

    def tf(cp: Element) -> None:
        # local_pose needs the element in a tree with the doc's defaults -> temporarily attach
        parent = e.getparent()
        parent.append(cp)
        try:
            mirror_subtree(ed, cp, axis)
        finally:
            parent.remove(cp)

    return duplicate(ed, [e], rename=rename, with_dependents=with_dependents, transform=tf,
                     text=f"좌우 대칭 복제 {ed.doc.label(e)}")


_SIDE_PAIRS = [("_left", "_right"), ("_l", "_r"), ("left_", "right_"), ("l_", "r_"), ("Left", "Right"),
               ("_L", "_R"), ("L_", "R_")]


def swap_side(name: str) -> str:
    for a, b in _SIDE_PAIRS:
        if a in name:
            return name.replace(a, b)
        if b in name:
            return name.replace(b, a)
    return name + "_mirror"


def set_joint_range_deg(ed: Editor, joint: Element, lo_deg: float, hi_deg: float) -> None:
    """Set a hinge/ball range given in degrees, written in the document's angle unit."""
    lo, hi = ed.doc.from_rad(math.radians(lo_deg)), ed.doc.from_rad(math.radians(hi_deg))
    ed.set_attr(joint, "range", mx.fmt_vec([lo, hi]), "관절 범위 설정")


def couple_joints(ed: Editor, joint1: Element | str, joint2: Element | str, ratio: float = 1.0,
                  offset: float = 0.0, name: str | None = None) -> Element:
    """Physical coupling q1 = offset + ratio * q2 via an MJCF <equality><joint> constraint
    (MuJoCo enforces it during simulation; positions are relative to the reference pose)."""
    def nm(j):
        e = ed.doc.find_named("joint", j) if isinstance(j, str) else j
        if e is None or not e.get("name"):
            raise ValueError("관절 연동에는 이름이 있는 관절 두 개가 필요합니다")
        return e.get("name")
    j1, j2 = nm(joint1), nm(joint2)
    eq = make("joint", name=name or ed.doc.unique_name("equality", f"couple_{j1}_{j2}"), joint1=j1, joint2=j2,
              polycoef=mx.fmt_vec([offset, ratio, 0, 0, 0]))
    sec = ed.doc.section("equality")
    with ed.batch(f"관절 연동 {j1} ← {j2}"):
        if sec is None:
            sec = make("equality")
            ed.push(Insert(ed.doc.root, _section_index(ed, "equality"), sec, "equality 섹션 추가"))
        ed.push(Insert(sec, len(sec), eq, "관절 연동 추가"))
    return eq
