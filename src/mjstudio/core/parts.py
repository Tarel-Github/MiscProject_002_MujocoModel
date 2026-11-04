"""Parts: extract a body subtree as a standalone MJCF and assemble parts into models.

A *part* is an ordinary MJCF document (it compiles on its own and can be edited
and saved like any model). Importing a part into another model handles:
  * name collisions (prefixing + reference remapping),
  * identical assets (reused) vs. different assets with the same name (renamed),
  * asset file paths (re-expressed relative to the target model),
  * compiler differences (angle unit, euler sequence, autolimits),
  * default classes (copied into an isolated class namespace so the target's
    defaults cannot silently change the part's behaviour).
"""
from __future__ import annotations

import copy
import logging
import math
import os
from pathlib import Path

import numpy as np
from lxml import etree

from . import mathx as mx
from .doc import FILE_ATTRS, Element, MjcfDoc, children, is_elem, parent_tag
from .editor import Editor, ref_kind
from .schema import ACTUATORS, ORIENT_KEYS, attrs_for, reference_attrs
from .undo import Insert, SetAttrs

log = logging.getLogger("mjstudio")
TREE_NAMED = ("body", "joint", "freejoint", "geom", "site", "camera", "light")
SECTION_ORDER = ("tendon", "equality", "contact", "actuator", "sensor")
DEFAULTED_SECTIONS = {"tendon", "equality", "actuator"}  # their children use default classes


def _kind(e: Element) -> str | None:
    if e.tag == "default":
        return "class"
    return ref_kind(e)


def _refs_of(e: Element) -> set[tuple[str, str]]:
    """All (kind, name) references inside subtree e."""
    out: set[tuple[str, str]] = set()
    sites = reference_attrs()
    for d in e.iter():
        if not is_elem(d):
            continue
        pt = parent_tag(d)
        for r in sites:
            if r.tag == d.tag and (r.parent_tag is None or r.parent_tag == pt):
                if r.parent_tag is None and pt in ("equality", "fixed", "spatial", "skin") and d.tag in (
                        "joint", "geom", "site", "connect", "weld", "tendon", "bone"):
                    continue
                v = d.get(r.attr)
                if v:
                    out.add((r.kind, v))
        for a_name, a_type in (("objname", "objtype"), ("refname", "reftype")):
            if d.get(a_name) and d.get(a_type):
                t = d.get(a_type)
                out.add(("body" if t in ("body", "xbody") else t, d.get(a_name)))
    return out


def _defined(e: Element, doc: MjcfDoc) -> set[tuple[str, str]]:
    out = set()
    for d in e.iter():
        if is_elem(d):
            k = _kind(d)
            n = doc.obj_name(d) if d.tag != "default" else d.get("class")
            if k and n and k != "equality":
                out.add((k, n))
    return out


# ============================================================== extract
def extract_part(doc: MjcfDoc, body: Element, at_origin: bool = True, name: str | None = None) -> MjcfDoc:
    """Standalone MJCF containing *body*'s subtree and everything it needs."""
    if body.tag != "body":
        raise ValueError("부품은 <body>에서만 추출할 수 있습니다")
    root = etree.Element("mujoco", model=name or body.get("name") or "part")
    for c in doc.sections("compiler"):
        root.append(copy.deepcopy(c))
    for d in doc.sections("default"):
        root.append(copy.deepcopy(d))
    cp = copy.deepcopy(body)
    # childclass inherited from an ancestor outside the subtree must be made explicit
    if not cp.get("childclass"):
        for a in body.iterancestors():
            if a.tag in ("body", "frame") and a.get("childclass"):
                cp.set("childclass", a.get("childclass"))
                break
    lp, lq = doc.local_pose(body)
    if at_origin:
        for k in ("pos",) + ORIENT_KEYS:
            cp.attrib.pop(k, None)
    # assets referenced by the subtree or by any default class (copied wholesale)
    need = _refs_of(cp)
    for d in doc.sections("default"):
        need |= _refs_of(d)
    asset_elems: list[Element] = []
    changed = True
    while changed:
        changed = False
        for a in doc.iter_tags("mesh", "material", "texture", "hfield"):
            if parent_tag(a) != "asset" or a in asset_elems:
                continue
            if (a.tag, doc.obj_name(a)) in need:
                asset_elems.append(a)
                need |= _refs_of(a)
                changed = True
    if asset_elems:
        asset = etree.SubElement(root, "asset")
        for a in doc.iter_tags("mesh", "material", "texture", "hfield"):
            if a in asset_elems:
                asset.append(copy.deepcopy(a))
    wb = etree.SubElement(root, "worldbody")
    wb.append(cp)
    inside = _defined(cp, doc)
    notes = []
    for sec_tag in SECTION_ORDER + ("deformable",):
        kept = []
        for sec in doc.sections(sec_tag):
            for c in children(sec):
                refs = _refs_of(c)
                if c.tag == "skin":
                    refs = {r for r in refs if r[0] == "body"}
                if refs and refs <= inside | {(k, n) for k, n in _asset_names(asset_elems, doc)}:
                    kept.append(copy.deepcopy(c))
                elif refs & inside:
                    notes.append(f"{c.tag} '{c.get('name', '')}'이(가) 부품 밖의 요소를 참조해 제외함")
        if kept:
            s = etree.SubElement(root, sec_tag)
            for c in kept:
                s.append(c)
                inside |= _defined(c, doc)
    part = MjcfDoc(root, base_dir=doc.base_dir)
    part.meta["part"] = {"source": str(doc.path) if doc.path else "", "root": body.get("name") or "",
                         "attach_pos": list(map(float, lp)), "attach_quat": list(map(float, lq)), "notes": notes}
    for n in notes:
        log.info("부품 추출: %s", n)
    return part


def _asset_names(elems, doc):
    return [(e.tag, doc.obj_name(e)) for e in elems]


# ============================================================== unit normalisation
def _normalize_units(part: MjcfDoc, unit: str, autolimits: str) -> None:
    """Rewrite the part so it means the same thing under the target compiler settings."""
    src_unit, seq = part.angle_unit, part.eulerseq
    src_auto = part.compiler_attr("autolimits", "true")
    for e in part.root.iter():
        if not is_elem(e):
            continue
        # orientation specifiers depending on units / euler sequence -> quaternion
        if ("euler" in e.attrib or "axisangle" in e.attrib) and e.tag != "compiler":
            q = part._orient_from({k: e.get(k) for k in ("euler", "axisangle") if e.get(k)})
            if q is not None:
                e.attrib.pop("euler", None)
                e.attrib.pop("axisangle", None)
                e.set("quat", mx.fmt_vec(q))
    if src_unit != unit:
        f = (math.pi / 180.0) if src_unit == "degree" else (180.0 / math.pi)
        for j in part.root.iter("joint"):
            if parent_tag(j) in ("fixed", "equality"):
                continue
            jt = part.get(j, "type") or "hinge"
            if jt not in ("hinge", "ball"):
                continue
            for k in ("range", "ref", "springref"):
                v = mx.parse_vec(j.get(k))
                if v is not None:
                    j.set(k, mx.fmt_vec(v * f))
    if src_auto != autolimits and src_auto == "false":
        # under autolimits=false a range without 'limited' means *not* limited
        for e in part.root.iter():
            if not is_elem(e) or part.in_defaults(e):
                continue
            pairs = []
            if e.tag == "joint" and parent_tag(e) not in ("fixed", "equality"):
                pairs = [("range", "limited")]
            elif e.tag in ACTUATORS:
                pairs = [("ctrlrange", "ctrllimited"), ("forcerange", "forcelimited")]
            for rk, lk in pairs:
                if part.get(e, rk, builtin=False) is not None and part.get(e, lk, builtin=False) is None:
                    e.set(lk, "false")
    for c in part.sections("compiler"):
        c.set("angle", unit)
        c.attrib.pop("eulerseq", None)
        c.set("autolimits", autolimits)


# ============================================================== import
def _asset_signature(doc: MjcfDoc, e: Element):
    attrs = {k: v for k, v in e.attrib.items() if k not in ("name",) and not k.startswith("file")}
    files = {}
    for k in FILE_ATTRS.get(e.tag, ((), ""))[0]:
        p = doc.resolve_file(e, k)
        if p is not None:
            files[k] = str(Path(p).resolve())
    return e.tag, tuple(sorted(attrs.items())), tuple(sorted(files.items()))


def import_part(ed: Editor, part: MjcfDoc, parent: Element | None = None, pos=None, quat=None,
                prefix: str | None = None, index: int | None = None) -> list[Element]:
    """Assemble *part* into the editor's model under *parent* (body or world).
    Returns the inserted top-level elements (first one is the part's root body)."""
    doc = ed.doc
    parent = parent if parent is not None else doc.worldbody
    if parent.tag not in ("body", "worldbody", "frame"):
        raise ValueError("부품은 바디나 월드에만 붙일 수 있습니다")
    p = part.clone()
    _normalize_units(p, doc.angle_unit, doc.compiler_attr("autolimits", "true"))
    pwb = p.section("worldbody")
    roots = [c for c in children(pwb) if c.tag == "body"] if pwb is not None else []
    if not roots:
        raise ValueError("부품의 worldbody에 바디가 없습니다")
    if prefix is None:
        prefix = (p.model_name or "part") + "_"
    if parent.tag != "worldbody":
        for r in roots:
            for fj in [c for c in children(r) if c.tag == "freejoint" or
                       (c.tag == "joint" and (p.get(c, "type") or "hinge") == "free")]:
                log.warning("부품 가져오기: 자유 관절 '%s' 제거 (월드 바로 아래 바디에만 허용)", fj.get("name", ""))
                r.remove(fj)
    # ---------- names
    mapping: dict[tuple[str, str], str] = {}
    taken: dict[str, set] = {}

    def fresh(kind: str, name: str) -> str:
        t = taken.setdefault(kind, set(doc.names(kind)))
        n = name if name not in t else f"{prefix}{name}"
        i = 1
        while n in t:
            n = f"{prefix}{name}_{i}"
            i += 1
        t.add(n)
        return n

    reuse_assets: set = set()
    target_assets = {}
    for a in doc.iter_tags("mesh", "material", "texture", "hfield"):
        if parent_tag(a) == "asset":
            target_assets[(a.tag, doc.obj_name(a))] = a
    passet = p.section("asset")
    if passet is not None:
        for a in list(children(passet)):
            key = (a.tag, p.obj_name(a))
            if key in target_assets and _asset_signature(p, a) == _asset_signature(doc, target_assets[key]):
                reuse_assets.add(a)
            elif key[1] is not None and key[1] in taken.setdefault(a.tag, set(doc.names(a.tag))):
                new = fresh(a.tag, key[1])
                mapping[(a.tag, key[1])] = new
                a.set("name", new)
            else:
                taken.setdefault(a.tag, set(doc.names(a.tag))).add(key[1])
            # express file paths relative to the target model
            for k in FILE_ATTRS.get(a.tag, ((), ""))[0]:
                fp = p.resolve_file(a, k)
                if fp is None:
                    continue
                if not a.get("name") and k == "file":
                    a.set("name", p.obj_name(a))  # keep the implicit name stable
                kind = FILE_ATTRS[a.tag][1]
                base = doc.asset_dir(kind)
                try:
                    rel = os.path.relpath(Path(fp).resolve(), base)
                except ValueError:
                    rel = str(Path(fp).resolve())
                a.set(k, Path(rel).as_posix())
    for top in roots + [c for s in SECTION_ORDER + ("deformable",) for sec in p.sections(s) for c in children(sec)]:
        for d in top.iter():
            if not is_elem(d) or d.tag == "default":
                continue
            k = _kind(d)
            n = d.get("name")
            if k and n and k != "equality":
                new = fresh(k, n)
                if new != n:
                    mapping[(k, n)] = new
                    d.set("name", new)
            elif k == "equality" and n:
                d.set("name", fresh("equality", n))
    # ---------- default classes (isolated namespace)
    pdefs = p.sections("default")
    tdefs = doc.sections("default")
    target_main_nonempty = any(children(d) for d in tdefs)
    part_has_defaults = any(children(d) for d in pdefs)
    ns_elem = None
    if part_has_defaults or target_main_nonempty:
        ns = fresh("class", f"{prefix}main".rstrip("_") if prefix else "part_main")
        ns_elem = etree.Element("default")
        ns_elem.set("class", ns)
        pmain = pdefs[0] if pdefs else None
        main_children = [c for c in children(pmain) if c.tag != "default"] if pmain is not None else []
        for c in main_children:
            ns_elem.append(copy.deepcopy(c))
        # neutralise attributes the target's main default sets but the part's main does not
        for td in tdefs:
            for tc in children(td):
                if tc.tag == "default":
                    continue
                mine = next((c for c in children(ns_elem) if c.tag == tc.tag), None)
                for k in tc.attrib:
                    if mine is not None and k in mine.attrib:
                        continue
                    spec = next((s for s in attrs_for(tc.tag) if s.name == k), None)
                    if spec is None or spec.default is None:
                        log.warning("부품 가져오기: 대상 모델의 기본값 %s/%s에 중립값이 없어 부품에도 적용됩니다",
                                    tc.tag, k)
                        continue
                    if mine is None:
                        mine = etree.SubElement(ns_elem, tc.tag)
                    mine.set(k, spec.default)
        # nested classes
        cls_map = {}
        if pmain is not None:
            for sub in children(pmain, "default"):
                for d in sub.iter("default"):
                    old = d.get("class")
                    new = fresh("class", old)
                    if new != old:
                        cls_map[old] = new
                        d.set("class", new)
                ns_elem.append(copy.deepcopy(sub))
        if pmain is not None and pmain.get("class") and pmain.get("class") != "main":
            cls_map[pmain.get("class")] = ns
        cls_map["main"] = ns
        # remap class / childclass references inside the part
        for top in roots + [c for s in SECTION_ORDER for sec in p.sections(s) for c in children(sec)]:
            for d in top.iter():
                if not is_elem(d):
                    continue
                for a in ("class", "childclass"):
                    v = d.get(a)
                    if v in cls_map:
                        d.set(a, cls_map[v])
        for r in roots:
            if not r.get("childclass"):
                r.set("childclass", ns)
        for s in SECTION_ORDER:
            if s not in DEFAULTED_SECTIONS:
                continue
            for sec in p.sections(s):
                for c in children(sec):
                    if not c.get("class"):
                        c.set("class", ns)
    # ---------- remap references and asset reuse
    _remap(p.root, mapping)
    # ---------- insert (one undoable step)
    inserted: list[Element] = []
    with ed.structural(f"부품 가져오기 {p.model_name}"):
        if ns_elem is not None:
            top_def = doc.section("default")
            if top_def is None:
                top_def = etree.Element("default")
                ed.push(Insert(doc.root, _section_index(doc, "default"), top_def, "default 섹션 추가"))
            ed.push(Insert(top_def, len(top_def), ns_elem, "기본 클래스 가져오기"))
        if passet is not None:
            new_assets = [a for a in children(passet) if a not in reuse_assets]
            if new_assets:
                tasset = doc.section("asset")
                if tasset is None:
                    tasset = etree.Element("asset")
                    ed.push(Insert(doc.root, _section_index(doc, "asset"), tasset, "asset 섹션 추가"))
                for a in new_assets:
                    ed.push(Insert(tasset, len(tasset), a, "에셋 가져오기"))
        for k, r in enumerate(roots):
            if k == 0 and (pos is not None or quat is not None):
                _set_pose_attrs(r, pos if pos is not None else np.zeros(3), quat if quat is not None else mx.QID)
            idx = (len(parent) if index is None else index + k)
            ed.push(Insert(parent, idx, r, "부품 바디 가져오기"))
            inserted.append(r)
        for s in SECTION_ORDER + ("deformable",):
            items = [c for sec in p.sections(s) for c in children(sec)]
            if not items:
                continue
            tsec = doc.section(s)
            if tsec is None:
                tsec = etree.Element(s)
                ed.push(Insert(doc.root, _section_index(doc, s), tsec, f"{s} 섹션 추가"))
            for c in items:
                ed.push(Insert(tsec, len(tsec), c, f"{c.tag} 가져오기"))
                inserted.append(c)
    if mapping:
        log.info("부품 가져오기: 이름이 겹치지 않도록 요소 %d개 이름 변경 (접두사 '%s')", len(mapping), prefix)
    return inserted


def _remap(root: Element, mapping: dict[tuple[str, str], str]) -> None:
    from .ops import _remap_refs
    _remap_refs(root, mapping)


def _set_pose_attrs(e: Element, pos, quat) -> None:
    for k in ("pos",) + ORIENT_KEYS:
        e.attrib.pop(k, None)
    pos = np.asarray(pos, float)
    if not np.allclose(pos, 0):
        e.set("pos", mx.fmt_vec(pos))
    q = mx.qcanon(quat)
    if mx.qangle(q, mx.QID) > 1e-12:
        e.set("quat", mx.fmt_vec(q))


def _section_index(doc: MjcfDoc, tag: str) -> int:
    from .schema import SECTIONS
    order = {t: i for i, t in enumerate(SECTIONS)}
    k = order.get(tag, 99)
    for i, c in enumerate(doc.root):
        if is_elem(c) and order.get(c.tag, 99) > k:
            return i
    return len(doc.root)


def replace_body(ed: Editor, old: Element, part: MjcfDoc, prefix: str | None = None) -> list[Element]:
    """Replace a body subtree with a part placed at the same local pose (e.g. swap a head)."""
    parent = old.getparent()
    idx = parent.index(old)
    lp, lq = ed.doc.local_pose(old)
    with ed.structural(f"부품으로 교체 {ed.doc.label(old)}"):
        ed.delete([old])
        out = import_part(ed, part, parent, lp, lq, prefix=prefix, index=idx)
    return out


def save_part(part: MjcfDoc, path: str | os.PathLike) -> list[str]:
    return part.save(path)


def load_part(path: str | os.PathLike) -> MjcfDoc:
    return MjcfDoc.load(path)
