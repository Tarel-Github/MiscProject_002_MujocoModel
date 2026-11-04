"""Project folders: model.xml + project.json + packaged resources.

project.json keeps everything MJCF cannot express (joint groups, expression rigs,
editor metadata); unknown keys written by newer versions are preserved on save.

    <project>/
        model.xml          MJCF (always loadable by plain MuJoCo)
        project.json       editor data
        meshes/ textures/  packaged asset files
        expressions/ animations/   expression data (Phase 4)
"""
from __future__ import annotations

import json
import logging
import math
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .doc import FILE_ATTRS, MjcfDoc
from .editor import Editor

log = logging.getLogger("mjstudio")
FORMAT = "mjstudio-project"
VERSION = 1


@dataclass
class JointGroup:
    """Editor-level linkage: one value drives several joints, q_i = q0_i + w_i * value."""
    name: str
    joints: dict[str, float] = field(default_factory=dict)  # joint name -> weight
    description: str = ""

    def to_json(self) -> dict:
        return {"name": self.name, "joints": self.joints, "description": self.description}

    @classmethod
    def from_json(cls, d: dict) -> "JointGroup":
        return cls(d["name"], {k: float(v) for k, v in d.get("joints", {}).items()}, d.get("description", ""))

    def apply(self, ed: Editor, value: float, forward: bool = True) -> list[str]:
        """Set member joints (hinge/slide). Returns joints that are missing in the model."""
        m = ed.model
        missing = []
        for jn, w in self.joints.items():
            j = ed.joint_id(jn)
            if j < 0:
                missing.append(jn)
                continue
            a = m.jnt_qposadr[j]
            q = m.qpos0[a] + w * value
            if m.jnt_limited[j]:
                q = float(np.clip(q, *m.jnt_range[j]))
            ed.data.qpos[a] = q
        if forward:
            ed.forward()
        return missing


class Project:
    FILE = "project.json"

    def __init__(self, root: str | os.PathLike, model_file: str = "model.xml"):
        self.root = Path(root).resolve()
        self.data: dict = {"format": FORMAT, "version": VERSION, "model": model_file,
                           "joint_groups": [], "expressions": {}, "meta": {}}

    # ------------------------------------------------------------ paths
    @property
    def model_path(self) -> Path:
        return self.root / self.data.get("model", "model.xml")

    @property
    def groups(self) -> list[JointGroup]:
        return [JointGroup.from_json(g) for g in self.data.get("joint_groups", [])]

    def set_group(self, g: JointGroup) -> None:
        gs = [x for x in self.data.setdefault("joint_groups", []) if x["name"] != g.name]
        gs.append(g.to_json())
        self.data["joint_groups"] = gs

    def remove_group(self, name: str) -> None:
        self.data["joint_groups"] = [x for x in self.data.get("joint_groups", []) if x["name"] != name]

    # ------------------------------------------------------------ io
    @classmethod
    def is_project(cls, path: str | os.PathLike) -> bool:
        p = Path(path)
        return (p.is_dir() and (p / cls.FILE).exists()) or p.name == cls.FILE

    @classmethod
    def load(cls, path: str | os.PathLike) -> tuple["Project", MjcfDoc]:
        p = Path(path)
        if p.is_dir():
            p = p / cls.FILE
        data = json.loads(p.read_text(encoding="utf-8"))
        if data.get("format") != FORMAT:
            raise ValueError(f"{p} is not an mjstudio project")
        if int(data.get("version", 1)) > VERSION:
            log.warning("project written by a newer version (%s); unknown data is preserved", data.get("version"))
        prj = cls(p.parent, data.get("model", "model.xml"))
        prj.data.update(data)
        doc = MjcfDoc.load(prj.model_path)
        return prj, doc

    def save(self, doc: MjcfDoc, package: bool = True) -> list[str]:
        """Write model.xml (+ packaged assets) and project.json. Returns notes."""
        self.root.mkdir(parents=True, exist_ok=True)
        notes: list[str] = []
        if package:
            notes += package_assets(doc, self.root)
        notes += doc.save(self.model_path)
        tmp = self.root / (self.FILE + ".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.root / self.FILE)
        return notes


def missing_resources(doc: MjcfDoc) -> list[tuple[str, str, str]]:
    """(element label, attribute, resolved path) of asset files that do not exist."""
    out = []
    for tag, (keys, _kind) in FILE_ATTRS.items():
        for e in doc.iter_tags(tag):
            for k in keys:
                v = e.get(k)
                if not v or v in doc.mem_assets:
                    continue
                p = doc.resolve_file(e, k)
                if p is not None and not Path(p).exists():
                    out.append((doc.label(e), k, str(p)))
    return out


def package_assets(doc: MjcfDoc, root: Path) -> list[str]:
    """Copy every referenced asset file into root/meshes or root/textures and point the
    model at the copies (compiler meshdir/texturedir). In-memory assets are written out."""
    root = Path(root).resolve()
    notes: list[str] = []
    dirs = {"mesh": root / "meshes", "texture": root / "textures"}
    used: dict[str, set] = {"mesh": set(), "texture": set()}
    plan = []
    for tag, (keys, kind) in FILE_ATTRS.items():
        for e in doc.iter_tags(tag):
            for k in keys:
                v = e.get(k)
                if not v:
                    continue
                if v in doc.mem_assets:
                    data, src = doc.mem_assets[v], None
                else:
                    src = doc.resolve_file(e, k)
                    if src is None or not Path(src).exists():
                        notes.append(f"missing resource: {v}")
                        continue
                    data = None
                name = Path(v).name
                stem, ext = os.path.splitext(name)
                i = 1
                while name in used[kind]:
                    name = f"{stem}_{i}{ext}"
                    i += 1
                used[kind].add(name)
                plan.append((e, k, kind, src, data, name))
    if not plan:
        return notes
    for kind in dirs:
        if used[kind]:
            dirs[kind].mkdir(parents=True, exist_ok=True)
    for e, k, kind, src, data, name in plan:
        dst = dirs[kind] / name
        if data is not None:
            dst.write_bytes(data)
        elif Path(src).resolve() != dst.resolve():
            shutil.copy2(src, dst)
        e.set(k, name)
    for c in doc.sections("compiler"):
        c.attrib.pop("assetdir", None)
    comp = doc.section("compiler", create=True)
    if used["mesh"]:
        comp.set("meshdir", "meshes")
    if used["texture"]:
        comp.set("texturedir", "textures")
    doc.mem_assets.clear()
    doc.base_dir = root
    notes.append(f"packaged {len(plan)} asset file(s) into {root}")
    return notes
