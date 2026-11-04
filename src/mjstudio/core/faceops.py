"""Attach expression rigs (face skins) to a model and import them from glTF/GLB."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from . import gltf
from .editor import Editor
from .expr import ExprParam, FaceRig
from .meshops import SetMemAsset, _unique_key
from .ops import _section_index, make
from .undo import Insert, Remove

log = logging.getLogger("mjstudio")


def attach_face_rig(ed: Editor, rig: FaceRig, group: int = 1) -> None:
    """Write the rig's skin (.skn, in memory until saved) under <deformable> and register
    the rig with the editor. The head (and jaw) bodies must exist in the compiled model."""
    m = ed.model
    if m is None:
        raise ValueError("model must compile before attaching a face")
    import mujoco
    if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, rig.head_body) < 0:
        raise ValueError(f"head body '{rig.head_body}' not found")
    if rig.jaw_body and mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, rig.jaw_body) < 0:
        log.warning("jaw body '%s' not found: the face is bound to the head only", rig.jaw_body)
        rig.jaw_body = None
        rig.jaw_w[:] = 0
    rig.skin = ed.doc.unique_name("skin", rig.skin)
    jr = rig._jaw_rest(m) if rig.jaw_body else None
    key = _unique_key(ed, rig.skin, "skn")
    sk = make("skin", name=rig.skin, file=key, rgba=" ".join(f"{x:g}" for x in rig.rgba), group=str(group))
    with ed.batch(f"Attach face {rig.name}"):
        ed.push(SetMemAsset(ed, key, rig.skn_bytes(jr), f"Face skin data {rig.name}"))
        sec = ed.doc.section("deformable")
        if sec is None:
            sec = make("deformable")
            ed.push(Insert(ed.doc.root, _section_index(ed, "deformable"), sec, "Add deformable section"))
        ed.push(Insert(sec, len(sec), sk, f"Add face skin {rig.skin}"))
    ed.rigs[rig.name] = rig
    rig._cache = None
    ed.rigs_changed.emit()


def detach_face_rig(ed: Editor, name: str) -> None:
    rig = ed.rigs.pop(name)
    sk = ed.doc.find_named("skin", rig.skin)
    if sk is not None:
        ed.push(Remove(sk, f"Remove face skin {rig.skin}"))
    ed.rigs_changed.emit()


def rig_from_gltf(path: str | Path, name: str | None = None, head_body: str = "head", jaw_body: str = "jaw",
                  mesh: str | None = None, offset=(0, 0, 0), scale: float = 1.0) -> FaceRig:
    """Face rig from a glTF/GLB authored with the head centre at the origin (Blender: Z up,
    face towards -Y). Shape keys -> blend shapes, one expression parameter per key; skin
    weights of joints whose name contains 'jaw' bind vertices to the jaw body."""
    sc = gltf.load(path)
    cands = [g for g in sc.meshes if mesh is None or g.name == mesh or g.node == mesh]
    if not cands:
        raise ValueError(f"no mesh{' ' + mesh if mesh else ''} in {path}")
    gm = max(cands, key=lambda g: (len(g.morph), len(g.V)))
    V = gltf.to_mujoco(gm.V, gm.world) * scale + np.asarray(offset, float)
    if gm.joints is not None:
        jaw_idx = [i for i, n in enumerate(gm.joint_names) if "jaw" in n.lower()]
        jw = np.zeros(len(V))
        for i in jaw_idx:
            jw += np.where(gm.joints == i, gm.weights, 0).sum(axis=1)
    else:
        jw = None
    rig = FaceRig(name or Path(path).stem, V, gm.F, head_body, jaw_body if (jw is not None and jw.any()) else None, jw)
    for k, d in gm.morph.items():
        rig.add_key(k, gltf.vec_to_mujoco(d, gm.world) * scale)
        rig.add_param(ExprParam(k, keys={k: 1.0}, group="shape keys"))
    rig.meta["source"] = str(path)
    rig.meta["joints"] = gm.joint_names
    log.info("face rig '%s' from %s: %d vertices, %d shape keys, jaw-weighted vertices: %d",
             rig.name, Path(path).name, len(V), len(gm.morph), int((rig.jaw_w > 0.5).sum()))
    return rig
