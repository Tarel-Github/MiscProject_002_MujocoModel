"""Face made in Blender (bpy 5.2.3, official glTF exporter) -> MuJoCo Model Studio."""
import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from mjstudio.core.editor import Editor
from mjstudio.core.expr import FaceController
from mjstudio.core.faceops import attach_face_rig, rig_from_gltf
from mjstudio.core.project import Project
from mjstudio.library import anatomy as A
from mjstudio.library.face import build_face_rig

GLB = Path(__file__).resolve().parents[1] / "examples" / "assets" / "face_blender.glb"
D = math.radians


def _match(A_pts, B_pts):
    """index into B of the nearest point for every point of A, and the distances"""
    idx = np.empty(len(A_pts), np.int64)
    dist = np.empty(len(A_pts))
    for s in range(0, len(A_pts), 512):
        d = np.linalg.norm(A_pts[s:s + 512, None, :] - B_pts[None, :, :], axis=2)
        idx[s:s + 512] = d.argmin(axis=1)
        dist[s:s + 512] = d.min(axis=1)
    return idx, dist


@pytest.fixture(scope="module")
def imported():
    return rig_from_gltf(GLB, "face")


def test_blender_asset_geometry_keys_and_weights(imported):
    ref = build_face_rig("ref", n_theta=104, n_phi=80)     # the data that was given to Blender
    assert len(imported.V0) == len(ref.V0) and len(imported.F) == len(ref.F)
    # lip-seam copies share positions: the jaw weight (scaled) disambiguates them
    idx, dist = _match(np.c_[imported.V0, imported.jaw_w], np.c_[ref.V0, ref.jaw_w])
    assert dist.max() < 1e-3
    assert np.linalg.norm(imported.V0 - ref.V0[idx], axis=1).max() < 1e-5   # Blender -> glTF (Y up) -> MuJoCo
    assert len(set(idx.tolist())) == len(idx)
    assert len(imported.keys) == 17 and "smile" in imported.params and "blink_left" in imported.params
    ref.set("smile", 1.0)
    d_ref = (ref.local_vertices() - ref.V0)[idx]
    assert np.abs(imported.keys["smile"] - d_ref).max() < 1e-5
    ref.reset()
    ref.set("blink_left", 1.0)
    d_ref = (ref.local_vertices() - ref.V0)[idx]
    assert np.abs(imported.keys["blink_left"] - d_ref).max() < 1e-5   # baked bone motion as a shape key
    assert np.abs(imported.jaw_w - ref.jaw_w[idx]).max() < 1e-3      # skin weights of the 'jaw' bone


def test_attach_blender_face_and_drive_it(imported, tmp_path):
    ed = Editor()
    ed.new("blender_face")
    ed.doc.section("option").set("gravity", "0 0 0")
    A.build_head(ed, None, pos=(0, 0, 1.6), neck=False)
    attach_face_rig(ed, imported)
    assert ed.result.ok, ed.result.error
    m, d = ed.model, ed.data
    assert m.nskin == 1
    fc = FaceController(m, d, imported)
    fc.set_jaw_open(0.0)
    base = imported.world_vertices(m, d)
    fc.set_expression("smile", 0.8)
    fc.set_expression("blink_left", 1.0)
    smiled = imported.world_vertices(m, d)
    assert np.linalg.norm(smiled - base, axis=1).max() > 0.004
    # jaw joint opens the mouth through the Blender skin weights
    imported.reset()
    ed.set_qpos("jaw_open", [D(20)])
    opened = imported.world_vertices(m, d)
    low = imported.jaw_w > 0.99
    assert (base[low] - opened[low])[:, 2].max() > 0.01
    # save as project and reload -> same face, same expression values
    imported.set("smile", 0.8)
    prj = Project(tmp_path / "p")
    prj.save(ed.doc, rigs=ed.rigs)
    prj2, doc2 = Project.load(tmp_path / "p")
    ed2 = Editor()
    ed2.set_doc(doc2, prj2.load_rigs())
    assert ed2.result.ok, ed2.result.error
    r2 = ed2.rigs["face"]
    assert np.allclose(r2.local_vertices(), imported.local_vertices(), atol=1e-9)
    assert mujoco.MjModel.from_xml_path(str(prj2.model_path)).nskin == 1
    imported.reset()
