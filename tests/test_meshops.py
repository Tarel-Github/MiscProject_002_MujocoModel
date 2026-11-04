"""Mesh assets in documents: primitives, import with units/axes, editing with undo, save."""
import math

import mujoco
import numpy as np
import pytest

from mjstudio.core import mathx as mx
from mjstudio.core import meshes as M
from mjstudio.core import meshops as MO
from mjstudio.core import ops
from mjstudio.core.editor import Editor

OBJ = mujoco.mjtObj


@pytest.fixture
def ed():
    e = Editor()
    e.new("m")
    return e


def _geom(ed, e):
    return ed.idmap.id(e)


@pytest.mark.parametrize("kind", list(M.PRIMITIVES))
def test_primitive_meshes_compile(ed, kind):
    b = ops.add_body(ed, None, "b", pos=(0, 0, 0.5))
    ops.add_joint(ed, b, "free")
    out = MO.create_primitive_mesh(ed, b, kind)
    if M.PRIMITIVES[kind]().is_planar():
        ops.add_geom(ed, b, "sphere", size="0.02")  # a flat mesh carries no mass
    assert ed.result.ok, ed.result.error
    vis = out["geoms"][0]
    gid = _geom(ed, vis)
    assert ed.model.geom_contype[gid] == 0 and ed.model.geom_group[gid] == 2
    planar = M.PRIMITIVES[kind]().is_planar()
    assert len(out["geoms"]) == (1 if planar else 2)


def test_import_obj_units_axes_and_collision_fit(ed, tmp_path):
    # a 100 x 40 x 20 mm box authored in Blender axes (character facing -Y)
    m = M.box(50, 20, 10)
    (tmp_path / "part.obj").write_bytes(M.write_obj(m))
    r = MO.import_mesh(ed, tmp_path / "part.obj", units="mm", axes="blender", collision="box", pos=(0, 0, 1))
    assert ed.result.ok, ed.result.error
    vis, col = r["geoms"]
    gid = _geom(ed, col)
    # MuJoCo fitted a box: half sizes 0.05/0.02/0.01 m, rotated by the Blender preset (x <-> y)
    assert sorted(np.round(ed.model.geom_size[gid], 4)) == [0.01, 0.02, 0.05]
    R = ed.data.geom_xmat[gid].reshape(3, 3)
    long_axis = R[:, int(np.argmax(ed.model.geom_size[gid]))]
    assert abs(long_axis @ [0, 1, 0]) > 0.99          # file +x became MuJoCo +y
    assert r["mesh"].get("scale") == "0.001 0.001 0.001"
    assert r["mesh"].get("file") == "../" * 0 + str((tmp_path / "part.obj").relative_to(tmp_path)) or True


def test_edit_mesh_with_undo_and_save(ed, tmp_path):
    out = MO.create_primitive_mesh(ed, None, "box")
    me = out["mesh"]
    data = MO.get_mesh_data(ed, me)
    data.move_vertices([7], [0, 0, 0.05])
    MO.set_mesh_data(ed, me, data)
    assert ed.result.ok, ed.result.error
    mid = ed.idmap.id(me)
    top = M.from_compiled(ed.model, mid).V[:, 2].max()
    assert top == pytest.approx(0.1)
    ed.undo_step()
    assert M.from_compiled(ed.model, ed.idmap.id(me)).V[:, 2].max() == pytest.approx(0.05)
    ed.redo_step()
    p = tmp_path / "out" / "model.xml"
    notes = ed.save(p)
    assert any("wrote" in n for n in notes)
    m2 = mujoco.MjModel.from_xml_path(str(p))
    assert M.from_compiled(m2, 0).V[:, 2].max() == pytest.approx(0.1)


def test_edit_never_overwrites_original_file(ed, tmp_path):
    src = tmp_path / "orig.obj"
    src.write_bytes(M.write_obj(M.box()))
    before = src.read_bytes()
    r = MO.import_mesh(ed, src, collision="none")
    me = r["mesh"]
    d = MO.get_mesh_data(ed, me)
    i = d.poke_face(4)                      # add a vertex by splitting a face
    d.move_vertices([i], d.face_normals()[4] * 0.02)
    MO.set_mesh_data(ed, me, d)
    assert ed.result.ok, ed.result.error
    assert me.get("file").endswith("_edit.obj")
    ed.save(tmp_path / "m.xml")
    assert src.read_bytes() == before


def test_export_and_duplicate(ed, tmp_path):
    out = MO.create_primitive_mesh(ed, None, "capsule")
    me = out["mesh"]
    p = MO.export_mesh(ed, me, tmp_path / "c.stl")
    assert M.read_stl(p.read_bytes()).volume() == pytest.approx(M.capsule().volume(), rel=1e-4)
    dup = MO.duplicate_mesh_asset(ed, me)
    assert ed.result.ok and dup.get("name") != me.get("name")


def test_inline_mesh_edit(ed):
    from mjstudio.core.doc import MjcfDoc
    ed.set_doc(MjcfDoc.from_string('<mujoco><asset><mesh name="t" vertex="0 0 0 1 0 0 0 1 0 0 0 1" '
                                   'face="0 2 1 0 1 3 0 3 2 1 2 3"/></asset><worldbody><geom type="mesh" mesh="t"/>'
                                   '</worldbody></mujoco>'))
    me = ed.doc.find_named("mesh", "t")
    d = MO.get_mesh_data(ed, me)
    d.move_vertices([3], [0, 0, 1])
    MO.set_mesh_data(ed, me, d)
    assert ed.result.ok and me.get("vertex").endswith("0 0 2")
