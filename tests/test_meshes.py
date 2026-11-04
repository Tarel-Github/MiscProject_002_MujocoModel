"""Mesh primitives, I/O and editing operations."""
import math

import mujoco
import numpy as np
import pytest

from mjstudio.core import mathx as mx
from mjstudio.core import meshes as M


def _compile_mesh(m: M.TriMesh, collide=True):
    v = " ".join(f"{x:.9g}" for x in m.V.ravel())
    f = " ".join(str(int(x)) for x in m.F.ravel())
    extra = "" if collide else ' contype="0" conaffinity="0" mass="0"'
    from mjstudio.core.compiler import supports
    shell = ' inertia="shell"' if (m.is_planar() and supports("mesh_inertia")) else ""
    xml = (f'<mujoco><asset><mesh name="m" vertex="{v}" face="{f}"{shell}/></asset><worldbody><body><freejoint/>'
           f'<geom size="0.01"/><geom type="mesh" mesh="m"{extra}/></body></worldbody></mujoco>')
    return mujoco.MjModel.from_xml_string(xml)


@pytest.mark.parametrize("name", ["box", "sphere", "ellipsoid", "cylinder", "capsule"])
def test_closed_primitives_compile_with_correct_volume(name):
    m = M.PRIMITIVES[name]()
    assert not m.validate()
    model = _compile_mesh(m)
    assert model.nmesh == 1
    vol = m.volume()
    assert vol > 0, "outward orientation"
    expect = {"box": 0.1 ** 3, "sphere": 4 / 3 * math.pi * 0.05 ** 3, "ellipsoid": 4 / 3 * math.pi * 0.06 * 0.04 * 0.03,
              "cylinder": math.pi * 0.03 ** 2 * 0.2,
              "capsule": math.pi * 0.03 ** 2 * 0.2 + 4 / 3 * math.pi * 0.03 ** 3}[name]
    assert vol == pytest.approx(expect, rel=0.05)


@pytest.mark.parametrize("name", ["plane", "triangle", "quad"])
def test_open_primitives_are_visual_only(name):
    m = M.PRIMITIVES[name]()
    assert m.is_planar()
    assert _compile_mesh(m, collide=False).nmesh == 1


def test_obj_stl_roundtrip():
    m = M.capsule()
    for ext in ("obj", "stl"):
        data = M.encode(m, ext)
        r = M.read_obj(data) if ext == "obj" else M.read_stl(data)
        assert r.nv == m.nv and r.nf == m.nf
        assert r.volume() == pytest.approx(m.volume(), rel=1e-4)


def test_obj_polygons_and_negative_indices():
    obj = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf -4/1/1 -3/2/1 -2/3/1 -1/4/1\n"
    m = M.read_obj(obj)
    assert m.nf == 2 and m.F.max() == 3


def test_edit_operations():
    m = M.box()
    m.move_vertices([7], [0, 0, 0.05])
    assert m.V[7, 2] == pytest.approx(0.1)
    i = m.add_vertex([0, 0, 0.2])
    m.add_face(4, 5, i)
    assert m.nf == 13
    with pytest.raises(ValueError):
        m.add_face(1, 1, 2)
    m.delete_faces([12])
    m.delete_vertices([i])
    assert m.nv == 8 and m.nf == 12
    m.delete_vertices([0])
    assert m.nv == 7 and m.F.max() < 7 and m.nf < 12
    m2 = M.box()
    m2.flip()
    assert m2.volume() < 0


def test_transform_units_axes_and_mirror():
    m = M.box(1, 2, 3)
    t = m.transformed(scale=M.UNITS["mm"], quat=M.AXES["blender"])
    lo, hi = t.bounds()
    assert np.allclose(hi - lo, [0.004, 0.002, 0.006])   # x<->y swapped by the Blender preset
    mir = m.transformed(scale=[-1, 1, 1])
    assert mir.volume() == pytest.approx(m.volume())       # winding fixed after mirroring
    # glTF-style Y-up: file +Y (up) -> MuJoCo +Z, file +Z (front) -> MuJoCo +X
    R = mx.q2mat(M.AXES["yup"])
    assert np.allclose(R @ [0, 1, 0], [0, 0, 1]) and np.allclose(R @ [0, 0, 1], [1, 0, 0])


def test_from_compiled_recovers_file_frame():
    m = M.box(0.1, 0.05, 0.02).transformed(pos=[0.3, 0.2, 0.1], quat=mx.axang2q([1, 2, 3], 0.7))
    model = _compile_mesh(m)
    r = M.from_compiled(model, 0)
    assert np.allclose(r.V, m.V, atol=1e-6)


def test_ray_pick_face():
    m = M.box()
    f, t = m.pick_face([0.0, 0.0, 1.0], [0.0, 0.0, -1.0])
    assert f >= 0 and t == pytest.approx(0.95)
    assert m.face_normals()[f] @ [0, 0, 1] > 0.99
