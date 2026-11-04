"""Document model: load/save fidelity, defaults, poses, compile mapping."""
import math

import mujoco
import numpy as np
import pytest

from mjstudio.core import mathx as mx
from mjstudio.core.compiler import compile_doc
from mjstudio.core.doc import MjcfDoc, children

OBJ = mujoco.mjtObj

EXTRA = """<mujoco model="extra">
  <!-- a comment that must survive -->
  <compiler angle="degree" eulerseq="zyx"/>
  <size nuserdata="3"/>
  <custom><numeric name="answer" data="42"/><text name="note" data="hello"/></custom>
  <statistic center="0 0 1"/>
  <default>
    <geom rgba="0.1 0.2 0.3 1"/>
    <default class="arm"><joint damping="2" range="-90 90"/><geom type="capsule" size="0.03" fromto="0 0 0 0.3 0 0"/></default>
  </default>
  <worldbody>
    <body name="base" pos="0 0 1" euler="10 20 30">
      <geom type="box" size="0.1 0.1 0.1" axisangle="1 1 0 45"/>
      <body name="arm" childclass="arm" pos="0.1 0 0" xyaxes="0 1 0 -1 0 0">
        <joint name="j1" axis="0 0 1"/>
        <geom name="armgeom"/>
        <site name="tip" pos="0.3 0 0" zaxis="1 1 1"/>
        <body name="fore" pos="0.3 0 0" quat="0.9 0 0.1 0">
          <joint name="j2" axis="0 1 0" range="-10 120"/>
          <geom fromto="0 0 0 0 0 -0.25"/>
        </body>
      </body>
    </body>
  </worldbody>
  <tendon><fixed name="t"><joint joint="j1" coef="1"/><joint joint="j2" coef="-1"/></fixed></tendon>
  <equality><joint name="couple" joint1="j1" joint2="j2" polycoef="0 0.5 0 0 0" active="false"/></equality>
  <actuator><motor name="m1" joint="j1" gear="10"/><position joint="j2" kp="50"/></actuator>
  <sensor><jointpos joint="j1"/><framepos objtype="site" objname="tip"/><touch name="touch" site="tip"/></sensor>
  <keyframe><key name="k" qpos="0.1 0.2" ctrl="0 0"/></keyframe>
</mujoco>"""


def test_roundtrip_humanoid(humanoid_path, tmp_path):
    ref = mujoco.MjModel.from_xml_path(str(humanoid_path))
    doc = MjcfDoc.load(humanoid_path)
    out = tmp_path / "h.xml"
    doc.save(out)
    m2 = mujoco.MjModel.from_xml_path(str(out))
    for f in ("nbody", "njnt", "ngeom", "nsite", "nu", "nsensor", "nkey", "nq", "ncam", "nlight"):
        assert getattr(ref, f) == getattr(m2, f), f
    for f in ("body_pos", "body_quat", "geom_size", "geom_pos", "geom_quat", "jnt_range", "jnt_axis",
              "dof_damping", "actuator_gear", "key_qpos", "geom_friction", "body_mass"):
        assert np.allclose(getattr(ref, f), getattr(m2, f)), f
    # comment (license header) preserved
    assert "Licensed under the Apache License" in out.read_text()


def test_preserves_unknown_and_comments(tmp_path):
    doc = MjcfDoc.from_string(EXTRA, base_dir=tmp_path)
    p = tmp_path / "e.xml"
    doc.save(p)
    txt = p.read_text()
    for s in ("a comment that must survive", '<numeric name="answer" data="42"/>', "<statistic",
              'polycoef="0 0.5 0 0 0"', 'nuserdata="3"', 'eulerseq="zyx"'):
        assert s in txt, s
    m = mujoco.MjModel.from_xml_path(str(p))
    assert m.ntendon == 1 and m.neq == 1 and m.nnumeric == 1


def test_compile_idmap_and_defaults(humanoid_path):
    doc = MjcfDoc.load(humanoid_path)
    res = compile_doc(doc)
    assert res.ok, res.error
    m = res.model
    # every body/joint/geom/site maps back to the right element
    for tag, ot, n in (("body", OBJ.mjOBJ_BODY, m.nbody), ("geom", OBJ.mjOBJ_GEOM, m.ngeom),
                       ("site", OBJ.mjOBJ_SITE, m.nsite), ("camera", OBJ.mjOBJ_CAMERA, m.ncam)):
        mapped = [res.idmap.elem(int(ot), i) for i in range(n)]
        if tag == "body":
            assert mapped[0] is None  # world
            mapped = mapped[1:]
        assert all(e is not None and e.tag == tag for e in mapped), tag
    for j in range(m.njnt):
        e = res.idmap.elem(int(OBJ.mjOBJ_JOINT), j)
        assert e is not None and e.get("name") == mujoco.mj_id2name(m, OBJ.mjOBJ_JOINT, j)
    # default class resolution: knee joints get class 'knee' range
    knee = doc.find_named("joint", "knee_right")
    r = doc.resolved(knee)
    assert r["range"] == ("-160 2", "class:knee")
    assert r["damping"][1] == "class:body"
    assert r["name"][1] == "explicit"


def test_local_pose_matches_mujoco(humanoid_path):
    for src in (str(humanoid_path), None):
        doc = MjcfDoc.load(src) if src else MjcfDoc.from_string(EXTRA)
        res = compile_doc(doc)
        m = res.model
        for e, (ot, i) in res.idmap.by_elem.items():
            if ot == int(OBJ.mjOBJ_BODY):
                p, q = doc.local_pose(e)
                assert np.allclose(p, m.body_pos[i], atol=1e-6), doc.label(e)
                assert mx.qangle(q, m.body_quat[i]) < 1e-5, doc.label(e)
            elif ot == int(OBJ.mjOBJ_GEOM) and m.geom_type[i] != mujoco.mjtGeom.mjGEOM_MESH:
                p, q = doc.local_pose(e)
                assert np.allclose(p, m.geom_pos[i], atol=1e-6), doc.label(e)
                if m.geom_type[i] not in (mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_PLANE):
                    assert mx.qangle(q, m.geom_quat[i]) < 1e-5, doc.label(e)
            elif ot == int(OBJ.mjOBJ_SITE):
                p, q = doc.local_pose(e)
                assert np.allclose(p, m.site_pos[i], atol=1e-6)
                assert mx.qangle(q, m.site_quat[i]) < 1e-5
            elif ot == int(OBJ.mjOBJ_JOINT) and e.tag == "joint":
                p, q = doc.local_pose(e)
                assert np.allclose(p, m.jnt_pos[i], atol=1e-6)
                assert np.allclose(mx.q2mat(q)[:, 2], m.jnt_axis[i], atol=1e-6)


@pytest.mark.parametrize("style", ["quat", "euler", "axisangle", "xyaxes", "zaxis", None])
def test_pose_changes_roundtrip(style):
    rng = np.random.default_rng(7)
    doc = MjcfDoc.from_string(EXTRA)
    base = doc.find_named("body", "base")
    for k in ("euler",):
        del base.attrib[k]
    if style == "quat":
        base.set("quat", "1 0 0 0")
    elif style == "euler":
        base.set("euler", "0 0 0")
    elif style == "axisangle":
        base.set("axisangle", "0 0 1 0")
    elif style == "xyaxes":
        base.set("xyaxes", "1 0 0 0 1 0")
    elif style == "zaxis":
        base.set("zaxis", "0 0 1")
    for _ in range(5):
        p = rng.normal(size=3)
        q = mx.qnorm(rng.normal(size=4))
        for k, v in doc.pose_changes(base, p, q).items():
            if v is None:
                base.attrib.pop(k, None)
            else:
                base.set(k, v)
        res = compile_doc(doc)
        assert res.ok, res.error
        bid = res.idmap.id(base)
        assert np.allclose(res.model.body_pos[bid], p, atol=1e-6)
        assert mx.qangle(res.model.body_quat[bid], q) < 1e-6
        orient = [k for k in ("quat", "axisangle", "xyaxes", "zaxis", "euler") if k in base.attrib]
        assert len(orient) <= 1
        if style in ("euler", "axisangle", "xyaxes"):
            assert orient == [style]


def test_fromto_pose_change():
    doc = MjcfDoc.from_string(EXTRA)
    g = doc.find_named("geom", "armgeom")  # fromto inherited from class 'arm'
    p0, q0 = doc.local_pose(g)
    assert np.allclose(p0, [0.15, 0, 0])
    ch = doc.pose_changes(g, [0.2, 0.1, 0], mx.axang2q([0, 0, 1], math.pi / 2))
    assert "fromto" in ch
    for k, v in ch.items():
        g.set(k, v) if v is not None else g.attrib.pop(k, None)
    res = compile_doc(doc)
    gid = res.idmap.id(g)
    assert np.allclose(res.model.geom_pos[gid], [0.2, 0.1, 0], atol=1e-6)
    assert np.allclose(res.model.geom_size[gid][:2], [0.03, 0.15], atol=1e-6)


def test_compile_error_location():
    doc = MjcfDoc.from_string(EXTRA)
    g = doc.find_named("geom", "armgeom")
    g.set("mesh", "nonexistent")
    g.set("type", "mesh")
    res = compile_doc(doc)
    assert not res.ok
    assert "nonexistent" in res.error
    assert res.error_elem is g


def test_mem_assets_and_paths(tmp_path):
    # an on-disk mesh in a subdir referenced via meshdir, and an in-memory mesh
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "tet.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n")
    xml = """<mujoco><compiler meshdir="assets"/><asset><mesh file="tet.obj"/><mesh name="mem" file="mem.obj"/></asset>
    <worldbody><body><freejoint/><geom type="mesh" mesh="tet"/><geom type="mesh" mesh="mem"/></body></worldbody></mujoco>"""
    (tmp_path / "m.xml").write_text(xml)
    doc = MjcfDoc.load(tmp_path / "m.xml")
    doc.mem_assets["mem.obj"] = (tmp_path / "assets" / "tet.obj").read_bytes()
    res = compile_doc(doc)
    assert res.ok, res.error
    assert res.model.nmesh == 2
    # saving elsewhere rewrites meshdir so the file still resolves
    sub = tmp_path / "out" / "deep"
    doc.mem_assets.clear()
    doc.root.find("asset").remove(doc.root.find("asset")[1])
    doc.root.find("worldbody").find("body").remove(doc.root.find("worldbody").find("body")[2])
    notes = doc.save(sub / "m2.xml")
    assert notes and "meshdir" in notes[0]
    assert mujoco.MjModel.from_xml_path(str(sub / "m2.xml")).nmesh == 1


def test_include_flatten(tmp_path):
    (tmp_path / "part.xml").write_text('<mujoco><body name="inc"><geom size="0.1"/></body></mujoco>')
    (tmp_path / "main.xml").write_text('<mujoco><worldbody><include file="part.xml"/></worldbody></mujoco>')
    doc = MjcfDoc.load(tmp_path / "main.xml")
    assert doc.find_named("body", "inc") is not None
    assert doc.meta["flattened_includes"] == ["part.xml"]
    assert compile_doc(doc).ok
