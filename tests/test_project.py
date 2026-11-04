"""Project save/load, packaging, joint groups, coupling."""
import math

import mujoco
import numpy as np
import pytest

from mjstudio.core import ops
from mjstudio.core.editor import Editor
from mjstudio.core.project import JointGroup, Project, missing_resources
from mjstudio.library import anatomy as A

TET = "v 0 0 0\nv 0.1 0 0\nv 0 0.1 0\nv 0 0 0.1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n"


def test_project_roundtrip_with_packaging(tmp_path):
    src = tmp_path / "src"
    (src / "assets").mkdir(parents=True)
    (src / "assets" / "tet.obj").write_text(TET)
    (src / "m.xml").write_text('<mujoco><compiler meshdir="assets"/><asset><mesh file="tet.obj"/></asset>'
                               '<worldbody><body name="b"><freejoint/><geom type="mesh" mesh="tet"/></body></worldbody></mujoco>')
    ed = Editor()
    ed.open(src / "m.xml")
    assert not missing_resources(ed.doc)
    prj = Project(tmp_path / "proj")
    prj.set_group(JointGroup("all", {"b": 1.0}))
    notes = prj.save(ed.doc)
    assert (tmp_path / "proj" / "meshes" / "tet.obj").exists()
    assert any("packaged" in n for n in notes)
    # the packaged project no longer depends on the source folder
    import shutil
    shutil.rmtree(src)
    prj2, doc2 = Project.load(tmp_path / "proj")
    assert [g.name for g in prj2.groups] == ["all"]
    assert mujoco.MjModel.from_xml_path(str(prj2.model_path)).nmesh == 1
    assert not missing_resources(doc2)


def test_missing_resources_reported(tmp_path):
    (tmp_path / "m.xml").write_text('<mujoco><asset><mesh file="nope.stl"/></asset><worldbody/></mujoco>')
    ed = Editor(auto_compile=False)
    ed.set_doc(__import__("mjstudio.core.doc", fromlist=["MjcfDoc"]).MjcfDoc.load(tmp_path / "m.xml"))
    miss = missing_resources(ed.doc)
    assert miss and miss[0][2].endswith("nope.stl")


def test_joint_group_fist_and_unknown_keys_preserved(tmp_path):
    ed = Editor()
    ed.new("hand")
    A.build_hand(ed, None, "left", pos=(0, 0, 1))
    fist = JointGroup("fist", {f"{f}_{j}_left": w for f in ("index", "middle", "ring", "little")
                               for j, w in (("mcp_flex", 1.0), ("pip", 1.1), ("dip", 0.8))})
    fist.apply(ed, math.radians(70))
    assert ed.get_qpos("index_pip_left")[0] == pytest.approx(math.radians(77))
    assert ed.get_qpos("little_dip_left")[0] == pytest.approx(math.radians(56))
    prj = Project(tmp_path / "p")
    prj.set_group(fist)
    prj.data["future_feature"] = {"x": 1}
    prj.save(ed.doc)
    prj2, _ = Project.load(tmp_path / "p")
    assert prj2.data["future_feature"] == {"x": 1}
    assert prj2.groups[0].joints == fist.joints


def test_physical_coupling_equality():
    ed = Editor()
    ed.new("c")
    with ed.batch("b"):
        b = ops.add_body(ed, None, "a", pos=(0, 0, 1))
        j1 = ops.add_joint(ed, b, "hinge", "j1", axis=(0, 1, 0), damping="1")
        ops.add_geom(ed, b, "capsule", size="0.02", fromto="0 0 0 0.2 0 0")
        c = ops.add_body(ed, b, "c", pos=(0.2, 0, 0))
        ops.add_joint(ed, c, "hinge", "j2", axis=(0, 1, 0), damping="1")
        ops.add_geom(ed, c, "capsule", size="0.02", fromto="0 0 0 0.2 0 0")
    ops.couple_joints(ed, "j2", "j1", ratio=0.5)
    ops.add_actuator(ed, j1, "position", kp="20")
    assert ed.result.ok, ed.result.error
    m = ed.model
    m.opt.gravity[:] = 0
    d = mujoco.MjData(m)
    d.ctrl[0] = 0.6
    for _ in range(1500):
        mujoco.mj_step(m, d)
    q1, q2 = d.qpos[0], d.qpos[1]
    assert q1 == pytest.approx(0.6, abs=0.03) and q2 == pytest.approx(0.5 * q1, abs=0.01)
