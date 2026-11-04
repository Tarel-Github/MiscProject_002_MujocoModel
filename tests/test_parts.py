"""Part extraction / assembly / replacement."""
import math

import mujoco
import numpy as np
import pytest

from mjstudio.core import mathx as mx
from mjstudio.core import ops
from mjstudio.core.compiler import compile_doc
from mjstudio.core.doc import MjcfDoc
from mjstudio.core.editor import Editor
from mjstudio.core.parts import extract_part, import_part, replace_body

OBJ = mujoco.mjtObj
ARM_JOINTS = ("shoulder1_left", "shoulder2_left", "elbow_left")


@pytest.fixture
def hum(humanoid_path):
    ed = Editor()
    ed.open(humanoid_path)
    return ed


def _jparams(m, name):
    j = mujoco.mj_name2id(m, OBJ.mjOBJ_JOINT, name)
    d = m.jnt_dofadr[j]
    return np.concatenate([m.jnt_range[j], [m.dof_damping[d], m.dof_armature[d], m.jnt_stiffness[j],
                                             m.jnt_limited[j]], m.jnt_axis[j], m.jnt_pos[j]])


def test_extract_arm_standalone(hum, tmp_path):
    arm = hum.doc.find_named("body", "upper_arm_left")
    part = extract_part(hum.doc, arm)
    res = compile_doc(part)
    assert res.ok, res.error
    m = res.model
    assert m.nbody == 1 + 3 and m.njnt == 3
    acts = {mujoco.mj_id2name(m, OBJ.mjOBJ_ACTUATOR, i) for i in range(m.nu)}
    assert acts == set(ARM_JOINTS)
    # same joint physics as inside the humanoid (childclass 'body' inherited from torso)
    for j in ARM_JOINTS:
        assert np.allclose(_jparams(m, j), _jparams(hum.model, j))
    # save as an independent part file and reload
    p = tmp_path / "arm_part.xml"
    part.save(p)
    assert mujoco.MjModel.from_xml_path(str(p)).njnt == 3


def test_import_into_radian_model_preserves_physics(hum, tmp_path):
    part = extract_part(hum.doc, hum.doc.find_named("body", "upper_arm_left"))
    ed = Editor()
    ed.new("robot")
    # target main default that must NOT leak into the part
    ed.doc.root.insert(1, MjcfDoc.from_string('<mujoco><default><joint damping="99" armature="7"/></default></mujoco>').root[0])
    ed.recompile()
    torso = ops.add_body(ed, None, "chest", pos=(0, 0, 1))
    ops.add_joint(ed, torso, "free")
    ops.add_geom(ed, torso, "box", size="0.1 0.1 0.2")
    out = import_part(ed, part, torso, pos=(0, 0.2, 0.1))
    assert ed.result.ok, ed.result.error
    assert ed.doc.angle_unit == "radian"
    for j in ARM_JOINTS:
        assert np.allclose(_jparams(ed.model, j), _jparams(hum.model, j)), j
    assert out[0].get("name") == "upper_arm_left"
    ed.undo_step()
    assert ed.doc.find_named("body", "upper_arm_left") is None and ed.result.ok


def test_import_twice_and_into_self_renames(hum):
    part = extract_part(hum.doc, hum.doc.find_named("body", "upper_arm_left"))
    nu0, nj0 = hum.model.nu, hum.model.njnt
    torso = hum.doc.find_named("body", "torso")
    import_part(hum, part, torso, pos=(0, 0.4, 0), prefix="extra_")
    assert hum.result.ok, hum.result.error
    import_part(hum, part, torso, pos=(0, -0.4, 0), prefix="extra_")
    assert hum.result.ok, hum.result.error
    m = hum.model
    assert m.nu == nu0 + 6 and m.njnt == nj0 + 6
    a = hum.doc.find_named("actuator", "extra_elbow_left")
    assert a is not None and a.get("joint") == "extra_elbow_left"
    assert hum.doc.find_named("actuator", "extra_elbow_left_1").get("joint") == "extra_elbow_left_1"
    # keyframes were extended with qpos0 for the new joints
    assert m.key_qpos.shape[1] == m.nq


def test_replace_head(hum):
    head = hum.doc.find_named("body", "head")
    lp, lq = hum.doc.local_pose(head)
    new = Editor()
    new.new("head_v2")
    h = ops.add_body(new, None, "head", pos=(0, 0, 0))
    ops.add_geom(new, h, "ellipsoid", "skull", size="0.08 0.07 0.1")
    for side, y in (("left", 0.03), ("right", -0.03)):
        eye = ops.add_body(new, h, f"eye_{side}", pos=(0.07, y, 0.02))
        ops.add_joint(new, eye, "hinge", f"eye_{side}_yaw", axis=(0, 0, 1), range_rad=(-0.7, 0.7))
        ops.add_joint(new, eye, "hinge", f"eye_{side}_pitch", axis=(0, 1, 0), range_rad=(-0.5, 0.6))
        ops.add_geom(new, eye, "sphere", size="0.012", contype="0", conaffinity="0", mass="0.007")
    nj0 = hum.model.njnt
    replace_body(hum, head, new.doc)
    assert hum.result.ok, hum.result.error
    newhead = hum.doc.find_named("body", "head")
    assert newhead is not None and newhead is not head
    p2, q2 = hum.doc.local_pose(newhead)
    assert np.allclose(p2, lp) and mx.qangle(q2, lq) < 1e-9
    assert hum.model.njnt == nj0 + 4
    assert hum.doc.find_named("geom", "skull") is not None
    # the egocentric camera lived in the old head and is gone with it
    assert hum.doc.find_named("camera", "egocentric") is None


def test_free_joint_removed_when_attached_to_body(hum):
    part = extract_part(hum.doc, hum.doc.find_named("body", "torso"))
    ed = Editor()
    ed.new("x")
    b = ops.add_body(ed, None, "base", pos=(0, 0, 2))
    ops.add_geom(ed, b, "box")
    import_part(ed, part, b)
    assert ed.result.ok, ed.result.error
    assert not any(e.tag == "freejoint" for e in b.iter())


def test_nested_structural_edits_keep_keyframes_valid(hum):
    """Several replacements inside one batch: keyframes must be remapped consecutively."""
    from mjstudio.library import anatomy as A
    k0 = hum.model.key_qpos.copy()
    knee = hum.model.jnt_qposadr[hum.joint_id("knee_left")]
    with hum.batch("many"):
        for side in ("left", "right"):
            part = A.make_part(A.build_hand, f"h{side}", side=side)
            replace_body(hum, hum.doc.find_named("body", f"hand_{side}"), part, prefix=f"{side}_")
            part = A.make_part(A.build_foot, f"f{side}", side=side)
            replace_body(hum, hum.doc.find_named("body", f"foot_{side}"), part, prefix=f"{side}_")
    assert hum.result.ok, hum.result.error
    m = hum.model
    assert m.key_qpos.shape == (2, m.nq)
    assert np.allclose(m.key_qpos[:, m.jnt_qposadr[hum.joint_id("knee_left")]], k0[:, knee])
