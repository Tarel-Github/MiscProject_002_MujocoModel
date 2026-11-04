"""Building operations: add elements, duplicate with dependents, mirror."""
import math

import mujoco
import numpy as np
import pytest

from mjstudio.core import mathx as mx
from mjstudio.core import ops
from mjstudio.core.editor import Editor


@pytest.fixture
def ed():
    e = Editor()
    e.new("t")
    return e


def test_build_arm(ed):
    with ed.batch("build"):
        b = ops.add_body(ed, None, "upper", pos=(0, 0.2, 1.0))
        ops.add_joint(ed, b, "ball", "shoulder", range_rad=(0, math.radians(90)))
        ops.add_geom(ed, b, "capsule", "upper_g", size="0.04", fromto="0 0 0 0 0.3 0")
        f = ops.add_body(ed, b, "fore_left", pos=(0, 0.3, 0))
        j = ops.add_joint(ed, f, "hinge", "elbow_left", axis=(1, 0, 0), range_rad=(0, math.radians(140)))
        ops.add_geom(ed, f, "capsule", size="0.035", fromto="0 0 0 0 0.25 0")
        ops.add_site(ed, f, "tip_left", pos=(0, 0.27, 0))
    a = ops.add_actuator(ed, j, "position", kp=20)
    ops.add_sensor(ed, "jointpos", joint="elbow_left")
    ops.add_sensor(ed, "touch", site="tip_left")
    assert ed.result.ok, ed.result.error
    m = ed.model
    assert m.nbody == 3 and m.njnt == 2 and m.nu == 1 and m.nsensor == 2
    jid = ed.joint_id("elbow_left")
    assert m.jnt_range[jid][1] == pytest.approx(math.radians(140))
    assert m.actuator_ctrlrange[0][1] == pytest.approx(math.radians(140))
    assert a.get("ctrlrange") is not None
    # free joint only for world children
    with pytest.raises(ValueError):
        ops.add_joint(ed, f, "free")
    ops.add_joint(ed, b, "free") if False else None


def test_duplicate_with_dependents(humanoid_path):
    ed = Editor()
    ed.open(humanoid_path)
    nu0, ns0 = ed.model.nu, ed.model.nsensor
    arm = ed.doc.find_named("body", "upper_arm_left")
    new = ops.duplicate(ed, [arm])
    assert ed.result.ok, ed.result.error
    m = ed.model
    # 3 arm motors duplicated (shoulder1, shoulder2, elbow) referencing renamed joints
    assert m.nu == nu0 + 3
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
    assert "elbow_left_copy" in names
    act = ed.doc.find_named("actuator", "elbow_left_copy")
    assert act.get("joint") == "elbow_left_copy"
    assert new[0].get("name") == "upper_arm_left_copy"
    ed.undo_step()
    assert ed.model.nu == nu0 and ed.model.nsensor == ns0


def test_mirror_duplicate_symmetry(ed):
    with ed.batch("build"):
        b = ops.add_body(ed, None, "torso", pos=(0, 0, 1))
        sh = ops.add_body(ed, b, "arm_left", pos=(0, 0.2, 0.3), quat=mx.axang2q([1, 0.3, 0.2], 0.4))
        ops.add_joint(ed, sh, "hinge", "shoulder_left", axis=(0.3, 1, 0.2), range_rad=(-1, 2))
        ops.add_geom(ed, sh, "capsule", size="0.04", fromto="0 0 0 0.05 0.3 -0.1")
        ops.add_site(ed, sh, "hand_left", pos=(0.02, 0.33, -0.1))
    ops.add_actuator(ed, "shoulder_left", "motor")
    out = ops.mirror_duplicate(ed, sh)
    assert ed.result.ok, ed.result.error
    m, d = ed.model, ed.data
    assert ed.doc.find_named("joint", "shoulder_right") is not None
    assert ed.doc.find_named("actuator", "mot_shoulder_right") is not None
    for q in (0.0, 0.7, -0.5):
        ed.set_qpos("shoulder_left", [q])
        ed.set_qpos("shoulder_right", [q])
        L = d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "hand_left")]
        R = d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "hand_right")]
        # mirrored across the torso's XZ plane (torso at y=0)
        assert np.allclose(R, L * [1, -1, 1], atol=1e-6), (q, L, R)
    assert out[0].get("name") == "arm_right"
