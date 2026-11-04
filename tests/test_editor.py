"""Editor session: undo/redo, rename refs, cascade delete, keyframe remap, reparent."""
import mujoco
import numpy as np
import pytest

from mjstudio.core import mathx as mx
from mjstudio.core.doc import MjcfDoc
from mjstudio.core.editor import Editor

OBJ = mujoco.mjtObj


@pytest.fixture
def ed(humanoid_path):
    e = Editor()
    e.open(humanoid_path)
    assert e.result.ok
    return e


def test_undo_redo_exact(ed):
    before = ed.doc.to_string()
    torso = ed.doc.find_named("body", "torso")
    knee = ed.doc.find_named("joint", "knee_left")
    ed.set_attr(knee, "damping", "3")
    ed.set_pose(torso, [0, 0, 1.5], mx.axang2q([0, 0, 1], 0.3))
    with ed.batch("multi"):
        ed.set_attr(knee, "range", "-120 0")
        ed.set_attr(knee, "armature", "0.02")
    assert ed.model.jnt_range[ed.joint_id("knee_left")][0] == pytest.approx(np.radians(-120))
    assert ed.dirty
    for _ in range(3):
        ed.undo_step()
    assert ed.doc.to_string() == before
    for _ in range(3):
        ed.redo_step()
    assert knee.get("armature") == "0.02"
    assert ed.model.dof_armature[ed.model.jnt_dofadr[ed.joint_id("knee_left")]] == pytest.approx(0.02)


def test_merge_key(ed):
    knee = ed.doc.find_named("joint", "knee_left")
    for v in ("1", "2", "3"):
        ed.set_attr(knee, "damping", v, merge_key="drag")
    assert knee.get("damping") == "3"
    ed.undo_step()
    assert knee.get("damping") is None


def test_rename_updates_references(ed):
    j = ed.doc.find_named("joint", "hip_x_right")
    refs = ed.references_to("joint", "hip_x_right")
    assert refs, "humanoid actuators reference the hip joint"
    ed.rename(j, "hipR_x")
    assert ed.result.ok, ed.result.error
    assert all(r.get(a) == "hipR_x" for r, a in refs)
    with pytest.raises(ValueError):
        ed.rename(j, "knee_left")  # duplicate
    ed.undo_step()
    assert j.get("name") == "hip_x_right" and all(r.get(a) == "hip_x_right" for r, a in refs)
    # body rename updates light target
    ed.rename(ed.doc.find_named("body", "torso"), "chest")
    assert ed.doc.find_named("light", "spotlight").get("target") == "chest"
    assert ed.result.ok


def test_delete_cascade_and_keyframes(ed):
    m0 = ed.model
    nq0, nu0, nkey = m0.nq, m0.nu, m0.nkey
    key_q = m0.key_qpos.copy()
    hip_qadr = m0.jnt_qposadr[ed.joint_id("hip_x_left")]
    # value of a joint *after* the deleted subtree must survive the remap
    elbow_qadr = m0.jnt_qposadr[ed.joint_id("elbow_left")]
    thigh = ed.doc.find_named("body", "thigh_right")
    removed = ed.delete([thigh])
    assert ed.result.ok, ed.result.error
    m = ed.model
    assert m.nq == nq0 - 6 and m.nu < nu0 and m.nkey == nkey
    assert any(r.tag == "motor" for r in removed)
    new_elbow = m.jnt_qposadr[ed.joint_id("elbow_left")]
    assert np.allclose(m.key_qpos[:, new_elbow], key_q[:, elbow_qadr])
    new_hip = m.jnt_qposadr[ed.joint_id("hip_x_left")]
    assert np.allclose(m.key_qpos[:, new_hip], key_q[:, hip_qadr])
    ed.undo_step()
    assert ed.model.nq == nq0 and np.allclose(ed.model.key_qpos, key_q)


def test_move_keeps_world_pose(ed):
    hand = ed.doc.find_named("body", "hand_left")
    torso = ed.doc.find_named("body", "torso")
    wp, wq = ed.world_pose(hand)
    ed.move(hand, torso)
    assert ed.result.ok, ed.result.error
    assert hand.getparent() is torso
    wp2, wq2 = ed.world_pose(hand)
    assert np.allclose(wp, wp2, atol=1e-6) and mx.qangle(wq, wq2) < 1e-6


def test_state_transfer_on_recompile(ed):
    ed.set_qpos("knee_left", [-1.0])
    ed.set_attr(ed.doc.find_named("geom", "head"), "rgba", "1 0 0 1")
    assert ed.get_qpos("knee_left")[0] == pytest.approx(-1.0)


def test_new_doc_and_save(tmp_path):
    ed = Editor()
    ed.new("blank")
    assert ed.result.ok
    p = tmp_path / "blank.xml"
    ed.save(p)
    assert not ed.dirty
    assert mujoco.MjModel.from_xml_path(str(p)).ngeom == 1


def test_world_delta_with_posed_joints(ed):
    """Translating a posed child body in world space moves it exactly by that vector."""
    ed.set_qpos("knee_left", [-1.2])
    ed.set_qpos("hip_y_left", [-0.7])
    foot = ed.doc.find_named("body", "foot_left")
    bid = ed.idmap.id(foot)
    x0 = ed.data.xpos[bid].copy()
    q0 = ed.data.xquat[bid].copy()
    t = np.array([0.05, -0.02, 0.03])
    Q = mx.axang2q([0, 1, 0], 0.4)
    lp, lq = ed.local_after_world_delta(foot, t, Q, x0)
    ed.set_pose(foot, lp, lq)
    bid = ed.idmap.id(foot)
    assert np.allclose(ed.data.xpos[bid], x0 + t, atol=1e-6)
    assert mx.qangle(ed.data.xquat[bid], mx.qmul(Q, q0)) < 1e-6


def test_move_keeps_world_pose_when_posed(ed):
    ed.set_qpos("elbow_left", [0.8])
    ed.set_qpos("shoulder1_left", [0.5])
    hand = ed.doc.find_named("body", "hand_left")
    bid = ed.idmap.id(hand)
    x0, q0 = ed.data.xpos[bid].copy(), ed.data.xquat[bid].copy()
    ed.move(hand, ed.doc.find_named("body", "torso"))
    bid = ed.idmap.id(hand)
    assert np.allclose(ed.data.xpos[bid], x0, atol=1e-6)
    assert mx.qangle(ed.data.xquat[bid], q0) < 1e-6


def test_transform_session_preview_matches_commit(ed):
    from mjstudio.core.preview import TransformSession
    ed.set_qpos("knee_left", [-0.9])
    targets = [ed.doc.find_named("body", "foot_left"), ed.doc.find_named("geom", "head"),
               ed.doc.find_named("joint", "elbow_right")]
    ts = TransformSession(ed, targets)
    pivot = ed.world_pose(targets[0])[0]
    t, Q = np.array([0.02, 0.0, 0.05]), mx.axang2q([1, 0, 0], 0.3)
    ts.update(t, Q, pivot)
    expect = {it["e"]: mx.pose_mul(*it["P0"], *it["L"]) for it in ts.items}
    foot_preview = ed.data.xpos[ed.idmap.id(targets[0])].copy()
    prev_geom = ed.data.geom_xpos[ed.idmap.id(targets[1])].copy()
    prev_axis = ed.data.xaxis[ed.idmap.id(targets[2])].copy()
    ts.commit("drag")
    assert ed.result.ok
    for e in targets:
        p, q = ed.frame_world(e)
        assert np.allclose(p, expect[e][0], atol=1e-6), e.get("name")
        assert mx.qangle(q, expect[e][1]) < 1e-6
    assert np.allclose(ed.data.xpos[ed.idmap.id(targets[0])], foot_preview, atol=1e-6)
    assert np.allclose(ed.data.xaxis[ed.idmap.id(targets[2])], prev_axis, atol=1e-6)
    assert np.allclose(ed.data.geom_xpos[ed.idmap.id(targets[1])], prev_geom, atol=1e-6)
    ed.undo_step()  # one undo step reverts the whole drag
    assert targets[0].get("pos") == "0 0 -.39"  # exact original text restored


def test_transform_session_free_body(ed):
    from mjstudio.core.preview import TransformSession
    torso = ed.doc.find_named("body", "torso")
    ts = TransformSession(ed, [torso])
    x0 = ed.world_pose(torso)[0]
    ts.update([0.3, 0, 0], mx.QID, x0)
    assert np.allclose(ed.world_pose(torso)[0], x0 + [0.3, 0, 0])
    ts.commit()
    assert np.allclose(ed.world_pose(torso)[0], x0 + [0.3, 0, 0])
    assert torso.get("pos") == "0.3 0 1.282"


def test_joint_type_change_and_add_joint_remap_keyframes(ed):
    from mjstudio.core import ops
    nq0 = ed.model.nq
    knee = ed.doc.find_named("joint", "knee_left")
    ed.set_attrs(knee, {"type": "ball", "range": "0 60"})
    assert ed.result.ok, ed.result.error
    assert ed.model.nq == nq0 + 3 and ed.model.key_qpos.shape[1] == ed.model.nq
    ed.undo_step()
    assert ed.result.ok and ed.model.nq == nq0
    ops.add_joint(ed, ed.doc.find_named("body", "head"), "hinge", "nod", axis=(0, 1, 0))
    assert ed.result.ok, ed.result.error
    assert ed.model.nq == nq0 + 1
