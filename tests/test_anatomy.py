"""Requirement checks 8.2-8.5 as physics tests (hand, foot, eyes, jaw)."""
import math

import mujoco
import numpy as np
import pytest

from mjstudio.core import mathx as mx
from mjstudio.core import ops
from mjstudio.core.editor import Editor
from mjstudio.library import anatomy as A

OBJ = mujoco.mjtObj
D = math.radians


def _sens(m, d, name):
    i = mujoco.mj_name2id(m, OBJ.mjOBJ_SENSOR, name)
    return d.sensordata[m.sensor_adr[i]:m.sensor_adr[i] + m.sensor_dim[i]].copy()


def _act(m, joint):
    for i in range(m.nu):
        if m.actuator_trnid[i, 0] == mujoco.mj_name2id(m, OBJ.mjOBJ_JOINT, joint):
            return i
    raise KeyError(joint)


def _q(m, d, joint):
    return d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, OBJ.mjOBJ_JOINT, joint)]]


# ------------------------------------------------------------------- foot rig
@pytest.fixture(scope="module")
def foot_rig():
    """Loaded leg: a 40 kg 'shin' that can only slide vertically, segmented foot on a floor."""
    ed = Editor()
    ed.new("foot_rig")
    with ed.batch("rig"):
        shin = ops.add_body(ed, None, "shin", pos=(0, 0, 0.078))
        ops.add_joint(ed, shin, "slide", "shin_z", axis=(0, 0, 1), damping="200")
        ops.add_geom(ed, shin, "capsule", size="0.04", fromto="0 0 0.05 0 0 0.4", mass="40", contype="0",
                     conaffinity="0")
        A.build_foot(ed, shin, "left", pos=(0, 0, 0), kp_ankle=600)
    assert ed.result.ok, ed.result.error
    return ed


def _settle(m, d, t):
    for _ in range(int(t / m.opt.timestep)):
        mujoco.mj_step(m, d)


def test_foot_segments_present(foot_rig):
    m = foot_rig.model
    for b in ("talus_left", "hindfoot_left", "midfoot_left", "forefoot_left", "toes_left"):
        assert mujoco.mj_name2id(m, OBJ.mjOBJ_BODY, b) >= 0, b
    # separate collision geoms for heel and forefoot (ball) and toes
    for g in ("heel_left", "ball_left", "toes_pad_left"):
        gid = mujoco.mj_name2id(m, OBJ.mjOBJ_GEOM, g)
        assert gid >= 0 and (m.geom_contype[gid] or m.geom_conaffinity[gid]), g


def test_ankle_moves_alone(foot_rig):
    """Q: 발목만 움직일 수 있는가? - ankle angle changes, distal segments stay rigid."""
    ed = foot_rig
    m, d = ed.model, mujoco.MjData(ed.model)
    mujoco.mj_forward(m, d)
    ids = [mujoco.mj_name2id(m, OBJ.mjOBJ_BODY, b) for b in ("hindfoot_left", "forefoot_left", "toes_left")]
    rel0 = d.xpos[ids[2]] - d.xpos[ids[0]]
    d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, OBJ.mjOBJ_JOINT, "ankle_flex_left")]] = D(-30)
    mujoco.mj_forward(m, d)
    rel1 = d.xpos[ids[2]] - d.xpos[ids[0]]
    assert np.isclose(np.linalg.norm(rel0), np.linalg.norm(rel1))  # rigid
    assert not np.allclose(rel0, rel1)                               # but rotated
    shin = mujoco.mj_name2id(m, OBJ.mjOBJ_BODY, "shin")
    assert np.allclose(d.xpos[shin], [0, 0, 0.078])                  # shank unaffected


def test_flat_stance_then_heel_rise(foot_rig):
    """Q: 뒤꿈치가 들리면서 발 앞부분이 지면과 접촉? / 뒤꿈치와 전족부 접촉 구분?"""
    m = foot_rig.model
    d = mujoco.MjData(m)
    _settle(m, d, 1.0)
    heel, fore, toes = (_sens(m, d, f"touch_{z}_left")[0] for z in ("heel_zone", "forefoot_zone", "toes_zone"))
    assert heel > 20 and fore + toes > 20, (heel, fore, toes)   # flat foot: both regions loaded
    z0 = _q(m, d, "shin_z")
    # plantarflex the ankle; MTP dorsiflexes so the toes stay on the floor
    d.ctrl[_act(m, "ankle_flex_left")] = D(-30)
    d.ctrl[_act(m, "mtp_left")] = D(30)
    _settle(m, d, 2.0)
    heel, fore, toes = (_sens(m, d, f"touch_{z}_left")[0] for z in ("heel_zone", "forefoot_zone", "toes_zone"))
    assert heel == 0.0, heel                                       # heel off the ground
    assert fore + toes > 100, (fore, toes)                         # weight on the front of the foot
    assert _q(m, d, "shin_z") > z0 + 0.03                          # body raised by the heel rise
    assert _q(m, d, "mtp_left") > D(15)                            # bending at the MTP line
    hid = mujoco.mj_name2id(m, OBJ.mjOBJ_GEOM, "heel_left")
    assert d.geom_xpos[hid][2] - m.geom_size[hid][0] > 0.02          # heel clearly lifted


def test_forefoot_bending_control(foot_rig):
    """Q: 발 앞부분의 굽힘을 제어할 수 있는가? (foot in the air)"""
    m = foot_rig.model
    d = mujoco.MjData(m)
    d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, OBJ.mjOBJ_JOINT, "shin_z")]] = 0.3
    m2 = m.__copy__() if hasattr(m, "__copy__") else m
    for target in (D(40), D(-20)):
        d.ctrl[_act(m2, "mtp_left")] = target
        g = m2.opt.gravity.copy()
        m2.opt.gravity[:] = 0
        _settle(m2, d, 1.0)
        m2.opt.gravity[:] = g
        assert abs(_q(m2, d, "mtp_left") - target) < D(3)


# ------------------------------------------------------------------- hand
@pytest.fixture(scope="module")
def hand():
    ed = Editor()
    ed.new("hand")
    ed.doc.section("option").set("gravity", "0 0 0")
    A.build_hand(ed, None, "right", pos=(0, 0, 0.5))
    assert ed.result.ok, ed.result.error
    return ed


def test_every_finger_joint_independent(hand):
    m = hand.model
    joints = [mujoco.mj_id2name(m, OBJ.mjOBJ_JOINT, j) for j in range(m.njnt)]
    assert len(joints) == 22
    for name in ("index", "middle", "ring", "little"):
        for j in ("mcp_flex", "mcp_abd", "pip", "dip"):
            assert f"{name}_{j}_right" in joints
    for j in ("cmc_rot", "cmc_abd", "mcp", "ip"):
        assert f"thumb_{j}_right" in joints
    for jn in joints:
        if jn.startswith("wrist"):
            continue
        d = mujoco.MjData(m)
        j = mujoco.mj_name2id(m, OBJ.mjOBJ_JOINT, jn)
        lo, hi = m.jnt_range[j]
        target = lo + 0.7 * (hi - lo)
        d.ctrl[_act(m, jn)] = target
        _settle(m, d, 0.6)
        assert abs(_q(m, d, jn) - target) < D(4), jn
        # the palm hangs on actuated wrist joints which take the reaction torque; the
        # independence requirement concerns the finger joints themselves
        others = [abs(d.qpos[m.jnt_qposadr[k]]) for k in range(m.njnt)
                  if k != j and not mujoco.mj_id2name(m, OBJ.mjOBJ_JOINT, k).startswith("wrist")]
        assert max(others) < D(2.5), (jn, max(others))


def test_finger_range_limit(hand):
    m = hand.model
    d = mujoco.MjData(m)
    d.ctrl[_act(m, "index_pip_right")] = D(170)  # beyond the 105 deg limit
    _settle(m, d, 1.0)
    assert _q(m, d, "index_pip_right") < D(108)


def test_thumb_base_rotation_and_abduction_differ(hand):
    m = hand.model
    tip = mujoco.mj_name2id(m, OBJ.mjOBJ_BODY, "thumb_distal_right")
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    p0 = d.xpos[tip].copy()
    deltas = []
    for jn in ("thumb_cmc_rot_right", "thumb_cmc_abd_right"):
        d2 = mujoco.MjData(m)
        d2.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, OBJ.mjOBJ_JOINT, jn)]] = D(30)
        mujoco.mj_forward(m, d2)
        deltas.append(d2.xpos[tip] - p0)
    c = deltas[0] @ deltas[1] / (np.linalg.norm(deltas[0]) * np.linalg.norm(deltas[1]))
    assert abs(c) < 0.8, c   # two genuinely different base motions


def test_left_right_hands_mirror():
    ed = Editor()
    ed.new("hands")
    A.build_hand(ed, None, "left", pos=(0, 0.3, 1.0))
    A.build_hand(ed, None, "right", pos=(0, -0.3, 1.0))
    m, d = ed.model, ed.data
    for jn in ("index_pip", "thumb_cmc_rot", "little_mcp_abd"):
        ed.set_qpos(f"{jn}_left", [D(40)])
        ed.set_qpos(f"{jn}_right", [D(40)])
    L = d.xpos[mujoco.mj_name2id(m, OBJ.mjOBJ_BODY, "thumb_distal_left")]
    R = d.xpos[mujoco.mj_name2id(m, OBJ.mjOBJ_BODY, "thumb_distal_right")]
    assert np.allclose(R, L * [1, -1, 1], atol=1e-6)


# ------------------------------------------------------------------- eyes / jaw
@pytest.fixture(scope="module")
def head():
    ed = Editor()
    ed.new("head")
    ed.doc.section("option").set("gravity", "0 0 0")
    A.build_head(ed, None, pos=(0, 0, 1.6), neck=False)
    assert ed.result.ok, ed.result.error
    return ed


def test_eyes_independent_with_direction_readout(head):
    m = head.model
    d = mujoco.MjData(m)
    d.ctrl[_act(m, "eye_left_yaw")] = D(20)
    d.ctrl[_act(m, "eye_right_pitch")] = D(-15)
    _settle(m, d, 0.5)
    gl, gr = _sens(m, d, "gaze_dir_left"), _sens(m, d, "gaze_dir_right")
    assert np.allclose(gl, [math.cos(D(20)), math.sin(D(20)), 0], atol=0.02)          # left looks left
    assert np.allclose(gr, [math.cos(D(-15)), 0, math.sin(D(-15))], atol=0.02)        # right looks down
    assert abs(_q(m, d, "eye_left_pitch")) < D(1) and abs(_q(m, d, "eye_right_yaw")) < D(1)
    # joint angle sensors
    assert _sens(m, d, "jointpos_eye_left_yaw")[0] == pytest.approx(D(20), abs=D(1))


def test_eye_range_limits(head):
    m = head.model
    d = mujoco.MjData(m)
    d.ctrl[_act(m, "eye_left_yaw")] = D(90)
    d.ctrl[_act(m, "eye_left_pitch")] = D(80)
    _settle(m, d, 0.5)
    assert _q(m, d, "eye_left_yaw") < D(47) and _q(m, d, "eye_left_pitch") < D(32)


def test_jaw_open_close_and_readout(head):
    m = head.model
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    chin = mujoco.mj_name2id(m, OBJ.mjOBJ_SITE, "chin")
    z0 = d.site_xpos[chin][2]
    d.ctrl[_act(m, "jaw_open")] = D(20)
    _settle(m, d, 0.5)
    assert d.site_xpos[chin][2] < z0 - 0.02
    assert _sens(m, d, "jointpos_jaw_open")[0] == pytest.approx(D(20), abs=D(1.5))
    d.ctrl[_act(m, "jaw_open")] = 0
    _settle(m, d, 0.5)
    assert abs(_q(m, d, "jaw_open")) < D(1.5)


def test_full_humanoid_structure_and_stability():
    from mjstudio.library.humanoid import make_humanoid_doc
    ed = make_humanoid_doc()
    m = ed.model
    names = {mujoco.mj_id2name(m, OBJ.mjOBJ_BODY, b) for b in range(m.nbody)}
    for b in ("pelvis", "abdomen", "chest", "head", "jaw", "eye_left", "eye_right", "shoulder_left",
              "upper_arm_left", "forearm_left", "hand_left", "thigh_left", "shank_left", "talus_left",
              "hindfoot_left", "midfoot_left", "forefoot_left", "toes_left", "index_distal_right",
              "thumb_distal_right"):
        assert b in names, b
    assert m.nskin == 1 and m.nkey == 1
    assert 70 < m.body_subtreemass[0] < 80
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    for _ in range(1000):                       # 2 s
        mujoco.mj_step(m, d)
    assert np.all(np.isfinite(d.qpos)) and sum(int(w.number) for w in d.warning) == 0
    feet = [mujoco.mj_name2id(m, OBJ.mjOBJ_SENSOR, f"touch_{z}_{s}") for z in ("heel_zone", "forefoot_zone")
            for s in ("left", "right")]
    assert all(d.sensordata[m.sensor_adr[i]] > 0 for i in feet)   # standing on heels and forefeet
