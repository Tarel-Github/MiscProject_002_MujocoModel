"""Expression system: skin binding, shape keys + bones, independence, sync, persistence."""
import math

import mujoco
import numpy as np
import pytest

from mjstudio.core.editor import Editor
from mjstudio.core.expr import FaceController, Track
from mjstudio.core.project import Project
from mjstudio.library import anatomy as A

D = math.radians
OBJ = mujoco.mjtObj


@pytest.fixture(scope="module")
def head():
    ed = Editor()
    ed.new("face")
    ed.doc.section("option").set("gravity", "0 0 0")
    A.build_head(ed, None, pos=(0, 0, 1.6), neck=False, face=True)
    assert ed.result.ok, ed.result.error
    return ed


def _scene_skin(m, d):
    scn = mujoco.MjvScene(m, 2000)
    opt, cam, pert = mujoco.MjvOption(), mujoco.MjvCamera(), mujoco.MjvPerturb()
    mujoco.mjv_updateScene(m, d, opt, pert, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
    return scn


def test_skin_binding_matches_mujoco_skinning(head):
    """Neutral rig deformation == MuJoCo's own skinning of the .skn (head + jaw bones)."""
    m, d = head.model, head.data
    rig = head.rigs["face"]
    assert m.nskin == 1 and m.skin_vertnum[0] == len(rig.V0)
    rig.reset()
    for jaw in (0.0, D(25)):
        head.set_qpos("jaw_open", [jaw])
        scn = _scene_skin(m, d)
        a, n = scn.skinvertadr[0], scn.skinvertnum[0]
        mj = scn.skinvert[3 * a:3 * (a + n)].reshape(-1, 3)
        ours = rig.world_vertices(m, d)
        assert np.abs(mj - ours).max() < 2e-5, jaw     # float32 precision


def test_jaw_opens_mouth_slit(head):
    m, d = head.model, head.data
    rig = head.rigs["face"]
    rig.reset()
    head.set_qpos("jaw_open", [0.0])
    closed = rig.world_vertices(m, d)
    head.set_qpos("jaw_open", [D(20)])
    opened = rig.world_vertices(m, d)
    lower = rig.jaw_w > 0.99
    assert (closed[lower] - opened[lower])[:, 2].max() > 0.01      # lower lip/chin dropped > 1 cm
    upper = rig.jaw_w == 0
    assert np.abs(closed[upper] - opened[upper]).max() < 1e-9       # rest of the face fixed
    head.set_qpos("jaw_open", [0.0])


def test_smile_with_closed_jaw_and_independent_controls(head):
    rig = head.rigs["face"]
    rig.reset()
    base = rig.local_vertices()
    fc = FaceController(head.model, head.data, rig)
    fc.set_jaw_open(0.0)
    fc.set_expression("smile", 0.8)
    smile = rig.local_vertices() - base
    assert np.linalg.norm(smile, axis=1).max() > 0.003              # visible deformation
    rig.reset()
    rig.set("brow_raise_left", 1.0)
    brow = np.linalg.norm(rig.local_vertices() - base, axis=1)
    rig.reset()
    rig.set("mouth_smile_left", 1.0)
    corner = np.linalg.norm(rig.local_vertices() - base, axis=1)
    # brows and mouth corners move disjoint vertex sets (independent controls)
    assert ((brow > 1e-4) & (corner > 1e-4)).sum() == 0
    rig.reset()


def test_left_right_blink_separately_and_eye_motion_together(head):
    m, d = head.model, head.data
    rig = head.rigs["face"]
    fc = FaceController(m, d, rig)
    rig.reset()
    base = rig.local_vertices()
    fc.set_expression("blink_left", 1.0)
    mv = np.linalg.norm(rig.local_vertices() - base, axis=1) > 1e-4
    y = rig.V0[:, 1]
    assert mv.any() and (y[mv] > 0).all()                       # only the left side (y > 0) moves
    rig.reset()
    fc.set_expression("blink_right", 1.0)
    mv = np.linalg.norm(rig.local_vertices() - base, axis=1) > 1e-4
    assert mv.any() and (y[mv] < 0).all()
    # eyes keep rotating while blinking (eye joints are independent of the face mesh)
    fc.set_expression("blink_left", 1.0)
    for _ in range(300):
        fc.set_eye_rotation("left", D(25), D(10))
        mujoco.mj_step(m, d)
    yaw, pitch = fc.eye_rotation("left")
    assert yaw == pytest.approx(D(25), abs=D(2)) and pitch == pytest.approx(D(10), abs=D(2))
    gaze = fc.eye_direction("left")
    assert gaze[1] > 0.3                                        # looking to the left
    rig.reset()


def test_presets_and_blink_closes_eye(head):
    rig = head.rigs["face"]
    for p in ("neutral", "smile", "frown", "surprise", "blink", "wink_left", "wink_right"):
        rig.apply_preset(p)
    rig.apply_preset("blink")
    V = rig.local_vertices()
    # upper-lid vertices moved down onto the eye: their lowest point is below the eye centre
    up = rig.bones["lid_upper_L"].weights > 0.9
    assert V[up][:, 2].min() < 0.03 - 0.004
    rig.reset()


def test_tracks_follow_simulation_time(head):
    m = head.model
    d = mujoco.MjData(m)
    rig = head.rigs["face"]
    rig.reset()
    rig.tracks = [Track("blink_left", [0.0, 0.2, 0.4], [0.0, 1.0, 0.0])]
    vals = []
    for _ in range(int(0.4 / m.opt.timestep)):
        mujoco.mj_step(m, d)
        rig.evaluate_tracks(d.time)
        vals.append((d.time, rig.get("blink_left")))
    t, v = max(vals, key=lambda x: x[1])
    assert abs(t - 0.2) < 2 * m.opt.timestep and v > 0.99
    rig.tracks = []
    rig.reset()


def test_project_roundtrip_reproduces_expression(head, tmp_path):
    rig = head.rigs["face"]
    rig.apply_preset("surprise")
    rig.set("mouth_smile_right", 0.4)
    before = rig.local_vertices()
    prj = Project(tmp_path / "p")
    prj.save(head.doc, rigs=head.rigs)
    prj2, doc = Project.load(tmp_path / "p")
    rigs = prj2.load_rigs()
    ed2 = Editor()
    ed2.set_doc(doc, rigs)
    assert ed2.result.ok, ed2.result.error
    r2 = ed2.rigs["face"]
    assert np.allclose(r2.local_vertices(), before, atol=1e-9)
    assert r2.values == pytest.approx(rig.values)
    # plain MuJoCo loads the saved model with its skin
    m = mujoco.MjModel.from_xml_path(str(prj2.model_path))
    assert m.nskin == 1
    rig.reset()


def test_sensor_bus_groups_and_custom_channels(head):
    from mjstudio.core.sensing import SensorBus
    m, d = head.model, head.data
    rig = head.rigs["face"]
    fc = FaceController(m, d, rig)
    bus = SensorBus(m)
    bus.register("expression", lambda m_, d_: [rig.get(p) for p in sorted(rig.params)], group="expression")
    bus.register("gaze_left", lambda m_, d_: fc.eye_direction("left"), group="vision")
    rig.set("smile", 0.5)
    obs = bus.read_groups(d)
    assert {"proprio_pos", "vestibular", "pose", "expression", "vision"} <= set(obs)
    spec = bus.spec(d)
    g, off, dim, kind = spec["jointpos_jaw_open"]
    assert g == "proprio_pos" and dim == 1 and kind == "jointpos"
    assert obs["expression"][sorted(rig.params).index("smile")] == 0.5
    assert np.isclose(np.linalg.norm(obs["vision"]), 1.0)
    rig.reset()
