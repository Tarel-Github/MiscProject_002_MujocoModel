"""Final evaluation scenario, end to end (headless):

 "Load an existing MuJoCo humanoid, modify a specific finger segment, add a new joint at the
  front of the foot, import and assemble a jaw/face mesh made in Blender, rotate both eyes
  separately and move the jaw, control expressions (smile, blink) at the same time, save the
  modified model, reload it and simulate normally."

    python examples/final_scenario.py      -> examples/output/final_scenario/
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from mjstudio.core import ops  # noqa: E402
from mjstudio.core.editor import Editor  # noqa: E402
from mjstudio.core.expr import FaceController, Track  # noqa: E402
from mjstudio.core.faceops import attach_face_rig, rig_from_gltf  # noqa: E402
from mjstudio.core.project import Project  # noqa: E402
from mjstudio.library.humanoid import make_humanoid_doc  # noqa: E402

D = math.radians
OUT = HERE / "output" / "final_scenario"
GLB = HERE / "assets" / "face_blender.glb"
EYES = {"left": (D(20), D(5)), "right": (D(-12), D(-8))}


def drive(m, d, rig, seconds: float) -> dict:
    """Simulate with eye/jaw commands and expression tracks evaluated on data.time."""
    fc = FaceController(m, d, rig)
    for side, (yaw, pitch) in EYES.items():
        fc.set_eye_rotation(side, yaw, pitch)
    fc.set_jaw_open(0.6)
    log = {"t": [], "smile": [], "blink_left": []}
    issues = 0
    for _ in range(int(seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)
        rig.evaluate_tracks(d.time)
        log["t"].append(d.time)
        log["smile"].append(rig.get("smile"))
        log["blink_left"].append(rig.get("blink_left"))
        if not np.all(np.isfinite(d.qpos)):
            issues += 1
            break
    return {"fc": fc, "log": log, "issues": issues + sum(int(w.number) for w in d.warning)}


def run(out: Path = OUT, verbose: bool = True) -> dict:
    say = print if verbose else (lambda *a, **k: None)
    out.mkdir(parents=True, exist_ok=True)
    # 0. an existing humanoid MJCF (built once with the library, then treated as a plain file)
    src = out / "existing_humanoid.xml"
    make_humanoid_doc(face=False).save(src)
    # 1. load it
    ed = Editor()
    ed.open(src)
    m0 = ed.model
    say(f"1. loaded {src.name}: {m0.njnt} joints, {m0.nu} actuators")
    # 2. modify a specific finger segment (left index PIP range, middle phalanx length +10 %)
    pip = ed.doc.find_named("joint", "index_pip_left")
    ops.set_joint_range_deg(ed, pip, -5, 95)
    dist = ed.doc.find_named("body", "index_distal_left")
    p, q = ed.doc.local_pose(dist)
    ed.set_pose(dist, p * 1.10, q)
    say("2. index finger PIP range -> -5..95 deg, middle phalanx +10 %")
    # 3. new joint at the front of the left foot (hallux interphalangeal joint)
    with ed.batch("hallux IP"):
        hal = ops.add_body(ed, ed.doc.find_named("body", "toes_left"), "hallux_distal_left", pos=(0.052, -0.022, 0))
        ops.add_joint(ed, hal, "hinge", "hallux_ip_left", axis=(0, -1, 0), range_rad=(D(-10), D(60)),
                      damping="0.05", armature="0.0005")
        ops.add_geom(ed, hal, "capsule", size="0.0095", fromto="0 0 -0.064 0.024 0 -0.064", mass="0.02")
    ops.add_actuator(ed, "hallux_ip_left", "position", kp="1")
    say(f"3. forefoot joint added: hallux_ip_left (model now {ed.model.njnt} joints)")
    # 4. face + jaw mesh made in Blender (shape keys, jaw bone weights) assembled on head/jaw
    rig = rig_from_gltf(GLB, "face", head_body="head", jaw_body="jaw")
    attach_face_rig(ed, rig)
    for g in ("skull", "mandible"):
        ed.set_attr(ed.doc.find_named("geom", g), "group", "3")      # the skin replaces these visuals
    say(f"4. Blender face assembled: {len(rig.V0)} vertices, {len(rig.keys)} shape keys, "
        f"{int((rig.jaw_w > 0.5).sum())} vertices follow the jaw")
    assert ed.result.ok, ed.result.error
    # 5./6. eyes separately + jaw + expressions at the same time (simulation time tracks)
    rig.tracks = [Track("smile", [0.0, 0.5, 2.0], [0.0, 0.8, 0.8]),
                  Track("blink_left", [0.0, 0.6, 0.7, 0.8, 2.0], [0.0, 0.0, 1.0, 0.0, 0.0])]
    m, d = ed.model, ed.data
    mujoco.mj_resetDataKeyframe(m, d, 0)
    r1 = drive(m, d, rig, 1.0)
    fc = r1["fc"]
    eyes1 = {s: fc.eye_rotation(s) for s in EYES}
    jaw1 = fc.jaw_open()
    say(f"5. eyes: left {np.degrees(eyes1['left']).round(1)} deg, right {np.degrees(eyes1['right']).round(1)} deg;"
        f" jaw {math.degrees(jaw1):.1f} deg")
    i07 = int(np.argmin(np.abs(np.array(r1['log']['t']) - 0.7)))
    say(f"6. smile at t=1.0: {r1['log']['smile'][-1]:.2f}; blink_left at t=0.70: {r1['log']['blink_left'][i07]:.2f}")
    W1 = rig.world_vertices(m, d)
    # 7. save as project, reload with a fresh editor, simulate again
    prj = Project(out / "project")
    for n in prj.save(ed.doc, rigs=ed.rigs):
        say("   ", n)
    prj2, doc2 = Project.load(out / "project")
    ed2 = Editor()
    ed2.set_doc(doc2, prj2.load_rigs())
    assert ed2.result.ok, ed2.result.error
    m2, d2 = ed2.model, ed2.data
    rig2 = ed2.rigs["face"]
    mujoco.mj_resetDataKeyframe(m2, d2, 0)
    r2 = drive(m2, d2, rig2, 1.0)
    eyes2 = {s: r2["fc"].eye_rotation(s) for s in EYES}
    W2 = rig2.world_vertices(m2, d2)
    same = (max(abs(a - b) for s in EYES for a, b in zip(eyes1[s], eyes2[s])) < 1e-9
            and abs(jaw1 - r2["fc"].jaw_open()) < 1e-9 and np.abs(W1 - W2).max() < 1e-9)
    plain = mujoco.MjModel.from_xml_path(str(prj2.model_path))
    say(f"7. reloaded project: {m2.njnt} joints, skins {m2.nskin}; physics issues {r1['issues']}/{r2['issues']};"
        f" identical replay: {same}; plain MuJoCo loads it: nq={plain.nq}")
    return {"eyes": eyes1, "jaw": jaw1, "r1": r1, "r2": r2, "same": same, "ed": ed, "ed2": ed2,
            "issues": r1["issues"] + r2["issues"], "project": prj2}


if __name__ == "__main__":
    run()
