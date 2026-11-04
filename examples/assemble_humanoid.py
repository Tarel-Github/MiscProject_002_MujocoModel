"""End-to-end assembly scenario on the *official* MuJoCo humanoid (Apache-2.0).

 1. load examples/models/humanoid.xml (single-sphere hands, two-capsule feet)
 2. replace both hands with 5-finger hands, both feet with segmented feet and the
    head with a head that has independently actuated eyes and a hinged jaw
 3. edit a specific finger segment (left index PIP range + middle phalanx length)
 4. add a new joint at the front of the left foot (hallux interphalangeal joint)
 5. define joint groups (fists, conjugate gaze) and save everything as a project
 6. reload the project with plain MuJoCo and simulate

    python examples/assemble_humanoid.py   -> examples/output/humanoid_assembled/
"""
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from mjstudio.core import mathx as mx  # noqa: E402
from mjstudio.core import ops, parts  # noqa: E402
from mjstudio.core.editor import Editor  # noqa: E402
from mjstudio.core.project import JointGroup, Project  # noqa: E402
from mjstudio.library import anatomy as A  # noqa: E402

D = math.radians
OUT = HERE / "output" / "humanoid_assembled"


def main() -> Path:
    ed = Editor()
    ed.open(HERE / "models" / "humanoid.xml")
    m0 = ed.model
    print(f"original: joints={m0.njnt} actuators={m0.nu} keyframes={m0.nkey}")

    with ed.batch("assemble anatomy"):
        for side, sgn in (("left", -1), ("right", 1)):
            # hands: fingers (-z in the part) aligned with the forearm direction
            old = ed.doc.find_named("body", f"hand_{side}")
            part = A.make_part(A.build_hand, f"hand_{side}", side=side)
            new = parts.replace_body(ed, old, part, prefix=f"{side}_")[0]
            fwd = np.array([1.0, sgn * 1.0, 1.0]) / math.sqrt(3)
            ed.set_pose(new, [0.16, sgn * 0.16, 0.16], mx.zaxis2q(-fwd))
            # feet: keep the original sole height (-0.417 in the shin frame)
            old = ed.doc.find_named("body", f"foot_{side}")
            part = A.make_part(A.build_foot, f"foot_{side}", side=side)
            new = parts.replace_body(ed, old, part, prefix=f"{side}_")[0]
            ed.set_pose(new, [0.0, 0.0, -0.342], mx.QID)
        part = A.make_part(A.build_head, "head", neck=True)
        parts.replace_body(ed, ed.doc.find_named("body", "head"), part)
    assert ed.result.ok, ed.result.error

    # 3. modify a specific finger segment: left index PIP range and middle phalanx length
    pip = ed.doc.find_named("joint", "index_pip_left")
    ops.set_joint_range_deg(ed, pip, -5, 95) if ed.doc.angle_unit == "radian" else \
        ed.set_attr(pip, "range", "-5 95")
    dist = ed.doc.find_named("body", "index_distal_left")
    p, q = ed.doc.local_pose(dist)
    ed.set_pose(dist, p * 1.1, q)                                  # 10 % longer middle phalanx
    # 4. new joint at the front of the left foot: hallux interphalangeal joint
    toes = ed.doc.find_named("body", "toes_left")
    with ed.batch("add hallux IP joint"):
        hal = ops.add_body(ed, toes, "hallux_distal_left", pos=(0.052, -0.022, 0.0))
        ops.add_joint(ed, hal, "hinge", "hallux_ip_left", axis=(0, -1, 0), range_rad=(D(-10), D(60)),
                      damping="0.05", armature="0.0005")
        ops.add_geom(ed, hal, "capsule", size="0.0095", fromto="0 0 -0.064 0.024 0 -0.064", mass="0.02",
                     rgba=A.SKIN_RGBA)
    ops.add_actuator(ed, "hallux_ip_left", "position", kp="1")
    ops.add_sensor(ed, "jointpos", joint="hallux_ip_left")
    assert ed.result.ok, ed.result.error
    m = ed.model
    print(f"assembled: joints={m.njnt} actuators={m.nu} sensors={m.nsensor} keyframes={m.nkey} (qpos remapped)")

    # 5. joint groups (editor level) + save as a project folder
    prj = Project(OUT)
    for side in ("left", "right"):
        prj.set_group(JointGroup(f"fist_{side}", {f"{f}_{j}_{side}": w for f in ("index", "middle", "ring", "little")
                                                  for j, w in (("mcp_flex", 1.0), ("pip", 1.15), ("dip", 0.8))},
                                 "curl all four fingers"))
    prj.set_group(JointGroup("gaze_yaw", {"eye_left_yaw": 1.0, "eye_right_yaw": 1.0}, "conjugate horizontal gaze"))
    prj.set_group(JointGroup("gaze_pitch", {"eye_left_pitch": 1.0, "eye_right_pitch": 1.0}, "conjugate vertical gaze"))
    for n in prj.save(ed.doc):
        print(" ", n)

    # 6. reload with plain MuJoCo and simulate
    m2 = mujoco.MjModel.from_xml_path(str(prj.model_path))
    d2 = mujoco.MjData(m2)
    mujoco.mj_resetDataKeyframe(m2, d2, 0)
    for _ in range(500):
        mujoco.mj_step(m2, d2)
    assert np.all(np.isfinite(d2.qpos))
    print(f"reloaded {prj.model_path.name}: nq={m2.nq}, simulated {d2.time:.2f}s OK")
    return OUT


if __name__ == "__main__":
    main()
