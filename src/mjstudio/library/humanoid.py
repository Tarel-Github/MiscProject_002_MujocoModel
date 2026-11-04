"""Full-body humanoid assembled from the anatomical parts (built with editor operations).

Segment lengths and mass fractions follow de Leva (1996, male, adjusted Zatsiorsky
data) scaled to the requested stature/mass. Standing pose = qpos0 (anatomical
position: arms at the sides, palms forward). Joint sign conventions: flexion,
abduction (away from the midline) and the library part conventions are positive.

Chain: pelvis (free) - abdomen (lumbar 3 DOF) - chest (thoracic 3 DOF) - head
(neck 3 DOF, eyes 2x2, jaw, optional face) ; chest - shoulder girdle (2 DOF) -
upper arm (glenohumeral 3 DOF) - forearm (elbow + pronation/supination) - hand
(wrist 2 DOF + fingers) ; pelvis - thigh (hip 3 DOF) - shank (knee) - foot.
"""
from __future__ import annotations

import math

import numpy as np

from ..core import mathx as mx
from ..core import ops
from ..core.editor import Editor
from ..core.undo import Insert
from . import anatomy as A

D = math.radians
# de Leva 1996 (male): mass fraction, length at stature 1.741 m
SEG = {"pelvis": (0.1117, 0.1457), "abdomen": (0.1633, 0.2155), "chest": (0.1596, 0.1707),
       "upper_arm": (0.0271, 0.2817), "forearm": (0.0162, 0.2689), "thigh": (0.1416, 0.4222),
       "shank": (0.0433, 0.4340)}


def build_humanoid(ed: Editor, stature: float = 1.75, mass: float = 75.0, face: bool = True,
                   toes: str = "single", name_prefix: str = "") -> dict:
    """Insert a full humanoid under the world body. Returns {name: body element}."""
    k = stature / 1.741
    L = {n: v[1] * k for n, v in SEG.items()}
    Mx = {n: v[0] * mass for n, v in SEG.items()}
    hip_y, ankle_h = 0.088 * k, 0.075 * k
    hip_z = L["thigh"] + L["shank"] + ankle_h + 0.002
    out: dict = {}
    sk = A.SKIN_RGBA

    def hinge(body, name, axis, lo, hi, damping, armature=0.02, **kw):
        return A._hinge(ed, body, name, axis, lo, hi, damping=damping, armature=armature, **kw)

    with ed.structural("Build humanoid"):
        pelvis = ops.add_body(ed, None, f"{name_prefix}pelvis", pos=(0, 0, hip_z))
        ops.add_joint(ed, pelvis, "free", f"{name_prefix}root")
        ops.add_geom(ed, pelvis, "capsule", "pelvis", size=f"{0.085 * k:.4g}",
                     fromto=mx.fmt_vec([-0.01, -0.07 * k, 0.03 * k, -0.01, 0.07 * k, 0.03 * k]), mass=f"{Mx['pelvis']:.4g}",
                     rgba=sk)
        ops.add_site(ed, pelvis, "pelvis_imu", pos=(0, 0, 0.05 * k), size=0.006)
        out["pelvis"] = pelvis
        abd = ops.add_body(ed, pelvis, "abdomen", pos=(0, 0, 0.095 * k))
        hinge(abd, "lumbar_flex", (0, 1, 0), -25, 60, 8)
        hinge(abd, "lumbar_lat", (1, 0, 0), -25, 25, 8)
        hinge(abd, "lumbar_rot", (0, 0, 1), -20, 20, 8)
        ops.add_geom(ed, abd, "capsule", "abdomen", size=f"{0.095 * k:.4g}",
                     fromto=mx.fmt_vec([0, -0.045 * k, 0.1 * k, 0, 0.045 * k, 0.1 * k]), mass=f"{Mx['abdomen']:.4g}",
                     rgba=sk)
        out["abdomen"] = abd
        chest = ops.add_body(ed, abd, "chest", pos=(0, 0, L["abdomen"]))
        hinge(chest, "thoracic_flex", (0, 1, 0), -20, 40, 8)
        hinge(chest, "thoracic_lat", (1, 0, 0), -20, 20, 8)
        hinge(chest, "thoracic_rot", (0, 0, 1), -30, 30, 8)
        ops.add_geom(ed, chest, "ellipsoid", "ribcage", size=mx.fmt_vec(np.array([0.1, 0.13, 0.15]) * k),
                     pos=(0.0, 0, 0.12 * k), mass=f"{Mx['chest']:.4g}", rgba=sk)
        ops.add_site(ed, chest, "chest_imu", pos=(0.05 * k, 0, 0.15 * k), size=0.006)
        out["chest"] = chest
        # head (neck joints are part of the head part); C7 ~ chest origin + 0.27
        head = A.build_head(ed, chest, pos=(0.01 * k, 0, 0.36 * k), scale=k, face=face, attach_face=False)
        ops.add_geom(ed, chest, "capsule", "neck", size=f"{0.048 * k:.4g}",
                     fromto=mx.fmt_vec(np.array([0.005, 0, 0.22, 0.01, 0, 0.29]) * k), rgba=sk, mass="0",
                     contype="0", conaffinity="0")
        out["head"] = head
        for side in ("left", "right"):
            S = A.Side(side)
            gird = ops.add_body(ed, chest, S.n("shoulder"), pos=S.p(np.array([0.0, 0.03, 0.23]) * k))
            hinge(gird, S.n("shoulder_elev"), S.ax((1, 0, 0)), -10, 30, 6)
            hinge(gird, S.n("shoulder_prot"), S.ax((0, 0, 1)), -20, 20, 6)
            ops.add_geom(ed, gird, "capsule", size=f"{0.035 * k:.4g}",
                         fromto=S.ft(np.array([0.0, 0.0, 0.0]) * k, np.array([0.0, 0.15, 0.0]) * k), mass=f"{0.9 * k:.4g}",
                         rgba=sk)
            ua = ops.add_body(ed, gird, S.n("upper_arm"), pos=S.p(np.array([0.0, 0.17, -0.02]) * k))
            hinge(ua, S.n("shoulder_flex"), S.ax((0, -1, 0)), -50, 170, 3)
            hinge(ua, S.n("shoulder_abd"), S.ax((1, 0, 0)), -20, 160, 3)
            hinge(ua, S.n("shoulder_rot"), S.ax((0, 0, 1)), -80, 80, 3)
            ops.add_geom(ed, ua, "capsule", S.n("upper_arm"), size=f"{0.045 * k:.4g}",
                         fromto=S.ft((0, 0, -0.02 * k), (0, 0, -L["upper_arm"] + 0.03 * k)), mass=f"{Mx['upper_arm']:.4g}",
                         rgba=sk)
            fa = ops.add_body(ed, ua, S.n("forearm"), pos=S.p((0, 0, -L["upper_arm"])))
            hinge(fa, S.n("elbow"), S.ax((0, -1, 0)), -5, 150, 2)
            hinge(fa, S.n("pronation"), S.ax((0, 0, 1)), -85, 85, 1)
            ops.add_geom(ed, fa, "capsule", S.n("forearm"), size=f"{0.037 * k:.4g}",
                         fromto=S.ft((0, 0, -0.02 * k), (0, 0, -L["forearm"] + 0.02 * k)), mass=f"{Mx['forearm']:.4g}",
                         rgba=sk)
            A.build_hand(ed, fa, side, pos=S.p((0, 0, -L["forearm"])), scale=k)
            th = ops.add_body(ed, pelvis, S.n("thigh"), pos=S.p((0, hip_y, 0)))
            hinge(th, S.n("hip_flex"), S.ax((0, -1, 0)), -20, 120, 6)
            hinge(th, S.n("hip_abd"), S.ax((1, 0, 0)), -25, 45, 6)
            hinge(th, S.n("hip_rot"), S.ax((0, 0, 1)), -40, 40, 6)
            ops.add_geom(ed, th, "capsule", S.n("thigh"), size=f"{0.07 * k:.4g}",
                         fromto=S.ft((0, 0, -0.04 * k), (0, 0, -L["thigh"] + 0.05 * k)), mass=f"{Mx['thigh']:.4g}", rgba=sk)
            sh = ops.add_body(ed, th, S.n("shank"), pos=S.p((0, 0, -L["thigh"])))
            hinge(sh, S.n("knee"), S.ax((0, 1, 0)), -5, 150, 6)
            ops.add_geom(ed, sh, "capsule", S.n("shank"), size=f"{0.05 * k:.4g}",
                         fromto=S.ft((0, 0, -0.03 * k), (0, 0, -L["shank"] + 0.06 * k)), mass=f"{Mx['shank']:.4g}", rgba=sk)
            A.build_foot(ed, sh, side, pos=S.p((0, 0, -L["shank"])), scale=k, toes=toes, kp_ankle=300.0)
            for nm, kp in (("shoulder_elev", 150), ("shoulder_prot", 150), ("shoulder_flex", 150), ("shoulder_abd", 150),
                           ("shoulder_rot", 80), ("elbow", 80), ("pronation", 20), ("hip_flex", 500), ("hip_abd", 500),
                           ("hip_rot", 200), ("knee", 500)):
                ops.add_actuator(ed, S.n(nm), "position", kp=str(kp))
            for nm in ("shoulder_flex", "shoulder_abd", "elbow", "hip_flex", "knee"):
                ops.add_sensor(ed, "jointpos", joint=S.n(nm))
            out[S.n("thigh")], out[S.n("upper_arm")] = th, ua
        for nm, kp in (("lumbar_flex", 600), ("lumbar_lat", 600), ("lumbar_rot", 400), ("thoracic_flex", 500),
                       ("thoracic_lat", 500), ("thoracic_rot", 300)):
            ops.add_actuator(ed, nm, "position", kp=str(kp))
        for site in ("pelvis_imu", "chest_imu"):
            ops.add_sensor(ed, "gyro", name=f"{site}_gyro", site=site)
            ops.add_sensor(ed, "accelerometer", name=f"{site}_accel", site=site)
        ops.add_sensor(ed, "framequat", name="pelvis_orientation", objtype="site", objname="pelvis_imu")
        # contacts between non-adjacent but touching segments
        pairs = [("chest", "upper_arm_left"), ("chest", "upper_arm_right"), ("abdomen", "thigh_left"),
                 ("abdomen", "thigh_right"), ("abdomen", "upper_arm_left"), ("abdomen", "upper_arm_right"),
                 ("pelvis", "forearm_left"), ("pelvis", "forearm_right"), ("thigh_left", "thigh_right")]
        sec = ed.doc.section("contact")
        if sec is None:
            sec = ops.make("contact")
            ed.push(Insert(ed.doc.root, ops._section_index(ed, "contact"), sec, "Add contact section"))
        for a, b in pairs:
            ed.push(Insert(sec, len(sec), ops.make("exclude", body1=a, body2=b), "exclude contact"))
    if face:
        from ..core.faceops import attach_face_rig
        from .face import build_face_rig
        attach_face_rig(ed, build_face_rig("face", k, "head", "jaw"))
    ops.add_keyframe(ed, "stand")
    return out


def make_humanoid_doc(**kw) -> Editor:
    ed = Editor()
    ed.new("humanoid")
    ed.doc.section("option").set("timestep", "0.002")
    build_humanoid(ed, **kw)
    return ed
