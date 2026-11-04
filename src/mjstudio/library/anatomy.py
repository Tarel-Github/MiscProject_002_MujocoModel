"""Parametric anatomical parts built with the generic editor operations.

Conventions (MuJoCo humanoid convention): x forward, y left, z up.
Left-side parts are defined explicitly; right-side parts are their mirror image
(positions y -> -y, hinge axes transformed as axial vectors), so a positive joint
angle means the same anatomical motion on both sides.

Dimensions are approximate adult values (≈1.75 m male) and can be scaled.
Joint ranges are typical active ranges of motion; the module docstrings of each
builder state where the model deliberately simplifies anatomy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..core import mathx as mx
from ..core import ops
from ..core.doc import Element
from ..core.editor import Editor

D = math.radians
SKIN_RGBA = "0.85 0.67 0.55 1"
# Hand geoms collide with the world/objects (conaffinity 1) but not with each other:
# (2 & 1) | (2 & 1) == 0 for hand-hand pairs, (1 & 1) == 1 for hand-floor/object pairs.
HAND = {"contype": "2", "conaffinity": "1"}
NAIL_RGBA = "0.92 0.78 0.72 1"


class Side:
    def __init__(self, side: str):
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        self.side = side
        self.s = 1.0 if side == "left" else -1.0
        self.tag = "L" if side == "left" else "R"

    def p(self, v):
        v = np.asarray(v, float)
        return np.array([v[0], self.s * v[1], v[2]])

    def ax(self, a):
        """Hinge axis (axial vector) mirrored so equal angles give mirrored motion."""
        a = np.asarray(a, float)
        return a if self.s > 0 else np.array([-a[0], a[1], -a[2]])

    def q(self, q):
        q = np.asarray(q, float)
        return q if self.s > 0 else np.array([q[0], -q[1], q[2], -q[3]])

    def ft(self, a, b):
        return mx.fmt_vec(np.concatenate([self.p(a), self.p(b)]))

    def n(self, base: str) -> str:
        return f"{base}_{self.side}"


def _hinge(ed, body, name, axis, lo_deg, hi_deg, pos=(0, 0, 0), damping=0.05, armature=0.0005, **kw):
    return ops.add_joint(ed, body, "hinge", name, axis=axis, pos=pos, range_rad=(D(lo_deg), D(hi_deg)),
                         damping=damping, armature=armature, **kw)


def _touch(ed, body, name, pos, size, sensors=True, quat=None):
    s = ops.add_site(ed, body, name, pos=pos, stype="box", size=mx.fmt_vec(size), rgba="0.2 0.8 1 0.25",
                     group="4", quat=None if quat is None else mx.fmt_vec(quat))
    if sensors:
        ops.add_sensor(ed, "touch", name=f"touch_{name}", site=name)
    return s


# ======================================================================== hand
@dataclass
class FingerSpec:
    name: str
    mcp: tuple[float, float, float]      # MCP joint centre in palm frame (left hand)
    lengths: tuple[float, float, float]  # proximal, middle, distal phalanx (m)
    radius: float
    abd: tuple[float, float]             # abduction range (deg); +: away from the middle finger
    abd_axis_sign: float                 # +1 index side (radial), -1 ulnar side


# Palm frame (left hand, anatomical position): origin at the wrist joint centre,
# fingers point to -z, palm faces +x, thumb side (radial) is +y.
# Phalanx lengths after Alexander & Viktor (2010) / Buryanov & Kotiuk (2010) adult averages.
FINGERS = [
    FingerSpec("index", (0.006, 0.025, -0.092), (0.040, 0.023, 0.019), 0.0090, (-15, 20), 1.0),
    FingerSpec("middle", (0.006, 0.006, -0.095), (0.045, 0.027, 0.020), 0.0092, (-15, 15), 1.0),
    FingerSpec("ring", (0.006, -0.012, -0.091), (0.042, 0.026, 0.019), 0.0087, (-15, 20), -1.0),
    FingerSpec("little", (0.004, -0.029, -0.083), (0.033, 0.019, 0.017), 0.0078, (-15, 25), -1.0),
]


def build_finger(ed: Editor, palm: Element, spec: FingerSpec, side: str = "left", scale: float = 1.0,
                 sensors: bool = True, actuators: bool = True) -> Element:
    """Index/middle/ring/little finger: MCP (flexion + ab/adduction, 2 DOF), PIP and DIP
    flexion (1 DOF each) and three phalanges. Flexion axis is -y so +angle = flexion."""
    S = Side(side)
    nm = lambda b: S.n(f"{spec.name}_{b}")  # noqa: E731
    L = [x * scale for x in spec.lengths]
    r = spec.radius * scale
    flex = S.ax((0, -1, 0))
    abd = S.ax((spec.abd_axis_sign, 0, 0))
    prox = ops.add_body(ed, palm, nm("proximal"), pos=S.p(np.array(spec.mcp) * scale))
    _hinge(ed, prox, nm("mcp_flex"), flex, -20, 90)
    _hinge(ed, prox, nm("mcp_abd"), abd, *spec.abd)
    ops.add_geom(ed, prox, "capsule", size=mx.fmt_num(r), fromto=S.ft((0, 0, -0.004), (0, 0, -L[0] + 0.004)),
                 rgba=SKIN_RGBA, density="1100", **HAND)
    mid = ops.add_body(ed, prox, nm("middle"), pos=S.p((0, 0, -L[0])))
    _hinge(ed, mid, nm("pip"), flex, -5, 105)
    ops.add_geom(ed, mid, "capsule", size=mx.fmt_num(r * 0.92), fromto=S.ft((0, 0, -0.003), (0, 0, -L[1] + 0.003)),
                 rgba=SKIN_RGBA, density="1100", **HAND)
    dist = ops.add_body(ed, mid, nm("distal"), pos=S.p((0, 0, -L[1])))
    _hinge(ed, dist, nm("dip"), flex, -5, 85)
    ops.add_geom(ed, dist, "capsule", size=mx.fmt_num(r * 0.85), fromto=S.ft((0, 0, -0.003), (0, 0, -L[2] + r * 0.85)),
                 rgba=SKIN_RGBA, density="1100", **HAND)
    ops.add_geom(ed, dist, "box", size=mx.fmt_vec([r * 0.15, r * 0.6, L[2] * 0.3]), pos=S.p((-r * 0.75, 0, -L[2] * 0.62)),
                 rgba=NAIL_RGBA, contype="0", conaffinity="0", mass="0")
    for b, ln, rr, part in ((prox, L[0], r, "proximal"), (mid, L[1], r * 0.92, "middle"), (dist, L[2], r * 0.85, "distal")):
        _touch(ed, b, nm(f"{part}_pad"), S.p((rr * 0.5, 0, -ln / 2)), (rr * 0.75, rr * 1.1, ln / 2), sensors)
    if actuators:
        for j in ("mcp_flex", "mcp_abd", "pip", "dip"):
            ops.add_actuator(ed, nm(j), "position", kp="0.6" if j != "mcp_abd" else "0.4")
    if sensors:
        for j in ("mcp_flex", "mcp_abd", "pip", "dip"):
            ops.add_sensor(ed, "jointpos", joint=nm(j))
    return prox


def build_thumb(ed: Editor, palm: Element, side: str = "left", scale: float = 1.0, sensors: bool = True,
                actuators: bool = True) -> Element:
    """Thumb: metacarpal on a 2-DOF carpometacarpal (CMC) base - *rotation* (flexion /
    opposition sweep across the palm) and *abduction* (palmar abduction) - then MCP and
    IP flexion. Anatomical note: the trapeziometacarpal saddle joint has two
    non-orthogonal, non-intersecting axes that produce conjunct axial rotation; here
    they are approximated by two orthogonal hinges at one point."""
    S = Side(side)
    nm = lambda b: S.n(f"thumb_{b}")  # noqa: E731
    sc = scale
    # resting thumb metacarpal: ~35 deg radial from the index ray, pronated ~70 deg
    q_rest = mx.qmul(mx.axang2q((1, 0, 0), D(35)), mx.axang2q((0, 0, 1), D(-70)))
    meta = ops.add_body(ed, palm, nm("metacarpal"), pos=S.p(np.array([0.012, 0.022, -0.022]) * sc), quat=S.q(q_rest))
    _hinge(ed, meta, nm("cmc_rot"), S.ax((0, -1, 0)), -20, 50)   # flexion / opposition sweep
    _hinge(ed, meta, nm("cmc_abd"), S.ax((1, 0, 0)), -10, 55)    # palmar abduction
    L = (0.045 * sc, 0.031 * sc, 0.024 * sc)
    r = 0.0105 * sc
    ops.add_geom(ed, meta, "capsule", size=mx.fmt_num(r * 1.1), fromto=S.ft((0, 0, -0.006), (0, 0, -L[0] + 0.004)),
                 rgba=SKIN_RGBA, density="1100", **HAND)
    prox = ops.add_body(ed, meta, nm("proximal"), pos=S.p((0, 0, -L[0])))
    _hinge(ed, prox, nm("mcp"), S.ax((0, -1, 0)), -10, 55)
    ops.add_geom(ed, prox, "capsule", size=mx.fmt_num(r), fromto=S.ft((0, 0, -0.003), (0, 0, -L[1] + 0.003)),
                 rgba=SKIN_RGBA, density="1100", **HAND)
    dist = ops.add_body(ed, prox, nm("distal"), pos=S.p((0, 0, -L[1])))
    _hinge(ed, dist, nm("ip"), S.ax((0, -1, 0)), -15, 80)
    ops.add_geom(ed, dist, "capsule", size=mx.fmt_num(r * 0.95), fromto=S.ft((0, 0, -0.003), (0, 0, -L[2] + r)),
                 rgba=SKIN_RGBA, density="1100", **HAND)
    for b, ln, part in ((meta, L[0], "metacarpal"), (prox, L[1], "proximal"), (dist, L[2], "distal")):
        _touch(ed, b, nm(f"{part}_pad"), S.p((r * 0.5, 0, -ln / 2)), (r * 0.8, r * 1.1, ln / 2), sensors)
    if actuators:
        for j, kp in (("cmc_rot", "1.0"), ("cmc_abd", "1.0"), ("mcp", "0.6"), ("ip", "0.4")):
            ops.add_actuator(ed, nm(j), "position", kp=kp)
    if sensors:
        for j in ("cmc_rot", "cmc_abd", "mcp", "ip"):
            ops.add_sensor(ed, "jointpos", joint=nm(j))
    return meta


def build_hand(ed: Editor, parent: Element | None, side: str = "left", pos=(0, 0, 0), quat=None,
               scale: float = 1.0, wrist: bool = True, sensors: bool = True, actuators: bool = True) -> Element:
    """Palm + four 3-phalanx fingers + thumb. With wrist=True the palm gets wrist flexion
    and radial/ulnar deviation hinges (forearm pronation belongs to the forearm)."""
    S = Side(side)
    with ed.structural(f"Build {side} hand"):
        palm = ops.add_body(ed, parent, S.n("hand"), pos=pos, quat=None if quat is None else S.q(quat))
        if wrist:
            _hinge(ed, palm, S.n("wrist_flex"), S.ax((0, -1, 0)), -70, 80, damping=0.2)
            _hinge(ed, palm, S.n("wrist_dev"), S.ax((1, 0, 0)), -30, 20, damping=0.2)
        ops.add_geom(ed, palm, "box", S.n("palm"), size=mx.fmt_vec(np.array([0.014, 0.041, 0.045]) * scale),
                     pos=S.p(np.array([0.0, -0.001, -0.048]) * scale), rgba=SKIN_RGBA, density="1100", **HAND)
        ops.add_geom(ed, palm, "ellipsoid", size=mx.fmt_vec(np.array([0.016, 0.022, 0.022]) * scale),
                     pos=S.p(np.array([0.006, 0.019, -0.035]) * scale), rgba=SKIN_RGBA, density="1100", **HAND)  # thenar
        _touch(ed, palm, S.n("palm_pad"), S.p(np.array([0.012, 0.0, -0.05]) * scale),
               np.array([0.006, 0.042, 0.045]) * scale, sensors)
        for f in FINGERS:
            build_finger(ed, palm, f, side, scale, sensors, actuators)
        build_thumb(ed, palm, side, scale, sensors, actuators)
        if actuators and wrist:
            ops.add_actuator(ed, S.n("wrist_flex"), "position", kp="8")
            ops.add_actuator(ed, S.n("wrist_dev"), "position", kp="8")
    return palm


# ======================================================================== foot
def build_foot(ed: Editor, parent: Element | None, side: str = "left", pos=(0, 0, 0), scale: float = 1.0,
               toes: str = "single", sensors: bool = True, actuators: bool = True,
               kp_ankle: float = 120.0) -> Element:
    """Segmented foot. Frame: origin at the talocrural joint centre, x anterior, floor at
    z = -0.075*scale. Chain and joints (positive angle in brackets):

      ankle (talus)    talocrural hinge   ankle_flex   [dorsiflexion]  -50..25 deg
      hindfoot         subtalar hinge     subtalar     [inversion]     -10..25 deg
      midfoot          transverse tarsal  midtarsal    [dorsiflexion]   -8..8 deg (stiff)
      forefoot         tarsometatarsal    tmt          [dorsiflexion]   -5..5 deg (stiff)
      toes             metatarsophalang.  mtp          [dorsiflexion]  -35..70 deg

    The 'dorsum' (발등) is not a segment: it is the upper surface of the midfoot and
    forefoot geoms. Simplifications vs. anatomy: one talocrural axis (Inman: ~8 deg
    inferolateral tilt, ~8 deg posterolateral), one subtalar axis (~42 deg elevation,
    ~23 deg medial deviation; real values vary widely), Chopart and Lisfranc joints
    reduced to one stiff sagittal hinge each, and - with toes='single' - the five MTP
    joints merged into one hinge on the oblique metatarsal break line (≈58 deg to the
    long axis). toes='detailed' gives each toe its own MTP and IP joints."""
    S = Side(side)
    s = scale
    zf = -0.075 * s
    nm = S.n
    with ed.structural(f"Build {side} foot"):
        ankle = ops.add_body(ed, parent, nm("talus"), pos=pos)
        # talocrural axis, medial->lateral is +y for the left foot; lateral malleolus posterior & inferior
        ax_tc = -mx.qrot(mx.QID, np.array([-0.14, 1.0, -0.14]) / np.linalg.norm([-0.14, 1.0, -0.14]))
        _hinge(ed, ankle, nm("ankle_flex"), S.ax(ax_tc), -50, 25, damping=1.0, armature=0.01)
        ops.add_geom(ed, ankle, "sphere", size=mx.fmt_num(0.022 * s), rgba=SKIN_RGBA, mass=f"{0.1 * s ** 3:.4g}",
                     contype="0", conaffinity="0")
        hind = ops.add_body(ed, ankle, nm("hindfoot"), pos=S.p(np.array([-0.012, 0.0, -0.028]) * s))
        v = np.array([math.cos(D(42)) * math.cos(D(23)), -math.cos(D(42)) * math.sin(D(23)), math.sin(D(42))])
        _hinge(ed, hind, nm("subtalar"), S.ax(-v), -10, 25, damping=1.0, armature=0.005)
        zh = zf + 0.028 * s  # hindfoot frame is 0.028 below the ankle
        ops.add_geom(ed, hind, "capsule", nm("heel"), size=mx.fmt_num(0.024 * s),
                     fromto=S.ft(np.array([-0.035, -0.008, zh / s + 0.024]) * s, np.array([-0.035, 0.008, zh / s + 0.024]) * s),
                     rgba=SKIN_RGBA, mass=f"{0.3 * s ** 3:.4g}", friction="1.0 0.005 0.0001")
        ops.add_geom(ed, hind, "box", size=mx.fmt_vec(np.array([0.03, 0.022, 0.022]) * s), pos=S.p(np.array([0.0, 0.0, 0.0]) * s),
                     rgba=SKIN_RGBA, mass=f"{0.08 * s ** 3:.4g}", contype="0", conaffinity="0")
        _touch(ed, hind, nm("heel_zone"), S.p(np.array([-0.035, 0.0, zh / s + 0.012]) * s),
               np.array([0.04, 0.03, 0.02]) * s, sensors)
        mid = ops.add_body(ed, hind, nm("midfoot"), pos=S.p(np.array([0.04, 0.0, 0.004]) * s))
        # arch stiffness: ~2 deg deflection under body weight (the real midfoot 'locks' in push-off)
        _hinge(ed, mid, nm("midtarsal"), S.ax((0.1, -1, 0.15)), -8, 8, damping=3.0, armature=0.002,
               stiffness=f"{800 * s ** 4:.4g}", solreflimit="0.005 1")
        zm = zh - 0.004 * s
        ops.add_geom(ed, mid, "box", nm("midfoot_dorsum"), size=mx.fmt_vec(np.array([0.026, 0.03, 0.018]) * s),
                     pos=S.p(np.array([0.02, 0.003, zm / s + 0.035]) * s), rgba=SKIN_RGBA, mass=f"{0.25 * s ** 3:.4g}",
                     contype="0", conaffinity="0")
        # lateral column (cuboid) touches lightly; the medial arch stays elevated
        ops.add_geom(ed, mid, "capsule", nm("midfoot_lateral"), size=mx.fmt_num(0.008 * s),
                     fromto=S.ft(np.array([0.0, 0.022, zm / s + 0.014]) * s, np.array([0.04, 0.026, zm / s + 0.014]) * s),
                     rgba=SKIN_RGBA, mass="0")
        fore = ops.add_body(ed, mid, nm("forefoot"), pos=S.p(np.array([0.045, 0.0, 0.0]) * s))
        _hinge(ed, fore, nm("tmt"), S.ax((0.0, -1, 0)), -5, 5, damping=3.0, armature=0.002,
               stiffness=f"{800 * s ** 4:.4g}", solreflimit="0.005 1")
        zfo = zm
        ops.add_geom(ed, fore, "box", nm("forefoot_dorsum"), size=mx.fmt_vec(np.array([0.035, 0.034, 0.014]) * s),
                     pos=S.p(np.array([0.03, 0.002, zfo / s + 0.03]) * s), rgba=SKIN_RGBA, mass=f"{0.25 * s ** 3:.4g}",
                     contype="0", conaffinity="0")
        # ball of the foot along the oblique metatarsal heads (1st MT head medial-anterior)
        mt1, mt5 = np.array([0.065, -0.026, zfo / s + 0.011]), np.array([0.035, 0.034, zfo / s + 0.011])
        ops.add_geom(ed, fore, "capsule", nm("ball"), size=mx.fmt_num(0.011 * s), fromto=S.ft(mt1 * s, mt5 * s),
                     rgba=SKIN_RGBA, mass=f"{0.05 * s ** 3:.4g}", friction="1.0 0.005 0.0001")
        _touch(ed, fore, nm("forefoot_zone"), S.p(((mt1 + mt5) / 2 + [0, 0, -0.002]) * s),
               np.array([0.03, 0.045, 0.016]) * s, sensors,
               quat=S.q(mx.axang2q((0, 0, 1), math.atan2(mt5[1] - mt1[1], mt5[0] - mt1[0]) - math.pi / 2)))
        if toes == "detailed":
            _detailed_toes(ed, fore, S, s, zfo, mt1, mt5, sensors, actuators)
        else:
            toe = ops.add_body(ed, fore, nm("toes"), pos=S.p(((mt1 + mt5) / 2 * [1, 1, 0] + [0, 0, 0.0]) * s))
            line = (mt1 - mt5)[:2]
            u = np.array([line[0], line[1], 0.0]) / np.linalg.norm(line)   # lateral -> medial (left foot)
            _hinge(ed, toe, nm("mtp"), S.ax(u), -35, 70, damping=0.3, armature=0.001)
            ops.add_geom(ed, toe, "capsule", nm("toes_pad"), size=mx.fmt_num(0.01 * s),
                         fromto=S.ft(np.array([0.045, -0.03, zfo / s + 0.01]) * s, np.array([0.02, 0.03, zfo / s + 0.01]) * s),
                         rgba=SKIN_RGBA, mass=f"{0.06 * s ** 3:.4g}", friction="1.0 0.005 0.0001")
            _touch(ed, toe, nm("toes_zone"), S.p(np.array([0.033, 0.0, zfo / s + 0.006]) * s),
                   np.array([0.022, 0.04, 0.012]) * s, sensors)
            if actuators:
                ops.add_actuator(ed, nm("mtp"), "position", kp=f"{8 * s:.4g}")
            if sensors:
                ops.add_sensor(ed, "jointpos", joint=nm("mtp"))
        if actuators:
            ops.add_actuator(ed, nm("ankle_flex"), "position", kp=f"{kp_ankle:.4g}")
            ops.add_actuator(ed, nm("subtalar"), "position", kp=f"{kp_ankle * 0.5:.4g}")
        if sensors:
            for j in ("ankle_flex", "subtalar", "midtarsal", "tmt"):
                ops.add_sensor(ed, "jointpos", joint=nm(j))
    return ankle


def _detailed_toes(ed, fore, S, s, zfo, mt1, mt5, sensors, actuators):
    """Optional extension: hallux (MTP + IP) and four lesser toes (MTP + PIP + DIP)."""
    heads = [mt1 + (mt5 - mt1) * t for t in (0.0, 0.3, 0.52, 0.74, 1.0)]
    lens = [(0.032, 0.026), (0.022, 0.012, 0.010), (0.019, 0.010, 0.009), (0.017, 0.009, 0.009), (0.015, 0.008, 0.008)]
    radii = [0.011, 0.0065, 0.006, 0.0058, 0.0055]
    for i, (h, ls, r) in enumerate(zip(heads, lens, radii)):
        nm = S.n(f"toe{i + 1}")
        b = ops.add_body(ed, fore, f"{nm}_proximal", pos=S.p(h * [1, 1, 0] * s))
        _hinge(ed, b, f"{nm}_mtp", S.ax((0, -1, 0)), -35, 70, damping=0.1, armature=0.0005)
        parent = b
        for k, ln in enumerate(ls):
            if k > 0:
                parent = ops.add_body(ed, parent, f"{nm}_{'distal' if k == len(ls) - 1 else 'middle'}",
                                      pos=S.p((ls[k - 1] * s, 0, 0)))
                _hinge(ed, parent, f"{nm}_{'ip' if len(ls) == 2 else ('pip' if k == 1 else 'dip')}",
                       S.ax((0, -1, 0)), -10, 60, damping=0.05, armature=0.0002)
            ops.add_geom(ed, parent, "capsule", size=mx.fmt_num(r * s),
                         fromto=S.ft(np.array([0.002, 0, h[2] + 0.0]) * s, np.array([ln - 0.002, 0, h[2]]) * s),
                         rgba=SKIN_RGBA, mass=f"{0.01 * s ** 3:.4g}", friction="1.0 0.005 0.0001")
        if actuators:
            ops.add_actuator(ed, f"{nm}_mtp", "position", kp=f"{2 * s:.4g}")


# ======================================================================== head
def build_head(ed: Editor, parent: Element | None, pos=(0, 0, 0), scale: float = 1.0, jaw_slide: bool = False,
               sensors: bool = True, actuators: bool = True, neck: bool = True, face: bool = False,
               face_name: str = "face") -> Element:
    """Head with two independently actuated eyes (yaw + pitch hinges each) and a
    hinged mandible. Frame: origin at the head centre, x forward, y left, z up.
      eye_<side>_yaw   [look left]  -45..45 deg      eye_<side>_pitch [look up] -45..30 deg
      jaw_open         [open]        0..30 deg (temporomandibular rotation)
      jaw_protrude     (optional slide, 0..10 mm) for the translational TMJ phase.
    Eye rotation is a joint-level control independent of facial expressions."""
    s = scale
    with ed.structural("Build head"):
        head = ops.add_body(ed, parent, "head", pos=pos)
        if neck:
            _hinge(ed, head, "neck_pitch", (0, 1, 0), -40, 50, pos=(0, 0, -0.09 * s), damping=1.0, armature=0.01)
            _hinge(ed, head, "neck_yaw", (0, 0, 1), -70, 70, pos=(0, 0, -0.09 * s), damping=1.0, armature=0.01)
            _hinge(ed, head, "neck_roll", (1, 0, 0), -35, 35, pos=(0, 0, -0.09 * s), damping=1.0, armature=0.01)
        ops.add_geom(ed, head, "ellipsoid", "skull", size=mx.fmt_vec(np.array([0.095, 0.075, 0.1]) * s),
                     pos=(-0.01 * s, 0, 0.01 * s), rgba=SKIN_RGBA, mass=f"{4.0 * s ** 3:.4g}",
                     group="3" if face else None)
        if face:  # dark oral cavity visible when the mouth opens
            ops.add_geom(ed, head, "ellipsoid", "oral_cavity", size=mx.fmt_vec(np.array([0.024, 0.026, 0.016]) * s),
                         pos=(0.034 * s, 0, -0.058 * s), rgba="0.22 0.05 0.06 1", contype="0", conaffinity="0",
                         mass="0")
        ops.add_site(ed, head, "head_imu", pos=(0, 0, 0.03 * s), size=0.005)
        for side in ("left", "right"):
            S = Side(side)
            eye = ops.add_body(ed, head, f"eye_{side}", pos=S.p(np.array([0.07, 0.0315, 0.03]) * s))
            _hinge(ed, eye, f"eye_{side}_yaw", (0, 0, 1), -45, 45, damping=0.002, armature=1e-5)
            _hinge(ed, eye, f"eye_{side}_pitch", (0, -1, 0), -45, 30, damping=0.002, armature=1e-5)
            ops.add_geom(ed, eye, "sphere", f"eyeball_{side}", size=mx.fmt_num(0.012 * s), rgba="0.96 0.96 0.94 1",
                         contype="0", conaffinity="0", mass=f"{0.0075 * s ** 3:.4g}")
            ops.add_geom(ed, eye, "cylinder", f"iris_{side}", size=mx.fmt_vec([0.0058 * s, 0.0008 * s]),
                         pos=(0.0113 * s, 0, 0), quat=mx.fmt_vec(mx.axang2q((0, 1, 0), D(90))),
                         rgba="0.25 0.38 0.55 1", contype="0", conaffinity="0", mass="0")
            ops.add_geom(ed, eye, "cylinder", f"pupil_{side}", size=mx.fmt_vec([0.0026 * s, 0.0009 * s]),
                         pos=(0.0117 * s, 0, 0), quat=mx.fmt_vec(mx.axang2q((0, 1, 0), D(90))),
                         rgba="0.02 0.02 0.02 1", contype="0", conaffinity="0", mass="0")
            # gaze site: its x axis is the line of sight
            ops.add_site(ed, eye, f"gaze_{side}", pos=(0.012 * s, 0, 0), size=0.001)
        jaw = ops.add_body(ed, head, "jaw", pos=(0.005 * s, 0, -0.035 * s))
        _hinge(ed, jaw, "jaw_open", (0, 1, 0), 0, 30, damping=0.2, armature=0.001)
        if jaw_slide:
            ops.add_joint(ed, jaw, "slide", "jaw_protrude", axis=(1, 0, 0), range_rad=(0.0, 0.010),
                          damping="2", armature="0.001")
        hide = {"group": "3"} if face else {}   # the face skin shows the jaw instead
        ops.add_geom(ed, jaw, "capsule", "mandible", size=mx.fmt_num(0.012 * s),
                     fromto=mx.fmt_vec(np.array([0.0, 0.045, -0.01, 0.07, 0.0, -0.06]) * s), rgba=SKIN_RGBA,
                     mass=f"{0.1 * s ** 3:.4g}", contype="0", conaffinity="0", **hide)
        ops.add_geom(ed, jaw, "capsule", size=mx.fmt_num(0.012 * s),
                     fromto=mx.fmt_vec(np.array([0.0, -0.045, -0.01, 0.07, 0.0, -0.06]) * s), rgba=SKIN_RGBA,
                     mass=f"{0.1 * s ** 3:.4g}", contype="0", conaffinity="0", **hide)
        ops.add_site(ed, jaw, "chin", pos=(0.075 * s, 0, -0.065 * s), size=0.004)
        if actuators:
            for side in ("left", "right"):
                ops.add_actuator(ed, f"eye_{side}_yaw", "position", kp="0.05")
                ops.add_actuator(ed, f"eye_{side}_pitch", "position", kp="0.05")
            ops.add_actuator(ed, "jaw_open", "position", kp="3")
            if neck:
                for j in ("neck_pitch", "neck_yaw", "neck_roll"):
                    ops.add_actuator(ed, j, "position", kp="30")
        if sensors:
            for side in ("left", "right"):
                ops.add_sensor(ed, "jointpos", joint=f"eye_{side}_yaw")
                ops.add_sensor(ed, "jointpos", joint=f"eye_{side}_pitch")
                ops.add_sensor(ed, "framexaxis", name=f"gaze_dir_{side}", objtype="site", objname=f"gaze_{side}")
            ops.add_sensor(ed, "jointpos", joint="jaw_open")
            ops.add_sensor(ed, "gyro", name="head_gyro", site="head_imu")
            ops.add_sensor(ed, "accelerometer", name="head_accel", site="head_imu")
    if face:
        from ..core.faceops import attach_face_rig
        from .face import build_face_rig
        attach_face_rig(ed, build_face_rig(face_name, scale, head.get("name"), "jaw"))
    return head


# ======================================================================== standalone parts
def make_part(builder, name: str, **kw):
    """Run a builder in a fresh document and return it as a part (MjcfDoc)."""
    ed = Editor()
    ed.new(name)
    wb = ed.doc.worldbody
    for g in [c for c in wb if getattr(c, "tag", None) in ("geom", "light")]:
        wb.remove(g)
    ed.recompile()
    builder(ed, None, **kw)
    return ed.doc
