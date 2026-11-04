"""Procedural stylised face for the expression system (head-local frame: x forward, y left, z up).

The surface covers the whole head. It has
  * eye openings with spherical eyelids wrapped around the eyeballs (bone driven blink),
  * a mouth slit whose lower lip is bound to the jaw body (opens with the jaw joint),
  * nose, brow ridge, cheekbones, lips and chin shaped by smooth displacement fields,
  * shape keys (blend shapes) for brows, mouth corners, lips and cheeks.
Dimensions match ``anatomy.build_head`` (eye centres (0.07, ±0.0315, 0.03), eyeball radius
12 mm, jaw joint at (0.005, 0, -0.035)) and scale with ``scale``.
"""
from __future__ import annotations

import math

import numpy as np

from ..core.expr import ExprParam, FaceRig

EYE = np.array([0.07, 0.0315, 0.03])
CENTER = np.array([-0.01, 0.0, 0.01])
RADII = np.array([0.097, 0.077, 0.105])
LID_R = 0.0135
MOUTH_W = -0.055       # lip line height
MOUTH_HALF = 0.024     # half mouth width
TMJ = np.array([0.005, 0.0, -0.035])


def _g(x, mu, s):
    return np.exp(-0.5 * ((x - mu) / s) ** 2)


def _k(x, mu, s):
    """Compact smooth bump (1-r^2)^3 with support |x-mu| < 2.5 s: regions that do not
    overlap stay exactly independent (unlike Gaussian tails)."""
    r2 = ((x - mu) / (2.5 * s)) ** 2
    return np.where(r2 < 1, (1 - r2) ** 3, 0.0)


def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _density_samples(n: int, lo: float, hi: float, peaks) -> np.ndarray:
    """n samples on [lo, hi] (both included) with density 1 + sum gain*gauss(mu, sigma)."""
    x = np.linspace(lo, hi, 8001)
    dens = np.ones_like(x)
    for mu, sig, gain in peaks:
        dens += gain * np.exp(-0.5 * ((x - mu) / sig) ** 2)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(x))])
    return np.interp(np.linspace(0, 1, n), cdf / cdf[-1], x)


def face_geometry(n_theta: int = 128, n_phi: int = 96, scale: float = 1.0):
    """Returns (V, F, info) in head-local coordinates (metres). The grid is denser on the
    face (around azimuth 0 and between brows and chin) and symmetric left/right."""
    c, a = CENTER, RADII
    phi_m = math.asin((MOUTH_W - c[2]) / a[2])
    phis = list(_density_samples(n_phi + 1, -math.pi / 2, math.pi / 2, [(-0.25, 0.42, 5.0)])[1:-1])
    jm = int(np.argmin([abs(p - phi_m) for p in phis]))
    phis[jm] = phi_m                                     # one grid row exactly on the lip line
    k = n_theta // 2
    pos = _density_samples(k + 1, 0.0, math.pi, [(0.0, 0.5, 7.0)])
    thetas = np.concatenate([-pos[::-1][:-1], pos[:-1]])  # symmetric, periodic (-pi included)
    n_theta = len(thetas)
    P = np.array([[c[0] + a[0] * math.cos(p) * math.cos(t), c[1] + a[1] * math.cos(p) * math.sin(t),
                   c[2] + a[2] * math.sin(p)] for p in phis for t in thetas])
    # head proportions: shorter, broader lower head (jaw line) instead of an egg-shaped point
    low = P[:, 2] < -0.06
    P[low, 2] = -0.06 + 0.72 * (P[low, 2] + 0.06)
    P[:, 1] *= 1 + 0.22 * _smooth((-0.03 - P[:, 2]) / 0.05)
    u, w = P[:, 1], P[:, 2]
    front = _smooth((P[:, 0] - 0.0) / 0.05)               # 0 behind the ears, 1 on the face
    dx = np.zeros(len(P))
    dx += 0.006 * _g(w, -0.058, 0.025) * _g(u, 0, 0.035)                                # lower face forward
    dx += 0.015 * _g(u, 0, 0.0068) * _g(w, -0.009, 0.0125)                              # nose
    dx += 0.005 * _g(u, 0, 0.013) * _g(w, -0.021, 0.005)                                # alae
    dx += 0.004 * _g(w, 0.05, 0.008) * _g(np.abs(u), 0.03, 0.022)                       # brow ridge
    dx += 0.004 * _g(w, 0.003, 0.014) * _g(np.abs(u), 0.046, 0.012)                     # cheekbones
    dx += 0.0045 * _g(w, MOUTH_W, 0.0065) * _g(u, 0, 0.02) * (np.abs(u) < MOUTH_HALF * 1.25)  # lips
    dx += 0.004 * _g(w, -0.082, 0.01) * _g(u, 0, 0.022)                                 # chin
    P[:, 0] += dx * front
    # eye sockets: pull the surface onto the eyelid sphere around each eyeball
    sock = np.zeros((2, len(P)))
    for k, s in enumerate((1.0, -1.0)):
        e = EYE * [1, s, 1]
        dt = np.hypot(u - e[1], w - e[2])
        wt = (1 - _smooth((dt - 0.0165) / 0.011)) * (P[:, 0] > 0.03)
        sock[k] = wt
        d = P - e
        on = e + LID_R * d / np.linalg.norm(d, axis=1, keepdims=True)
        P = P + wt[:, None] * (on - P)
    # faces (grid + poles); eye openings removed; mouth slit duplicated
    n = len(P)
    rows = len(phis)
    idx = lambda j, i: j * n_theta + (i % n_theta)  # noqa: E731
    keep = np.ones(n, bool)
    holes = []
    for s in (1.0, -1.0):
        e = EYE * [1, s, 1]
        hole = (((u - e[1]) / 0.0135) ** 2 + ((w - e[2] - 0.0008) / 0.0062) ** 2 < 1) & (P[:, 0] > 0.05)
        keep &= ~hole
        holes.append((e, hole))
    for e, hole in holes:
        grid = hole.reshape(rows, n_theta)
        nb = np.zeros_like(grid)
        nb[1:] |= grid[:-1]
        nb[:-1] |= grid[1:]
        nb |= np.roll(grid, 1, axis=1) | np.roll(grid, -1, axis=1)
        rim = (nb & ~grid).ravel()
        du, dw = u[rim] - e[1], w[rim] - e[2] - 0.0008
        ang = np.arctan2(dw / 0.0062, du / 0.0135)
        tu, tw = e[1] + 0.0135 * np.cos(ang), e[2] + 0.0008 + 0.0062 * np.sin(ang)
        q = P[rim].copy()
        q[:, 1], q[:, 2] = tu, tw
        d = q - e
        P[rim] = e + LID_R * d / np.linalg.norm(d, axis=1, keepdims=True)
    slit_cols = [i for i in range(n_theta) if abs(P[idx(jm, i), 1]) < MOUTH_HALF and P[idx(jm, i), 0] > 0.04]
    slit_cols = slit_cols[1:-1] if len(slit_cols) > 2 else []      # corners stay shared
    dup = {}
    V = [p for p in P]
    for i in slit_cols:
        dup[i] = len(V)
        V.append(P[idx(jm, i)].copy())
    bottom, top = len(V), len(V) + 1
    V.append([c[0], c[1], c[2] - a[2]])
    V.append([c[0], c[1], c[2] + a[2]])
    V = np.array(V)
    F = []
    for j in range(rows - 1):
        for i in range(n_theta):
            q = [idx(j, i), idx(j, i + 1), idx(j + 1, i + 1), idx(j + 1, i)]
            if not all(keep[x] for x in q):
                continue
            if j + 1 == jm:      # faces just below the lip line use the lower-lip copies
                q[2] = dup.get((i + 1) % n_theta, q[2])
                q[3] = dup.get(i, q[3])
            F += [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]
    for i in range(n_theta):
        F.append([bottom, idx(0, i + 1), idx(0, i)])
        F.append([top, idx(rows - 1, i), idx(rows - 1, i + 1)])
    F = np.array(F, np.int64)
    used = np.zeros(len(V), bool)
    used[F.ravel()] = True
    remap = -np.ones(len(V), np.int64)
    remap[used] = np.arange(int(used.sum()))
    lower = np.zeros(len(V), bool)
    lower[list(dup.values())] = True
    info = {"lower_lip": lower[used], "socket_L": np.r_[sock[0], np.zeros(len(V) - n)][used],
            "socket_R": np.r_[sock[1], np.zeros(len(V) - n)][used], "mouth_w": MOUTH_W}
    return V[used] * scale, remap[F], info


def jaw_weights(V: np.ndarray, lower_lip: np.ndarray, scale: float = 1.0) -> np.ndarray:
    x, y, z = (V / scale).T
    # line from the jaw joint (TMJ) to the lip line at the front
    t = np.clip((x - TMJ[0]) / (0.085 - TMJ[0]), 0, 1)
    zl = TMJ[2] + t * (MOUTH_W - TMJ[2])
    below = _smooth((zl - z + 0.004) / 0.014)
    behind = _smooth((x + 0.02) / 0.025)
    neck = 1 - _smooth((-0.112 - z) / 0.015)
    w = below * behind * neck
    near_mouth = (np.abs(y) < MOUTH_HALF) & (np.abs(z - MOUTH_W) < 0.004) & (x > 0.04)
    w[near_mouth & ~lower_lip] = 0.0      # upper-lip edge stays on the head
    w[lower_lip] = 1.0
    return np.clip(w, 0, 1)


def build_face_rig(name: str = "face", scale: float = 1.0, head_body: str = "head", jaw_body: str = "jaw",
                   n_theta: int = 128, n_phi: int = 96) -> FaceRig:
    V, F, info = face_geometry(n_theta, n_phi, scale)
    rig = FaceRig(name, V, F, head_body, jaw_body, jaw_weights(V, info["lower_lip"], scale))
    x, y, z = (V / scale).T
    low = info["lower_lip"] | (z < MOUTH_W - 1e-4)
    front = _smooth((x - 0.02) / 0.04)
    s = scale
    for side, sg in (("L", 1.0), ("R", -1.0)):
        e = EYE * [1, sg, 1]
        uu = sg * y          # lateral coordinate, positive towards this side
        # brows (shape keys)
        br = _k(uu, e[1], 0.013) * _k(z, 0.052, 0.011) * _smooth((x - 0.045) / 0.02)
        rig.add_key(f"brow_raise_{side}", np.c_[np.zeros_like(x), np.zeros_like(x), 0.006 * br] * s)
        rig.add_key(f"brow_lower_{side}", np.c_[0.001 * br, -sg * 0.003 * br * _k(uu, 0.02, 0.02), -0.0045 * br] * s)
        # mouth corners
        cg = _k(uu, MOUTH_HALF, 0.011) * _k(z, MOUTH_W, 0.011) * front * (uu > -0.005)
        ck = _k(uu, 0.042, 0.015) * _k(z, -0.02, 0.015) * front
        rig.add_key(f"mouth_smile_{side}", (np.c_[-0.003 * cg + 0.003 * ck, sg * 0.003 * cg, 0.0055 * cg + 0.003 * ck]) * s)
        rig.add_key(f"mouth_frown_{side}", np.c_[-0.001 * cg, sg * 0.001 * cg, -0.005 * cg] * s)
        pf = _k(uu, 0.045, 0.016) * _k(z, -0.045, 0.016) * front
        dirn = np.array([0.35, sg * 0.94, 0.0])
        rig.add_key(f"cheek_puff_{side}", (0.007 * pf)[:, None] * dirn * s)
        # eyelid control bones (rotation about the eye centre)
        sock = info[f"socket_{side}"]
        lid = np.clip(sock * 1.6, 0, 1)
        up = lid * _smooth((z - e[2] + 0.002) / 0.004) * (x > 0.045)
        lo_ = lid * _smooth((e[2] - z - 0.002) / 0.004) * (x > 0.045)
        rig.add_bone(f"lid_upper_{side}", e * s, up)
        rig.add_bone(f"lid_lower_{side}", e * s, lo_)
    lip = _k(y, 0, 0.016) * _k(z - MOUTH_W, 0, 0.0065) * front
    rig.add_key("lips_part", np.c_[np.zeros_like(x), np.zeros_like(x), np.where(low, -0.0025, 0.0025) * lip] * s)
    pk = _k(y, 0, 0.02) * _k(z - MOUTH_W, 0, 0.012) * front
    rig.add_key("lip_pucker", np.c_[0.008 * pk, -0.35 * y * pk, np.zeros_like(x)] * s)
    D = math.radians
    for side, nm in (("L", "left"), ("R", "right")):
        rig.add_param(ExprParam(f"blink_{nm}", bones={f"lid_upper_{side}": {"r": [0, D(60), 0]},
                                                       f"lid_lower_{side}": {"r": [0, -D(13), 0]}}, group="eyes"))
        rig.add_param(ExprParam(f"eye_wide_{nm}", bones={f"lid_upper_{side}": {"r": [0, -D(14), 0]}}, group="eyes"))
        rig.add_param(ExprParam(f"eye_squint_{nm}", bones={f"lid_lower_{side}": {"r": [0, -D(14), 0]},
                                                            f"lid_upper_{side}": {"r": [0, D(8), 0]}}, group="eyes"))
        rig.add_param(ExprParam(f"brow_raise_{nm}", keys={f"brow_raise_{side}": 1.0}, group="brows"))
        rig.add_param(ExprParam(f"brow_lower_{nm}", keys={f"brow_lower_{side}": 1.0}, group="brows"))
        rig.add_param(ExprParam(f"mouth_smile_{nm}", keys={f"mouth_smile_{side}": 1.0}, group="mouth"))
        rig.add_param(ExprParam(f"mouth_frown_{nm}", keys={f"mouth_frown_{side}": 1.0}, group="mouth"))
        rig.add_param(ExprParam(f"cheek_puff_{nm}", keys={f"cheek_puff_{side}": 1.0}, group="cheeks"))
    rig.add_param(ExprParam("smile", keys={"mouth_smile_L": 1.0, "mouth_smile_R": 1.0}, group="mouth"))
    rig.add_param(ExprParam("lips_part", keys={"lips_part": 1.0}, group="mouth"))
    rig.add_param(ExprParam("lip_pucker", keys={"lip_pucker": 1.0}, group="mouth"))
    rig.presets = default_presets()
    rig.meta["source"] = "procedural (mjstudio.library.face)"
    return rig


def default_presets() -> dict:
    return {
        "neutral": {},
        "smile": {"smile": 1.0, "eye_squint_left": 0.35, "eye_squint_right": 0.35},
        "frown": {"brow_lower_left": 1.0, "brow_lower_right": 1.0, "mouth_frown_left": 0.8,
                  "mouth_frown_right": 0.8, "eye_squint_left": 0.3, "eye_squint_right": 0.3},
        "surprise": {"brow_raise_left": 1.0, "brow_raise_right": 1.0, "eye_wide_left": 1.0, "eye_wide_right": 1.0,
                     "lips_part": 0.7},
        "blink": {"blink_left": 1.0, "blink_right": 1.0},
        "wink_left": {"blink_left": 1.0, "mouth_smile_left": 0.5},
        "wink_right": {"blink_right": 1.0, "mouth_smile_right": 0.5},
    }
