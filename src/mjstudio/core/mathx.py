"""Math helpers: number formatting, quaternions, MJCF orientation conventions.

Quaternions use MuJoCo order (w, x, y, z). Angles are radians unless stated.
The MJCF orientation conversions follow the MuJoCo XML reference semantics
(they are verified against the MuJoCo compiler in tests/test_mathx.py).
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

EPS = 1e-12
QID = np.array([1.0, 0.0, 0.0, 0.0])


# ----------------------------------------------------------------- formatting
def fmt_num(x: float, prec: int = 8) -> str:
    """Compact MJCF number: '%.8g' with '-0' normalised to '0'."""
    x = float(x)
    if not math.isfinite(x):
        return "inf" if x > 0 else ("-inf" if x < 0 else "nan")
    if abs(x) < 10 ** (-(prec + 4)):
        return "0"
    s = f"{x:.{prec}g}"
    if s == "-0":
        s = "0"
    return s


def fmt_vec(v: Iterable[float], prec: int = 8) -> str:
    return " ".join(fmt_num(float(x), prec) for x in v)


def parse_vec(s: str | None, n: int | None = None) -> np.ndarray | None:
    """Parse whitespace separated numbers. Returns None if s is None/empty or invalid."""
    if s is None:
        return None
    parts = s.replace(",", " ").split()
    if not parts:
        return None
    try:
        arr = np.array([float(p) for p in parts], dtype=float)
    except ValueError:
        return None
    if n is not None and arr.size != n:
        return None
    return arr


def is_number_list(s: str, n: int | None = None, nmin: int | None = None, nmax: int | None = None) -> bool:
    v = parse_vec(s)
    if v is None:
        return False
    if n is not None and v.size != n:
        return False
    if nmin is not None and v.size < nmin:
        return False
    if nmax is not None and v.size > nmax:
        return False
    return True


# ----------------------------------------------------------------- quaternion
def qnorm(q: Sequence[float]) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = float(np.linalg.norm(q))
    if n < EPS:
        return QID.copy()
    return q / n


def qcanon(q: Sequence[float]) -> np.ndarray:
    """Normalise and force w >= 0 (q and -q are the same rotation)."""
    q = qnorm(q)
    return -q if q[0] < 0 else q


def qmul(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def qconj(q: Sequence[float]) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def q2mat(q: Sequence[float]) -> np.ndarray:
    w, x, y, z = qnorm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def mat2q(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=float).reshape(3, 3)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    return qcanon(q)


def qrot(q: Sequence[float], v: Sequence[float]) -> np.ndarray:
    return q2mat(q) @ np.asarray(v, dtype=float)


def axang2q(axis: Sequence[float], ang: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    n = float(np.linalg.norm(a))
    if n < EPS:
        return QID.copy()
    a = a / n
    s = math.sin(ang / 2)
    return np.array([math.cos(ang / 2), a[0] * s, a[1] * s, a[2] * s])


def q2axang(q: Sequence[float]) -> tuple[np.ndarray, float]:
    q = qcanon(q)
    s = float(np.linalg.norm(q[1:]))
    if s < 1e-12:
        return np.array([0.0, 0.0, 1.0]), 0.0
    ang = 2 * math.atan2(s, q[0])
    return q[1:] / s, ang


def qangle(a: Sequence[float], b: Sequence[float]) -> float:
    """Rotation angle between two orientations (radians)."""
    d = abs(float(np.dot(qnorm(a), qnorm(b))))
    return 2 * math.acos(min(1.0, d))


# ------------------------------------------------------- MJCF orientation specs
_AX = {"x": 0, "y": 1, "z": 2}


def euler2q(euler: Sequence[float], seq: str = "xyz") -> np.ndarray:
    """MuJoCo euler semantics: lower case = intrinsic (rotating axes, post-multiply),
    upper case = extrinsic (fixed axes, pre-multiply); characters may be mixed."""
    if len(seq) != 3:
        raise ValueError(f"eulerseq must have 3 characters: {seq!r}")
    q = QID.copy()
    for ang, ch in zip(euler, seq):
        t = np.array([math.cos(ang / 2), 0.0, 0.0, 0.0])
        t[1 + _AX[ch.lower()]] = math.sin(ang / 2)
        q = qmul(q, t) if ch.islower() else qmul(t, q)
    return q


def q2euler(q: Sequence[float], seq: str = "xyz") -> np.ndarray | None:
    """Inverse of euler2q for pure-intrinsic or pure-extrinsic sequences.
    Returns None for mixed-case sequences (caller should fall back to quat).
    Method: Bernardes & Viollet (2022), quaternion to Euler for any sequence."""
    if len(seq) != 3:
        return None
    if not (seq.islower() or seq.isupper()):
        return None
    intrinsic = seq.islower()
    ax = [_AX[c.lower()] for c in seq]
    if ax[0] == ax[1] or ax[1] == ax[2]:
        return None
    # euler2q(intrinsic a,b,c) == extrinsic (c,b,a) with reversed angles
    if intrinsic:
        i, j, k = ax[2], ax[1], ax[0]
    else:
        i, j, k = ax[0], ax[1], ax[2]
    q = qnorm(q)
    ext_seq = "".join("XYZ"[t] for t in (i, j, k))
    proper = i == k
    if proper:
        k = 3 - i - j
    sign = int((i - j) * (j - k) * (k - i) / 2)
    a, b, c, d = q[0], q[i + 1], q[j + 1], q[k + 1] * sign
    if not proper:
        a, b, c, d = a - c, b + d, c + a, d - b
    th2 = 2 * math.atan2(math.hypot(c, d), math.hypot(a, b))
    tp = math.atan2(b, a)
    tm = math.atan2(d, c)
    eps = 1e-7
    degenerate = abs(th2) < eps or abs(th2 - math.pi) < eps
    th1, th3 = tp - tm, tp + tm
    if not proper:
        th3 *= sign
        th2 -= math.pi / 2
    out = [th1, th2, th3]  # extrinsic angles about (i, j, k): first i
    if degenerate:
        # gimbal lock: fix third angle at 0 and pick the first angle that reproduces q
        best = None
        for cand in (2 * tp, -2 * tp, 2 * tm, -2 * tm, tp - tm, tp + tm):
            trial = [cand, out[1], 0.0]
            err = qangle(euler2q(trial, ext_seq), q)
            if best is None or err < best[0]:
                best = (err, trial)
        out = best[1]
    out = [(x + math.pi) % (2 * math.pi) - math.pi for x in out]
    if intrinsic:
        out = out[::-1]
    return np.array(out)


def xyaxes2q(xy: Sequence[float]) -> np.ndarray:
    xy = np.asarray(xy, dtype=float)
    x = xy[:3] / max(np.linalg.norm(xy[:3]), EPS)
    y = xy[3:6] - x * float(np.dot(x, xy[3:6]))
    y = y / max(np.linalg.norm(y), EPS)
    z = np.cross(x, y)
    return mat2q(np.column_stack([x, y, z]))


def zaxis2q(zv: Sequence[float]) -> np.ndarray:
    """Minimal rotation taking (0,0,1) to zv (MuJoCo mjuu_z2quat semantics)."""
    v = np.asarray(zv, dtype=float)
    n = float(np.linalg.norm(v))
    if n < EPS:
        return QID.copy()
    v = v / n
    axis = np.cross([0.0, 0.0, 1.0], v)
    s = float(np.linalg.norm(axis))
    if s < 1e-10:
        axis = np.array([1.0, 0.0, 0.0])
    else:
        axis = axis / s
    ang = math.atan2(s, v[2])
    return axang2q(axis, ang)


# ---------------------------------------------------------------- rigid poses
def pose_mul(p1, q1, p2, q2):
    """(p1,q1) o (p2,q2): transform expressed in frame 2 into frame 0."""
    return np.asarray(p1, float) + qrot(q1, p2), qmul(q1, q2)


def pose_inv(p, q):
    qi = qconj(qnorm(q))
    return -qrot(qi, p), qi


def look_rot(forward: Sequence[float], up: Sequence[float]) -> np.ndarray:
    f = np.asarray(forward, float)
    f = f / max(np.linalg.norm(f), EPS)
    u = np.asarray(up, float)
    r = np.cross(f, u)
    if np.linalg.norm(r) < 1e-9:
        r = np.cross(f, [1.0, 0.0, 0.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return np.column_stack([r, u, -f])


def closest_on_line_to_ray(lp, ld, ro, rd) -> float:
    """Parameter t of the point lp + t*ld closest to the ray ro + s*rd."""
    lp, ld, ro, rd = (np.asarray(a, float) for a in (lp, ld, ro, rd))
    w0 = lp - ro
    a = ld @ ld
    b = ld @ rd
    c = rd @ rd
    d = ld @ w0
    e = rd @ w0
    den = a * c - b * b
    if abs(den) < 1e-12:
        return 0.0
    return (b * e - c * d) / den


def ray_plane(ro, rd, pp, pn):
    """Intersection point of ray with plane (point pp, normal pn) or None."""
    ro, rd, pp, pn = (np.asarray(a, float) for a in (ro, rd, pp, pn))
    den = rd @ pn
    if abs(den) < 1e-12:
        return None
    t = ((pp - ro) @ pn) / den
    if t < 0:
        return None
    return ro + t * rd
