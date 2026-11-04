"""Orientation conventions must match the MuJoCo compiler exactly."""
import itertools
import math

import mujoco
import numpy as np
import pytest

from mjstudio.core import mathx as mx

RNG = np.random.default_rng(1234)


def _compiled_body_quat(attr: str, angle: str = "radian", eulerseq: str = "xyz") -> np.ndarray:
    xml = (f'<mujoco><compiler angle="{angle}" eulerseq="{eulerseq}"/><worldbody>'
           f'<body {attr}><geom size="0.1"/></body></worldbody></mujoco>')
    m = mujoco.MjModel.from_xml_string(xml)
    return m.body_quat[1].copy()


def _same(q1, q2, tol=1e-6):
    return mx.qangle(q1, q2) < tol


SEQS = ["".join(p) for p in itertools.product("xyz", repeat=3) if p[0] != p[1] and p[1] != p[2]]
SEQS = SEQS + [s.upper() for s in SEQS] + ["xYz", "XyZ"]


@pytest.mark.parametrize("seq", SEQS)
def test_euler_matches_mujoco(seq):
    for _ in range(5):
        e = RNG.uniform(-math.pi, math.pi, 3)
        qm = _compiled_body_quat(f'euler="{mx.fmt_vec(e, 12)}"', eulerseq=seq)
        assert _same(mx.euler2q(e, seq), qm), seq


@pytest.mark.parametrize("seq", [s for s in SEQS if s.islower() or s.isupper()])
def test_q2euler_roundtrip(seq):
    for _ in range(20):
        q = mx.qnorm(RNG.normal(size=4))
        e = mx.q2euler(q, seq)
        assert e is not None
        assert _same(mx.euler2q(e, seq), q, 1e-6), (seq, q, e)


@pytest.mark.parametrize("seq", ["xyz", "XYZ", "zyx", "zxz", "ZXZ", "yxy"])
def test_q2euler_gimbal(seq):
    ax = seq.lower()
    for mid in (0.0, math.pi / 2, -math.pi / 2, math.pi):
        e = np.array([0.3, mid, -0.7])
        q = mx.euler2q(e, seq)
        e2 = mx.q2euler(q, seq)
        assert _same(mx.euler2q(e2, seq), q, 1e-6), (seq, mid, e2)
    assert mx.q2euler(mx.QID, "xYz") is None
    _ = ax


def test_axisangle_xyaxes_zaxis_degree():
    for _ in range(10):
        ax = RNG.normal(size=3)
        ang = RNG.uniform(-170, 170)
        qm = _compiled_body_quat(f'axisangle="{mx.fmt_vec(ax, 12)} {ang:.12g}"', angle="degree")
        assert _same(mx.axang2q(ax, math.radians(ang)), qm)
        xy = RNG.normal(size=6)
        qm = _compiled_body_quat(f'xyaxes="{mx.fmt_vec(xy, 12)}"')
        assert _same(mx.xyaxes2q(xy), qm)
        z = RNG.normal(size=3)
        qm = _compiled_body_quat(f'zaxis="{mx.fmt_vec(z, 12)}"')
        assert _same(mx.zaxis2q(z), qm)
    # special: zaxis anti-parallel to +z
    qm = _compiled_body_quat('zaxis="0 0 -1"')
    assert _same(mx.zaxis2q([0, 0, -1]), qm)


def test_mat_quat_roundtrip():
    for _ in range(50):
        q = mx.qcanon(RNG.normal(size=4))
        assert _same(mx.mat2q(mx.q2mat(q)), q, 1e-9)
        v = RNG.normal(size=3)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q)
        assert np.allclose(R.reshape(3, 3) @ v, mx.qrot(q, v))


def test_fmt():
    assert mx.fmt_num(-0.0) == "0"
    assert mx.fmt_num(1.0) == "1"
    assert mx.fmt_num(0.1 + 0.2) == "0.3"
    assert mx.fmt_vec([1, 2.5, -3]) == "1 2.5 -3"
    assert mx.parse_vec("1 2  3").tolist() == [1, 2, 3]
    assert mx.parse_vec("1 a") is None
