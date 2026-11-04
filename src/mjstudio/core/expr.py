"""Facial expression rigs: shape keys + control bones on a MuJoCo <skin>.

Physics and appearance are separated:
  * eyes and jaw are MuJoCo joints (simulated, actuated, sensed),
  * the face surface is an MJCF <skin> bound to the head and jaw bodies, so plain
    MuJoCo already shows it following the head and the jaw,
  * expressions (shape keys = blend shapes, and control bones = method A) are
    evaluated by this module and written into ``MjvScene.skinvert`` right before
    rendering. The head/jaw poses come from the same MjData, so the face is in sync
    with the simulation time; time-varying expression tracks use ``data.time``.

All rig geometry is stored in the head body's local frame (x forward, y left, z up).
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mujoco
import numpy as np
from lxml import etree

from . import mathx as mx

OBJ = mujoco.mjtObj

# expression presets (parameter -> value); presets referring to missing parameters skip them
DEFAULT_PRESETS = {
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


@dataclass
class ControlBone:
    """A virtual bone (not a MuJoCo body) that rotates/translates weighted vertices."""
    name: str
    pivot: list                      # head-local position
    weights: np.ndarray = field(repr=False, default=None)   # (V,) influence 0..1


@dataclass
class ExprParam:
    """value in [lo, hi] -> shape key weights and bone motions (linear in value)."""
    name: str
    keys: dict = field(default_factory=dict)      # shape key -> coefficient at value 1
    bones: dict = field(default_factory=dict)     # bone -> {"t": [x,y,z] m, "r": [rx,ry,rz] rotation vector rad}
    lo: float = 0.0
    hi: float = 1.0
    group: str = ""


@dataclass
class Track:
    """Piecewise-linear expression curve over simulation time."""
    param: str
    times: list
    values: list

    def at(self, t: float) -> float:
        return float(np.interp(t, self.times, self.values))


class FaceRig:
    def __init__(self, name: str, V0, F, head_body: str = "head", jaw_body: str | None = "jaw",
                 jaw_weights=None, skin: str | None = None):
        self.name = name
        self.skin = skin or f"{name}_skin"
        self.head_body, self.jaw_body = head_body, jaw_body
        self.V0 = np.asarray(V0, float).reshape(-1, 3)
        self.F = np.asarray(F, np.int64).reshape(-1, 3)
        n = len(self.V0)
        self.jaw_w = np.zeros(n) if jaw_weights is None else np.clip(np.asarray(jaw_weights, float), 0, 1)
        self.keys: dict[str, np.ndarray] = {}
        self.bones: dict[str, ControlBone] = {}
        self.params: dict[str, ExprParam] = {}
        self.presets: dict[str, dict[str, float]] = {}
        self.values: dict[str, float] = {}
        self.tracks: list[Track] = []
        self.rgba = [0.85, 0.67, 0.55, 1.0]
        self.meta: dict = {}
        self._cache = None

    # ------------------------------------------------------------- definition
    def add_key(self, name: str, delta) -> None:
        d = np.asarray(delta, float).reshape(-1, 3)
        if d.shape != self.V0.shape:
            raise ValueError(f"Shape Key '{name}'의 정점 수({len(d)})가 메시({len(self.V0)})와 다릅니다")
        self.keys[name] = d

    def add_bone(self, name: str, pivot, weights) -> None:
        w = np.clip(np.asarray(weights, float).reshape(-1), 0, 1)
        if len(w) != len(self.V0):
            raise ValueError("본 가중치 길이가 맞지 않습니다")
        self.bones[name] = ControlBone(name, list(map(float, pivot)), w)

    def add_param(self, p: ExprParam) -> None:
        for k in p.keys:
            if k not in self.keys:
                raise KeyError(f"파라미터 '{p.name}'이(가) 없는 Shape Key '{k}'을(를) 씁니다")
        for b in p.bones:
            if b not in self.bones:
                raise KeyError(f"파라미터 '{p.name}'이(가) 없는 본 '{b}'을(를) 씁니다")
        self.params[p.name] = p
        self.values.setdefault(p.name, 0.0)

    # ------------------------------------------------------------- control
    def set(self, name: str, value: float) -> None:
        """Set an expression parameter, or apply a preset by name (value = amount)."""
        if name in self.params:
            p = self.params[name]
            self.values[name] = float(np.clip(value, p.lo, p.hi))
        elif name in self.presets:
            self.apply_preset(name, value)
        else:
            raise KeyError(f"알 수 없는 표정 '{name}'")

    def get(self, name: str) -> float:
        return self.values.get(name, 0.0)

    def reset(self) -> None:
        for k in self.values:
            self.values[k] = 0.0

    def apply_preset(self, name: str, amount: float = 1.0, additive: bool = False) -> None:
        if not additive:
            self.reset()
        for k, v in self.presets[name].items():
            if k in self.params:
                p = self.params[k]
                self.values[k] = float(np.clip(self.values.get(k, 0.0) + amount * v if additive else amount * v,
                                               p.lo, p.hi))

    def evaluate_tracks(self, t: float) -> None:
        for tr in self.tracks:
            if tr.param in self.params:
                self.set(tr.param, tr.at(t))

    # ------------------------------------------------------------- deformation
    def key_weights(self) -> dict[str, float]:
        w: dict[str, float] = {}
        for pn, v in self.values.items():
            if v == 0.0 or pn not in self.params:
                continue
            for k, c in self.params[pn].keys.items():
                w[k] = w.get(k, 0.0) + c * v
        return w

    def local_vertices(self) -> np.ndarray:
        """Deformed vertices in the head frame (jaw not applied)."""
        V = self.V0.copy()
        for k, w in self.key_weights().items():
            if w:
                V += w * self.keys[k]
        motions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for pn, v in self.values.items():
            if v == 0.0 or pn not in self.params:
                continue
            for b, mot in self.params[pn].bones.items():
                t, r = motions.get(b, (np.zeros(3), np.zeros(3)))
                motions[b] = (t + v * np.asarray(mot.get("t", [0, 0, 0]), float),
                              r + v * np.asarray(mot.get("r", [0, 0, 0]), float))
        if motions:
            base = V.copy()
            for b, (t, r) in motions.items():
                bone = self.bones[b]
                ang = float(np.linalg.norm(r))
                R = mx.q2mat(mx.axang2q(r / ang, ang)) if ang > 1e-12 else np.eye(3)
                piv = np.asarray(bone.pivot)
                moved = (base - piv) @ R.T + piv + t
                V += bone.weights[:, None] * (moved - base)
        return V

    def _jaw_rest(self, m: mujoco.MjModel):
        """Jaw body pose relative to the head body at qpos0 (bind pose)."""
        hid = mujoco.mj_name2id(m, OBJ.mjOBJ_BODY, self.head_body)
        jid = mujoco.mj_name2id(m, OBJ.mjOBJ_BODY, self.jaw_body) if self.jaw_body else -1
        if hid < 0 or jid < 0:
            return None
        d0 = mujoco.MjData(m)
        mujoco.mj_kinematics(m, d0)
        ip, iq = mx.pose_inv(d0.xpos[hid], d0.xquat[hid])
        return mx.pose_mul(ip, iq, d0.xpos[jid], d0.xquat[jid])

    def world_vertices(self, m: mujoco.MjModel, d: mujoco.MjData, local: np.ndarray | None = None) -> np.ndarray:
        if self._cache is None or self._cache[0] is not m:
            self._cache = (m, self._jaw_rest(m))
        jr = self._cache[1]
        V = self.local_vertices() if local is None else local
        hid = mujoco.mj_name2id(m, OBJ.mjOBJ_BODY, self.head_body)
        if hid < 0:
            raise KeyError(f"머리 바디 '{self.head_body}'을(를) 찾을 수 없습니다")
        Rh = d.xmat[hid].reshape(3, 3)
        W = V @ Rh.T + d.xpos[hid]
        if jr is not None and self.jaw_w.any():
            jid = mujoco.mj_name2id(m, OBJ.mjOBJ_BODY, self.jaw_body)
            Rjr = mx.q2mat(jr[1])
            vj = (V - jr[0]) @ Rjr             # head-local -> jaw-local at bind
            Wj = vj @ d.xmat[jid].reshape(3, 3).T + d.xpos[jid]
            w = self.jaw_w[:, None]
            W = (1 - w) * W + w * Wj
        return W

    def normals(self, W: np.ndarray) -> np.ndarray:
        a, b, c = W[self.F[:, 0]], W[self.F[:, 1]], W[self.F[:, 2]]
        fn = np.cross(b - a, c - a)
        n = np.zeros_like(W)
        for k in range(3):
            for ax in range(3):
                n[:, ax] += np.bincount(self.F[:, k], weights=fn[:, ax], minlength=len(W))
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        return n / np.maximum(ln, 1e-15)

    def write_scene(self, m: mujoco.MjModel, d: mujoco.MjData, scn: mujoco.MjvScene) -> bool:
        """Overwrite this rig's skin vertices/normals in an updated MjvScene."""
        sid = mujoco.mj_name2id(m, OBJ.mjOBJ_SKIN, self.skin)
        if sid < 0 or sid >= scn.nskin:
            return False
        n = int(scn.skinvertnum[sid])
        if n != len(self.V0):
            return False
        a = int(scn.skinvertadr[sid])
        W = self.world_vertices(m, d)
        scn.skinvert[3 * a:3 * (a + n)] = W.astype(np.float32).ravel()
        scn.skinnormal[3 * a:3 * (a + n)] = self.normals(W).astype(np.float32).ravel()
        return True

    # ------------------------------------------------------------- MJCF
    def mjcf_skin(self, m: mujoco.MjModel | None = None, jaw_rest=None, material: str | None = None,
                  group: int = 1) -> etree._Element:
        """<skin> element: rest vertices in the head frame, bones = head (+ jaw)."""
        e = etree.Element("skin", name=self.skin)
        e.set("rgba", mx.fmt_vec(self.rgba, 4))
        if material:
            e.set("material", material)
        e.set("group", str(group))
        e.set("vertex", mx.fmt_vec(np.round(self.V0, 6).ravel(), 7))
        e.set("face", " ".join(map(str, self.F.ravel())))
        hw = 1.0 - self.jaw_w
        ids = np.nonzero(hw > 1e-6)[0]
        b = etree.SubElement(e, "bone", body=self.head_body, bindpos="0 0 0", bindquat="1 0 0 0")
        b.set("vertid", " ".join(map(str, ids)))
        b.set("vertweight", mx.fmt_vec(hw[ids], 6))
        if self.jaw_body and self.jaw_w.any():
            if jaw_rest is None:
                if m is None:
                    raise ValueError("턱 기준 자세가 필요합니다 (컴파일된 모델 또는 jaw_rest 전달)")
                jaw_rest = self._jaw_rest(m)
            ids = np.nonzero(self.jaw_w > 1e-6)[0]
            jb = etree.SubElement(e, "bone", body=self.jaw_body, bindpos=mx.fmt_vec(jaw_rest[0]),
                                  bindquat=mx.fmt_vec(jaw_rest[1]))
            jb.set("vertid", " ".join(map(str, ids)))
            jb.set("vertweight", mx.fmt_vec(self.jaw_w[ids], 6))
        return e

    def skn_bytes(self, jaw_rest=None) -> bytes:
        """MuJoCo .skn skin file (format of mjCSkin::LoadSKN, MuJoCo 3.0.x): rest vertices in
        the head frame; bones: head (bind = identity) and jaw (bind = jaw pose in head frame)."""
        import struct
        bones = [(self.head_body, np.zeros(3), mx.QID, 1.0 - self.jaw_w)]
        if self.jaw_body and self.jaw_w.any():
            if jaw_rest is None:
                raise ValueError("턱 기준 자세가 필요합니다")
            bones.append((self.jaw_body, np.asarray(jaw_rest[0], float), np.asarray(jaw_rest[1], float), self.jaw_w))
        out = [struct.pack("<4i", len(self.V0), 0, len(self.F), len(bones)),
               self.V0.astype("<f4").tobytes(), self.F.astype("<i4").tobytes()]
        for name, bp, bq, w in bones:
            ids = np.nonzero(w > 1e-6)[0]
            out.append(name.encode("utf-8")[:39].ljust(40, b"\0"))
            out.append(np.asarray(bp, "<f4").tobytes() + np.asarray(bq, "<f4").tobytes())
            out.append(struct.pack("<i", len(ids)) + ids.astype("<i4").tobytes() + w[ids].astype("<f4").tobytes())
        return b"".join(out)

    # ------------------------------------------------------------- persistence
    def to_json(self, data_file: str) -> dict:
        return {"name": self.name, "skin": self.skin, "head_body": self.head_body, "jaw_body": self.jaw_body,
                "data": data_file, "rgba": self.rgba, "meta": self.meta,
                "bones": [{"name": b.name, "pivot": b.pivot} for b in self.bones.values()],
                "params": [asdict(p) for p in self.params.values()],
                "presets": self.presets, "values": self.values,
                "tracks": [asdict(t) for t in self.tracks]}

    def save(self, folder: str | Path) -> dict:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        arrays = {"V0": self.V0, "F": self.F, "jaw_w": self.jaw_w}
        for i, (k, d) in enumerate(self.keys.items()):
            arrays[f"key{i}"] = d
        for i, (k, b) in enumerate(self.bones.items()):
            arrays[f"bone{i}"] = b.weights
        arrays["key_names"] = np.array(list(self.keys.keys()), dtype=object)
        arrays["bone_names"] = np.array(list(self.bones.keys()), dtype=object)
        fn = folder / f"{self.name}.npz"
        np.savez_compressed(fn, **{k: (v.astype(str) if v.dtype == object else v) for k, v in arrays.items()})
        return self.to_json(fn.name)

    @classmethod
    def load(cls, meta: dict, folder: str | Path) -> "FaceRig":
        z = np.load(Path(folder) / meta["data"], allow_pickle=False)
        r = cls(meta["name"], z["V0"], z["F"], meta.get("head_body", "head"), meta.get("jaw_body"),
                z["jaw_w"], meta.get("skin"))
        r.rgba = meta.get("rgba", r.rgba)
        r.meta = meta.get("meta", {})
        for i, k in enumerate(z["key_names"].tolist()):
            r.keys[k] = z[f"key{i}"]
        piv = {b["name"]: b["pivot"] for b in meta.get("bones", [])}
        for i, k in enumerate(z["bone_names"].tolist()):
            r.bones[k] = ControlBone(k, piv.get(k, [0, 0, 0]), z[f"bone{i}"])
        for p in meta.get("params", []):
            r.params[p["name"]] = ExprParam(**p)
        r.presets = meta.get("presets", {})
        r.values = {k: float(v) for k, v in meta.get("values", {}).items()}
        for p in r.params:
            r.values.setdefault(p, 0.0)
        r.tracks = [Track(**t) for t in meta.get("tracks", [])]
        return r


# ===================================================================== controller API
class FaceController:
    """External control API (also used by the GUI):

        fc = FaceController(model, data, rig)
        fc.set_expression("smile", 0.8)
        fc.set_expression("blink_left", 1.0)
        fc.set_eye_rotation("left", yaw, pitch)     # radians; joint targets
        fc.set_jaw_open(0.5)                          # fraction of the jaw range
    Eye and jaw commands drive position actuators when present (simulation), otherwise
    they set the joint positions directly (kinematic preview)."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, rig: FaceRig | None,
                 eye_joints: dict | None = None, jaw_joint: str = "jaw_open", kinematic: bool = False):
        self.m, self.d, self.rig = model, data, rig
        self.kinematic = kinematic   # also set joint positions directly (editing without simulation)
        self.eye_joints = eye_joints or {s: (f"eye_{s}_yaw", f"eye_{s}_pitch") for s in ("left", "right")}
        self.jaw_joint = jaw_joint

    # -- expressions
    def set_expression(self, name: str, value: float) -> None:
        if self.rig is None:
            raise RuntimeError("얼굴 리그가 없습니다")
        self.rig.set(name, value)

    def expressions(self) -> list[str]:
        return list(self.rig.params) + list(self.rig.presets) if self.rig else []

    # -- joints
    def _jid(self, name: str) -> int:
        j = mujoco.mj_name2id(self.m, OBJ.mjOBJ_JOINT, name)
        if j < 0:
            raise KeyError(f"관절 '{name}'을(를) 찾을 수 없습니다")
        return j

    def _drive(self, joint: str, q: float) -> float:
        j = self._jid(joint)
        if self.m.jnt_limited[j]:
            q = float(np.clip(q, *self.m.jnt_range[j]))
        acts = [i for i in range(self.m.nu) if self.m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT
                and self.m.actuator_trnid[i, 0] == j]
        if acts:
            self.d.ctrl[acts[0]] = q
        if not acts or self.kinematic:
            self.d.qpos[self.m.jnt_qposadr[j]] = q
        return q

    def set_eye_rotation(self, side: str, yaw: float, pitch: float) -> tuple[float, float]:
        yj, pj = self.eye_joints[side]
        return self._drive(yj, yaw), self._drive(pj, pitch)

    def set_gaze(self, yaw: float, pitch: float) -> None:
        for s in self.eye_joints:
            self.set_eye_rotation(s, yaw, pitch)

    def set_jaw_open(self, amount: float) -> float:
        j = self._jid(self.jaw_joint)
        lo, hi = self.m.jnt_range[j] if self.m.jnt_limited[j] else (0.0, math.radians(30))
        return self._drive(self.jaw_joint, lo + float(np.clip(amount, 0, 1)) * (hi - lo))

    def eye_rotation(self, side: str) -> tuple[float, float]:
        yj, pj = self.eye_joints[side]
        return (float(self.d.qpos[self.m.jnt_qposadr[self._jid(yj)]]),
                float(self.d.qpos[self.m.jnt_qposadr[self._jid(pj)]]))

    def eye_direction(self, side: str, site: str | None = None) -> np.ndarray:
        s = mujoco.mj_name2id(self.m, OBJ.mjOBJ_SITE, site or f"gaze_{side}")
        return self.d.site_xmat[s].reshape(3, 3)[:, 0].copy()

    def jaw_open(self) -> float:
        return float(self.d.qpos[self.m.jnt_qposadr[self._jid(self.jaw_joint)]])
