"""Minimal glTF 2.0 / GLB reader (and writer) for meshes, shape keys and skins.

Supports what Blender's glTF exporter produces for characters: triangle
primitives, node transforms, morph targets (dense or *sparse* accessors) with
names in ``mesh.extras.targetNames``, skins (JOINTS_0/WEIGHTS_0 + inverse bind
matrices). Coordinates are returned in glTF space (Y up, +Z front); use
``meshes.AXES['yup']`` to convert to MuJoCo (Z up, +X front).
"""
from __future__ import annotations

import base64
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import mathx as mx

_CT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}


@dataclass
class GltfMesh:
    name: str
    V: np.ndarray                         # (n,3) in the mesh node's local frame
    F: np.ndarray                         # (m,3)
    N: np.ndarray | None = None
    morph: dict[str, np.ndarray] = field(default_factory=dict)   # name -> (n,3) position deltas
    morph_default: dict[str, float] = field(default_factory=dict)
    joints: np.ndarray | None = None      # (n,4) indices into joint_names
    weights: np.ndarray | None = None     # (n,4)
    joint_names: list[str] = field(default_factory=list)
    inverse_bind: np.ndarray | None = None  # (J,4,4)
    node: str = ""
    world: np.ndarray = field(default_factory=lambda: np.eye(4))  # node world matrix


@dataclass
class GltfNode:
    name: str
    parent: int
    local: np.ndarray
    world: np.ndarray


@dataclass
class GltfScene:
    meshes: list[GltfMesh]
    nodes: list[GltfNode]

    def node(self, name: str) -> GltfNode | None:
        return next((n for n in self.nodes if n.name == name), None)


def _trs(n: dict) -> np.ndarray:
    if "matrix" in n:
        return np.array(n["matrix"], float).reshape(4, 4).T
    M = np.eye(4)
    t = n.get("translation", [0, 0, 0])
    r = n.get("rotation", [0, 0, 0, 1])          # glTF quaternion order: x y z w
    s = n.get("scale", [1, 1, 1])
    M[:3, :3] = mx.q2mat([r[3], r[0], r[1], r[2]]) @ np.diag(s)
    M[:3, 3] = t
    return M


class _Reader:
    def __init__(self, gj: dict, bins: list[bytes]):
        self.g, self.bins = gj, bins

    def view(self, i: int) -> tuple[bytes, int, int]:
        bv = self.g["bufferViews"][i]
        return self.bins[bv["buffer"]], bv.get("byteOffset", 0), bv.get("byteStride", 0)

    def _raw(self, view, offset, ctype, ncomp, count, stride):
        buf, voff, vstride = self.view(view)
        dt = np.dtype(_CT[ctype])
        esz = dt.itemsize * ncomp
        stride = stride or vstride or esz
        start = voff + offset
        if stride == esz:
            return np.frombuffer(buf, dt, count * ncomp, start).reshape(count, ncomp).copy()
        out = np.empty((count, ncomp), dt)
        for k in range(count):
            out[k] = np.frombuffer(buf, dt, ncomp, start + k * stride)
        return out

    def accessor(self, i: int) -> np.ndarray:
        a = self.g["accessors"][i]
        nc, cnt, ct = _NC[a["type"]], a["count"], a["componentType"]
        if "bufferView" in a:
            arr = self._raw(a["bufferView"], a.get("byteOffset", 0), ct, nc, cnt, 0).astype(float if ct == 5126 else np.int64)
        else:
            arr = np.zeros((cnt, nc), float if ct == 5126 else np.int64)
        if a.get("normalized") and ct != 5126:
            arr = arr / float(np.iinfo(_CT[ct]).max)
        sp = a.get("sparse")
        if sp:
            idx = self._raw(sp["indices"]["bufferView"], sp["indices"].get("byteOffset", 0),
                            sp["indices"]["componentType"], 1, sp["count"], 0).ravel().astype(np.int64)
            val = self._raw(sp["values"]["bufferView"], sp["values"].get("byteOffset", 0), ct, nc, sp["count"], 0)
            arr = arr.astype(float) if ct == 5126 else arr
            arr[idx] = val
        return arr


def load(path: str | Path) -> GltfScene:
    p = Path(path)
    data = p.read_bytes()
    if data[:4] == b"glTF":
        _, _ver, length = struct.unpack_from("<4sII", data, 0)
        off, gj, bins = 12, None, []
        while off < length:
            clen, ctype = struct.unpack_from("<I4s", data, off)
            chunk = data[off + 8: off + 8 + clen]
            if ctype == b"JSON":
                gj = json.loads(chunk.decode("utf-8"))
            elif ctype == b"BIN\x00":
                bins.append(chunk)
            off += 8 + clen
        buffers = []
        for k, b in enumerate(gj.get("buffers", [])):
            if "uri" not in b:
                buffers.append(bins[0] if bins else b"")
            else:
                buffers.append(_uri(b["uri"], p.parent))
    else:
        gj = json.loads(data.decode("utf-8"))
        buffers = [_uri(b["uri"], p.parent) for b in gj.get("buffers", [])]
    return _parse(gj, buffers)


def _uri(uri: str, base: Path) -> bytes:
    if uri.startswith("data:"):
        return base64.b64decode(uri.split(",", 1)[1])
    return (base / uri).read_bytes()


def _parse(gj: dict, buffers: list[bytes]) -> GltfScene:
    R = _Reader(gj, buffers)
    nodes_j = gj.get("nodes", [])
    parent = [-1] * len(nodes_j)
    for i, n in enumerate(nodes_j):
        for c in n.get("children", []):
            parent[c] = i
    local = [_trs(n) for n in nodes_j]
    world: list = [None] * len(nodes_j)

    def wm(i):
        if world[i] is None:
            world[i] = local[i] if parent[i] < 0 else wm(parent[i]) @ local[i]
        return world[i]

    nodes = [GltfNode(n.get("name", f"node{i}"), parent[i], local[i], wm(i)) for i, n in enumerate(nodes_j)]
    out: list[GltfMesh] = []
    for ni, n in enumerate(nodes_j):
        if "mesh" not in n:
            continue
        mj = gj["meshes"][n["mesh"]]
        names = (mj.get("extras") or {}).get("targetNames", [])
        Vs, Fs, Ns, J, W = [], [], [], [], []
        morph: dict[str, list] = {}
        base = 0
        for prim in mj["primitives"]:
            if prim.get("mode", 4) != 4:
                continue
            at = prim["attributes"]
            V = R.accessor(at["POSITION"])
            Vs.append(V)
            if "indices" in prim:
                Fs.append(R.accessor(prim["indices"]).reshape(-1, 3) + base)
            else:
                Fs.append(np.arange(len(V)).reshape(-1, 3) + base)
            Ns.append(R.accessor(at["NORMAL"]) if "NORMAL" in at else np.zeros_like(V))
            if "JOINTS_0" in at:
                J.append(R.accessor(at["JOINTS_0"]).astype(np.int64))
                W.append(R.accessor(at["WEIGHTS_0"]).astype(float))
            for k, t in enumerate(prim.get("targets", [])):
                nm = names[k] if k < len(names) else f"target{k}"
                d = R.accessor(t["POSITION"]) if "POSITION" in t else np.zeros_like(V)
                morph.setdefault(nm, []).append(d)
            base += len(V)
        if not Vs:
            continue
        gm = GltfMesh(mj.get("name", n.get("name", "mesh")), np.vstack(Vs), np.vstack(Fs).astype(np.int64),
                      np.vstack(Ns), {k: np.vstack(v) for k, v in morph.items() if len(v) == len(Vs)},
                      node=n.get("name", ""), world=wm(ni))
        defaults = mj.get("weights", [])
        gm.morph_default = {names[k] if k < len(names) else f"target{k}": float(w) for k, w in enumerate(defaults)}
        if J and len(J) == len(Vs) and "skin" in n:
            sk = gj["skins"][n["skin"]]
            gm.joints, gm.weights = np.vstack(J), np.vstack(W)
            gm.joint_names = [nodes_j[j].get("name", f"node{j}") for j in sk["joints"]]
            if "inverseBindMatrices" in sk:
                ib = R.accessor(sk["inverseBindMatrices"]).reshape(-1, 4, 4)
                gm.inverse_bind = np.transpose(ib, (0, 2, 1))  # column-major -> row-major
        out.append(gm)
    return GltfScene(out, nodes)


def to_mujoco(points: np.ndarray, world: np.ndarray | None = None) -> np.ndarray:
    """glTF (Y up, +Z front) points -> MuJoCo frame (Z up, +X front), optionally applying
    a node world matrix first."""
    P = np.asarray(points, float)
    if world is not None:
        P = P @ world[:3, :3].T + world[:3, 3]
    return P @ np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float).T


def vec_to_mujoco(vectors: np.ndarray, world: np.ndarray | None = None) -> np.ndarray:
    """Like to_mujoco for displacement vectors (no translation)."""
    D = np.asarray(vectors, float)
    if world is not None:
        D = D @ world[:3, :3].T
    return D @ np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float).T


# ---------------------------------------------------------------------- writer
def save_glb(path: str | Path, V: np.ndarray, F: np.ndarray, morph: dict[str, np.ndarray] | None = None,
             name: str = "mesh") -> None:
    """Write a single-mesh GLB (positions, indices, dense morph targets with names).
    Coordinates are written as given (pass glTF-space data)."""
    V = np.asarray(V, np.float32)
    F = np.asarray(F, np.uint32).ravel()
    morph = morph or {}
    blobs, views, accs = [], [], []

    def add(arr: np.ndarray, ctype: int, typ: str, target: int | None, minmax: bool = False):
        b = arr.tobytes()
        off = sum(len(x) for x in blobs)
        pad = (-len(b)) % 4
        blobs.append(b + b"\x00" * pad)
        v = {"buffer": 0, "byteOffset": off, "byteLength": len(b)}
        if target:
            v["target"] = target
        views.append(v)
        a = {"bufferView": len(views) - 1, "componentType": ctype, "count": int(arr.shape[0]), "type": typ}
        if minmax:
            a["min"] = arr.min(axis=0).tolist()
            a["max"] = arr.max(axis=0).tolist()
        accs.append(a)
        return len(accs) - 1

    pos = add(V, 5126, "VEC3", 34962, True)
    idx = add(F.reshape(-1, 1).astype(np.uint32), 5125, "SCALAR", 34963)
    accs[idx]["count"] = int(F.size)
    targets = [{"POSITION": add(np.asarray(d, np.float32), 5126, "VEC3", 34962, True)} for d in morph.values()]
    prim = {"attributes": {"POSITION": pos}, "indices": idx, "mode": 4}
    if targets:
        prim["targets"] = targets
    mesh = {"name": name, "primitives": [prim]}
    if morph:
        mesh["weights"] = [0.0] * len(morph)
        mesh["extras"] = {"targetNames": list(morph.keys())}
    gj = {"asset": {"version": "2.0", "generator": "MuJoCo Model Studio"}, "scene": 0,
          "scenes": [{"nodes": [0]}], "nodes": [{"name": name, "mesh": 0}], "meshes": [mesh],
          "buffers": [{"byteLength": sum(len(x) for x in blobs)}], "bufferViews": views, "accessors": accs}
    js = json.dumps(gj).encode("utf-8")
    js += b" " * ((-len(js)) % 4)
    binb = b"".join(blobs)
    total = 12 + 8 + len(js) + 8 + len(binb)
    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, total))
        f.write(struct.pack("<I4s", len(js), b"JSON") + js)
        f.write(struct.pack("<I4s", len(binb), b"BIN\x00") + binb)
