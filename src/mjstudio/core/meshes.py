"""Triangle meshes: primitives, OBJ/STL I/O, simple editing, unit/axis conversion.

Only numpy is used. Meshes imported from files are referenced *as files* by the
MJCF (units/axes go to the <mesh scale> and the geom orientation), so UVs and
normals of the original file are kept; a mesh is re-encoded (as OBJ) only when it
is generated or edited in the editor.
"""
from __future__ import annotations

import io
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import mathx as mx

UNITS = {"m": 1.0, "cm": 0.01, "mm": 0.001, "in": 0.0254, "ft": 0.3048}

# axis presets: rotation (quat) taking file coordinates to MuJoCo (x forward, y left, z up)
#  - "mujoco":           identity
#  - "blender":          Blender scene axes (Z up, character facing -Y)  -> rotate +90 deg about Z
#  - "yup":              Y-up files facing +Z (glTF convention, OBJ/STL exported with "Y up")
AXES = {
    "mujoco": mx.QID.copy(),
    "blender": mx.axang2q((0, 0, 1), math.pi / 2),
    "yup": mx.mat2q(np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)),
}


@dataclass
class TriMesh:
    V: np.ndarray                       # (n, 3) float
    F: np.ndarray                       # (m, 3) int
    name: str = "mesh"
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.V = np.asarray(self.V, float).reshape(-1, 3)
        self.F = np.asarray(self.F, np.int64).reshape(-1, 3)

    # ------------------------------------------------------------ basics
    def copy(self, name: str | None = None) -> "TriMesh":
        return TriMesh(self.V.copy(), self.F.copy(), name or self.name, dict(self.meta))

    @property
    def nv(self) -> int:
        return len(self.V)

    @property
    def nf(self) -> int:
        return len(self.F)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.V.min(axis=0), self.V.max(axis=0)

    def face_normals(self, unit: bool = True) -> np.ndarray:
        a, b, c = (self.V[self.F[:, k]] for k in range(3))
        n = np.cross(b - a, c - a)
        if unit:
            ln = np.linalg.norm(n, axis=1, keepdims=True)
            n = n / np.maximum(ln, 1e-15)
        return n

    def vertex_normals(self) -> np.ndarray:
        fn = self.face_normals(unit=False)  # area weighted
        vn = np.zeros_like(self.V)
        for k in range(3):
            for ax in range(3):
                vn[:, ax] += np.bincount(self.F[:, k], weights=fn[:, ax], minlength=self.nv)
        ln = np.linalg.norm(vn, axis=1, keepdims=True)
        return vn / np.maximum(ln, 1e-15)

    def volume(self) -> float:
        a, b, c = (self.V[self.F[:, k]] for k in range(3))
        return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)

    def is_planar(self, tol: float = 1e-9) -> bool:
        if self.nv < 4:
            return True
        c = self.V - self.V.mean(axis=0)
        s = np.linalg.svd(c, compute_uv=False)
        return bool(s[-1] <= tol * max(s[0], 1e-12))

    def validate(self) -> list[str]:
        issues = []
        if self.nv == 0 or self.nf == 0:
            issues.append("빈 메시")
            return issues
        if self.nv < 4:
            issues.append("MuJoCo 메시는 정점이 4개 이상이어야 합니다")
        if self.F.min() < 0 or self.F.max() >= self.nv:
            issues.append("면의 정점 번호가 범위를 벗어남")
        deg = np.sum((self.F[:, 0] == self.F[:, 1]) | (self.F[:, 1] == self.F[:, 2]) | (self.F[:, 0] == self.F[:, 2]))
        if deg:
            issues.append(f"면적이 0인 면 {int(deg)}개")
        return issues

    # ------------------------------------------------------------ transforms
    def transformed(self, scale=1.0, quat=None, pos=None) -> "TriMesh":
        V = self.V * np.asarray(scale, float)
        if quat is not None:
            V = V @ mx.q2mat(quat).T
        if pos is not None:
            V = V + np.asarray(pos, float)
        out = self.copy()
        out.V = V
        if np.prod(np.sign(np.broadcast_to(np.asarray(scale, float), (3,)))) < 0:
            out.F = out.F[:, ::-1].copy()  # mirroring flips the winding
        return out

    # ------------------------------------------------------------ editing
    def move_vertices(self, idx, delta) -> None:
        self.V[np.asarray(idx, int)] += np.asarray(delta, float)

    def add_vertex(self, p) -> int:
        self.V = np.vstack([self.V, np.asarray(p, float).reshape(1, 3)])
        return self.nv - 1

    def add_face(self, i: int, j: int, k: int) -> int:
        if len({i, j, k}) < 3 or max(i, j, k) >= self.nv or min(i, j, k) < 0:
            raise ValueError("면에는 서로 다른 기존 정점 3개가 필요합니다")
        self.F = np.vstack([self.F, np.array([[i, j, k]], np.int64)])
        return self.nf - 1

    def poke_face(self, f: int, p=None) -> int:
        """Insert a vertex inside face f (default: centroid) and split the face into three
        (keeps the winding consistent). Returns the new vertex index."""
        a, b, c = (int(x) for x in self.F[f])
        if p is None:
            p = self.V[[a, b, c]].mean(axis=0)
        i = self.add_vertex(p)
        self.F[f] = [a, b, i]
        self.F = np.vstack([self.F, np.array([[b, c, i], [c, a, i]], np.int64)])
        return i

    def delete_faces(self, fidx) -> None:
        keep = np.ones(self.nf, bool)
        keep[np.asarray(fidx, int)] = False
        self.F = self.F[keep]

    def delete_vertices(self, idx) -> None:
        """Remove vertices and every face using them; indices are compacted."""
        idx = np.unique(np.asarray(idx, int))
        rm = np.zeros(self.nv, bool)
        rm[idx] = True
        self.F = self.F[~rm[self.F].any(axis=1)]
        remap = -np.ones(self.nv, np.int64)
        remap[~rm] = np.arange(int((~rm).sum()))
        self.V = self.V[~rm]
        self.F = remap[self.F]

    def remove_unused(self) -> None:
        used = np.zeros(self.nv, bool)
        used[self.F.ravel()] = True
        self.delete_vertices(np.nonzero(~used)[0])

    def flip(self) -> None:
        self.F = self.F[:, ::-1].copy()

    def weld(self, tol: float = 1e-9) -> None:
        """Merge vertices closer than tol (STL files store one vertex per face corner)."""
        if self.nv == 0:
            return
        q = np.round(self.V / max(tol, 1e-15)).astype(np.int64)
        _, first, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
        self.V = self.V[first]
        self.F = inv.reshape(-1)[self.F]
        good = (self.F[:, 0] != self.F[:, 1]) & (self.F[:, 1] != self.F[:, 2]) & (self.F[:, 0] != self.F[:, 2])
        self.F = self.F[good]

    def pick_face(self, origin, direction) -> tuple[int, float]:
        """Ray/triangle intersection (Moller-Trumbore). Returns (face, distance) or (-1, inf)."""
        o, d = np.asarray(origin, float), np.asarray(direction, float)
        a, b, c = (self.V[self.F[:, k]] for k in range(3))
        e1, e2 = b - a, c - a
        p = np.cross(d, e2)
        det = np.einsum("ij,ij->i", e1, p)
        ok = np.abs(det) > 1e-14
        inv = np.where(ok, 1.0 / np.where(ok, det, 1), 0)
        t0 = o - a
        u = np.einsum("ij,ij->i", t0, p) * inv
        q = np.cross(t0, e1)
        v = (q @ d) * inv
        t = np.einsum("ij,ij->i", e2, q) * inv
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
        if not hit.any():
            return -1, math.inf
        t = np.where(hit, t, np.inf)
        f = int(np.argmin(t))
        return f, float(t[f])


# ======================================================================== primitives
def box(hx=0.05, hy=0.05, hz=0.05) -> TriMesh:
    V = np.array([[sx * hx, sy * hy, sz * hz] for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)], float)
    F = [[0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6], [0, 1, 4], [1, 5, 4],
         [2, 6, 3], [3, 6, 7], [0, 4, 2], [2, 4, 6], [1, 3, 5], [3, 7, 5]]
    return TriMesh(V, F, "box")


def uv_sphere(r=0.05, nu=24, nv=12, radii=None) -> TriMesh:
    rad = np.array(radii if radii is not None else (r, r, r), float)
    V = [[0, 0, -1.0]]
    for i in range(1, nv):
        ph = -math.pi / 2 + math.pi * i / nv
        for j in range(nu):
            th = 2 * math.pi * j / nu
            V.append([math.cos(ph) * math.cos(th), math.cos(ph) * math.sin(th), math.sin(ph)])
    V.append([0, 0, 1.0])
    V = np.array(V) * rad
    F = []
    top = len(V) - 1
    for j in range(nu):
        F.append([0, 1 + (j + 1) % nu, 1 + j])
    for i in range(nv - 2):
        r0, r1 = 1 + i * nu, 1 + (i + 1) * nu
        for j in range(nu):
            a, b, c, d = r0 + j, r0 + (j + 1) % nu, r1 + j, r1 + (j + 1) % nu
            F += [[a, b, d], [a, d, c]]
    r0 = 1 + (nv - 2) * nu
    for j in range(nu):
        F.append([r0 + j, r0 + (j + 1) % nu, top])
    return TriMesh(V, F, "sphere")


def ellipsoid(rx=0.06, ry=0.04, rz=0.03, nu=24, nv=12) -> TriMesh:
    m = uv_sphere(1.0, nu, nv, radii=(rx, ry, rz))
    m.name = "ellipsoid"
    return m


def cylinder(r=0.03, half=0.1, n=24) -> TriMesh:
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    ring = np.stack([r * np.cos(t), r * np.sin(t)], 1)
    V = np.vstack([np.c_[ring, -half * np.ones(n)], np.c_[ring, half * np.ones(n)], [[0, 0, -half]], [[0, 0, half]]])
    F = []
    for j in range(n):
        a, b = j, (j + 1) % n
        F += [[a, b, n + b], [a, n + b, n + a], [2 * n, b, a], [2 * n + 1, n + a, n + b]]
    return TriMesh(V, F, "cylinder")


def capsule(r=0.03, half=0.1, n=24, nring=6) -> TriMesh:
    """Capsule along z: cylinder of half length `half` with hemispherical caps."""
    rows = []
    for i in range(nring + 1):   # bottom cap, from pole to equator
        ph = -math.pi / 2 + (math.pi / 2) * i / nring
        rows.append((r * math.cos(ph), -half + r * math.sin(ph)))
    for i in range(nring + 1):   # top cap, from equator to pole
        ph = (math.pi / 2) * i / nring
        rows.append((r * math.cos(ph), half + r * math.sin(ph)))
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    V, F = [], []
    for rr, z in rows:
        for th in t:
            V.append([rr * math.cos(th), rr * math.sin(th), z])
    nr = len(rows)
    for i in range(nr - 1):
        for j in range(n):
            a, b = i * n + j, i * n + (j + 1) % n
            c, d = a + n, b + n
            F += [[a, b, d], [a, d, c]]
    m = TriMesh(V, F, "capsule")
    m.weld(1e-12)   # merge the pole rings
    return m


def plane(sx=0.5, sy=0.5, nx=1, ny=1) -> TriMesh:
    xs, ys = np.linspace(-sx, sx, nx + 1), np.linspace(-sy, sy, ny + 1)
    V = np.array([[x, y, 0.0] for y in ys for x in xs])
    F = []
    for j in range(ny):
        for i in range(nx):
            a = j * (nx + 1) + i
            b, c, d = a + 1, a + nx + 1, a + nx + 2
            F += [[a, b, d], [a, d, c]]
    return TriMesh(V, F, "plane")


def triangle(s=0.05) -> TriMesh:
    """Flat triangle; split around its centroid because MuJoCo needs >= 4 vertices per mesh."""
    a, b, c = np.array([0, 0, 0.0]), np.array([s, 0, 0.0]), np.array([0, s, 0.0])
    return TriMesh([a, b, c, (a + b + c) / 3], [[0, 1, 3], [1, 2, 3], [2, 0, 3]], "triangle")


def quad(s=0.05) -> TriMesh:
    m = plane(s / 2, s / 2, 1, 1)
    m.name = "quad"
    return m


PRIMITIVES = {"box": box, "sphere": uv_sphere, "ellipsoid": ellipsoid, "cylinder": cylinder,
              "capsule": capsule, "plane": plane, "triangle": triangle, "quad": quad}


# ======================================================================== I/O
def read_obj(data: bytes | str, name: str = "mesh") -> TriMesh:
    text = data.decode("utf-8", "replace") if isinstance(data, (bytes, bytearray)) else data
    V, F = [], []
    for line in text.splitlines():
        s = line.strip()
        if not s or s[0] == "#":
            continue
        parts = s.split()
        if parts[0] == "v" and len(parts) >= 4:
            V.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif parts[0] == "f" and len(parts) >= 4:
            idx = []
            for p in parts[1:]:
                k = int(p.split("/")[0])
                idx.append(k - 1 if k > 0 else len(V) + k)
            for t in range(1, len(idx) - 1):   # fan triangulation of polygons
                F.append([idx[0], idx[t], idx[t + 1]])
    return TriMesh(np.array(V, float).reshape(-1, 3), np.array(F, np.int64).reshape(-1, 3), name)


def write_obj(m: TriMesh) -> bytes:
    out = io.StringIO()
    out.write(f"# exported by MuJoCo Model Studio\no {m.name}\n")
    for v in m.V:
        out.write(f"v {v[0]:.9g} {v[1]:.9g} {v[2]:.9g}\n")
    for f in m.F + 1:
        out.write(f"f {f[0]} {f[1]} {f[2]}\n")
    return out.getvalue().encode("utf-8")


def read_stl(data: bytes, name: str = "mesh") -> TriMesh:
    if len(data) >= 84:
        n = struct.unpack_from("<I", data, 80)[0]
        if 84 + 50 * n == len(data):
            rec = np.frombuffer(data, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]),
                                count=n, offset=84)
            V = rec["v"].reshape(-1, 3).astype(float)
            m = TriMesh(V, np.arange(len(V)).reshape(-1, 3), name)
            m.weld(1e-9)
            return m
    text = data.decode("utf-8", "replace")
    pts = [[float(x) for x in ln.split()[1:4]] for ln in text.splitlines() if ln.strip().startswith("vertex")]
    V = np.array(pts, float).reshape(-1, 3)
    m = TriMesh(V, np.arange(len(V)).reshape(-1, 3), name)
    m.weld(1e-9)
    return m


def write_stl(m: TriMesh) -> bytes:
    n = m.face_normals()
    rec = np.zeros(m.nf, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    rec["n"] = n
    rec["v"] = m.V[m.F]
    head = b"MuJoCo Model Studio".ljust(80, b" ")
    return head + struct.pack("<I", m.nf) + rec.tobytes()


def read_mesh_file(path: str | Path) -> TriMesh:
    p = Path(path)
    data = p.read_bytes()
    ext = p.suffix.lower()
    if ext == ".obj":
        return read_obj(data, p.stem)
    if ext == ".stl":
        return read_stl(data, p.stem)
    raise ValueError(f"지원하지 않는 메시 형식 {ext} (OBJ 또는 STL; glTF는 GLB 가져오기 사용)")


def encode(m: TriMesh, ext: str) -> bytes:
    ext = ext.lower().lstrip(".")
    if ext == "obj":
        return write_obj(m)
    if ext == "stl":
        return write_stl(m)
    raise ValueError(f".{ext} 형식으로 쓸 수 없습니다")


def from_compiled(model, mesh_id: int, scale=None) -> TriMesh:
    """Recover a mesh in its *file* frame from a compiled MjModel (undoing MuJoCo's
    re-centering v_c = R^T (s*v - p)): v = (R v_c + p) / s."""
    a, n = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
    fa, fn = model.mesh_faceadr[mesh_id], model.mesh_facenum[mesh_id]
    vc = model.mesh_vert[a:a + n].astype(float)
    V = vc @ mx.q2mat(model.mesh_quat[mesh_id]).T + model.mesh_pos[mesh_id]
    if scale is not None:
        V = V / np.asarray(scale, float)
    F = model.mesh_face[fa:fa + fn].astype(np.int64)
    return TriMesh(V, F, f"mesh{mesh_id}")
