"""Vertex/face editing of a mesh geom inside the viewport."""
from __future__ import annotations

import logging
import math

import mujoco
import numpy as np
from OpenGL import GL

from ..core import mathx as mx
from ..core import meshops as MO

log = logging.getLogger("mjstudio")


class MeshEditMode:
    """형상 하나가 쓰는 메시 에셋을 편집합니다. 선택: 정점(클릭, Shift 추가) 또는 면 하나(표면 클릭).
    이동/회전 도구로 끌면 선택한 정점이 움직입니다."""

    def __init__(self, view, geom):
        self.view, self.ed, self.geom = view, view.ed, geom
        self.mesh = MO.geom_mesh_elem(self.ed, geom)
        if self.mesh is None:
            raise ValueError("선택한 형상은 메시를 쓰지 않습니다 (먼저 '메시로 변환')")
        MO.get_mesh_data(self.ed, self.mesh)   # fails early if the data is not editable
        self.sel: list[int] = []
        self.face = -1
        self._drag0 = None

    # ------------------------------------------------------------ geometry
    def _ids(self):
        m = self.ed.model
        gid = self.ed.idmap.id(self.geom)
        if gid < 0:
            return None
        return gid, int(m.geom_dataid[gid])

    def world_vertices(self) -> np.ndarray | None:
        ids = self._ids()
        if ids is None:
            return None
        gid, mid = ids
        m, d = self.ed.model, self.ed.data
        a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        return d.geom_xpos[gid] + m.mesh_vert[a:a + n].astype(float) @ d.geom_xmat[gid].reshape(3, 3).T

    def faces(self) -> np.ndarray:
        gid, mid = self._ids()
        m = self.ed.model
        a, n = m.mesh_faceadr[mid], m.mesh_facenum[mid]
        return m.mesh_face[a:a + n].astype(np.int64)

    def world_to_file(self, dw: np.ndarray) -> np.ndarray:
        """World displacement -> displacement in the mesh file frame (before <mesh scale>)."""
        gid, mid = self._ids()
        m, d = self.ed.model, self.ed.data
        Rm = mx.q2mat(m.mesh_quat[mid])
        local = Rm @ (d.geom_xmat[gid].reshape(3, 3).T @ dw)
        return local / MO.mesh_scale(self.ed, self.mesh)

    # ------------------------------------------------------------ selection
    def pick(self, proj, sx: float, sy: float, add: bool) -> None:
        W = self.world_vertices()
        if W is None:
            return
        best = None
        for i, p in enumerate(W):
            pr = proj.project(p)
            if pr is None:
                continue
            dd = math.hypot(pr[0] - sx, pr[1] - sy)
            if dd < 9 and (best is None or (dd, pr[2]) < best[0]):
                best = ((dd, pr[2]), i)
        if best is not None:
            i = best[1]
            self.face = -1
            if add:
                self.sel.remove(i) if i in self.sel else self.sel.append(i)
            else:
                self.sel = [i]
            return
        ro, rd = proj.ray(sx, sy)
        from ..core.meshes import TriMesh
        f, _t = TriMesh(W, self.faces()).pick_face(ro, rd)
        self.face = f
        if not add:
            self.sel = []

    def frame(self):
        W = self.world_vertices()
        if W is None or not self.sel:
            return None
        return W[self.sel].mean(axis=0), np.eye(3)

    # ------------------------------------------------------------ drawing
    def draw(self, proj, dpr: float) -> None:
        W = self.world_vertices()
        if W is None:
            return
        F = self.faces()
        if len(F) <= 6000:
            GL.glLineWidth(1.0 * dpr)
            GL.glColor4f(0.75, 0.85, 1.0, 0.35)
            E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
            GL.glBegin(GL.GL_LINES)
            for a, b in E:
                GL.glVertex3dv(W[a])
                GL.glVertex3dv(W[b])
            GL.glEnd()
        if 0 <= self.face < len(F):
            GL.glColor4f(1.0, 0.6, 0.15, 0.55)
            GL.glBegin(GL.GL_TRIANGLES)
            for k in F[self.face]:
                GL.glVertex3dv(W[k])
            GL.glEnd()
        GL.glPointSize(5 * dpr)
        GL.glColor4f(1, 1, 1, 0.9)
        GL.glBegin(GL.GL_POINTS)
        for p in W:
            GL.glVertex3dv(p)
        GL.glEnd()
        if self.sel:
            GL.glPointSize(9 * dpr)
            GL.glColor4f(1.0, 0.6, 0.15, 1)
            GL.glBegin(GL.GL_POINTS)
            for i in self.sel:
                GL.glVertex3dv(W[i])
            GL.glEnd()

    # ------------------------------------------------------------ drag (preview + commit)
    def begin_drag(self) -> None:
        gid, mid = self._ids()
        m = self.ed.model
        a = m.mesh_vertadr[mid]
        self._drag0 = (mid, m.mesh_vert[a:a + m.mesh_vertnum[mid]].copy())

    def update_drag(self, dpos_world, dquat, pivot) -> None:
        if self._drag0 is None:
            return
        mid, v0 = self._drag0
        gid, _ = self._ids()
        m, d = self.ed.model, self.ed.data
        R = d.geom_xmat[gid].reshape(3, 3)
        W0 = d.geom_xpos[gid] + v0[self.sel].astype(float) @ R.T
        Wn = (W0 - pivot) @ mx.q2mat(dquat).T + pivot + dpos_world
        a = m.mesh_vertadr[mid]
        m.mesh_vert[a + np.array(self.sel)] = (Wn - d.geom_xpos[gid]) @ R
        self.view.request_mesh_upload(mid)

    def commit_drag(self, dpos_world, dquat, pivot) -> None:
        if self._drag0 is None:
            return
        mid, v0 = self._drag0
        gid, _ = self._ids()
        m, d = self.ed.model, self.ed.data
        R = d.geom_xmat[gid].reshape(3, 3)
        W0 = d.geom_xpos[gid] + v0[self.sel].astype(float) @ R.T
        Wn = (W0 - pivot) @ mx.q2mat(dquat).T + pivot + dpos_world
        a = m.mesh_vertadr[mid]
        m.mesh_vert[a:a + len(v0)] = v0       # restore; the recompile brings the edited data
        self._drag0 = None
        data = MO.get_mesh_data(self.ed, self.mesh)
        for k, i in enumerate(self.sel):
            data.V[i] += self.world_to_file(Wn[k] - W0[k])
        self._apply(data, "메시 정점 이동")

    def cancel_drag(self) -> None:
        if self._drag0 is not None:
            mid, v0 = self._drag0
            m = self.ed.model
            a = m.mesh_vertadr[mid]
            m.mesh_vert[a:a + len(v0)] = v0
            self.view.request_mesh_upload(mid)
            self._drag0 = None

    # ------------------------------------------------------------ edits
    def _say(self, msg: str) -> None:
        log.warning("%s", msg)
        self.view.status.emit(msg)

    def _apply(self, data, text: str) -> None:
        try:
            MO.set_mesh_data(self.ed, self.mesh, data, text)
        except ValueError as ex:
            log.error("%s: %s", text, ex)
            self.view.status.emit(f"{text} 실패: {ex}")

    def delete(self) -> None:
        data = MO.get_mesh_data(self.ed, self.mesh)
        if self.sel:
            data.delete_vertices(self.sel)
            self.sel = []
            self._apply(data, "메시 정점 삭제")
        elif self.face >= 0:
            data.delete_faces([self.face])
            self.face = -1
            self._apply(data, "메시 면 삭제")

    def make_face(self) -> None:
        if len(self.sel) != 3:
            self._say("면을 만들려면 정점을 정확히 3개 선택하세요 (Shift+클릭)")
            return
        data = MO.get_mesh_data(self.ed, self.mesh)
        i, j, k = self.sel
        n = np.cross(data.V[j] - data.V[i], data.V[k] - data.V[i])
        if n @ (data.V[[i, j, k]].mean(axis=0) - data.V.mean(axis=0)) < 0:
            j, k = k, j                         # face outwards (away from the mesh centre)
        data.add_face(i, j, k)
        self._apply(data, "메시 면 만들기")

    def poke(self) -> None:
        if self.face < 0:
            self._say("먼저 면을 클릭하세요 (그 면 안에 정점이 추가됩니다)")
            return
        data = MO.get_mesh_data(self.ed, self.mesh)
        i = data.poke_face(self.face)
        self.face = -1
        self.sel = [i]
        self._apply(data, "메시 정점 추가")
