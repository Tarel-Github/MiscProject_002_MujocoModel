"""Interactive transforms with live preview on the compiled model.

While dragging, the compiled MjModel fields (body_pos, geom_pos, ...) are
patched directly so the viewport updates without recompiling. On commit the
final poses are written to the document as one undo step (which recompiles and
discards the temporary patches); cancel restores the patched fields.
"""
from __future__ import annotations

import mujoco
import numpy as np

from . import mathx as mx
from .doc import Element
from .editor import Editor

OBJ = mujoco.mjtObj
MOVABLE = {"body", "geom", "site", "camera", "light", "joint"}


class TransformSession:
    def __init__(self, ed: Editor, elems: list[Element]):
        self.ed = ed
        m, d = ed.model, ed.data
        self.items: list[dict] = []
        self.ok = m is not None
        if not self.ok:
            return
        # do not move an element together with one of its ancestors (double motion)
        sel = [e for e in elems if e.tag in MOVABLE]
        sel = [e for e in sel if not any(a in sel for a in e.iterancestors())]
        for e in sel:
            it = {"e": e, "P0": ed.parent_world(e), "L0": ed.doc.local_pose(e), "id": ed.idmap.id(e)}
            it["L"] = it["L0"]
            i = it["id"]
            L0p, L0q = it["L0"]
            ip, iq = mx.pose_inv(L0p, L0q)
            if i >= 0:
                if e.tag == "body":
                    it["save"] = (m.body_pos[i].copy(), m.body_quat[i].copy())
                    it["mocap"] = int(m.body_mocapid[i])
                    j = m.body_jntadr[i]
                    it["free"] = (int(m.jnt_qposadr[j]) if m.body_jntnum[i] > 0 and
                                  m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE else -1)
                    if it["free"] >= 0:
                        a = it["free"]
                        it["qfree0"] = (d.qpos[a:a + 3].copy(), d.qpos[a + 3:a + 7].copy())
                elif e.tag == "geom":
                    it["save"] = (m.geom_pos[i].copy(), m.geom_quat[i].copy())
                    it["off"] = mx.pose_mul(ip, iq, m.geom_pos[i], m.geom_quat[i])
                    # kinematics skips geom_pos when the 'sameframe' optimisation is set
                    it["same"] = int(m.geom_sameframe[i])
                    m.geom_sameframe[i] = 0
                elif e.tag == "site":
                    it["save"] = (m.site_pos[i].copy(), m.site_quat[i].copy())
                    it["off"] = mx.pose_mul(ip, iq, m.site_pos[i], m.site_quat[i])
                    it["same"] = int(m.site_sameframe[i])
                    m.site_sameframe[i] = 0
                elif e.tag == "camera":
                    it["save"] = (m.cam_pos[i].copy(), m.cam_quat[i].copy())
                elif e.tag == "light":
                    it["save"] = (m.light_pos[i].copy(), m.light_dir[i].copy())
                elif e.tag == "joint":
                    it["save"] = (m.jnt_pos[i].copy(), m.jnt_axis[i].copy())
            self.items.append(it)
        self.mocap0 = (d.mocap_pos.copy(), d.mocap_quat.copy())

    def update(self, dpos, dquat, pivot) -> None:
        if not self.ok:
            return
        m, d = self.ed.model, self.ed.data
        D = (np.asarray(pivot, float) + np.asarray(dpos, float) - mx.qrot(dquat, pivot), mx.qnorm(dquat))
        for it in self.items:
            e = it["e"]
            Lp, Lq = self.ed.local_after_world_delta(e, dpos, dquat, pivot, it["P0"], it["L0"])
            it["L"] = (Lp, Lq)
            i = it["id"]
            if i < 0:
                continue
            if e.tag == "body":
                if it["mocap"] >= 0:
                    wp, wq = mx.pose_mul(*it["P0"], Lp, Lq)
                    d.mocap_pos[it["mocap"]] = wp
                    d.mocap_quat[it["mocap"]] = wq
                else:
                    m.body_pos[i] = Lp
                    m.body_quat[i] = Lq
                if it["free"] >= 0:
                    a = it["free"]
                    qp, qq = mx.pose_mul(D[0], D[1], *it["qfree0"])
                    d.qpos[a:a + 3] = qp
                    d.qpos[a + 3:a + 7] = qq
            elif e.tag == "geom":
                p, q = mx.pose_mul(Lp, Lq, *it["off"])
                m.geom_pos[i], m.geom_quat[i] = p, q
            elif e.tag == "site":
                p, q = mx.pose_mul(Lp, Lq, *it["off"])
                m.site_pos[i], m.site_quat[i] = p, q
            elif e.tag == "camera":
                m.cam_pos[i], m.cam_quat[i] = Lp, Lq
            elif e.tag == "light":
                m.light_pos[i] = Lp
                m.light_dir[i] = -mx.q2mat(Lq)[:, 2]
            elif e.tag == "joint":
                m.jnt_pos[i] = Lp
                m.jnt_axis[i] = mx.q2mat(Lq)[:, 2]
        mujoco.mj_forward(m, d)
        self.ed.state_changed.emit()

    def _restore(self) -> None:
        m, d = self.ed.model, self.ed.data
        for it in self.items:
            i, e = it["id"], it["e"]
            if i < 0 or "save" not in it:
                continue
            a, b = it["save"]
            if e.tag == "body":
                m.body_pos[i], m.body_quat[i] = a, b
                if it["free"] >= 0:
                    q = it["free"]
                    d.qpos[q:q + 3], d.qpos[q + 3:q + 7] = it["qfree0"]
            elif e.tag == "geom":
                m.geom_pos[i], m.geom_quat[i] = a, b
                m.geom_sameframe[i] = it["same"]
            elif e.tag == "site":
                m.site_pos[i], m.site_quat[i] = a, b
                m.site_sameframe[i] = it["same"]
            elif e.tag == "camera":
                m.cam_pos[i], m.cam_quat[i] = a, b
            elif e.tag == "light":
                m.light_pos[i], m.light_dir[i] = a, b
            elif e.tag == "joint":
                m.jnt_pos[i], m.jnt_axis[i] = a, b
        d.mocap_pos[:], d.mocap_quat[:] = self.mocap0

    def cancel(self) -> None:
        if not self.ok:
            return
        self._restore()
        mujoco.mj_forward(self.ed.model, self.ed.data)
        self.ed.state_changed.emit()

    def commit(self, text: str = "위치·회전 변경") -> None:
        if not self.ok or not self.items:
            return
        moved = [it for it in self.items
                 if not (np.allclose(it["L"][0], it["L0"][0], atol=1e-12) and mx.qangle(it["L"][1], it["L0"][1]) < 1e-12)]
        if not moved:
            self.cancel()
            return
        # restore compiled fields but keep free-joint qpos (transferred by the recompile)
        m, d = self.ed.model, self.ed.data
        keep_q = d.qpos.copy()
        self._restore()
        d.qpos[:] = keep_q
        with self.ed.batch(text):
            for it in moved:
                self.ed.set_pose(it["e"], *it["L"], text=text)
