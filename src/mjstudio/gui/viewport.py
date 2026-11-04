"""MuJoCo 3D viewport.

Rendering uses MuJoCo's own renderer (mjr_render) into MuJoCo's offscreen
framebuffer, then a small legacy-GL overlay pass (gizmos, joint ranges) and a
blit into the QOpenGLWidget framebuffer. Requires an OpenGL *compatibility*
profile context (see app.py).
"""
from __future__ import annotations

import math
import zlib

import mujoco
import numpy as np
from OpenGL import GL
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QMouseEvent, QWheelEvent
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from ..core import mathx as mx
from ..core.editor import Editor
from ..core.preview import MOVABLE, TransformSession

OBJ = mujoco.mjtObj
GT = mujoco.mjtGeom
HL = np.array([1.0, 0.6, 0.15], np.float32)
AXCOL = [(0.95, 0.32, 0.32), (0.45, 0.88, 0.38), (0.36, 0.58, 1.0)]


def vis_signature(m: mujoco.MjModel):
    """Changes whenever the GPU resources held by an MjrContext would differ."""
    crc = 0
    for arr in (m.mesh_vert, m.mesh_face, m.mesh_normal, m.mesh_texcoord, m.mesh_facenormal,
                m.mesh_facetexcoord, m.tex_rgb, m.tex_width, m.tex_height, m.tex_type, m.hfield_data,
                m.hfield_size, m.skin_vert, m.skin_face, m.skin_texcoord, m.mesh_vertadr, m.mesh_faceadr):
        a = np.ascontiguousarray(arr)
        if a.size:
            crc = zlib.crc32(memoryview(a).cast("B"), crc)
    q = m.vis.quality
    return (m.nmesh, m.ntex, m.nhfield, m.nskin, m.nflex, q.shadowsize, q.offsamples, q.numslices,
            q.numstacks, q.numquads, crc)


class CamProj:
    """World <-> pixel mapping identical to mjr_render's mono projection."""

    def __init__(self, scn: mujoco.MjvScene, w: int, h: int):
        cam = mujoco.mjv_averageCamera(scn.camera[0], scn.camera[1])
        self.w, self.h = max(w, 1), max(h, 1)
        self.pos = np.array(cam.pos, float)
        f = np.array(cam.forward, float)
        self.fwd = f / max(np.linalg.norm(f), 1e-12)
        r = np.cross(self.fwd, np.array(cam.up, float))
        self.right = r / max(np.linalg.norm(r), 1e-12)
        self.up = np.cross(self.right, self.fwd)
        self.near, self.far = float(cam.frustum_near), float(cam.frustum_far)
        self.top, self.bottom = float(cam.frustum_top), float(cam.frustum_bottom)
        hw = 0.5 * self.w / self.h * (self.top - self.bottom)
        self.left, self.rgt = float(cam.frustum_center) - hw, float(cam.frustum_center) + hw

    def project(self, p):
        v = np.asarray(p, float) - self.pos
        z = float(v @ self.fwd)
        if z <= 1e-9:
            return None
        xn, yn = float(v @ self.right) * self.near / z, float(v @ self.up) * self.near / z
        return ((xn - self.left) / (self.rgt - self.left) * self.w,
                (self.top - yn) / (self.top - self.bottom) * self.h, z)

    def ray(self, sx: float, sy: float):
        xn = self.left + (self.rgt - self.left) * sx / self.w
        yn = self.top - (self.top - self.bottom) * sy / self.h
        d = self.fwd * self.near + self.right * xn + self.up * yn
        return self.pos.copy(), d / np.linalg.norm(d)

    def px_world(self, p) -> float:
        """World length of one pixel at the depth of point p."""
        z = max(float((np.asarray(p, float) - self.pos) @ self.fwd), 1e-6)
        return (self.top - self.bottom) / self.h * z / self.near

    def gl_setup(self) -> None:
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.glFrustum(self.left, self.rgt, self.bottom, self.top, self.near, self.far)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        M = np.eye(4)
        M[0, :3], M[1, :3], M[2, :3] = self.right, self.up, -self.fwd
        M[:3, 3] = -M[:3, :3] @ self.pos
        GL.glLoadMatrixd(M.T.ravel())


# glClipControl (GL 4.5 / ARB_clip_control) constants; MuJoCo 3.x renders with reversed Z
# (clip depth mode ZERO_TO_ONE), which would clip half of a classic glOrtho/glFrustum volume.
_GL_LOWER_LEFT, _GL_CLIP_ORIGIN, _GL_CLIP_DEPTH_MODE = 0x8CA1, 0x935C, 0x935D
_GL_NEGATIVE_ONE_TO_ONE = 0x935E


def _clip_push():
    try:
        origin = int(GL.glGetIntegerv(_GL_CLIP_ORIGIN)[0])
        mode = int(GL.glGetIntegerv(_GL_CLIP_DEPTH_MODE)[0])
        GL.glClipControl(_GL_LOWER_LEFT, _GL_NEGATIVE_ONE_TO_ONE)
        return origin, mode
    except Exception:  # noqa: BLE001 - clip control unsupported: nothing to undo
        return None


def _clip_pop(state) -> None:
    if state is not None:
        try:
            GL.glClipControl(state[0], state[1])
        except Exception:  # noqa: BLE001
            pass


def _seg_dist(p, a, b) -> float:
    p, a, b = (np.asarray(x, float) for x in (p, a, b))
    ab = b - a
    t = 0.0 if ab @ ab < 1e-12 else float(np.clip((p - a) @ ab / (ab @ ab), 0, 1))
    return float(np.linalg.norm(p - (a + t * ab)))


def _perp(a):
    a = np.asarray(a, float)
    t = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(a, t)
    return u / np.linalg.norm(u)


class Gizmo:
    """Translate / rotate manipulator drawn in screen-constant size."""
    PX = 95  # handle length in pixels

    def __init__(self):
        self.mode = "translate"   # translate | rotate | none
        self.space = "world"      # world | local
        self.origin = np.zeros(3)
        self.R = np.eye(3)
        self.L = 0.1
        self.hover: str | None = None
        self.active: str | None = None
        self.visible = False

    def layout(self, origin, R, proj: CamProj) -> None:
        self.origin = np.asarray(origin, float)
        self.R = np.asarray(R, float) if self.space == "local" else np.eye(3)
        self.L = self.PX * proj.px_world(self.origin)

    def _ring(self, k: int, n: int = 64):
        a = self.R[:, (k + 1) % 3]
        b = self.R[:, (k + 2) % 3]
        t = np.linspace(0, 2 * math.pi, n, endpoint=False)
        return self.origin + self.L * 0.85 * (np.outer(np.cos(t), a) + np.outer(np.sin(t), b))

    def hit(self, proj: CamProj, sx: float, sy: float, tol: float = 9.0) -> str | None:
        if not self.visible or self.mode == "none":
            return None
        o = proj.project(self.origin)
        if o is None:
            return None
        m = np.array([sx, sy])
        if self.mode == "translate":
            if np.linalg.norm(m - o[:2]) < tol:
                return "view"
            best = None
            for k, nm in enumerate("xyz"):
                e = proj.project(self.origin + self.R[:, k] * self.L)
                if e is None:
                    continue
                d = _seg_dist(m, o[:2], e[:2])
                if d < tol and (best is None or d < best[0]):
                    best = (d, nm)
            for k, nm in ((2, "xy"), (0, "yz"), (1, "xz")):
                a, b = self.R[:, (k + 1) % 3], self.R[:, (k + 2) % 3]
                c = proj.project(self.origin + (a + b) * self.L * 0.3)
                if c is not None and np.linalg.norm(m - c[:2]) < tol and best is None:
                    best = (0, nm)
            return best[1] if best else None
        best = None
        for k, nm in enumerate(("rx", "ry", "rz")):
            pts = [proj.project(p) for p in self._ring(k)]
            pts = [p for p in pts if p is not None]
            for i in range(len(pts)):
                d = _seg_dist(m, pts[i][:2], pts[(i + 1) % len(pts)][:2])
                if d < tol and (best is None or d < best[0]):
                    best = (d, nm)
        return best[1] if best else None

    def draw(self, proj: CamProj, dpr: float) -> None:
        if not self.visible or self.mode == "none":
            return
        o = self.origin
        cur = self.active or self.hover
        GL.glLineWidth(2.5 * dpr)
        if self.mode == "translate":
            for k, nm in enumerate("xyz"):
                col = (1, 0.92, 0.3) if cur == nm else AXCOL[k]
                a = self.R[:, k]
                tip = o + a * self.L
                GL.glColor4f(*col, 1)
                GL.glBegin(GL.GL_LINES)
                GL.glVertex3dv(o)
                GL.glVertex3dv(tip)
                GL.glEnd()
                u = _perp(a)
                v = np.cross(a, u)
                GL.glBegin(GL.GL_TRIANGLE_FAN)
                GL.glVertex3dv(tip + a * self.L * 0.22)
                for i in range(13):
                    t = 2 * math.pi * i / 12
                    GL.glVertex3dv(tip + (u * math.cos(t) + v * math.sin(t)) * self.L * 0.06)
                GL.glEnd()
            for k, nm in ((2, "xy"), (0, "yz"), (1, "xz")):
                a, b = self.R[:, (k + 1) % 3], self.R[:, (k + 2) % 3]
                col = (1, 0.92, 0.3) if cur == nm else AXCOL[k]
                GL.glColor4f(*col, 0.45 if cur != nm else 0.8)
                s0, s1 = 0.2 * self.L, 0.4 * self.L
                GL.glBegin(GL.GL_QUADS)
                for p in (o + a * s0 + b * s0, o + a * s1 + b * s0, o + a * s1 + b * s1, o + a * s0 + b * s1):
                    GL.glVertex3dv(p)
                GL.glEnd()
            GL.glPointSize(9 * dpr)
            GL.glColor4f(*((1, 0.92, 0.3) if cur == "view" else (0.95, 0.95, 0.95)), 1)
            GL.glBegin(GL.GL_POINTS)
            GL.glVertex3dv(o)
            GL.glEnd()
        elif self.mode == "rotate":
            for k, nm in enumerate(("rx", "ry", "rz")):
                col = (1, 0.92, 0.3) if cur == nm else AXCOL[k]
                GL.glColor4f(*col, 1)
                GL.glBegin(GL.GL_LINE_LOOP)
                for p in self._ring(k):
                    GL.glVertex3dv(p)
                GL.glEnd()


class Viewport(QOpenGLWidget):
    picked = Signal(object, bool)          # element or None, additive
    status = Signal(str)

    def __init__(self, ed: Editor, parent=None):
        super().__init__(parent)
        self.ed = ed
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.pert = mujoco.MjvPerturb()
        self.scn: mujoco.MjvScene | None = None
        self.con: mujoco.MjrContext | None = None
        self._sig = None
        self._scn_model = None
        self._cam_init = False
        self.show_visual = True
        self.show_collision = True
        self.pick_mode = "body"        # body | geom
        self.gizmo = Gizmo()
        self.scene_hooks: list = []    # callables(model, data, scn) after mjv_updateScene
        self.hud = True
        self._proj: CamProj | None = None
        self._press = None
        self._last = None
        self._drag = None
        self._ts: TransformSession | None = None
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setMinimumSize(320, 240)
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = 0
        ed.state_changed.connect(self.update)
        ed.selection_changed.connect(lambda *_: self.update())
        ed.compiled.connect(lambda *_: self.update())

    # ------------------------------------------------------------- GL lifecycle
    def initializeGL(self) -> None:
        ctx = self.context()
        ctx.aboutToBeDestroyed.connect(self._free)
        GL.glClearColor(0.09, 0.1, 0.11, 1)

    def _free(self) -> None:
        self.makeCurrent()
        if self.con is not None:
            self.con.free()
            self.con = None
        self._sig = None
        self.doneCurrent()

    def _fb(self) -> tuple[int, int]:
        dpr = self.devicePixelRatioF()
        return max(1, int(round(self.width() * dpr))), max(1, int(round(self.height() * dpr)))

    def _ensure(self) -> bool:
        m = self.ed.model
        if m is None:
            return False
        sig = vis_signature(m)
        if self.con is None or sig != self._sig:
            if self.con is not None:
                self.con.free()
            w, h = self._fb()
            m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), w)
            m.vis.global_.offheight = max(int(m.vis.global_.offheight), h)
            scale = mujoco.mjtFontScale.mjFONTSCALE_150 if self.devicePixelRatioF() > 1.3 else mujoco.mjtFontScale.mjFONTSCALE_100
            self.con = mujoco.MjrContext(m, scale)
            self._sig = sig
        if self._scn_model is not m:
            self.scn = mujoco.MjvScene(m, maxgeom=20000)
            self._scn_model = m
            if not self._cam_init:
                mujoco.mjv_defaultFreeCamera(m, self.cam)
                self._cam_init = True
        return True

    def reset_camera(self) -> None:
        if self.ed.model is not None:
            mujoco.mjv_defaultFreeCamera(self.ed.model, self.cam)
            self.update()

    def paintGL(self) -> None:
        w, h = self._fb()
        if not self._ensure():
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.defaultFramebufferObject())
            GL.glClearColor(0.09, 0.1, 0.11, 1)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            return
        m, d, con, scn = self.ed.model, self.ed.data, self.con, self.scn
        if w > con.offWidth or h > con.offHeight:
            mujoco.mjr_resizeOffscreen(max(w, con.offWidth), max(h, con.offHeight), con)
        mujoco.mjv_updateScene(m, d, self.opt, self.pert, self.cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
        self._post_scene(m, d, scn)
        for hook in self.scene_hooks:
            hook(m, d, scn)
        vp = mujoco.MjrRect(0, 0, w, h)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, con)
        mujoco.mjr_render(vp, scn, con)
        self._proj = CamProj(scn, w, h)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, con.offFBO)
        self._overlay(m, d, self._proj)
        if self.hud:
            self._draw_hud(vp)
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, con.offFBO)
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.defaultFramebufferObject())
        GL.glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.defaultFramebufferObject())

    # ------------------------------------------------------------- scene edits
    def _sel_ids(self):
        geoms, bodies, sites = set(), set(), set()
        im = self.ed.idmap
        if im is None:
            return geoms, bodies, sites
        for e in self.ed.selection:
            o = im.obj(e)
            if o is None:
                continue
            if o[0] == int(OBJ.mjOBJ_GEOM):
                geoms.add(o[1])
            elif o[0] == int(OBJ.mjOBJ_BODY):
                bodies.add(o[1])
            elif o[0] == int(OBJ.mjOBJ_SITE):
                sites.add(o[1])
        return geoms, bodies, sites

    def _post_scene(self, m, d, scn) -> None:
        geoms, bodies, sites = self._sel_ids()
        hide = not (self.show_visual and self.show_collision)
        if not (geoms or bodies or sites or hide):
            return
        ct, ca, gb = m.geom_contype, m.geom_conaffinity, m.geom_bodyid
        for i in range(scn.ngeom):
            g = scn.geoms[i]
            if g.objtype == OBJ.mjOBJ_GEOM:
                gid = g.objid
                if hide:
                    coll = bool(ct[gid] or ca[gid])
                    if (coll and not self.show_collision) or (not coll and not self.show_visual):
                        g.type = GT.mjGEOM_NONE
                        continue
                strong = gid in geoms
                if strong or gb[gid] in bodies:
                    k = 0.55 if strong else 0.4
                    g.rgba[:3] = (1 - k) * g.rgba[:3] + k * HL
                    g.emission = max(g.emission, 0.35 if strong else 0.25)
            elif g.objtype == OBJ.mjOBJ_SITE and g.objid in sites:
                g.rgba[:] = (*HL, 1)
                g.emission = 0.6

    # ------------------------------------------------------------- overlay
    def _sel_frame(self):
        """(origin, R) of the first movable selected element, or None."""
        for e in self.ed.selection:
            if e.tag not in MOVABLE or self.ed.idmap is None:
                continue
            if e.tag == "body":
                bid = self.ed.idmap.id(e)
                if bid >= 0:
                    return self.ed.data.xpos[bid].copy(), self.ed.data.xmat[bid].reshape(3, 3).copy()
            if e.tag == "joint":
                jid = self.ed.idmap.id(e)
                if jid >= 0:
                    a = self.ed.data.xaxis[jid].copy()
                    return self.ed.data.xanchor[jid].copy(), mx.q2mat(mx.zaxis2q(a))
            p, q = self.ed.frame_world(e)
            return p, mx.q2mat(q)
        return None

    def _overlay(self, m, d, proj: CamProj) -> None:
        dpr = self.devicePixelRatioF()
        GL.glPushAttrib(GL.GL_ALL_ATTRIB_BITS)
        GL.glPushClientAttrib(GL.GL_CLIENT_ALL_ATTRIB_BITS)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glPushMatrix()
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glPushMatrix()
        clip = _clip_push()
        try:
            GL.glViewport(0, 0, proj.w, proj.h)
            proj.gl_setup()
            for cap in (GL.GL_LIGHTING, GL.GL_TEXTURE_2D, GL.GL_DEPTH_TEST, GL.GL_CULL_FACE, GL.GL_FOG,
                        GL.GL_STENCIL_TEST, GL.GL_ALPHA_TEST, GL.GL_COLOR_MATERIAL):
                GL.glDisable(cap)
            for i in range(6):
                GL.glDisable(GL.GL_CLIP_PLANE0 + i)
            # MuJoCo's fixed-function shadow mapping leaves a depth-compare texture with
            # eye-linear texgen enabled on another unit; it would modulate our colours.
            for unit in range(4):
                GL.glActiveTexture(GL.GL_TEXTURE0 + unit)
                for cap in (GL.GL_TEXTURE_2D, GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_GEN_S, GL.GL_TEXTURE_GEN_T,
                            GL.GL_TEXTURE_GEN_R, GL.GL_TEXTURE_GEN_Q):
                    GL.glDisable(cap)
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
            self._draw_selected_extras(m, d, proj, dpr)
            fr = self._sel_frame() if self._drag is None else (self.gizmo.origin, self.gizmo.R)
            self.gizmo.visible = fr is not None
            if fr is not None:
                if self._drag is None:
                    self.gizmo.layout(fr[0], fr[1], proj)
                self.gizmo.draw(proj, dpr)
            self._draw_triad(proj, dpr)
        finally:
            _clip_pop(clip)
            GL.glMatrixMode(GL.GL_MODELVIEW)
            GL.glPopMatrix()
            GL.glMatrixMode(GL.GL_PROJECTION)
            GL.glPopMatrix()
            GL.glPopClientAttrib()
            GL.glPopAttrib()

    def _draw_selected_extras(self, m, d, proj: CamProj, dpr: float) -> None:
        im = self.ed.idmap
        for e in self.ed.selection:
            o = im.obj(e) if im else None
            if o is None:
                continue
            ot, i = o
            if ot == int(OBJ.mjOBJ_JOINT):
                self._draw_joint(m, d, i, proj, dpr)
            elif ot == int(OBJ.mjOBJ_CAMERA):
                self._draw_camera(m, d, i, proj, dpr)
            elif ot == int(OBJ.mjOBJ_LIGHT):
                p, dv = d.light_xpos[i], d.light_xdir[i]
                L = 60 * proj.px_world(p)
                GL.glLineWidth(2 * dpr)
                GL.glColor4f(1, 0.9, 0.4, 1)
                GL.glBegin(GL.GL_LINES)
                GL.glVertex3dv(p)
                GL.glVertex3dv(p + dv * L)
                GL.glEnd()
            elif ot == int(OBJ.mjOBJ_BODY) and i > 0 and self.gizmo.mode == "none":
                p, R = d.xpos[i], d.xmat[i].reshape(3, 3)
                L = 60 * proj.px_world(p)
                GL.glLineWidth(2 * dpr)
                GL.glBegin(GL.GL_LINES)
                for k in range(3):
                    GL.glColor4f(*AXCOL[k], 1)
                    GL.glVertex3dv(p)
                    GL.glVertex3dv(p + R[:, k] * L)
                GL.glEnd()

    def _draw_joint(self, m, d, j, proj: CamProj, dpr: float) -> None:
        a, p = d.xaxis[j], d.xanchor[j]
        L = 70 * proj.px_world(p)
        jt = m.jnt_type[j]
        GL.glLineWidth(2.5 * dpr)
        GL.glColor4f(1, 0.45, 0.4, 1)
        GL.glBegin(GL.GL_LINES)
        GL.glVertex3dv(p - a * L * 0.6)
        GL.glVertex3dv(p + a * L)
        GL.glEnd()
        GL.glPointSize(8 * dpr)
        GL.glBegin(GL.GL_POINTS)
        GL.glVertex3dv(p)
        GL.glEnd()
        q = d.qpos[m.jnt_qposadr[j]]
        if jt == mujoco.mjtJoint.mjJNT_HINGE:
            b = m.jnt_bodyid[j]
            v = d.xipos[b] - p
            v = v - a * (v @ a)
            u = v / np.linalg.norm(v) if np.linalg.norm(v) > 1e-6 else _perp(a)
            u0 = mx.qrot(mx.axang2q(a, -q), u)  # direction at q = 0
            w = np.cross(a, u0)
            r = L * 0.7
            lo, hi = (m.jnt_range[j] if m.jnt_limited[j] else (-math.pi, math.pi))
            GL.glColor4f(1, 0.6, 0.3, 0.25)
            GL.glBegin(GL.GL_TRIANGLE_FAN)
            GL.glVertex3dv(p)
            for t in np.linspace(lo, hi, 48):
                GL.glVertex3dv(p + r * (math.cos(t) * u0 + math.sin(t) * w))
            GL.glEnd()
            GL.glColor4f(1, 0.6, 0.3, 0.9)
            GL.glBegin(GL.GL_LINE_STRIP)
            for t in np.linspace(lo, hi, 48):
                GL.glVertex3dv(p + r * (math.cos(t) * u0 + math.sin(t) * w))
            GL.glEnd()
            GL.glColor4f(1, 1, 1, 1)
            GL.glBegin(GL.GL_LINES)
            GL.glVertex3dv(p)
            GL.glVertex3dv(p + r * 1.15 * (math.cos(q) * u0 + math.sin(q) * w))
            GL.glEnd()
        elif jt == mujoco.mjtJoint.mjJNT_SLIDE and m.jnt_limited[j]:
            lo, hi = m.jnt_range[j]
            base = p - a * q
            GL.glColor4f(1, 0.6, 0.3, 0.9)
            GL.glBegin(GL.GL_LINES)
            GL.glVertex3dv(base + a * lo)
            GL.glVertex3dv(base + a * hi)
            GL.glEnd()

    def _draw_camera(self, m, d, c, proj: CamProj, dpr: float) -> None:
        p = d.cam_xpos[c]
        R = d.cam_xmat[c].reshape(3, 3)
        L = 80 * proj.px_world(p)
        f = math.tan(math.radians(m.cam_fovy[c]) / 2)
        pts = [p + L * (-R[:, 2] + sx * f * 1.33 * R[:, 0] + sy * f * R[:, 1]) for sx, sy in
               ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        GL.glLineWidth(1.5 * dpr)
        GL.glColor4f(0.6, 0.9, 0.5, 1)
        GL.glBegin(GL.GL_LINES)
        for q in pts:
            GL.glVertex3dv(p)
            GL.glVertex3dv(q)
        for k in range(4):
            GL.glVertex3dv(pts[k])
            GL.glVertex3dv(pts[(k + 1) % 4])
        GL.glEnd()

    def _draw_triad(self, proj: CamProj, dpr: float) -> None:
        s = int(70 * dpr)
        GL.glViewport(int(8 * dpr), int(8 * dpr), s, s)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.glOrtho(-1.3, 1.3, -1.3, 1.3, -2, 2)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        M = np.eye(4)
        M[0, :3], M[1, :3], M[2, :3] = proj.right, proj.up, -proj.fwd
        GL.glLoadMatrixd(M.T.ravel())
        GL.glLineWidth(2 * dpr)
        GL.glBegin(GL.GL_LINES)
        for k in range(3):
            GL.glColor4f(*AXCOL[k], 1)
            GL.glVertex3d(0, 0, 0)
            e = np.zeros(3)
            e[k] = 1
            GL.glVertex3dv(e)
        GL.glEnd()
        GL.glViewport(0, 0, proj.w, proj.h)

    def _draw_hud(self, vp) -> None:
        sim = getattr(self, "sim", None)
        left = []
        right = []
        if sim is not None and sim.running:
            left.append(("SIM", f"t = {self.ed.data.time:.3f} s"))
        if self.ed.result is not None and not self.ed.result.ok:
            left.append(("ERROR", "model has errors - showing last valid model"))
        if self.gizmo.mode != "none":
            right.append((self.gizmo.mode.upper(), self.gizmo.space))
        if left:
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, vp,
                               "\n".join(a for a, _ in left), "\n".join(b for _, b in left), self.con)
        if right:
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPRIGHT, vp,
                               "\n".join(a for a, _ in right), "\n".join(b for _, b in right), self.con)

    # ------------------------------------------------------------- picking
    def _dev(self, pos: QPointF) -> tuple[float, float]:
        dpr = self.devicePixelRatioF()
        return pos.x() * dpr, pos.y() * dpr

    def pick(self, sx: float, sy: float):
        """Element under the pixel (device coords) and the 3D point, honouring pick_mode."""
        m, d = self.ed.model, self.ed.data
        if m is None or self.scn is None or self._proj is None:
            return None, None
        w, h = self._proj.w, self._proj.h
        # small markers first: joints (when shown) and sites
        best = None
        if self.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT]:
            for j in range(m.njnt):
                if not self.opt.jointgroup[min(max(m.jnt_group[j], 0), 5)]:
                    continue
                pr = self._proj.project(d.xanchor[j])
                if pr and math.hypot(pr[0] - sx, pr[1] - sy) < 10 and (best is None or pr[2] < best[0]):
                    best = (pr[2], int(OBJ.mjOBJ_JOINT), j, d.xanchor[j].copy())
        for s in range(m.nsite):
            if not self.opt.sitegroup[min(max(m.site_group[s], 0), 5)]:
                continue
            pr = self._proj.project(d.site_xpos[s])
            if pr and math.hypot(pr[0] - sx, pr[1] - sy) < 8 and (best is None or pr[2] < best[0]):
                best = (pr[2], int(OBJ.mjOBJ_SITE), s, d.site_xpos[s].copy())
        if best is not None:
            return self.ed.idmap.elem(best[1], best[2]), best[3]
        selpnt = np.zeros(3)
        gid = np.zeros((1, 1), np.int32)
        fid = np.zeros((1, 1), np.int32)
        sid = np.zeros((1, 1), np.int32)
        bid = mujoco.mjv_select(m, d, self.opt, w / h, sx / w, (h - sy) / h, self.scn, selpnt, gid, fid, sid)
        g = int(gid[0, 0])
        if bid < 0 or g < 0:
            return None, None
        if not (self.show_visual and self.show_collision):
            coll = bool(m.geom_contype[g] or m.geom_conaffinity[g])
            if (coll and not self.show_collision) or (not coll and not self.show_visual):
                return None, None
        im = self.ed.idmap
        if self.pick_mode == "body" and bid > 0:
            e = im.elem(int(OBJ.mjOBJ_BODY), bid)
            if e is not None:
                return e, selpnt
        return im.elem(int(OBJ.mjOBJ_GEOM), g), selpnt

    # ------------------------------------------------------------- mouse
    def mousePressEvent(self, ev: QMouseEvent) -> None:
        self._press = ev.position()
        self._last = ev.position()
        sx, sy = self._dev(ev.position())
        if ev.button() == Qt.LeftButton and self._proj is not None and self.gizmo.visible:
            h = self.gizmo.hit(self._proj, sx, sy)
            if h:
                self._begin_drag(h, sx, sy)
                return
        self._drag = None

    def _begin_drag(self, handle: str, sx: float, sy: float) -> None:
        ro, rd = self._proj.ray(sx, sy)
        o, R = self.gizmo.origin.copy(), self.gizmo.R.copy()
        st = {"h": handle, "o": o, "R": R}
        if handle in ("x", "y", "z"):
            a = R[:, "xyz".index(handle)]
            st["a"] = a
            st["t0"] = mx.closest_on_line_to_ray(o, a, ro, rd)
        elif handle in ("xy", "yz", "xz", "view"):
            n = self._proj.fwd if handle == "view" else R[:, {"xy": 2, "yz": 0, "xz": 1}[handle]]
            st["n"] = n
            st["p0"] = mx.ray_plane(ro, rd, o, n)
            if st["p0"] is None:
                return
        else:
            a = R[:, "xyz".index(handle[1])]
            st["a"] = a
            p0 = mx.ray_plane(ro, rd, o, a)
            st["v0"] = (p0 - o) if p0 is not None else None
            st["s0"] = (sx, sy)
        self._ts = TransformSession(self.ed, list(self.ed.selection))
        if not self._ts.items:
            self._ts = None
            return
        self._drag = st
        self.gizmo.active = handle

    def _drag_to(self, sx: float, sy: float, snap: bool) -> None:
        st = self._drag
        ro, rd = self._proj.ray(sx, sy)
        o = st["o"]
        dpos, dq = np.zeros(3), mx.QID.copy()
        h = st["h"]
        if "t0" in st:
            t = mx.closest_on_line_to_ray(o, st["a"], ro, rd) - st["t0"]
            if snap:
                t = round(t / 0.01) * 0.01
            dpos = st["a"] * t
        elif "p0" in st:
            p = mx.ray_plane(ro, rd, o, st["n"])
            if p is None:
                return
            dpos = p - st["p0"]
            if snap:
                dpos = np.round(dpos / 0.01) * 0.01
        else:
            a = st["a"]
            p = mx.ray_plane(ro, rd, o, a)
            if st.get("v0") is not None and p is not None and abs(rd @ a) > 0.15:
                v0, v = st["v0"], p - o
                ang = math.atan2(a @ np.cross(v0, v), v0 @ v)
            else:  # ring seen edge-on: use horizontal mouse motion
                ang = (sx - st["s0"][0]) * 0.01
            if snap:
                step = math.radians(15)
                ang = round(ang / step) * step
            dq = mx.axang2q(a, ang)
        self._ts.update(dpos, dq, o)
        self.gizmo.origin = o + dpos
        if self.gizmo.space == "local":
            self.gizmo.R = mx.q2mat(dq) @ st["R"]
        txt = (f"Δ = {mx.fmt_vec(dpos, 4)} m" if h in ("x", "y", "z", "xy", "yz", "xz", "view")
               else f"Δθ = {math.degrees(mx.q2axang(dq)[1]):.1f}°")
        self.status.emit(txt)

    def mouseMoveEvent(self, ev: QMouseEvent) -> None:
        sx, sy = self._dev(ev.position())
        if self._drag is not None and self._ts is not None:
            self._drag_to(sx, sy, bool(ev.modifiers() & Qt.ControlModifier))
            return
        if self._last is None or ev.buttons() == Qt.NoButton:
            if self._proj is not None and self.gizmo.visible:
                h = self.gizmo.hit(self._proj, sx, sy)
                if h != self.gizmo.hover:
                    self.gizmo.hover = h
                    self.update()
            return
        m = self.ed.model
        if m is None or self.scn is None:
            return
        dx = ev.position().x() - self._last.x()
        dy = ev.position().y() - self._last.y()
        self._last = ev.position()
        hgt = max(self.height(), 1)
        shift = bool(ev.modifiers() & Qt.ShiftModifier)
        b = ev.buttons()
        A = mujoco.mjtMouse
        if b & Qt.LeftButton:
            act = A.mjMOUSE_ROTATE_H if shift else A.mjMOUSE_ROTATE_V
        elif b & Qt.RightButton:
            act = A.mjMOUSE_MOVE_H if shift else A.mjMOUSE_MOVE_V
        elif b & Qt.MiddleButton:
            act = A.mjMOUSE_ZOOM
        else:
            return
        mujoco.mjv_moveCamera(m, act, dx / hgt, dy / hgt, self.scn, self.cam)
        self.update()

    def mouseReleaseEvent(self, ev: QMouseEvent) -> None:
        if self._drag is not None and self._ts is not None:
            h = self._drag["h"]
            self._ts.commit("Rotate" if h.startswith("r") else "Move")
            self._ts = None
            self._drag = None
            self.gizmo.active = None
            self.update()
            return
        moved = self._press is not None and (ev.position() - self._press).manhattanLength() > 4
        if ev.button() == Qt.LeftButton and not moved:
            sx, sy = self._dev(ev.position())
            e, _ = self.pick(sx, sy)
            add = bool(ev.modifiers() & (Qt.ShiftModifier | Qt.ControlModifier))
            self.picked.emit(e, add)
        self._press = None
        self._last = None

    def mouseDoubleClickEvent(self, ev: QMouseEvent) -> None:
        sx, sy = self._dev(ev.position())
        _e, p = self.pick(sx, sy)
        if p is not None:
            self.cam.lookat[:] = p
            self.update()

    def wheelEvent(self, ev: QWheelEvent) -> None:
        if self.ed.model is None or self.scn is None:
            return
        dy = ev.angleDelta().y() / 120.0
        mujoco.mjv_moveCamera(self.ed.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.05 * dy, self.scn, self.cam)
        self.update()

    def keyPressEvent(self, ev) -> None:
        k = ev.key()
        if k == Qt.Key_Escape and self._ts is not None:
            self._ts.cancel()
            self._ts = None
            self._drag = None
            self.gizmo.active = None
            self.update()
            return
        super().keyPressEvent(ev)

    # ------------------------------------------------------------- camera helpers
    def frame_selection(self) -> None:
        m, d = self.ed.model, self.ed.data
        if m is None:
            return
        pts, rad = [], []
        for e in self.ed.selection:
            o = self.ed.idmap.obj(e) if self.ed.idmap else None
            if o and o[0] == int(OBJ.mjOBJ_BODY):
                gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == o[1]]
                for g in gs:
                    pts.append(d.geom_xpos[g])
                    rad.append(m.geom_rbound[g] if m.geom_rbound[g] > 0 else 0.1)
                if not gs:
                    pts.append(d.xpos[o[1]])
                    rad.append(0.1)
            elif o and o[0] == int(OBJ.mjOBJ_GEOM):
                pts.append(d.geom_xpos[o[1]])
                rad.append(m.geom_rbound[o[1]] if m.geom_rbound[o[1]] > 0 else 0.5)
            else:
                pts.append(self.ed.world_pose(e)[0])
                rad.append(0.1)
        if not pts:
            self.reset_camera()
            return
        P = np.array(pts)
        c = P.mean(axis=0)
        r = max(float(np.max(np.linalg.norm(P - c, axis=1) + np.array(rad))), 0.05)
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:] = c
        self.cam.distance = r * 3.0
        self.update()

    def grab_image(self):
        """Render and return a QImage of the viewport (works offscreen)."""
        return self.grabFramebuffer()
