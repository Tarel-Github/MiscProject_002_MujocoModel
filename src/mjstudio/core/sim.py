"""Simulation control on top of an Editor (GUI independent)."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import mujoco
import numpy as np

from .editor import Editor

log = logging.getLogger("mjstudio")
OBJ = mujoco.mjtObj


@dataclass
class ContactInfo:
    geom1: int
    geom2: int
    body1: int
    body2: int
    pos: np.ndarray
    normal_force: float
    dist: float


class Simulator:
    """Real-time stepping, reset and instability detection for editor.data."""

    MAX_STEPS_PER_TICK = 2000

    def __init__(self, ed: Editor):
        self.ed = ed
        self.running = False
        self.speed = 1.0
        self.last_issue: str | None = None
        self._warn = None
        self._wall = None
        self.on_issue = None  # optional callback(str)
        self.hooks_pre_step: list = []   # callables(model, data) run before each mj_step
        self.hooks_post_step: list = []  # callables(model, data) run after each mj_step

    # ------------------------------------------------------------------ control
    def play(self) -> None:
        if self.ed.model is None:
            return
        self.running = True
        self._wall = time.perf_counter()
        self._warn = self._warn_counts()

    def pause(self) -> None:
        self.running = False

    def toggle(self) -> None:
        self.pause() if self.running else self.play()

    def reset(self, keyframe: int | None = None) -> None:
        self.ed.reset_pose(keyframe)
        self.last_issue = None
        self._warn = self._warn_counts()
        self._wall = time.perf_counter()

    def step(self, n: int = 1) -> str | None:
        m, d = self.ed.model, self.ed.data
        if m is None:
            return None
        if self._warn is None:
            self._warn = self._warn_counts()
        for _ in range(n):
            for h in self.hooks_pre_step:
                h(m, d)
            mujoco.mj_step(m, d)
            for h in self.hooks_post_step:
                h(m, d)
        issue = self.check()
        self.ed.state_changed.emit()
        return issue

    def tick(self) -> int:
        """Advance simulated time to follow wall clock (called from a GUI timer)."""
        if not self.running or self.ed.model is None:
            return 0
        now = time.perf_counter()
        if self._wall is None:
            self._wall = now
        wall_dt = (now - self._wall) * self.speed
        self._wall = now
        dt = self.ed.model.opt.timestep
        n = int(min(self.MAX_STEPS_PER_TICK, max(1, round(wall_dt / dt))))
        issue = self.step(n)
        if issue:
            self.pause()
        return n

    # ------------------------------------------------------------------ checks
    def _warn_counts(self):
        d = self.ed.data
        if d is None:
            return None
        return [int(d.warning[i].number) for i in range(len(d.warning))]

    def check(self) -> str | None:
        d, m = self.ed.data, self.ed.model
        msgs = []
        if not (np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))):
            msgs.append("qpos/qvel 값이 유한하지 않음")
        cur = self._warn_counts()
        if self._warn is not None and cur is not None:
            for i, (a, b) in enumerate(zip(self._warn, cur)):
                if b > a:
                    msgs.append(f"{mujoco.mjtWarning(i).name} (+{b - a})")
        self._warn = cur
        if m.nv and float(np.max(np.abs(d.qvel))) > 1e4:
            msgs.append(f"속도 발산 |qvel|최대={float(np.max(np.abs(d.qvel))):.3g}")
        if msgs:
            self.last_issue = f"t={d.time:.3f}s: " + "; ".join(msgs)
            log.warning("물리 불안정: %s", self.last_issue)
            if self.on_issue:
                self.on_issue(self.last_issue)
            return self.last_issue
        return None

    # ------------------------------------------------------------------ readout
    def sensors(self) -> list[tuple[str, str, np.ndarray]]:
        m, d = self.ed.model, self.ed.data
        if m is None:
            return []
        out = []
        for i in range(m.nsensor):
            e = self.ed.idmap.elem(int(OBJ.mjOBJ_SENSOR), i)
            name = (e.get("name") if e is not None else None) or f"{e.tag if e is not None else 'sensor'} #{i}"
            a, n = m.sensor_adr[i], m.sensor_dim[i]
            out.append((name, mujoco.mjtSensor(m.sensor_type[i]).name.replace("mjSENS_", "").lower(),
                        d.sensordata[a:a + n].copy()))
        return out

    def contacts(self) -> list[ContactInfo]:
        m, d = self.ed.model, self.ed.data
        if m is None:
            return []
        out = []
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            mujoco.mj_contactForce(m, d, i, f6)
            out.append(ContactInfo(int(c.geom1), int(c.geom2), int(m.geom_bodyid[c.geom1]),
                                   int(m.geom_bodyid[c.geom2]), c.pos.copy(), float(f6[0]), float(c.dist)))
        return out
