"""Structured sensory read-out for learning agents (extension point for future senses).

    bus = SensorBus(model)
    bus.register("gaze_left", lambda m, d: fc.eye_direction("left"), group="vision")
    obs = bus.read_groups(data)      # {"touch": array, "proprio_pos": array, "vestibular": ...}
    layout = bus.spec()              # channel -> (group, offset in group, dim, kind)

MuJoCo sensors are grouped by type; custom channels (expression state, gaze, or a
future pain / skin model) are plain callables evaluated on the same MjData, so every
channel shares the simulation clock.
"""
from __future__ import annotations

from typing import Callable

import mujoco
import numpy as np

S = mujoco.mjtSensor
GROUP_OF = {
    "touch": (S.mjSENS_TOUCH,),
    "proprio_pos": (S.mjSENS_JOINTPOS, S.mjSENS_BALLQUAT, S.mjSENS_TENDONPOS, S.mjSENS_ACTUATORPOS),
    "proprio_vel": (S.mjSENS_JOINTVEL, S.mjSENS_BALLANGVEL, S.mjSENS_TENDONVEL, S.mjSENS_ACTUATORVEL),
    "vestibular": (S.mjSENS_GYRO, S.mjSENS_ACCELEROMETER),
    "pose": (S.mjSENS_FRAMEPOS, S.mjSENS_FRAMEQUAT, S.mjSENS_FRAMEXAXIS, S.mjSENS_FRAMEYAXIS,
             S.mjSENS_FRAMEZAXIS, S.mjSENS_FRAMELINVEL, S.mjSENS_FRAMEANGVEL),
    "force": (S.mjSENS_FORCE, S.mjSENS_TORQUE),
    "effort": (S.mjSENS_ACTUATORFRC,),
}


class SensorBus:
    def __init__(self, model: mujoco.MjModel):
        self.m = model
        self.channels: dict[str, dict] = {}
        kind_group = {int(t): g for g, ts in GROUP_OF.items() for t in ts}
        for i in range(model.nsensor):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i) or f"sensor{i}"
            t = int(model.sensor_type[i])
            self.channels[name] = {"kind": mujoco.mjtSensor(t).name[7:].lower(), "group": kind_group.get(t, "other"),
                                   "adr": int(model.sensor_adr[i]), "dim": int(model.sensor_dim[i]), "fn": None}

    def register(self, name: str, fn: Callable[[mujoco.MjModel, mujoco.MjData], np.ndarray], group: str = "custom",
                 kind: str = "custom") -> None:
        """Add a non-MuJoCo channel (evaluated on demand on the same MjData)."""
        self.channels[name] = {"kind": kind, "group": group, "adr": -1, "dim": None, "fn": fn}

    def read(self, d: mujoco.MjData) -> dict[str, np.ndarray]:
        out = {}
        for n, c in self.channels.items():
            if c["fn"] is not None:
                out[n] = np.atleast_1d(np.asarray(c["fn"](self.m, d), float))
            else:
                out[n] = d.sensordata[c["adr"]:c["adr"] + c["dim"]].copy()
        return out

    def read_groups(self, d: mujoco.MjData) -> dict[str, np.ndarray]:
        vals = self.read(d)
        groups: dict[str, list] = {}
        for n, c in self.channels.items():
            groups.setdefault(c["group"], []).append(vals[n])
        return {g: np.concatenate(v) if v else np.zeros(0) for g, v in groups.items()}

    def spec(self, d: mujoco.MjData | None = None) -> dict[str, tuple]:
        """channel -> (group, offset within the group vector, dim, kind)."""
        vals = self.read(d) if d is not None else {}
        off: dict[str, int] = {}
        out = {}
        for n, c in self.channels.items():
            dim = c["dim"] if c["dim"] is not None else (len(vals[n]) if n in vals else -1)
            g = c["group"]
            out[n] = (g, off.get(g, 0), dim, c["kind"])
            off[g] = off.get(g, 0) + max(dim, 0)
        return out
