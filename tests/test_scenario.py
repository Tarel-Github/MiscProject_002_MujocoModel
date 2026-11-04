"""The final evaluation scenario as a regression test (see examples/final_scenario.py)."""
import importlib.util
import math
from pathlib import Path

import mujoco
import numpy as np

EX = Path(__file__).resolve().parents[1] / "examples" / "final_scenario.py"


def test_final_scenario(tmp_path):
    spec = importlib.util.spec_from_file_location("final_scenario", EX)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    r = mod.run(tmp_path, verbose=False)
    ed = r["ed"]
    # finger segment modified, new forefoot joint present, Blender face assembled
    j = ed.joint_id("index_pip_left")
    assert np.allclose(np.degrees(ed.model.jnt_range[j]), [-5, 95])
    assert ed.joint_id("hallux_ip_left") >= 0
    assert ed.model.nskin == 1 and len(ed.rigs["face"].keys) == 17
    # eyes rotated separately, jaw moved
    for side, (yaw, pitch) in mod.EYES.items():
        assert np.allclose(r["eyes"][side], (yaw, pitch), atol=math.radians(1.5)), side
    assert math.radians(14) < r["jaw"] < math.radians(22)
    # expressions on the simulation clock at the same time
    t = np.array(r["r1"]["log"]["t"])
    assert r["r1"]["log"]["smile"][-1] == 0.8
    assert r["r1"]["log"]["blink_left"][int(np.argmin(np.abs(t - 0.7)))] > 0.99
    assert r["r1"]["log"]["blink_left"][int(np.argmin(np.abs(t - 0.95)))] == 0.0
    # saved, reloaded and simulated normally (and deterministically)
    assert r["issues"] == 0 and r["same"]
    assert mujoco.MjModel.from_xml_path(str(r["project"].model_path)).nskin == 1
