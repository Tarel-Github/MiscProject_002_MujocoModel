"""Render the library humanoid (run under a display / xvfb): python examples/humanoid_screenshot.py out.png"""
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from mjstudio.gui.app import configure_gl  # noqa: E402

configure_gl()
from PySide6.QtGui import QImage, QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from mjstudio.gui.viewport import Viewport  # noqa: E402
from mjstudio.library.humanoid import make_humanoid_doc  # noqa: E402

app = QApplication(sys.argv)
ed = make_humanoid_doc()
v = Viewport(ed)
v.resize(640, 800)
v.gizmo.mode = "none"
v.show()
rig = ed.rigs["face"]
D = math.radians
views = []
for name, pose, expr, cam in (
        ("선 자세 (qpos0)", {}, None, (0.0, 1.0, 2.6, 150, -8)),
        ("자세: 주먹·미소·시선·턱", {"shoulder_flex_left": 70, "elbow_left": 80, "shoulder_abd_right": 40,
                                          "hip_flex_right": 40, "knee_right": 60, "ankle_flex_right": -20,
                                          "eye_left_yaw": 25, "eye_right_yaw": 25, "jaw_open": 15,
                                          **{f"{f}_{j}_left": 75 for f in ("index", "middle", "ring", "little")
                                             for j in ("mcp_flex", "pip", "dip")}},
         "smile", (0.0, 1.05, 2.4, 140, -5)),
        ("얼굴 확대", {"eye_left_yaw": 25, "eye_right_yaw": 25, "jaw_open": 15}, "smile",
         (0.0, 1.62, 0.55, 170, -5))):
    ed.reset_pose()
    for j, deg in pose.items():
        ed.set_qpos(j, [D(deg)], forward=False)
    ed.forward()
    rig.reset()
    if expr:
        rig.apply_preset(expr)
    v.cam.lookat[:] = [cam[0], 0, cam[1]]
    v.cam.distance, v.cam.azimuth, v.cam.elevation = cam[2], cam[3], cam[4]
    v.repaint()
    app.processEvents()
    views.append((name, v.grabFramebuffer()))
sheet = QImage(640 * len(views), 800, QImage.Format_RGB32)
p = QPainter(sheet)
for k, (name, img) in enumerate(views):
    p.drawImage(640 * k, 0, img)
    p.setPen(0xFFFFFF)
    p.drawText(640 * k + 12, 785, name)
p.end()
out = sys.argv[1] if len(sys.argv) > 1 else "humanoid.png"
sheet.save(out)
print("saved", out)
