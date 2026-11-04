"""Render the procedural face in several expressions (run under a display / xvfb).

    python examples/face_screenshots.py docs/screenshots
"""
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from mjstudio.gui.app import configure_gl  # noqa: E402

configure_gl()
import mujoco  # noqa: E402
from PySide6.QtGui import QImage, QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from mjstudio.core.editor import Editor  # noqa: E402
from mjstudio.core.expr import FaceController  # noqa: E402
from mjstudio.gui.viewport import Viewport  # noqa: E402
from mjstudio.library import anatomy as A  # noqa: E402

out = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/screenshots")
out.mkdir(parents=True, exist_ok=True)
app = QApplication(sys.argv)
ed = Editor()
ed.new("face_demo")
A.build_head(ed, None, pos=(0, 0, 0.3), neck=False, face=True)
v = Viewport(ed)
v.resize(420, 420)
v.gizmo.mode = "none"
v.show()
rig = ed.rigs["face"]
fc = FaceController(ed.model, ed.data, rig)
D = math.radians
shots = [("neutral", {}, 0.0, (0, 0)), ("smile", {"preset": "smile"}, 0.0, (0, 0)),
         ("frown", {"preset": "frown"}, 0.0, (0, 0)), ("surprise", {"preset": "surprise"}, 0.45, (0, 8)),
         ("blink", {"preset": "blink"}, 0.0, (0, 0)), ("wink_left", {"preset": "wink_left"}, 0.0, (0, 0)),
         ("jaw_open_smile", {"preset": "smile"}, 0.8, (0, 0)), ("eyes_left_up", {}, 0.0, (30, 15))]
CAPTION = {"neutral": "무표정 (neutral)", "smile": "미소 (smile)", "frown": "찡그림 (frown)",
           "surprise": "놀람 (surprise) + 턱 벌림", "blink": "눈 감기 (blink)", "wink_left": "윙크 왼쪽 (wink_left)",
           "jaw_open_smile": "미소 + 턱 크게 벌림", "eyes_left_up": "시선: 왼쪽 위 (눈 관절)"}
tiles = []
for name, ex, jaw, (yaw, pitch) in shots:
    rig.reset()
    if "preset" in ex:
        rig.apply_preset(ex["preset"])
    for side in ("left", "right"):
        ed.set_qpos(f"eye_{side}_yaw", [D(yaw)], forward=False)
        ed.set_qpos(f"eye_{side}_pitch", [D(pitch)], forward=False)
    j = ed.joint_id("jaw_open")
    ed.data.qpos[ed.model.jnt_qposadr[j]] = jaw * ed.model.jnt_range[j][1]
    ed.forward()
    v.cam.lookat[:] = [0.05, 0, 0.3]
    v.cam.distance, v.cam.azimuth, v.cam.elevation = 0.42, 180 - 20, -5
    v.repaint()
    app.processEvents()
    img = v.grabFramebuffer()
    tiles.append((name, img))
W = 4
sheet = QImage(420 * W, 420 * ((len(tiles) + W - 1) // W), QImage.Format_RGB32)
p = QPainter(sheet)
for k, (name, img) in enumerate(tiles):
    x, y = 420 * (k % W), 420 * (k // W)
    p.drawImage(x, y, img)
    p.setPen(0xFFFFFF)
    p.drawText(x + 10, y + 405, CAPTION.get(name, name))
p.end()
sheet.save(str(out / "face_expressions.png"))
print("saved", out / "face_expressions.png")
