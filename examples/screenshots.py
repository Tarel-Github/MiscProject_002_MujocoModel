"""Render documentation screenshots of the assembled humanoid (run under a display/xvfb).

    python examples/assemble_humanoid.py && python examples/screenshots.py docs/screenshots
"""
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from mjstudio.gui.app import configure_gl  # noqa: E402

configure_gl()
import mujoco  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from mjstudio.core.project import Project  # noqa: E402
from mjstudio.gui.mainwin import MainWindow  # noqa: E402
from mjstudio.gui.theme import apply_dark  # noqa: E402

out = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/screenshots")
out.mkdir(parents=True, exist_ok=True)
app = QApplication(sys.argv)
apply_dark(app)
w = MainWindow()
w.resize(1600, 1000)
w.show()
prj, doc = Project.load(HERE / "output" / "humanoid_assembled")
w.project = prj
w.ed.set_doc(doc)
w.joints.rebuild()
ed, v = w.ed, w.view
D = math.radians


def snap(name, full=False):
    import time
    for _ in range(8):                 # 패널 레이아웃이 다 잡힌 뒤에 찍기
        app.processEvents()
        time.sleep(0.03)
    v.repaint()
    app.processEvents()
    (w.grab() if full else v.grabFramebuffer()).save(str(out / f"{name}.png"))
    print("saved", name)


def look(body, dist, az, el):
    v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    v.cam.lookat[:] = ed.world_pose(ed.doc.find_named("body", body))[0]
    v.cam.distance, v.cam.azimuth, v.cam.elevation = dist, az, el


ed.reset_pose()
v.reset_camera()
ed.select([ed.doc.find_named("joint", "index_pip_left")])
snap("editor_overview", full=True)
# 폴리곤 편집: 정강이 캡슐을 편집 가능한 메시로 변환하고 정점 몇 개를 선택한 화면 (끝나면 되돌림)
from PySide6.QtWidgets import QMessageBox  # noqa: E402
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
shin = ed.doc.find_named("geom", "shin_left")
ed.select([shin])
look("shin_left", 0.9, 150, -15)
w.toggle_mesh_edit()
mode = v.mesh_mode
W = mode.world_vertices()
mode.sel = [int(i) for i in W[:, 2].argsort()[-6:]]    # 위쪽 끝 정점 6개
snap("polygon_edit", full=True)
w.toggle_mesh_edit()
ed.undo_step()
ed.select([ed.doc.find_named("joint", "index_pip_left")])
# hand: fist group half closed + thumb opposition
for g in prj.groups:
    if g.name == "fist_left":
        g.apply(ed, D(55))
ed.set_qpos("thumb_cmc_rot_left", [D(35)])
ed.set_qpos("thumb_cmc_abd_left", [D(30)])
look("hand_left", 0.45, 200, -15)
snap("hand_left_fist")
# head: eyes look to the left/up independently, jaw open
ed.set_qpos("eye_left_yaw", [D(30)])
ed.set_qpos("eye_right_yaw", [D(10)])
ed.set_qpos("eye_right_pitch", [D(20)])
ed.set_qpos("jaw_open", [D(22)])
ed.select([ed.doc.find_named("joint", "jaw_open")])
look("head", 0.45, 160, -5)
snap("head_eyes_jaw")
# foot: plantarflexed ankle with MTP dorsiflexion (heel rise shape)
ed.set_qpos("ankle_flex_left", [D(-30)])
ed.set_qpos("mtp_left", [D(30)])
ed.select([ed.doc.find_named("joint", "mtp_left")])
look("forefoot_left", 0.55, -90, -8)
snap("foot_heel_rise")
