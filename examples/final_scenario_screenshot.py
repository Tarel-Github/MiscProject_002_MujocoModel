"""Open the saved final-scenario project in the editor and take screenshots (display / xvfb).

    python examples/final_scenario.py && python examples/final_scenario_screenshot.py docs/screenshots
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

from mjstudio.core.expr import FaceController  # noqa: E402
from mjstudio.core.project import Project  # noqa: E402
from mjstudio.gui.mainwin import MainWindow  # noqa: E402
from mjstudio.gui.theme import apply_dark  # noqa: E402

out = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/screenshots")
app = QApplication(sys.argv)
apply_dark(app)
w = MainWindow()
w.resize(1600, 1000)
w.show()
prj, doc = Project.load(HERE / "output" / "final_scenario" / "project")
w.project = prj
w.ed.set_doc(doc, prj.load_rigs())
ed, v = w.ed, w.view
rig = ed.rigs["face"]
fc = FaceController(ed.model, ed.data, rig, kinematic=True)
D = math.radians
fc.set_eye_rotation("left", D(20), D(5))
fc.set_eye_rotation("right", D(-12), D(-8))
fc.set_jaw_open(0.5)
rig.set("smile", 0.8)
rig.set("blink_left", 0.0)
ed.forward()
ed.select([ed.doc.find_named("joint", "jaw_open")])
w.face.rig_box.setCurrentText("face")
w.face._sync_sliders()
v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
v.cam.lookat[:] = ed.world_pose(ed.doc.find_named("body", "head"))[0]
v.cam.distance, v.cam.azimuth, v.cam.elevation = 0.6, 165, -5
w.face.parent().raise_()
import time  # noqa: E402
for _ in range(10):                 # let Qt finish layout passes before grabbing
    app.processEvents()
    time.sleep(0.05)
v.repaint()
app.processEvents()
w.grab().save(str(out / "final_scenario_editor.png"))
print("saved", out / "final_scenario_editor.png")
