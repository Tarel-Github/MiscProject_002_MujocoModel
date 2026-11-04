"""GUI smoke run (needs a display; in CI use xvfb-run). Saves screenshots.

usage: python tests/gui_smoke.py <out_dir> [model.xml]
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mjstudio.gui.app import configure_gl  # noqa: E402

configure_gl()
from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from mjstudio.gui.mainwin import MainWindow  # noqa: E402
from mjstudio.gui.theme import apply_dark  # noqa: E402

out = Path(sys.argv[1] if len(sys.argv) > 1 else "screens")
out.mkdir(parents=True, exist_ok=True)
model = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "examples" / "models" / "humanoid.xml")

app = QApplication(sys.argv)
apply_dark(app)
w = MainWindow()
w.resize(1600, 980)
w.show()
steps = []


def step(fn):
    steps.append(fn)


def run_next():
    if not steps:
        app.quit()
        return
    fn = steps.pop(0)
    try:
        fn()
    except Exception as ex:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print("STEP FAILED:", ex)
        os._exit(1)
    QTimer.singleShot(400, run_next)


def shot(name):
    def f():
        app.processEvents()
        img = w.grab()
        img.save(str(out / f"{name}.png"))
        vimg = w.view.grabFramebuffer()
        vimg.save(str(out / f"{name}_viewport.png"))
        print("saved", name, img.width(), img.height())
    return f


step(lambda: w.open_path(model))
step(lambda: w.ed.select([w.ed.doc.find_named("body", "thigh_left")]))
step(lambda: w.view.frame_selection())
step(lambda: (w.ed.set_qpos("knee_left", [-1.0]), w.ed.set_qpos("hip_y_left", [-0.6])))
step(shot("01_humanoid_body_selected"))
step(lambda: w.ed.select([w.ed.doc.find_named("joint", "knee_left")]))
step(lambda: w._gizmo("rotate"))
step(shot("02_joint_selected_range_arc"))
step(lambda: w.view.reset_camera())
step(lambda: w.toggle_sim())
step(lambda: None)
step(lambda: None)
step(shot("03_simulating"))
step(lambda: w.toggle_sim())


drag_info = {}


def gizmo_drag():
    """Drag the X handle of the move gizmo with real mouse events."""
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    v = w.view
    hand = w.ed.doc.find_named("body", "hand_left")
    w.ed.select([hand])
    w._gizmo("translate")
    v.repaint()
    app.processEvents()
    g, P = v.gizmo, v._proj
    a = P.project(g.origin + g.R[:, 0] * g.L * 0.7)
    b = P.project(g.origin + g.R[:, 0] * g.L * 1.4)
    drag_info["x0"] = w.ed.world_pose(hand)[0].copy()
    drag_info["pos0"] = hand.get("pos")
    p0 = QPoint(int(a[0]), int(a[1]))
    p1 = QPoint(int(b[0]), int(b[1]))
    QTest.mouseMove(v, p0)
    QTest.mousePress(v, Qt.LeftButton, Qt.NoModifier, p0)
    for k in range(1, 6):
        QTest.mouseMove(v, p0 + (p1 - p0) * k / 5)
    QTest.mouseRelease(v, Qt.LeftButton, Qt.NoModifier, p1)
    x1 = w.ed.world_pose(hand)[0]
    d = x1 - drag_info["x0"]
    print("gizmo drag moved hand by", d.round(4), "pos attr", drag_info["pos0"], "->", hand.get("pos"))
    assert d[0] > 0.01 and abs(d[1]) < 1e-6 and abs(d[2]) < 1e-6, "drag should move along +x only"
    assert w.ed.undo.undo_text() == "Move"


def undo_drag():
    hand = w.ed.doc.find_named("body", "hand_left")
    w.ed.undo_step()
    assert hand.get("pos") == drag_info["pos0"], hand.get("pos")
    print("undo restored", hand.get("pos"))


def report():
    ed = w.ed
    print("compile ok:", ed.result.ok, "nbody", ed.model.nbody, "sim time", round(ed.data.time, 3))
    print("tree items:", len(w.tree.items), "joint rows:", len(w.joints.rows))


step(lambda: w.sim.reset())
step(lambda: w.view.reset_camera())
step(gizmo_drag)
step(shot("04_after_gizmo_drag"))
step(undo_drag)
step(report)
QTimer.singleShot(500, run_next)
app.exec()
print("GUI smoke finished")
