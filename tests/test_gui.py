"""GUI integration test (needs an X display with OpenGL; run under xvfb-run on headless Linux)."""
import os
import sys

import numpy as np
import pytest

if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") or sys.platform == "win32"):
    pytest.skip("no display available (use xvfb-run)", allow_module_level=True)

from mjstudio.gui.app import configure_gl  # noqa: E402

configure_gl()
from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def app():
    from mjstudio.gui.theme import apply_dark
    a = QApplication.instance() or QApplication([])
    apply_dark(a)
    return a


@pytest.fixture
def win(app, humanoid_path):
    from mjstudio.gui.mainwin import MainWindow
    w = MainWindow()
    w.resize(1400, 900)
    w.show()
    assert w.open_path(str(humanoid_path))
    app.processEvents()
    yield w
    w.ed.undo.set_clean()
    w.close()


def _render(app, w):
    w.view.repaint()
    app.processEvents()


def test_viewport_renders_model(app, win):
    _render(app, win)
    img = win.view.grabFramebuffer()
    arr = np.array([[img.pixelColor(x, y).red() for x in range(0, img.width(), 25)]
                    for y in range(0, img.height(), 25)])
    assert arr.std() > 5, "viewport looks blank"
    assert win.view.con is not None and win.view._proj is not None


def test_projection_matches_render(app, win):
    """A projected geom centre must land on pixels of that geom (pick it back)."""
    _render(app, win)
    m, d = win.ed.model, win.ed.data
    gid = [i for i in range(m.ngeom) if (win.ed.idmap.elem(5, i) is not None and
                                         win.ed.idmap.elem(5, i).get("name") == "head")][0]
    sx, sy, _ = win.view._proj.project(d.geom_xpos[gid])
    win.view.pick_mode = "geom"
    e, _p = win.view.pick(sx, sy)
    assert e is not None and e.get("name") == "head"


def test_click_select_and_props(app, win):
    _render(app, win)
    m, d = win.ed.model, win.ed.data
    bid = win.ed.idmap.id(win.ed.doc.find_named("body", "head"))
    sx, sy, _ = win.view._proj.project(d.xpos[bid])
    QTest.mouseClick(win.view, Qt.LeftButton, Qt.NoModifier, QPoint(int(sx), int(sy)))
    app.processEvents()
    assert [e.get("name") for e in win.ed.selection] == ["head"]
    assert win.props.elem is win.ed.selection[0]
    assert win.tree.items[win.ed.selection[0]].isSelected()


def test_gizmo_drag_and_undo(app, win):
    hand = win.ed.doc.find_named("body", "hand_left")
    win.ed.select([hand])
    win._gizmo("translate")
    _render(app, win)
    g, P = win.view.gizmo, win.view._proj
    a = P.project(g.origin + g.R[:, 2] * g.L * 0.7)
    b = P.project(g.origin + g.R[:, 2] * g.L * 1.3)
    x0 = win.ed.world_pose(hand)[0].copy()
    p0, p1 = QPoint(int(a[0]), int(a[1])), QPoint(int(b[0]), int(b[1]))
    QTest.mouseMove(win.view, p0)
    QTest.mousePress(win.view, Qt.LeftButton, Qt.NoModifier, p0)
    for k in range(1, 5):
        QTest.mouseMove(win.view, p0 + (p1 - p0) * k / 4)
    QTest.mouseRelease(win.view, Qt.LeftButton, Qt.NoModifier, p1)
    dx = win.ed.world_pose(hand)[0] - x0
    assert dx[2] > 0.01 and np.allclose(dx[:2], 0, atol=1e-6)
    win.ed.undo_step()
    assert hand.get("pos") == ".18 -.18 .18"


def test_property_edit_roundtrip(app, win, tmp_path):
    knee = win.ed.doc.find_named("joint", "knee_left")
    win.ed.select([knee])
    app.processEvents()
    row = next(r for r in win.props.rows if r.spec.name == "damping")
    row.w.setText("7.5")
    row._line_done()
    assert knee.get("damping") == "7.5"
    jid = win.ed.joint_id("knee_left")
    assert win.ed.model.dof_damping[win.ed.model.jnt_dofadr[jid]] == pytest.approx(7.5)
    out = tmp_path / "edited.xml"
    assert win._save_to(out)
    import mujoco
    m2 = mujoco.MjModel.from_xml_path(str(out))
    j2 = mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_JOINT, "knee_left")
    assert m2.dof_damping[m2.jnt_dofadr[j2]] == pytest.approx(7.5)


def test_joint_slider_moves_model(app, win):
    rows = [r for r in win.joints.rows if r.info["label"] == "knee_left"]
    assert rows
    rows[0]._set(-45.0)
    assert win.ed.get_qpos("knee_left")[0] == pytest.approx(np.radians(-45.0))


def test_simulation_runs(app, win):
    win.sim.play()
    for _ in range(5):
        win.sim.tick()
    win.sim.step(50)
    win.sim.pause()
    assert win.ed.data.time > 0
