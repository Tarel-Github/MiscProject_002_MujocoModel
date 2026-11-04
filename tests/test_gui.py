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


def test_phase2_handlers_with_scripted_dialogs(app, win, tmp_path, monkeypatch):
    """Library insert, part import/export/replace, coupling, joint group -> project."""
    from pathlib import Path
    from PySide6.QtWidgets import QDialog, QFileDialog, QInputDialog, QMessageBox
    from mjstudio.gui import dialogs
    root = Path(__file__).resolve().parents[1]
    for cls in (dialogs.LibraryDialog, dialogs.PickBodyDialog, dialogs.JointGroupDialog):
        monkeypatch.setattr(cls, "exec", lambda self: QDialog.Accepted)
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: QMessageBox.Ok))
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: QMessageBox.Ok))
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(root / "examples/parts/hand_left.xml"), "")))
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(tmp_path / "arm.xml"), "")))
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(tmp_path / "proj")))
    monkeypatch.setattr(QInputDialog, "getDouble", staticmethod(lambda *a, **k: (0.5, True)))
    ed = win.ed
    n0 = ed.model.njnt
    ed.select([ed.doc.find_named("body", "torso")])
    win.insert_library("hand")                 # 'hand_left' already exists -> renamed
    assert ed.result.ok and ed.model.njnt == n0 + 22, ed.result.error
    win.insert_library("foot")
    win.import_part_file()
    assert ed.result.ok and ed.model.njnt == n0 + 22 + 5 + 22, ed.result.error
    ed.select([ed.doc.find_named("body", "upper_arm_left")])
    win.export_part()
    assert (tmp_path / "arm.xml").exists()
    ed.select([ed.doc.find_named("joint", "knee_left"), ed.doc.find_named("joint", "knee_right")])
    win.couple_selected()
    assert ed.doc.section("equality") is not None and ed.result.ok
    win.group_selected()
    assert win.project is not None and win.project.groups and (tmp_path / "proj" / "project.json").exists()
    ed.select([ed.doc.find_named("body", "head")])
    win.replace_with_part()
    assert ed.result.ok, ed.result.error
    win.project = None


def test_mesh_edit_mode_and_face_panel(app, win, monkeypatch):
    """Primitive mesh -> vertex edit mode (pick, gizmo drag, delete, poke, undo); face panel."""
    import mujoco
    from PySide6.QtWidgets import QMessageBox
    from mjstudio.core import meshops
    from mjstudio.gui.meshedit import MeshEditMode
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: QMessageBox.Ok))
    ed = win.ed
    out = meshops.create_primitive_mesh(ed, ed.doc.find_named("body", "torso"), "box", pos=(0.3, 0, 0))
    geom = out["geoms"][0]
    ed.select([geom])
    win.toggle_mesh_edit()
    mode = win.view.mesh_mode
    assert isinstance(mode, MeshEditMode)
    _render(app, win)
    W = mode.world_vertices()
    top = int(np.argmax(W[:, 2] + 0.01 * W[:, 0] + 0.001 * W[:, 1]))
    sx, sy, _ = win.view._proj.project(W[top])
    mode.pick(win.view._proj, sx, sy, False)
    assert mode.sel == [top]
    _render(app, win)
    g, P = win.view.gizmo, win.view._proj
    a = P.project(g.origin + g.R[:, 2] * g.L * 0.7)
    b = P.project(g.origin + g.R[:, 2] * g.L * 1.4)
    p0, p1 = QPoint(int(a[0]), int(a[1])), QPoint(int(b[0]), int(b[1]))
    QTest.mouseMove(win.view, p0)
    QTest.mousePress(win.view, Qt.LeftButton, Qt.NoModifier, p0)
    for k in range(1, 5):
        QTest.mouseMove(win.view, p0 + (p1 - p0) * k / 4)
    QTest.mouseRelease(win.view, Qt.LeftButton, Qt.NoModifier, p1)
    assert ed.result.ok, ed.result.error
    W2 = win.view.mesh_mode.world_vertices()
    assert W2[top, 2] > W[top, 2] + 0.005 and np.allclose(np.delete(W2, top, 0), np.delete(W, top, 0), atol=1e-6)
    ed.undo_step()
    assert np.allclose(win.view.mesh_mode.world_vertices(), W, atol=1e-6)
    # add a vertex inside a face (poke) and delete it again
    mode.face, mode.sel = 0, []
    mode.poke()
    assert ed.result.ok and meshops.get_mesh_data(ed, mode.mesh).nv == 9
    mode.delete()                       # deletes the new (selected) vertex and its 3 faces
    d = meshops.get_mesh_data(ed, mode.mesh)
    assert d.nv == 8 and d.nf == 11
    win.toggle_mesh_edit()
    assert win.view.mesh_mode is None
    # face panel: procedural face on a fresh head, preset buttons and sliders
    from mjstudio.library import anatomy as A
    ed.new("face")
    A.build_head(ed, None, pos=(0, 0, 1.6), neck=False, face=True)
    app.processEvents()
    fp = win.face
    assert fp.rig is not None and "smile" in fp.sliders
    fp.apply_preset("smile")
    assert fp.sliders["smile"].value() == 100
    fp.sliders["blink_left"].setValue(100)
    assert fp.rig.get("blink_left") == pytest.approx(1.0)
    fp.jaw.setValue(50)
    j = ed.joint_id("jaw_open")
    assert ed.data.qpos[ed.model.jnt_qposadr[j]] == pytest.approx(0.5 * ed.model.jnt_range[j][1], rel=1e-3)
    fp.eye[("left", "yaw")].setValue(200)      # 20 deg, linked -> both eyes
    assert ed.get_qpos("eye_right_yaw")[0] == pytest.approx(np.radians(20), abs=1e-3)
    _render(app, win)
    m, dd = ed.model, ed.data
    assert m.nskin == 1


def test_add_panel_finger_mesh_import_and_context_menu(app, win, tmp_path, monkeypatch):
    """'추가·편집' 패널: 손가락 하나 추가, 내 3D 모델 붙이기, 기본 형상 → 메시 변환 후 폴리곤 편집, 우클릭 메뉴."""
    from PySide6.QtWidgets import QDialog, QDockWidget, QFileDialog, QMessageBox, QPushButton
    from mjstudio.core import meshes as M
    from mjstudio.gui import dialogs
    from mjstudio.gui.mainwin import GUIDE
    from mjstudio.library import anatomy as A
    def fail(*a, **k):
        raise AssertionError(f"unexpected message box: {a[1:3]}")
    for nm in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, nm, staticmethod(fail))
    ed = win.ed
    docks = {d.objectName(): d for d in win.findChildren(QDockWidget)}
    assert docks["addedit"].windowTitle() == "추가·편집" and docks["addedit"].widget() is win.addp
    buttons = {b.text(): b for b in win.addp.findChildren(QPushButton)}
    for t in ("손가락 하나…", "3D 모델 파일 붙이기 (OBJ·STL·GLB)…", "폴리곤 편집 시작 / 끝내기", "형상을 메시로 변환"):
        assert t in buttons, t
    assert "손가락" in GUIDE and "폴리곤" in GUIDE
    # 1) 손가락 하나: 손 바디를 선택하고 패널 버튼 클릭 (대화상자는 기본값으로 확인)
    hand = ed.doc.find_named("body", "hand_left")
    ed.select([hand])
    app.processEvents()
    assert "hand_left" in win.addp.target.text()
    dlg = dialogs.FingerDialog("hand_left", lambda k, s: A.finger_default_pos(k, s))
    dlg.kind.setCurrentIndex(4)                        # 엄지: 기본 위치가 바뀐다
    assert np.allclose([s.value() for s in dlg.pos], A.finger_default_pos("thumb", "left"), atol=1e-4)
    monkeypatch.setattr(dialogs.FingerDialog, "exec", lambda self: QDialog.Accepted)
    nj, nu = ed.model.njnt, ed.model.nu
    buttons["손가락 하나…"].click()
    assert ed.result.ok, ed.result.error
    assert ed.model.njnt == nj + 4 and ed.model.nu == nu + 4    # 검지: MCP 굽힘·벌림, PIP, DIP
    new = ed.selection[-1]
    assert new.tag == "body" and new.getparent() is hand
    assert any((j.get("name") or "").endswith("index_pip_left") for j in new.iter("joint"))
    # 2) 내 3D 모델(OBJ, mm 단위)을 선택한 바디에 붙이기
    obj = tmp_path / "bracket.obj"
    obj.write_bytes(M.write_obj(M.box(20, 10, 5)))     # mm 단위로 만든 모델
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(obj), "")))

    def accept_mm(self):
        self.units.setCurrentIndex(2)                  # 밀리미터
        return QDialog.Accepted
    monkeypatch.setattr(dialogs.MeshImportDialog, "exec", accept_mm)
    torso = ed.doc.find_named("body", "torso")
    ed.select([torso])
    ng = ed.model.ngeom
    buttons["3D 모델 파일 붙이기 (OBJ·STL·GLB)…"].click()
    assert ed.result.ok, ed.result.error
    assert ed.model.ngeom == ng + 2                     # 보이는 형상 + 충돌 형상
    g = ed.selection[-1]
    assert g.tag == "geom" and g.getparent() is torso
    gid = ed.idmap.id(g)
    assert ed.model.geom_rbound[gid] == pytest.approx(np.linalg.norm([0.02, 0.01, 0.005]), rel=0.05)
    # 3) 기본 형상(구) → 메시로 변환 → 폴리곤 편집 → Esc
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes))
    head = ed.doc.find_named("geom", "head")
    ed.select([head])
    buttons["폴리곤 편집 시작 / 끝내기"].click()
    assert ed.doc.get(head, "type") == "mesh" and win.view.mesh_mode is not None
    assert "편집 중" in win.addp.poly_state.text()
    assert "폴리곤 편집" in win.view.hud_texts()[1]
    QTest.keyClick(win.view, Qt.Key_Escape)
    assert win.view.mesh_mode is None and "편집 중" not in win.addp.poly_state.text()
    ed.undo_step()
    assert (ed.doc.get(head, "type") or "sphere") == "sphere"
    # 4) 3D 화면 오른쪽 클릭(끌지 않음) → 그 자리 요소의 메뉴
    seen = []
    monkeypatch.setattr(win, "_exec_menu", lambda menu, gpos: seen.append([x.text() for x in menu.actions()]))
    _render(app, win)
    bid = ed.idmap.id(ed.doc.find_named("body", "head"))
    sx, sy, _ = win.view._proj.project(ed.data.xpos[bid])
    dpr = win.view.devicePixelRatioF()
    QTest.mouseClick(win.view, Qt.RightButton, Qt.NoModifier, QPoint(int(sx / dpr), int(sy / dpr)))
    assert seen and "추가" in seen[-1] and "폴리곤 편집 시작 / 끝내기" in seen[-1] and "삭제" in seen[-1]
    assert [e.get("name") for e in ed.selection] == ["head"]
    # 빈 모델에서는 화면 아래에 '추가·편집' 안내
    ed.new("empty")
    app.processEvents()
    assert "추가·편집" in win.view.hud_texts()[2]
