"""메인 창: 메뉴, 도구막대, 도크 패널, 파일 처리."""
from __future__ import annotations

import logging
import os
from pathlib import Path

import mujoco
import numpy as np
from lxml import etree
from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QGuiApplication, QKeySequence
from PySide6.QtWidgets import (QDockWidget, QFileDialog, QInputDialog, QLabel, QMainWindow, QMenu, QMessageBox,
                               QToolBar)

from .. import __version__
from ..core import meshes as M
from ..core import meshops, ops, parts
from ..core.compiler import compile_doc, validate_steps
from ..core.doc import MjcfDoc, MjcfError
from ..core.editor import Editor
from ..core.project import Project, missing_resources
from ..core.sim import Simulator
from ..library import anatomy
from .addpanel import AddPanel
from .dialogs import FingerDialog, JointGroupDialog, LibraryDialog, MeshImportDialog, PickBodyDialog
from .facepanel import FacePanel
from .jointpanel import JointPanel
from .logpanel import LogPanel
from .props import PropsPanel
from .simpanel import SimPanel
from .theme import C
from .tree import TreePanel
from .viewport import Viewport

log = logging.getLogger("mjstudio")
VF = mujoco.mjtVisFlag
RF = mujoco.mjtRndFlag
MJCF_FILTER = "MuJoCo MJCF (*.xml *.mjcf);;모든 파일 (*)"
GEOM_KO = {"sphere": "구", "box": "박스", "capsule": "캡슐", "cylinder": "원기둥", "ellipsoid": "타원체",
           "plane": "평면"}
JOINT_KO = {"hinge": "경첩", "slide": "슬라이드", "ball": "볼", "free": "자유"}

GUIDE = """<h3>어디서 무엇을 하나요?</h3>
<p><b>왼쪽 아래 '추가·편집' 패널</b>에 모델에 무언가를 더하는 기능이 모여 있습니다.
먼저 계층 트리나 3D 화면에서 <b>바디를 클릭</b>하면 그 바디에 추가됩니다.</p>
<ul>
<li><b>손가락 추가</b>: 추가·편집 › 신체 부품 › <i>손가락 하나…</i> (손바닥 바디를 먼저 선택)</li>
<li><b>손·발·머리·전신</b>: 추가·편집 › 신체 부품 (또는 메뉴 모델 › 라이브러리 부품)</li>
<li><b>내가 만든 3D 모델 붙이기</b>: 추가·편집 › 내 3D 모델 › <i>3D 모델 파일 붙이기 (OBJ·STL·GLB)</i>
    — 단위(mm 등)와 좌표축(Blender)을 고를 수 있고, 보이는 형상과 충돌 형상이 따로 만들어집니다.</li>
<li><b>폴리곤 수정</b>: 형상(또는 바디)을 선택 → <i>폴리곤 편집 시작</i> (Tab).
    캡슐·박스 같은 기본 형상은 편집 가능한 메시로 변환할지 묻습니다.
    정점 클릭(Shift 추가 선택) → 이동 도구(W)로 끌기, Del 삭제, F 세 정점으로 면 만들기, 면 클릭 후 P 정점 추가, Esc 끝내기.</li>
<li><b>얼굴 표정</b>: 아래 '얼굴' 탭 (프리셋, 표정 슬라이더, 눈·턱)</li>
<li><b>관절 움직여 보기</b>: 아래 '관절' 탭 슬라이더, 선택한 관절의 범위는 같은 탭에서 수정</li>
<li><b>속성 수정</b>: 오른쪽 '속성' 패널 (굵은 글씨 = 직접 지정한 값, 회색 = 상속값)</li>
</ul>
<p>3D 화면에서 <b>오른쪽 클릭</b>(끌지 않고)하면 그 자리의 요소에 대한 메뉴가 열립니다.</p>"""


class MainWindow(QMainWindow):
    def __init__(self, ed: Editor | None = None):
        super().__init__()
        self.settings = QSettings("mjstudio", "MuJoCoModelStudio")
        self.log_panel = LogPanel()  # 가장 먼저: 초기 로그도 받도록
        self.ed = ed or Editor()
        self.sim = Simulator(self.ed)
        self.protected = False  # 외부에서 연 원본 파일이고 아직 우리가 저장한 적 없음
        self.project: Project | None = None
        self.view = Viewport(self.ed)
        self.view.sim = self.sim
        self.setCentralWidget(self.view)
        self.tree = TreePanel(self.ed)
        self.props = PropsPanel(self.ed)
        self.joints = JointPanel(self.ed, groups=lambda: self.project.groups if self.project else [])
        self.simp = SimPanel(self.ed, self.sim)
        self.face = FacePanel(self.ed, self.sim)
        self.addp = AddPanel(self)
        dt = self._dock("계층", "hierarchy", self.tree, Qt.LeftDockWidgetArea)
        da = self._dock("추가·편집", "addedit", self.addp, Qt.LeftDockWidgetArea)
        self.splitDockWidget(dt, da, Qt.Vertical)
        dp = self._dock("속성", "properties", self.props, Qt.RightDockWidgetArea)
        self.resizeDocks([dt, dp], [360, 350], Qt.Horizontal)
        self.resizeDocks([dt, da], [420, 480], Qt.Vertical)
        dj = self._dock("관절", "joints", self.joints, Qt.BottomDockWidgetArea)
        ds = self._dock("시뮬레이션", "simulation", self.simp, Qt.BottomDockWidgetArea)
        df = self._dock("얼굴", "face", self.face, Qt.BottomDockWidgetArea)
        dl = self._dock("로그", "log", self.log_panel, Qt.BottomDockWidgetArea)
        self.tabifyDockWidget(dj, ds)
        self.tabifyDockWidget(ds, df)
        self.tabifyDockWidget(df, dl)
        dj.raise_()
        self.status_compile = QLabel()
        self.statusBar().addPermanentWidget(self.status_compile)
        self._build_actions()
        self.view.picked.connect(self._picked)
        self.view.status.connect(lambda t: self.statusBar().showMessage(t, 3000))
        self.view.contextRequested.connect(self._context_menu)
        self.tree.contextRequested.connect(self._context_menu)
        self.ed.compiled.connect(self._on_compiled)
        self.ed.doc_changed.connect(lambda *_: self._update_title())
        self.ed.doc_loaded.connect(self._update_title)
        self.ed.undo.changed.connect(lambda *_: self._update_undo())
        self.sim.on_issue = lambda msg: self.statusBar().showMessage(f"물리 불안정: {msg}", 8000)
        self.timer = QTimer(self)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self._tick)
        self.timer.start()
        self.resize(1600, 1000)
        geo = self.settings.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        if self.ed.result is not None:
            self._on_compiled(self.ed.result)
        self._update_title()
        self._update_undo()
        log.info("MuJoCo Model Studio %s — MuJoCo %s", __version__, mujoco.__version__)
        log.info("도움말 › 사용 안내 (F1)에서 기능 위치를 볼 수 있습니다.")

    def _dock(self, title, oid, w, area) -> QDockWidget:
        d = QDockWidget(title, self)
        d.setObjectName(oid)
        d.setWidget(w)
        self.addDockWidget(area, d)
        return d

    # ================================================================ 동작(액션)
    def _act(self, menu, text, slot=None, key=None, checkable=False, checked=False, tip=None) -> QAction:
        a = QAction(text, self)
        if key:
            a.setShortcut(QKeySequence(key))
        if checkable:
            a.setCheckable(True)
            a.setChecked(checked)
            if slot:
                a.toggled.connect(slot)
        elif slot:
            a.triggered.connect(lambda *_: slot())
        if tip:
            a.setToolTip(tip)
            a.setStatusTip(tip)
        if menu is not None:
            menu.addAction(a)
        self.addAction(a)
        return a

    def _build_actions(self) -> None:
        mb = self.menuBar()
        # ---- 파일
        f = mb.addMenu("파일(&F)")
        self._act(f, "새 모델(&N)", self.file_new, "Ctrl+N")
        self._act(f, "열기(&O)…", self.file_open, "Ctrl+O")
        self._act(f, "프로젝트 열기…", self.project_open, "Ctrl+Shift+O")
        self.recent_menu = f.addMenu("최근 파일")
        self._fill_recent()
        f.addSeparator()
        self._act(f, "저장(&S)", self.file_save, "Ctrl+S")
        self._act(f, "다른 이름으로 저장(&A)…", self.file_save_as, "Ctrl+Shift+S")
        self._act(f, "프로젝트(폴더)로 저장…", self.project_save_as)
        self._act(f, "컴파일된 MJCF 내보내기 (MuJoCo가 해석한 그대로)…", self.export_compiled)
        f.addSeparator()
        self._act(f, "파일에서 부품 가져오기…", self.import_part_file, "Ctrl+I")
        self._act(f, "선택 바디를 부품으로 저장…", self.export_part)
        self._act(f, "누락된 리소스 검사", self.check_resources)
        f.addSeparator()
        self._act(f, "모델 검증 (컴파일 + 200스텝)", self.validate, "F7")
        f.addSeparator()
        self._act(f, "끝내기(&Q)", self.close, "Ctrl+Q")
        # ---- 편집
        e = mb.addMenu("편집(&E)")
        self.a_undo = self._act(e, "실행 취소", self.ed.undo_step, QKeySequence.Undo)
        self.a_redo = self._act(e, "다시 실행", self.ed.redo_step, QKeySequence.Redo)
        e.addSeparator()
        self._act(e, "삭제", self.delete_sel, QKeySequence.Delete)
        self._act(e, "복제", self.duplicate_sel, "Ctrl+D")
        self._act(e, "좌우 대칭 복제 (왼쪽 ↔ 오른쪽)", self.mirror_sel, "Ctrl+Shift+M")
        self._act(e, "이름 바꾸기…", self.rename_sel, "F2")
        self._act(e, "부모 선택", self.select_parent, "Alt+Up")
        # ---- 추가 (모델)
        mm = mb.addMenu("추가(&A)")
        self._act(mm, "바디(Body)", lambda: self._add(lambda p: ops.add_body(self.ed, p)), "Ctrl+B")
        g = mm.addMenu("형상(Geom)")
        for t in ("sphere", "box", "capsule", "cylinder", "ellipsoid", "plane"):
            self._act(g, GEOM_KO[t], lambda t=t: self._add(lambda p: ops.add_geom(self.ed, p, t)))
        jm = mm.addMenu("관절(Joint)")
        for t in ("hinge", "slide", "ball", "free"):
            self._act(jm, f"{JOINT_KO[t]} 관절", lambda t=t: self._add_joint(t))
        self._act(mm, "사이트(Site)", lambda: self._add(lambda p: ops.add_site(self.ed, p)))
        self._act(mm, "카메라", lambda: self._add(lambda p: ops.add_camera(self.ed, p)))
        self._act(mm, "조명", lambda: self._add(lambda p: ops.add_light(self.ed, p)))
        mm.addSeparator()
        lib = mm.addMenu("신체 부품 (라이브러리)")
        self._act(lib, "손가락 하나…", self.insert_finger)
        self._act(lib, "손 전체 (손가락 5개, 관절 22개)…", lambda: self.insert_library("hand"))
        self._act(lib, "발 (발목·후족부·중족부·전족부·발가락)…", lambda: self.insert_library("foot"))
        self._act(lib, "머리 (눈·턱·목)…", lambda: self.insert_library("head"))
        lib.addSeparator()
        self._act(lib, "전신 인간형 (손가락, 분절 발, 눈, 턱, 얼굴)", self.insert_humanoid)
        self._act(lib, "얼굴 표정 리그 (기본 얼굴)", self.add_face)
        mm.addSeparator()
        self._act(mm, "3D 모델 파일 붙이기 (OBJ·STL·GLB)…", self.import_mesh_file, "Ctrl+Shift+I")
        self._act(mm, "Blender 얼굴(GLB) 가져오기…", self.import_face_glb)
        self._act(mm, "선택 바디를 부품으로 교체…", self.replace_with_part)
        mm.addSeparator()
        self._act(mm, "다시 컴파일", self.ed.recompile, "F5")
        # ---- 메시
        me = mb.addMenu("메시(&M)")
        self._act(me, "3D 모델 파일 붙이기 (OBJ·STL·GLB)…", self.import_mesh_file)
        pm = me.addMenu("편집용 메시 도형 추가")
        for t, ko in (("box", "박스"), ("sphere", "구"), ("capsule", "캡슐"), ("cylinder", "원기둥"),
                      ("ellipsoid", "타원체"), ("plane", "평면"), ("triangle", "삼각형"), ("quad", "사각형")):
            self._act(pm, ko, lambda t=t: self._add(lambda p: meshops.create_primitive_mesh(self.ed, p, t)["geoms"][0]))
        me.addSeparator()
        self.a_meshedit = self._act(me, "폴리곤 편집 시작 / 끝내기", self.toggle_mesh_edit, "Tab")
        self._act(me, "선택 정점·면 삭제", self.mesh_delete)
        self._act(me, "세 정점으로 면 만들기", self.mesh_make_face)
        self._act(me, "선택 면에 정점 추가", self.mesh_poke, "P")
        self._act(me, "형상을 편집 가능한 메시로 변환", self.convert_to_mesh)
        me.addSeparator()
        self._act(me, "메시 파일로 내보내기 (OBJ/STL)…", self.export_mesh)
        self._act(me, "메시 복제", self.duplicate_mesh)
        # ---- 관절
        j = mb.addMenu("관절(&J)")
        for t in ("hinge", "slide", "ball", "free"):
            self._act(j, f"{JOINT_KO[t]} 관절 추가", lambda t=t: self._add_joint(t))
        j.addSeparator()
        self._act(j, "선택 관절에 위치 제어 액추에이터 추가", lambda: self._add_actuator("position"))
        self._act(j, "선택 관절에 모터 추가", lambda: self._add_actuator("motor"))
        j.addSeparator()
        self._act(j, "선택한 두 관절 연동 (equality)…", self.couple_selected)
        self._act(j, "선택한 관절들로 관절 그룹 만들기…", self.group_selected)
        j.addSeparator()
        self._act(j, "자세 초기화 (qpos0)", self.ed.reset_pose, "Ctrl+R")
        self._act(j, "현재 자세를 키프레임으로 저장…", self.joints._save_key)
        # ---- 센서
        s = mb.addMenu("센서(&N)")
        for t, ko in (("jointpos", "관절 위치"), ("jointvel", "관절 속도")):
            self._act(s, f"{ko} 센서 추가 (선택 관절)", lambda t=t: self._add_sensor(t, "joint"))
        s.addSeparator()
        for t, ko in (("touch", "접촉(터치)"), ("accelerometer", "가속도"), ("gyro", "자이로"), ("velocimeter", "속도"),
                      ("force", "힘"), ("torque", "토크"), ("framepos", "위치"), ("framequat", "방향")):
            self._act(s, f"{ko} 센서 추가 (선택 사이트)", lambda t=t: self._add_sensor(t, "site"))
        # ---- 시뮬레이션
        sm = mb.addMenu("시뮬레이션(&S)")
        self.a_play = self._act(sm, "재생 / 일시정지", self.toggle_sim, "Space")
        self._act(sm, "한 스텝", lambda: self.sim.step(1), ".")
        self._act(sm, "초기 상태로", lambda: self.sim.reset(), "Backspace")
        # ---- 보기
        v = mb.addMenu("보기(&V)")
        self._act(v, "보이는 형상 표시", lambda on: self._setv("show_visual", on), checkable=True, checked=True)
        self._act(v, "충돌 형상 표시", lambda on: self._setv("show_collision", on), checkable=True, checked=True)
        v.addSeparator()
        for name, flag, on in (("관절 축", VF.mjVIS_JOINT, False), ("접촉점", VF.mjVIS_CONTACTPOINT, False),
                               ("접촉력", VF.mjVIS_CONTACTFORCE, False), ("질량 중심", VF.mjVIS_COM, False),
                               ("관성 상자", VF.mjVIS_INERTIA, False), ("볼록 껍질", VF.mjVIS_CONVEXHULL, False),
                               ("반투명", VF.mjVIS_TRANSPARENT, False), ("액추에이터", VF.mjVIS_ACTUATOR, False),
                               ("텐던", VF.mjVIS_TENDON, True), ("스킨(얼굴)", VF.mjVIS_SKIN, True)):
            self._act(v, name, lambda c, flag=flag: self._visflag(flag, c), checkable=True, checked=on)
        v.addSeparator()
        for name, flag, on in (("와이어프레임", RF.mjRND_WIREFRAME, False), ("그림자", RF.mjRND_SHADOW, True),
                               ("반사", RF.mjRND_REFLECTION, True), ("하늘 배경", RF.mjRND_SKYBOX, True)):
            self._act(v, name, lambda c, flag=flag: self._rndflag(flag, c), checkable=True, checked=on)
        gg = v.addMenu("형상 그룹")
        for i in range(6):
            self._act(gg, f"그룹 {i}", lambda c, i=i: self._group("geomgroup", i, c), checkable=True,
                      checked=bool(self.view.opt.geomgroup[i]))
        sg = v.addMenu("사이트 그룹")
        for i in range(6):
            self._act(sg, f"그룹 {i}", lambda c, i=i: self._group("sitegroup", i, c), checkable=True,
                      checked=bool(self.view.opt.sitegroup[i]))
        v.addSeparator()
        self._act(v, "선택 요소에 맞춰 보기 (폴리곤 편집 중: 면 만들기)", self._key_f, "F")
        self._act(v, "카메라 초기화", self.view.reset_camera, "Home")
        # ---- 도구막대
        tb = QToolBar("기본")
        tb.setObjectName("maintb")
        self.addToolBar(tb)
        for txt, slot, tip in (("새로 만들기", self.file_new, "새 모델"), ("열기", self.file_open, "MJCF 열기"),
                               ("저장", self.file_save, "저장")):
            a = QAction(txt, self)
            a.setToolTip(tip)
            a.triggered.connect(lambda *_, s=slot: s())
            tb.addAction(a)
        tb.addSeparator()
        self.tb_undo = QAction("실행 취소", self)          # 도구막대는 짧은 이름 고정, 내용은 툴팁으로
        self.tb_undo.triggered.connect(lambda *_: self.ed.undo_step())
        self.tb_redo = QAction("다시 실행", self)
        self.tb_redo.triggered.connect(lambda *_: self.ed.redo_step())
        tb.addAction(self.tb_undo)
        tb.addAction(self.tb_redo)
        tb.addSeparator()
        grp = QActionGroup(self)
        self.gizmo_acts = {}
        for txt, mode, key in (("선택", "none", "Q"), ("이동", "translate", "W"), ("회전", "rotate", "E")):
            a = self._act(None, txt, lambda c, mode=mode: c and self._gizmo(mode), key=key, checkable=True,
                          checked=(mode == "translate"))
            a.setToolTip(f"{txt} 도구 ({key})")
            grp.addAction(a)
            tb.addAction(a)
            self.gizmo_acts[mode] = a
        self.a_local = self._act(None, "로컬 축", lambda c: self._space(c), "L", checkable=True,
                                 tip="이동/회전 도구를 요소 자신의 축 기준으로 (L) — 끄면 월드 축")
        tb.addAction(self.a_local)
        pick = QActionGroup(self)
        for txt, mode, tip in (("바디 선택", "body", "클릭하면 바디(강체)를 선택"),
                               ("형상 선택", "geom", "클릭하면 개별 형상(geom)을 선택 — 폴리곤 편집에 편리")):
            a = self._act(None, txt, lambda c, mode=mode: c and setattr(self.view, "pick_mode", mode),
                          checkable=True, checked=(mode == "body"), tip=tip)
            pick.addAction(a)
            tb.addAction(a)
        tb.addSeparator()
        a = QAction("보이는 형상", self, checkable=True, checked=True)
        a.toggled.connect(lambda on: self._setv("show_visual", on))
        tb.addAction(a)
        a = QAction("충돌 형상", self, checkable=True, checked=True)
        a.toggled.connect(lambda on: self._setv("show_collision", on))
        tb.addAction(a)
        tb.addSeparator()
        self.tb_play = QAction("▶ 재생", self)
        self.tb_play.triggered.connect(self.toggle_sim)
        tb.addAction(self.tb_play)
        a = QAction("초기화", self)
        a.triggered.connect(lambda: self.sim.reset())
        tb.addAction(a)
        tb.addAction(self._act(None, "검증", self.validate))
        tb.addSeparator()
        tb.addAction(self._act(None, "사용 안내", self.show_guide))
        # ---- 도움말
        h = mb.addMenu("도움말(&H)")
        self._act(h, "사용 안내 (기능 위치)", self.show_guide, "F1")
        self._act(h, "3D 화면 조작법", self.show_controls)
        self._act(h, "정보", self.about)

    # ================================================================ 도우미
    def _setv(self, attr, on):
        setattr(self.view, attr, on)
        self.view.update()

    def _visflag(self, flag, on):
        self.view.opt.flags[flag] = 1 if on else 0
        self.view.update()

    def _rndflag(self, flag, on):
        if self.view.scn is not None:
            self.view.scn.flags[flag] = 1 if on else 0
        self.view.update()

    def _group(self, field, i, on):
        getattr(self.view.opt, field)[i] = 1 if on else 0
        self.view.update()

    def _gizmo(self, mode):
        self.view.gizmo.mode = mode
        self.view.update()

    def _space(self, local: bool):
        self.view.gizmo.space = "local" if local else "world"
        self.view.update()

    def _picked(self, e, add):
        self.ed.select(e if e is not None else [], add=add)

    def _target(self):
        return self.ed.selection[-1] if self.ed.selection else None

    def _guard(self, fn):
        try:
            return fn()
        except (ValueError, KeyError, MjcfError) as ex:
            QMessageBox.warning(self, "작업 실패", str(ex))
            log.error("%s", ex)
            return None

    def _add(self, fn):
        e = self._guard(lambda: fn(self._target()))
        if e is not None:
            self.ed.select([e])

    def _body_of(self, t):
        while t is not None and t.tag != "body":
            t = t.getparent()
        return t

    def _add_joint(self, jtype):
        body = self._body_of(self._target())
        if body is None:
            QMessageBox.information(self, "관절 추가", "먼저 바디(또는 바디 안의 요소)를 선택하세요.")
            return
        e = self._guard(lambda: ops.add_joint(self.ed, body, jtype))
        if e is not None:
            self.ed.select([e])

    def _add_actuator(self, kind):
        t = self._target()
        if t is None or t.tag != "joint":
            QMessageBox.information(self, "액추에이터 추가", "먼저 이름이 있는 관절을 선택하세요.")
            return
        e = self._guard(lambda: ops.add_actuator(self.ed, t, kind))
        if e is not None:
            self.ed.select([e])

    def _add_sensor(self, kind, need):
        t = self._target()
        if t is None or t.tag != need or not t.get("name"):
            QMessageBox.information(self, "센서 추가",
                                    f"먼저 이름이 있는 {'관절' if need == 'joint' else '사이트'}을(를) 선택하세요.")
            return
        kw = {need: t.get("name")}
        if kind.startswith("frame"):
            kw = {"objtype": "site", "objname": t.get("name")}
        e = self._guard(lambda: ops.add_sensor(self.ed, kind, **kw))
        if e is not None:
            self.ed.select([e])

    def delete_sel(self):
        if self.view.mesh_mode is not None:      # 폴리곤 편집 중에는 정점/면 삭제
            self.view.mesh_mode.delete()
            self.view.update()
            return
        sel = [e for e in self.ed.selection if e.getparent() is not None and e.tag not in ("worldbody",)]
        if not sel:
            return
        n = sum(1 for e in sel for _ in e.iter())
        if n > 20:
            r = QMessageBox.warning(
                self, "요소 삭제",
                f"선택한 요소 {len(sel)}개와 그 안의 XML 노드 총 {n}개, 그리고 이들을 참조하는\n"
                "액추에이터·센서가 함께 삭제됩니다. Ctrl+Z로 되돌릴 수 있습니다.\n\n계속할까요?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if r != QMessageBox.Yes:
                return
        self._guard(lambda: self.ed.delete(sel))

    def duplicate_sel(self):
        sel = [e for e in self.ed.selection if e.getparent() is not None]
        out = self._guard(lambda: ops.duplicate(self.ed, sel)) if sel else None
        if out:
            self.ed.select([o for o in out if o.getparent() is not None and
                            o.getparent().tag in ("body", "worldbody")][:1] or out[:1])

    def mirror_sel(self):
        t = self._target()
        if t is None or t.tag != "body":
            QMessageBox.information(self, "좌우 대칭 복제", "대칭 복제할 바디를 선택하세요 (부모의 XZ 평면 기준, y → -y).")
            return
        out = self._guard(lambda: ops.mirror_duplicate(self.ed, t))
        if out:
            self.ed.select([out[0]])

    def rename_sel(self):
        t = self._target()
        if t is None:
            return
        key = "class" if t.tag == "default" else "name"
        name, ok = QInputDialog.getText(self, "이름 바꾸기", f"{t.tag}의 새 이름:", text=t.get(key) or "")
        if ok:
            self._guard(lambda: self.ed.rename(t, name))

    def select_parent(self):
        t = self._target()
        if t is not None and t.getparent() is not None and t.getparent().tag not in ("worldbody", "mujoco"):
            self.ed.select([t.getparent()])

    def toggle_sim(self):
        self.sim.toggle()
        self.simp.sync_play()
        self.tb_play.setText("❚❚ 일시정지" if self.sim.running else "▶ 재생")

    def _tick(self):
        if self.sim.running:
            self.sim.tick()
            if not self.sim.running:  # 불안정 감지로 멈춤
                self.simp.sync_play()
                self.tb_play.setText("▶ 재생")

    # ================================================================ 부품 / 라이브러리 / 프로젝트
    def _attach_parent(self):
        t = self._target()
        while t is not None and t.tag not in ("body", "worldbody"):
            t = t.getparent()
        return t if t is not None else self.ed.doc.worldbody

    def import_part_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "파일에서 부품 가져오기", os.getcwd(), MJCF_FILTER)
        if not path:
            return
        try:
            src = MjcfDoc.load(path)
        except (MjcfError, OSError) as ex:
            QMessageBox.critical(self, "부품 가져오기", str(ex))
            return
        dlg = PickBodyDialog(src, self)
        if dlg.exec() != PickBodyDialog.Accepted or dlg.body() is None:
            return
        part = self._guard(lambda: parts.extract_part(src, dlg.body()))
        if part is None:
            return
        out = self._guard(lambda: parts.import_part(self.ed, part, self._attach_parent(),
                                                    prefix=dlg.prefix.text().strip() or None))
        if out:
            self.ed.select([out[0]])
            log.info("부품 '%s'을(를) %s에서 가져왔습니다", out[0].get("name"), path)

    def export_part(self):
        t = self._target()
        if t is None or t.tag != "body":
            QMessageBox.information(self, "부품으로 저장", "먼저 바디를 선택하세요.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "부품을 MJCF로 저장", f"{t.get('name') or 'part'}.xml",
                                              MJCF_FILTER)
        if not path:
            return
        part = self._guard(lambda: parts.extract_part(self.ed.doc, t))
        if part is not None:
            part.save(path)
            log.info("부품 저장: %s (%s)", path, "; ".join(part.meta["part"]["notes"]) or "단독으로 동작")

    def replace_with_part(self):
        t = self._target()
        if t is None or t.tag != "body":
            QMessageBox.information(self, "부품으로 교체", "교체할 바디를 선택하세요 (예: 머리, 손).")
            return
        path, _ = QFileDialog.getOpenFileName(self, "교체할 부품 (MJCF)", os.getcwd(), MJCF_FILTER)
        if not path:
            return
        try:
            src = MjcfDoc.load(path)
        except (MjcfError, OSError) as ex:
            QMessageBox.critical(self, "부품으로 교체", str(ex))
            return
        dlg = PickBodyDialog(src, self, "교체할 부품 선택")
        if dlg.exec() != PickBodyDialog.Accepted or dlg.body() is None:
            return
        part = self._guard(lambda: parts.extract_part(src, dlg.body()))
        if part is not None:
            out = self._guard(lambda: parts.replace_body(self.ed, t, part, prefix=dlg.prefix.text().strip() or None))
            if out:
                self.ed.select([out[0]])

    def insert_library(self, kind):
        """부품을 별도 문서에서 만든 뒤 조립합니다 (이름은 자동으로 겹치지 않게)."""
        dlg = LibraryDialog(kind, self)
        if dlg.exec() != LibraryDialog.Accepted:
            return
        side, sc = dlg.side_value(), dlg.scale.value()
        if kind == "hand":
            part = anatomy.make_part(anatomy.build_hand, f"hand_{side}", side=side, scale=sc)
        elif kind == "foot":
            part = anatomy.make_part(anatomy.build_foot, f"foot_{side}", side=side, scale=sc, toes=dlg.toes_value())
        else:
            part = anatomy.make_part(anatomy.build_head, "head", scale=sc)
        out = self._guard(lambda: parts.import_part(self.ed, part, self._attach_parent(), prefix=f"{kind}2_"))
        if out:
            self.ed.select([out[0]])
            log.info("라이브러리 부품 추가: %s → '%s'", kind, out[0].get("name"))

    def insert_finger(self):
        parent = self._attach_parent()
        pname = "월드" if parent.tag == "worldbody" else (parent.get("name") or "(이름 없는 바디)")
        on_hand = parent.tag == "body" and "hand" in (parent.get("name") or "")

        def default(kind, side):
            return anatomy.finger_default_pos(kind, side) if on_hand else np.zeros(3)

        dlg = FingerDialog(pname, default, self)
        if dlg.exec() != FingerDialog.Accepted:
            return
        kind, side, pos, sc = dlg.values()
        part = self._guard(lambda: anatomy.finger_part(kind, side, sc))
        if part is None:
            return
        out = self._guard(lambda: parts.import_part(self.ed, part, parent, pos=pos, prefix="new_"))
        if out:
            self.ed.select([out[0]])
            log.info("손가락 추가: %s (%s) → '%s'", anatomy.FINGER_NAMES_KO[kind],
                     "왼쪽" if side == "left" else "오른쪽", out[0].get("name"))

    def insert_humanoid(self):
        if self.ed.doc.find_named("body", "pelvis") is not None or self.ed.doc.find_named("body", "head"):
            QMessageBox.information(self, "전신 인간형", "이 모델에는 이미 'pelvis'/'head' 바디가 있습니다. "
                                    "새 모델(파일 › 새 모델)에서 만들어 주세요.")
            return
        from ..library.humanoid import build_humanoid
        out = self._guard(lambda: build_humanoid(self.ed))
        if out:
            self.ed.select([out["pelvis"]])
            self.view.reset_camera()
            log.info("전신 인간형 생성: 관절 %d개, 액추에이터 %d개", self.ed.model.njnt, self.ed.model.nu)

    def add_face(self):
        self.face.add_procedural()
        self._raise_dock("face")

    def import_face_glb(self):
        self.face.import_glb()
        self._raise_dock("face")

    def _raise_dock(self, oid):
        for d in self.findChildren(QDockWidget):
            if d.objectName() == oid:
                d.raise_()

    def couple_selected(self):
        js = [e for e in self.ed.selection if e.tag == "joint" and e.get("name")]
        if len(js) != 2:
            QMessageBox.information(self, "관절 연동", "이름이 있는 관절 두 개를 선택하세요 (Ctrl+클릭).")
            return
        ratio, ok = QInputDialog.getDouble(self, "관절 연동",
                                           f"{js[0].get('name')} = 비율 × {js[1].get('name')}", 1.0, -100, 100, 4)
        if ok:
            self._guard(lambda: ops.couple_joints(self.ed, js[0], js[1], ratio))

    def group_selected(self):
        js = [e.get("name") for e in self.ed.selection if e.tag == "joint" and e.get("name")]
        if not js:
            QMessageBox.information(self, "관절 그룹", "이름이 있는 관절을 하나 이상 선택하세요.")
            return
        dlg = JointGroupDialog(js, parent=self)
        if dlg.exec() != JointGroupDialog.Accepted:
            return
        if self.project is None:
            QMessageBox.information(self, "관절 그룹", "관절 그룹은 프로젝트 파일에 저장됩니다. "
                                    "이어서 모델을 프로젝트 폴더로 저장합니다.")
            if not self.project_save_as():
                return
        self.project.set_group(dlg.result_group())
        self.project.save(self.ed.doc, package=False, rigs=self.ed.rigs)
        self.joints.rebuild()

    def project_open(self):
        if not self._confirm_discard():
            return
        d = QFileDialog.getExistingDirectory(self, "프로젝트 폴더 열기", os.getcwd())
        if not d:
            return
        try:
            prj, doc = Project.load(d)
        except (OSError, ValueError, MjcfError) as ex:
            QMessageBox.critical(self, "프로젝트 열기", str(ex))
            return
        self.sim.pause()
        self.project = prj
        self.view.set_mesh_mode(None)
        self.ed.set_doc(doc, prj.load_rigs())
        self.protected = False
        self.view.reset_camera()
        self.joints.rebuild()
        log.info("프로젝트 열기: %s", prj.root)

    def project_save_as(self) -> bool:
        d = QFileDialog.getExistingDirectory(self, "프로젝트 폴더 선택 (빈 폴더 권장)", os.getcwd())
        if not d:
            return False
        prj = self.project if self.project is not None else Project(d)
        prj.root = Path(d).resolve()
        try:
            notes = prj.save(self.ed.doc, package=True, rigs=self.ed.rigs)
        except OSError as ex:
            QMessageBox.critical(self, "프로젝트 저장", str(ex))
            return False
        for n in notes:
            log.info("%s", n)
        self.project = prj
        self.protected = False
        self.ed.undo.set_clean()
        self._update_title()
        return True

    def check_resources(self):
        miss = missing_resources(self.ed.doc)
        if not miss:
            QMessageBox.information(self, "리소스 검사", "참조하는 에셋 파일이 모두 있습니다.")
            log.info("리소스 검사: 모든 에셋 파일 확인")
            return
        for lbl, k, p in miss:
            log.warning("누락된 리소스: %s/%s -> %s", lbl, k, p)
        QMessageBox.warning(self, "리소스 검사", f"누락된 파일 {len(miss)}개:\n" +
                            "\n".join(f"{lbl} ({k}): {p}" for lbl, k, p in miss[:20]))

    def _key_f(self):
        if self.view.mesh_mode is not None:
            self.mesh_make_face()
        else:
            self.view.frame_selection()

    # ================================================================ 메시
    def _sel_mesh_geom(self):
        t = self._target()
        if t is not None and t.tag == "geom" and meshops.geom_mesh_elem(self.ed, t) is not None:
            return t
        return None

    def _pick_geom_for_edit(self):
        """선택에서 편집할 형상을 고릅니다: 형상이면 그것, 바디면 그 바디의 형상 중 하나."""
        t = self._target()
        if t is None:
            return None
        if t.tag == "geom":
            return t
        if t.tag == "body":
            geoms = [c for c in t if getattr(c, "tag", None) == "geom"]
            if len(geoms) == 1:
                return geoms[0]
            if geoms:
                labels = [self.ed.doc.label(g) for g in geoms]
                item, ok = QInputDialog.getItem(self, "폴리곤 편집", "편집할 형상을 고르세요:", labels, 0, False)
                return geoms[labels.index(item)] if ok else None
        return None

    def import_mesh_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "3D 모델 파일 붙이기", os.getcwd(),
                                              "3D 모델 (*.obj *.stl *.glb *.gltf);;모든 파일 (*)")
        if not path:
            return
        parent = self._attach_parent()
        dlg = MeshImportDialog(path, parent.tag == "body", self)
        if dlg.exec() != MeshImportDialog.Accepted:
            return
        new_body = bool(dlg.mode.currentData())
        out = self._guard(lambda: meshops.import_mesh(
            self.ed, path, parent=parent, new_body=new_body, name=dlg.name.text().strip() or None,
            units=dlg.units_value(), axes=dlg.axes.currentData(), collision=dlg.collision_value()))
        if out:
            self.ed.select([out["body"] if new_body else out["geoms"][0]])

    def toggle_mesh_edit(self):
        if self.view.mesh_mode is not None:
            self.view.set_mesh_mode(None)
            self.statusBar().showMessage("폴리곤 편집을 끝냈습니다", 2000)
            self.addp.refresh()
            return
        g = self._pick_geom_for_edit()
        if g is None:
            QMessageBox.information(self, "폴리곤 편집", "편집할 형상(또는 그 형상이 있는 바디)을 먼저 선택하세요.\n"
                                    "도구막대의 '형상 선택'을 켜면 클릭으로 형상을 바로 고를 수 있습니다.")
            return
        if meshops.geom_mesh_elem(self.ed, g) is None:
            typ = self.ed.doc.get(g, "type") or "sphere"
            r = QMessageBox.question(self, "폴리곤 편집",
                                     f"'{self.ed.doc.label(g)}'은(는) 기본 형상({GEOM_KO.get(typ, typ)})이라 정점이 없습니다.\n"
                                     "같은 모양의 편집 가능한 메시로 변환한 뒤 편집할까요? (Ctrl+Z로 되돌릴 수 있음)")
            if r != QMessageBox.Yes:
                return
            if self._guard(lambda: meshops.convert_geom_to_mesh(self.ed, g)) is None:
                return
        from .meshedit import MeshEditMode
        mode = self._guard(lambda: MeshEditMode(self.view, g))
        if mode is None:
            return
        self.ed.select([g])
        self.view.set_mesh_mode(mode)
        self.view.setFocus()
        self.addp.refresh()
        self.statusBar().showMessage("폴리곤 편집: 정점 클릭(Shift 추가) · W 이동 · Del 삭제 · F 면 · P 정점 추가 · Esc 끝", 8000)

    def _need_mesh_mode(self) -> bool:
        if self.view.mesh_mode is None:
            QMessageBox.information(self, "폴리곤 편집", "먼저 '폴리곤 편집 시작'(Tab)을 눌러 편집 모드로 들어가세요.")
            return False
        return True

    def mesh_delete(self):
        if self._need_mesh_mode():
            self.view.mesh_mode.delete()
            self.view.update()

    def mesh_make_face(self):
        if self._need_mesh_mode():
            self.view.mesh_mode.make_face()
            self.view.update()

    def mesh_poke(self):
        if self._need_mesh_mode():
            self.view.mesh_mode.poke()
            self.view.update()

    def convert_to_mesh(self):
        g = self._pick_geom_for_edit()
        if g is None:
            QMessageBox.information(self, "메시로 변환", "변환할 형상(또는 그 바디)을 먼저 선택하세요.")
            return
        me = self._guard(lambda: meshops.convert_geom_to_mesh(self.ed, g))
        if me is not None:
            self.ed.select([g])
            log.info("메시로 변환: %s → %s", self.ed.doc.label(g), me.get("name"))

    def export_mesh(self):
        g = self._sel_mesh_geom()
        if g is None:
            QMessageBox.information(self, "메시 내보내기", "먼저 메시 형상을 선택하세요.")
            return
        me = meshops.geom_mesh_elem(self.ed, g)
        path, _ = QFileDialog.getSaveFileName(self, "메시 내보내기", f"{self.ed.doc.obj_name(me)}.obj",
                                              "OBJ (*.obj);;STL (*.stl)")
        if path:
            p = self._guard(lambda: meshops.export_mesh(self.ed, me, path))
            if p:
                log.info("메시 내보내기: %s", p)

    def duplicate_mesh(self):
        g = self._sel_mesh_geom()
        if g is None:
            QMessageBox.information(self, "메시 복제", "먼저 메시 형상을 선택하세요.")
            return
        me = self._guard(lambda: meshops.duplicate_mesh_asset(self.ed, meshops.geom_mesh_elem(self.ed, g)))
        if me is not None:
            self._guard(lambda: self.ed.set_attr(g, "mesh", me.get("name"), "복제한 메시 사용"))

    # ================================================================ 오른쪽 클릭 메뉴
    def _context_menu(self, e, gpos):
        menu = QMenu(self)
        if self.view.mesh_mode is not None and self.sender() is self.view:
            t = menu.addAction(f"폴리곤 편집: {self.ed.doc.label(self.view.mesh_mode.geom)}")
            t.setEnabled(False)
            menu.addSeparator()
            menu.addAction("선택 정점·면 삭제 (Del)", self.mesh_delete)
            menu.addAction("세 정점으로 면 만들기 (F)", self.mesh_make_face)
            menu.addAction("선택 면에 정점 추가 (P)", self.mesh_poke)
            menu.addSeparator()
            menu.addAction("폴리곤 편집 끝내기 (Esc)", self.toggle_mesh_edit)
            self._exec_menu(menu, gpos)
            return
        if e is not None and e.tag not in ("worldbody",) and e not in self.ed.selection:
            self.ed.select([e])
        tgt = e
        if e is not None:
            title = menu.addAction(self.ed.doc.label(e))
            title.setEnabled(False)
            menu.addSeparator()
        add = menu.addMenu("추가")
        add.addAction("바디", lambda: self._add(lambda p: ops.add_body(self.ed, tgt)))
        gm = add.addMenu("형상")
        for t in ("sphere", "box", "capsule", "cylinder", "ellipsoid"):
            gm.addAction(GEOM_KO[t], lambda t=t: self._add(lambda p: ops.add_geom(self.ed, tgt, t)))
        jm = add.addMenu("관절")
        for t in ("hinge", "slide", "ball", "free"):
            jm.addAction(f"{JOINT_KO[t]} 관절", lambda t=t: self._add_joint(t))
        add.addAction("사이트", lambda: self._add(lambda p: ops.add_site(self.ed, tgt)))
        add.addAction("카메라", lambda: self._add(lambda p: ops.add_camera(self.ed, tgt)))
        add.addAction("조명", lambda: self._add(lambda p: ops.add_light(self.ed, tgt)))
        add.addSeparator()
        add.addAction("손가락 하나…", self.insert_finger)
        add.addAction("손 전체…", lambda: self.insert_library("hand"))
        add.addAction("발…", lambda: self.insert_library("foot"))
        add.addAction("3D 모델 파일 붙이기…", self.import_mesh_file)
        if e is not None and e.tag in ("geom", "body"):
            menu.addAction("폴리곤 편집 시작 / 끝내기", self.toggle_mesh_edit)
            if e.tag == "geom" and meshops.geom_mesh_elem(self.ed, e) is None:
                menu.addAction("편집 가능한 메시로 변환", self.convert_to_mesh)
        if e is not None and e.getparent() is not None and e.tag != "worldbody":
            menu.addSeparator()
            menu.addAction("복제", self.duplicate_sel)
            if e.tag == "body":
                menu.addAction("좌우 대칭 복제", self.mirror_sel)
                menu.addAction("부품으로 저장…", self.export_part)
                menu.addAction("부품으로 교체…", self.replace_with_part)
            menu.addAction("이름 바꾸기…", self.rename_sel)
            menu.addAction("XML 복사", lambda: QGuiApplication.clipboard().setText(
                etree.tostring(e, pretty_print=True, encoding="unicode")))
            menu.addAction("화면에 맞춰 보기", self.view.frame_selection)
            menu.addSeparator()
            menu.addAction("삭제", self.delete_sel)
        self._exec_menu(menu, gpos)

    def _exec_menu(self, menu, gpos):
        menu.exec(gpos)

    # ================================================================ 파일
    def _confirm_discard(self) -> bool:
        if not self.ed.dirty:
            return True
        box = QMessageBox(QMessageBox.Question, "저장하지 않은 변경", "저장하지 않은 변경 사항이 있습니다. 먼저 저장할까요?",
                          QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, self)
        box.button(QMessageBox.Save).setText("저장")
        box.button(QMessageBox.Discard).setText("저장 안 함")
        box.button(QMessageBox.Cancel).setText("취소")
        box.setDefaultButton(QMessageBox.Save)
        r = box.exec()
        if r == QMessageBox.Cancel:
            return False
        if r == QMessageBox.Save:
            return self.file_save()
        return True

    def file_new(self):
        if not self._confirm_discard():
            return
        self.sim.pause()
        self.view.set_mesh_mode(None)
        self.ed.new("new_model")
        self.protected = False
        self.project = None
        self.view.reset_camera()

    def file_open(self):
        if not self._confirm_discard():
            return
        start = str(self.ed.doc.base_dir) if self.ed.doc.path else os.getcwd()
        path, _ = QFileDialog.getOpenFileName(self, "MJCF 모델 열기", start, MJCF_FILTER)
        if path:
            self.open_path(path)

    def open_path(self, path) -> bool:
        self.sim.pause()
        self.view.set_mesh_mode(None)
        try:
            self.ed.open(path)
        except (MjcfError, OSError) as ex:
            QMessageBox.critical(self, "열 수 없음", str(ex))
            log.error("열 수 없음 %s: %s", path, ex)
            return False
        self.protected = True
        self.project = None
        self._push_recent(str(Path(path).resolve()))
        self.view.reset_camera()
        if self.ed.result is not None and not self.ed.result.ok:
            QMessageBox.warning(self, "모델 오류", "파일은 열었지만 MuJoCo가 오류를 보고했습니다:\n\n" +
                                self.ed.result.pretty_error(self.ed.doc))
        return True

    def file_save(self) -> bool:
        if self.project is not None:
            try:
                for n in self.project.save(self.ed.doc, package=True, rigs=self.ed.rigs):
                    log.info("%s", n)
            except OSError as ex:
                QMessageBox.critical(self, "프로젝트 저장", str(ex))
                return False
            self.ed.undo.set_clean()
            self._update_title()
            self.statusBar().showMessage(f"프로젝트 저장: {self.project.root}", 4000)
            return True
        if self.ed.doc.path is None:
            return self.file_save_as()
        if self.protected:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Warning)
            box.setWindowTitle("원본 파일 보호")
            box.setText(f"원본 파일을 덮어쓸까요?\n\n{self.ed.doc.path}")
            box.setInformativeText("다른 이름으로 저장하면 원본은 그대로 남습니다.")
            sa = box.addButton("다른 이름으로 저장… (권장)", QMessageBox.AcceptRole)
            ow = box.addButton("원본 덮어쓰기", QMessageBox.DestructiveRole)
            cancel = box.addButton(QMessageBox.Cancel)
            cancel.setText("취소")
            box.setDefaultButton(sa)
            box.exec()
            if box.clickedButton() is sa:
                return self.file_save_as()
            if box.clickedButton() is not ow:
                return False
        return self._save_to(self.ed.doc.path)

    def file_save_as(self) -> bool:
        start = str(self.ed.doc.path.with_name(self.ed.doc.path.stem + "_edited.xml")) if self.ed.doc.path \
            else os.path.join(os.getcwd(), "model.xml")
        path, _ = QFileDialog.getSaveFileName(self, "MJCF로 저장", start, MJCF_FILTER)
        if not path:
            return False
        return self._save_to(path)

    def _save_to(self, path) -> bool:
        try:
            notes = self.ed.save(path)
        except OSError as ex:
            QMessageBox.critical(self, "저장 실패", str(ex))
            return False
        self.protected = False
        self._push_recent(str(Path(path).resolve()))
        self.statusBar().showMessage(f"저장함: {path}" + (f" (에셋 경로 {len(notes)}개 재작성)" if notes else ""), 5000)
        self._update_title()
        return True

    def export_compiled(self):
        res = compile_doc(self.ed.doc)
        if not res.ok:
            QMessageBox.warning(self, "내보내기", "모델이 컴파일되지 않습니다:\n" + res.pretty_error(self.ed.doc))
            return
        path, _ = QFileDialog.getSaveFileName(self, "MuJoCo가 저장한 XML 내보내기", "compiled.xml", MJCF_FILTER)
        if path:
            mujoco.mj_saveLastXML(path, res.model)
            log.info("MuJoCo가 생성한 XML 내보내기: %s", path)

    def validate(self):
        res = compile_doc(self.ed.doc)
        if not res.ok:
            msg = res.pretty_error(self.ed.doc)
            log.error("검증: 컴파일 오류: %s", msg)
            QMessageBox.warning(self, "모델 검증", "컴파일 오류:\n\n" + msg)
            return
        issues = validate_steps(res.model, 200)
        m = res.model
        summary = (f"컴파일 성공 ({res.elapsed_ms:.1f} ms) — 바디 {m.nbody}, 관절 {m.njnt}, 자유도 {m.nv}, "
                   f"형상 {m.ngeom}, 액추에이터 {m.nu}, 센서 {m.nsensor}")
        log.info("검증: %s", summary)
        if issues:
            for i in issues:
                log.warning("검증: %s", i)
            QMessageBox.warning(self, "모델 검증", summary + "\n\n시뮬레이션 문제:\n- " + "\n- ".join(issues))
        else:
            log.info("검증: 200 스텝 동안 수치 문제 없음")
            QMessageBox.information(self, "모델 검증", summary + "\n\n200 스텝 시뮬레이션: 문제 없음.")

    def _push_recent(self, path):
        rec = [p for p in (self.settings.value("recent") or []) if p != path]
        rec.insert(0, path)
        self.settings.setValue("recent", rec[:10])
        self._fill_recent()

    def _fill_recent(self):
        self.recent_menu.clear()
        for p in (self.settings.value("recent") or []):
            self.recent_menu.addAction(p, lambda p=p: self._confirm_discard() and self.open_path(p))

    # ================================================================ 상태 표시
    def _on_compiled(self, res):
        if res.ok:
            m = res.model
            self.status_compile.setText(f"<span style='color:{C['ok']}'>● 컴파일 성공</span> {res.elapsed_ms:.0f} ms · "
                                        f"바디 {m.nbody - 1} · 관절 {m.njnt} · 형상 {m.ngeom} · "
                                        f"액추에이터 {m.nu} · 센서 {m.nsensor}")
        else:
            self.status_compile.setText(f"<span style='color:{C['err']}'>● 컴파일 오류</span> (로그 탭 참고)")

    def _update_title(self):
        p = self.ed.doc.path
        name = p.name if p else "제목 없음"
        self.setWindowTitle(f"{name}{' *' if self.ed.dirty else ''} — MuJoCo Model Studio")

    def _update_undo(self):
        self.a_undo.setEnabled(self.ed.undo.can_undo())
        self.a_redo.setEnabled(self.ed.undo.can_redo())
        ut, rt = self.ed.undo.undo_text(), self.ed.undo.redo_text()
        self.a_undo.setText(f"실행 취소: {ut}" if ut else "실행 취소")
        self.a_redo.setText(f"다시 실행: {rt}" if rt else "다시 실행")
        self.tb_undo.setEnabled(self.ed.undo.can_undo())
        self.tb_redo.setEnabled(self.ed.undo.can_redo())
        self.tb_undo.setToolTip(f"실행 취소 (Ctrl+Z): {ut}" if ut else "실행 취소할 작업 없음")
        self.tb_redo.setToolTip(f"다시 실행 (Ctrl+Y): {rt}" if rt else "다시 실행할 작업 없음")
        self._update_title()

    def show_guide(self):
        box = QMessageBox(self)
        box.setWindowTitle("사용 안내")
        box.setTextFormat(Qt.RichText)
        box.setText(GUIDE)
        box.exec()

    def show_controls(self):
        QMessageBox.information(self, "3D 화면 조작법", (
            "왼쪽 드래그: 회전   ·   오른쪽 드래그: 이동   ·   휠 / 가운데 드래그: 확대·축소\n"
            "Shift + 드래그: 수평 방향 변형\n"
            "클릭: 선택 (Shift/Ctrl: 추가 선택)   ·   더블클릭: 그 지점을 화면 중심으로\n"
            "오른쪽 클릭 (끌지 않음): 그 자리 요소의 메뉴\n"
            "Q / W / E: 선택 / 이동 / 회전 도구   ·   L: 로컬/월드 축\n"
            "도구를 끄는 동안 Ctrl: 1 cm / 15° 단위로 맞춤   ·   Esc: 취소\n"
            "F: 선택 요소에 맞춰 보기   ·   Home: 카메라 초기화   ·   Space: 재생/일시정지\n"
            "Tab: 폴리곤 편집 시작/끝"))

    def about(self):
        QMessageBox.about(self, "정보", f"MuJoCo Model Studio {__version__}\nMuJoCo {mujoco.__version__}\n"
                                      "MuJoCo 모델을 눈으로 보며 만들고 고치고 조립하는 편집기")

    def closeEvent(self, ev):
        if not self._confirm_discard():
            ev.ignore()
            return
        self.settings.setValue("geometry", self.saveGeometry())
        self.timer.stop()
        self.log_panel.detach()
        ev.accept()
