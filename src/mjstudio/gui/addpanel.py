"""'추가·편집' 패널: 모델에 무언가를 더하거나 고치는 기능을 한곳에 모아 보여 줍니다."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QGridLayout, QGroupBox, QLabel, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout,
                               QWidget)

from ..core import meshops
from ..core import ops
from .theme import C


class AddPanel(QScrollArea):
    def __init__(self, win, parent=None):
        super().__init__(parent)
        self.w = win
        self.ed = win.ed
        self.setWidgetResizable(True)
        body = QWidget()
        self.setWidget(body)
        lay = QVBoxLayout(body)
        lay.setContentsMargins(6, 6, 6, 6)
        self.target = QLabel()
        self.target.setWordWrap(True)
        self.target.setTextFormat(Qt.RichText)
        lay.addWidget(self.target)

        def section(title: str, items, cols: int = 2, note: str | None = None) -> QGroupBox:
            box = QGroupBox(title)
            g = QGridLayout(box)
            g.setContentsMargins(6, 6, 6, 6)
            g.setSpacing(4)
            r = 0
            if note:
                lbl = QLabel(note)
                lbl.setWordWrap(True)
                lbl.setStyleSheet(f"color:{C['dim']}")
                g.addWidget(lbl, 0, 0, 1, cols)
                r = 1
            for k, (text, slot, tip) in enumerate(items):
                b = QPushButton(text)
                b.setToolTip(tip or text)
                b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                b.clicked.connect(lambda _=False, s=slot: s())
                g.addWidget(b, r + k // cols, k % cols)
            lay.addWidget(box)
            return box

        w = self.w
        add = w._add
        section("기본 요소", [
            ("바디(Body)", lambda: add(lambda p: ops.add_body(self.ed, p)), "선택 위치에 새 바디(강체)를 만듭니다"),
            ("박스", lambda: add(lambda p: ops.add_geom(self.ed, p, "box")), "박스 형상(geom)"),
            ("구", lambda: add(lambda p: ops.add_geom(self.ed, p, "sphere")), "구 형상"),
            ("캡슐", lambda: add(lambda p: ops.add_geom(self.ed, p, "capsule")), "캡슐 형상"),
            ("원기둥", lambda: add(lambda p: ops.add_geom(self.ed, p, "cylinder")), "원기둥 형상"),
            ("타원체", lambda: add(lambda p: ops.add_geom(self.ed, p, "ellipsoid")), "타원체 형상"),
            ("경첩 관절", lambda: w._add_joint("hinge"), "한 축 회전 관절 (선택한 바디에)"),
            ("슬라이드 관절", lambda: w._add_joint("slide"), "한 축 직선 이동 관절"),
            ("볼 관절", lambda: w._add_joint("ball"), "세 축 회전 관절"),
            ("자유 관절", lambda: w._add_joint("free"), "6자유도 (월드 바로 아래 바디만)"),
            ("사이트(Site)", lambda: add(lambda p: ops.add_site(self.ed, p)), "센서 위치 표시용 점"),
            ("카메라", lambda: add(lambda p: ops.add_camera(self.ed, p)), "카메라"),
            ("조명", lambda: add(lambda p: ops.add_light(self.ed, p)), "조명"),
        ], cols=2)
        section("신체 부품", [
            ("손가락 하나…", w.insert_finger, "검지·중지·약지·소지·엄지 중 하나를 선택한 바디(손바닥)에 붙입니다"),
            ("손 전체…", lambda: w.insert_library("hand"), "손바닥 + 손가락 5개 (관절 22개)"),
            ("발…", lambda: w.insert_library("foot"), "발목·후족부·중족부·전족부·발가락"),
            ("머리 (눈·턱)…", lambda: w.insert_library("head"), "좌우 눈 관절, 턱 관절, 목"),
            ("전신 인간형", w.insert_humanoid, "손가락·분절 발·눈·턱·얼굴이 있는 전신 (새 모델에서)"),
            ("얼굴 표정 리그", w.add_face, "머리/턱 바디에 표정 가능한 얼굴 메시를 씌웁니다"),
        ], cols=2, note="선택한 바디의 자식으로 붙습니다. 이름이 겹치면 자동으로 바뀝니다.")
        section("내 3D 모델 (Blender 등)", [
            ("3D 모델 파일 붙이기 (OBJ·STL·GLB)…", w.import_mesh_file,
             "직접 만든 3D 모델을 선택한 바디에 붙이거나 새 바디로 만듭니다"),
            ("Blender 얼굴(GLB) 가져오기…", w.import_face_glb, "Shape Key가 있는 얼굴 메시를 표정 리그로"),
        ], cols=1)
        section("편집용 메시 도형 (폴리곤 편집 가능)", [
            (t_ko, (lambda t=t: add(lambda p: meshops.create_primitive_mesh(self.ed, p, t)["geoms"][0])),
             f"{t_ko} 모양의 편집 가능한 메시")
            for t, t_ko in (("box", "박스"), ("sphere", "구"), ("capsule", "캡슐"), ("cylinder", "원기둥"),
                            ("ellipsoid", "타원체"), ("plane", "평면"), ("triangle", "삼각형"), ("quad", "사각형"))
        ], cols=4)
        poly = section("폴리곤(메시) 편집", [
            ("폴리곤 편집 시작 / 끝내기", w.toggle_mesh_edit, "선택한 형상의 정점·면을 편집합니다 (Tab)"),
            ("선택 정점·면 삭제", w.mesh_delete, "편집 중: 선택한 정점(또는 면) 삭제 (Del)"),
            ("세 정점으로 면 만들기", w.mesh_make_face, "편집 중: 정점 3개를 고른 뒤 (F)"),
            ("선택 면에 정점 추가", w.mesh_poke, "편집 중: 면을 클릭한 뒤 (P)"),
            ("형상을 메시로 변환", w.convert_to_mesh, "캡슐·박스 같은 기본 형상을 편집 가능한 메시로 바꿉니다"),
            ("메시 파일로 내보내기…", w.export_mesh, "OBJ / STL"),
            ("메시 복제", w.duplicate_mesh, "같은 모양의 독립 메시로 복사"),
        ], cols=1)
        self.poly_state = QLabel()
        self.poly_state.setWordWrap(True)
        self.poly_state.setTextFormat(Qt.RichText)
        poly.layout().addWidget(self.poly_state, poly.layout().rowCount(), 0, 1, 1)
        section("부품·구조", [
            ("파일에서 부품 가져오기…", w.import_part_file, "다른 MJCF 파일의 일부(바디 하위 트리)를 가져옵니다"),
            ("선택 바디를 부품으로 저장…", w.export_part, "독립 MJCF 파일로 저장해 다른 모델에 재사용"),
            ("선택 바디를 부품으로 교체…", w.replace_with_part, "예: 머리만 다른 머리로 교체"),
            ("복제", w.duplicate_sel, "선택 요소 복사 (Ctrl+D)"),
            ("좌우 대칭 복제", w.mirror_sel, "왼쪽 팔 → 오른쪽 팔처럼 대칭으로 복사"),
            ("삭제", w.delete_sel, "선택 요소 삭제 (Del)"),
        ], cols=1)
        lay.addStretch(1)
        self.ed.selection_changed.connect(lambda *_: self.refresh())
        self.ed.doc_loaded.connect(self.refresh)
        win.view.meshModeChanged.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        p = self.w._attach_parent()
        name = "월드 (최상위)" if p.tag == "worldbody" else (p.get("name") or "(이름 없는 바디)")
        self.target.setText(f"추가 위치: <b>{name}</b><br><span style='color:{C['dim']}'>"
                            "계층 트리나 3D 화면에서 바디를 클릭하면 그 바디에 추가됩니다.</span>")
        mode = self.w.view.mesh_mode
        if mode is None:
            self.poly_state.setText(f"<span style='color:{C['dim']}'>편집하려면 형상을 선택(도구막대의 "
                                    "'형상 선택'이 편리)한 뒤 '폴리곤 편집 시작'을 누르세요. 기본 형상은 "
                                    "메시로 변환할지 묻습니다.</span>")
        else:
            self.poly_state.setText(
                f"<b style='color:{C['hl']}'>편집 중:</b> {self.ed.doc.label(mode.geom)}<br>"
                "정점 클릭 = 선택 · Shift+클릭 = 추가 선택 · 이동 도구(W)로 끌기 · Del 삭제 · "
                "F 면 만들기 · 면 클릭 후 P = 정점 추가 · Esc/Tab 끝내기")
