"""대화상자: 부품 가져오기, 관절 그룹, 라이브러리 부품, 손가락, 메시 가져오기."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QHBoxLayout,
                               QLabel, QLineEdit, QTableWidget, QTableWidgetItem, QTreeWidget, QTreeWidgetItem,
                               QVBoxLayout)

from ..core.doc import MjcfDoc, children
from .theme import C, icon_for

SIDES = (("왼쪽", "left"), ("오른쪽", "right"))


def _buttons(dlg: QDialog, lay) -> None:
    bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    bb.button(QDialogButtonBox.Ok).setText("확인")
    bb.button(QDialogButtonBox.Cancel).setText("취소")
    bb.accepted.connect(dlg.accept)
    bb.rejected.connect(dlg.reject)
    lay.addWidget(bb)


def _note(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setStyleSheet(f"color:{C['dim']}")
    return lbl


def _side_box() -> QComboBox:
    b = QComboBox()
    for ko, en in SIDES:
        b.addItem(ko, en)
    return b


class PickBodyDialog(QDialog):
    """외부 MJCF 파일에서 부품으로 가져올 바디(하위 트리)를 고릅니다."""

    def __init__(self, doc: MjcfDoc, parent=None, title="부품 가져오기"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(480, 580)
        self.doc = doc
        lay = QVBoxLayout(self)
        name = doc.path.name if doc.path else "모델"
        info = QLabel(f"<b>{name}</b> — 부품으로 가져올 바디를 고르세요. 그 아래의 모든 요소와 필요한 에셋, "
                      "기본 클래스, 해당 부품만 참조하는 액추에이터·센서가 함께 들어옵니다.")
        info.setWordWrap(True)
        lay.addWidget(info)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        lay.addWidget(self.tree)
        wb = doc.section("worldbody")

        def add(parent_item, e):
            for c in children(e, "body"):
                it = QTreeWidgetItem([doc.label(c)])
                it.setIcon(0, icon_for("body"))
                it.setData(0, Qt.UserRole, c)
                (parent_item.addChild(it) if parent_item is not None else self.tree.addTopLevelItem(it))
                add(it, c)

        if wb is not None:
            add(None, wb)
        self.tree.expandToDepth(2)
        form = QFormLayout()
        self.prefix = QLineEdit()
        self.prefix.setPlaceholderText("이름이 겹칠 때만 붙는 접두사 (비우면 자동)")
        form.addRow("이름 접두사", self.prefix)
        lay.addLayout(form)
        _buttons(self, lay)
        if self.tree.topLevelItemCount():
            self.tree.setCurrentItem(self.tree.topLevelItem(0))

    def body(self):
        it = self.tree.currentItem()
        return it.data(0, Qt.UserRole) if it is not None else None


class JointGroupDialog(QDialog):
    """편집기 수준의 관절 그룹(관절별 가중치)."""

    def __init__(self, joints: list[str], name: str = "group", weights: dict | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("관절 그룹")
        self.resize(440, 500)
        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.name = QLineEdit(name)
        form.addRow("그룹 이름", self.name)
        lay.addLayout(form)
        lay.addWidget(_note("그룹 값 v는 각 관절을  q = q0 + 가중치 · v  로 움직입니다 (project.json에 저장)"))
        self.table = QTableWidget(len(joints), 2)
        self.table.setHorizontalHeaderLabels(["관절", "가중치"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.spins = []
        for r, j in enumerate(joints):
            it = QTableWidgetItem(j)
            it.setFlags(it.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(r, 0, it)
            s = QDoubleSpinBox()
            s.setRange(-10, 10)
            s.setDecimals(3)
            s.setValue((weights or {}).get(j, 1.0))
            self.table.setCellWidget(r, 1, s)
            self.spins.append((j, s))
        lay.addWidget(self.table)
        _buttons(self, lay)

    def result_group(self):
        from ..core.project import JointGroup
        return JointGroup(self.name.text().strip() or "group", {j: s.value() for j, s in self.spins})


class LibraryDialog(QDialog):
    """라이브러리 부품(손, 발, 머리) 삽입 옵션."""
    TITLES = {"hand": "손 추가", "foot": "발 추가", "head": "머리 추가"}

    def __init__(self, kind: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.TITLES.get(kind, "부품 추가"))
        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.side = _side_box()
        self.toes = QComboBox()
        self.toes.addItem("하나로 (MTP 관절 1개)", "single")
        self.toes.addItem("발가락마다 따로 (확장)", "detailed")
        self.scale = QDoubleSpinBox()
        self.scale.setRange(0.2, 5)
        self.scale.setValue(1.0)
        self.scale.setSingleStep(0.05)
        if kind in ("hand", "foot"):
            form.addRow("좌/우", self.side)
        if kind == "foot":
            form.addRow("발가락", self.toes)
        form.addRow("크기 배율", self.scale)
        lay.addLayout(form)
        lay.addWidget(_note("선택한 바디(없으면 월드)의 자식으로 붙습니다. 이름이 겹치면 자동으로 바뀝니다."))
        _buttons(self, lay)

    def side_value(self) -> str:
        return self.side.currentData()

    def toes_value(self) -> str:
        return self.toes.currentData()


class FingerDialog(QDialog):
    """손가락 하나 추가: 종류, 좌/우, 붙일 위치(부모 바디 기준)."""

    def __init__(self, parent_name: str, default_pos, parent=None):
        super().__init__(parent)
        self.setWindowTitle("손가락 추가")
        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.kind = QComboBox()
        for ko, en in (("검지", "index"), ("중지", "middle"), ("약지", "ring"), ("소지(새끼)", "little"),
                       ("엄지", "thumb")):
            self.kind.addItem(ko, en)
        self.side = _side_box()
        self.pos = [QDoubleSpinBox() for _ in range(3)]
        row = QHBoxLayout()
        for s in self.pos:
            s.setRange(-2, 2)
            s.setDecimals(4)
            s.setSingleStep(0.005)
            row.addWidget(s)
        self.scale = QDoubleSpinBox()
        self.scale.setRange(0.2, 5)
        self.scale.setValue(1.0)
        self.scale.setSingleStep(0.05)
        form.addRow("붙일 바디", QLabel(f"<b>{parent_name}</b>"))
        form.addRow("손가락", self.kind)
        form.addRow("좌/우", self.side)
        form.addRow("위치 x y z (m)", row)
        form.addRow("크기 배율", self.scale)
        lay.addLayout(form)
        lay.addWidget(_note("검지·중지·약지·소지: 기저관절(굽힘+벌림)·근위·중위·원위 지골, 마디별 굽힘 관절. "
                            "엄지: 중수골(기저부 회전+벌림)·근위·원위 지골. 각 관절에 액추에이터와 센서가 함께 생깁니다. "
                            "기본 위치는 라이브러리 손바닥 기준이며, 추가 후 이동 도구로 옮길 수 있습니다."))
        self._default = default_pos
        self.kind.currentIndexChanged.connect(self._fill)
        self.side.currentIndexChanged.connect(self._fill)
        self._fill()
        _buttons(self, lay)

    def _fill(self, *_):
        p = self._default(self.kind.currentData(), self.side.currentData())
        for s, v in zip(self.pos, p):
            s.setValue(float(v))

    def values(self):
        return self.kind.currentData(), self.side.currentData(), [s.value() for s in self.pos], self.scale.value()


class MeshImportDialog(QDialog):
    """3D 모델(OBJ/STL/GLB) 가져오기 옵션."""

    def __init__(self, path: str, has_body: bool, parent=None):
        super().__init__(parent)
        self.setWindowTitle("3D 모델 가져오기")
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(f"<b>{path}</b>"))
        form = QFormLayout()
        self.units = QComboBox()
        for ko, en in (("미터 (m)", "m"), ("센티미터 (cm)", "cm"), ("밀리미터 (mm)", "mm"), ("인치 (in)", "in"),
                       ("피트 (ft)", "ft")):
            self.units.addItem(ko, en)
        self.axes = QComboBox()
        self.axes.addItem("MuJoCo (위 Z, 정면 +X)", "mujoco")
        self.axes.addItem("Blender 장면 (위 Z, 정면 -Y)", "blender")
        self.axes.addItem("위 Y (정면 +Z)", "yup")
        self.mode = QComboBox()
        if has_body:
            self.mode.addItem("선택한 바디에 붙이기", False)
        self.mode.addItem("새 바디로 만들기", True)
        self.collision = QComboBox()
        for ko, en in (("볼록 껍질 (메시 모양 그대로)", "hull"), ("박스로 근사", "box"), ("캡슐로 근사", "capsule"),
                       ("구로 근사", "sphere"), ("원기둥으로 근사", "cylinder"), ("타원체로 근사", "ellipsoid"),
                       ("충돌 없음 (보이기만)", "none")):
            self.collision.addItem(ko, en)
        self.name = QLineEdit()
        self.name.setPlaceholderText("비우면 파일 이름 사용")
        if path.lower().endswith((".glb", ".gltf")):
            self.axes.setEnabled(False)
            self.axes.setToolTip("glTF는 축이 정해져 있어(위 Y, 정면 +Z) 자동으로 변환합니다")
        form.addRow("단위", self.units)
        form.addRow("좌표축", self.axes)
        form.addRow("붙이는 방식", self.mode)
        form.addRow("충돌 형상", self.collision)
        form.addRow("이름", self.name)
        lay.addLayout(form)
        lay.addWidget(_note("보이는 형상(그룹 2, 접촉 없음)과 충돌 형상(그룹 3)이 따로 만들어집니다. "
                            "OBJ/STL은 원본 파일을 그대로 참조하므로 원본이 바뀌지 않습니다."))
        _buttons(self, lay)

    def units_value(self) -> str:
        return self.units.currentData()

    def collision_value(self) -> str:
        return self.collision.currentData()
