"""얼굴 패널: 표정 프리셋과 파라미터, 눈·턱 조작.

화면에는 한국어 이름을 보이고, 외부 API에서 쓰는 원래 이름(예: blink_left)은 툴팁으로 보여 줍니다.
"""
from __future__ import annotations

import logging
import math
import os

import mujoco
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                               QMessageBox, QPushButton, QScrollArea, QSlider, QVBoxLayout, QWidget)

from ..core.editor import Editor
from ..core.expr import FaceController
from .theme import C

log = logging.getLogger("mjstudio")
SIDE_KO = {"left": "왼쪽", "right": "오른쪽"}
PARAM_KO = {"blink": "눈 감기", "eye_wide": "눈 크게 뜨기", "eye_squint": "눈 찡그리기", "brow_raise": "눈썹 올리기",
            "brow_lower": "눈썹 내리기", "mouth_smile": "입꼬리 올리기", "mouth_frown": "입꼬리 내리기",
            "cheek_puff": "볼 부풀리기", "smile": "미소 (양쪽)", "lips_part": "입술 벌리기", "lip_pucker": "입술 오므리기"}
GROUP_KO = {"eyes": "눈꺼풀", "brows": "눈썹", "mouth": "입", "cheeks": "볼", "other": "기타"}
PRESET_KO = {"neutral": "무표정", "smile": "미소", "frown": "찡그림", "surprise": "놀람", "blink": "눈 감기",
             "wink_left": "윙크 (왼쪽)", "wink_right": "윙크 (오른쪽)"}


def param_ko(name: str) -> str:
    """표정 파라미터의 한국어 표시 이름 (모르는 이름은 그대로)."""
    if name in PARAM_KO:
        return PARAM_KO[name]
    base, _, side = name.rpartition("_")
    if side in SIDE_KO and base in PARAM_KO:
        return f"{PARAM_KO[base]} ({SIDE_KO[side]})"
    return name


def _slider(lo=0, hi=100, val=0):
    s = QSlider(Qt.Horizontal)
    s.setRange(lo, hi)
    s.setValue(val)
    return s


class FacePanel(QWidget):
    def __init__(self, ed: Editor, sim=None, parent=None):
        super().__init__(parent)
        self.ed, self.sim = ed, sim
        self.sliders: dict[str, QSlider] = {}
        self._block = False
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("표정 리그"))
        self.rig_box = QComboBox()
        self.rig_box.currentTextChanged.connect(lambda *_: self.rebuild())
        bar.addWidget(self.rig_box, 1)
        b = QPushButton("기본 얼굴 추가")
        b.setToolTip("기본 얼굴(눈꺼풀, 입, Shape Key)을 'head'·'jaw' 바디에 씌웁니다")
        b.clicked.connect(self.add_procedural)
        bar.addWidget(b)
        b = QPushButton("Blender 얼굴(GLB) 가져오기…")
        b.setToolTip("Blender에서 만든 얼굴 메시: Shape Key → 표정, 'jaw' 본 가중치 → 턱 바디")
        b.clicked.connect(self.import_glb)
        bar.addWidget(b)
        outer.addLayout(bar)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        outer.addWidget(self.scroll)
        ed.rigs_changed.connect(self._rigs_changed)
        ed.compiled.connect(lambda r: r.ok and self._refresh_joints())
        self._timer = QTimer(self)
        self._timer.setInterval(200)
        self._timer.timeout.connect(self._refresh_joints)
        self._timer.start()
        self._rigs_changed()

    # ------------------------------------------------------------------ data
    @property
    def rig(self):
        return self.ed.rigs.get(self.rig_box.currentText())

    def controller(self) -> FaceController | None:
        if self.ed.model is None:
            return None
        kin = not (self.sim is not None and self.sim.running)
        return FaceController(self.ed.model, self.ed.data, self.rig, kinematic=kin)

    def _rigs_changed(self) -> None:
        cur = self.rig_box.currentText()
        self.rig_box.blockSignals(True)
        self.rig_box.clear()
        self.rig_box.addItems(list(self.ed.rigs))
        if cur in self.ed.rigs:
            self.rig_box.setCurrentText(cur)
        self.rig_box.blockSignals(False)
        self.rebuild()

    # ------------------------------------------------------------------ ui
    def rebuild(self) -> None:
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(2, 2, 2, 2)
        self.sliders = {}
        rig = self.rig
        # eyes + jaw (joints) are always available when the model has them
        jb = QGroupBox("눈과 턱 (MuJoCo 관절)")
        g = QGridLayout(jb)
        self.eye = {}
        r = 0
        for side in ("left", "right"):
            for k, (axis, lo, hi) in enumerate((("yaw", -45, 45), ("pitch", -45, 30))):
                s = _slider(lo * 10, hi * 10, 0)
                s.valueChanged.connect(lambda v, side=side, axis=axis: self._eye(side, axis, v / 10.0))
                lbl = QLabel(f"{SIDE_KO[side]} 눈 {'좌우 (yaw)' if axis == 'yaw' else '상하 (pitch)'}")
                lbl.setToolTip(f"eye_{side}_{axis} (도)")
                g.addWidget(lbl, r, 0)
                g.addWidget(s, r, 1)
                self.eye[(side, axis)] = s
                r += 1
        self.link = QCheckBox("두 눈 함께 움직이기 (공동 주시)")
        self.link.setChecked(True)
        g.addWidget(self.link, r, 0, 1, 2)
        r += 1
        self.jaw = _slider(0, 100, 0)
        self.jaw.valueChanged.connect(lambda v: self._jaw(v / 100.0))
        g.addWidget(QLabel("턱 벌리기"), r, 0)
        g.addWidget(self.jaw, r, 1)
        lay.addWidget(jb)
        if rig is None:
            lbl = QLabel("표정 리그가 없습니다. 위의 '기본 얼굴 추가'를 누르거나 Blender에서 만든 얼굴을 가져오세요 "
                         "(Shape Key가 있는 GLB, 선택적으로 'jaw' 본이 있는 아마추어).")
            lbl.setWordWrap(True)
            lbl.setStyleSheet(f"color:{C['dim']}")
            lay.addWidget(lbl)
        else:
            pb = QGroupBox("표정 프리셋")
            pl = QHBoxLayout(pb)
            for name in rig.presets:
                b = QPushButton(PRESET_KO.get(name, name))
                b.setToolTip(f"프리셋 이름: {name}")
                b.clicked.connect(lambda _=False, n=name: self.apply_preset(n))
                pl.addWidget(b)
            lay.addWidget(pb)
            groups: dict[str, list] = {}
            for p in rig.params.values():
                groups.setdefault(p.group or "other", []).append(p)
            for gname, plist in groups.items():
                box = QGroupBox(GROUP_KO.get(gname, gname))
                gl = QGridLayout(box)
                for k, p in enumerate(plist):
                    s = _slider(int(p.lo * 100), int(p.hi * 100), int(rig.get(p.name) * 100))
                    s.valueChanged.connect(lambda v, n=p.name: self._param(n, v / 100.0))
                    lbl = QLabel(param_ko(p.name))
                    lbl.setToolTip(f"API 이름: {p.name}")
                    gl.addWidget(lbl, k // 2, (k % 2) * 2)
                    gl.addWidget(s, k // 2, (k % 2) * 2 + 1)
                    self.sliders[p.name] = s
                lay.addWidget(box)
        lay.addStretch(1)
        self.scroll.setWidget(body)
        self._refresh_joints()

    def _sync_sliders(self) -> None:
        rig = self.rig
        if rig is None:
            return
        self._block = True
        for n, s in self.sliders.items():
            s.setValue(int(round(rig.get(n) * 100)))
        self._block = False

    def _refresh_joints(self) -> None:
        if not self.isVisible() or self.ed.model is None:
            return
        fc = self.controller()
        self._block = True
        try:
            for (side, axis), s in self.eye.items():
                if s.isSliderDown():
                    continue
                try:
                    yaw, pitch = fc.eye_rotation(side)
                except KeyError:
                    s.setEnabled(False)
                    continue
                s.setEnabled(True)
                s.setValue(int(round(math.degrees(yaw if axis == "yaw" else pitch) * 10)))
            try:
                j = fc._jid(fc.jaw_joint)
                lo, hi = self.ed.model.jnt_range[j]
                if not self.jaw.isSliderDown() and hi > lo:
                    self.jaw.setValue(int(round((fc.jaw_open() - lo) / (hi - lo) * 100)))
                self.jaw.setEnabled(True)
            except KeyError:
                self.jaw.setEnabled(False)
        finally:
            self._block = False

    # ------------------------------------------------------------------ actions
    def _param(self, name: str, v: float) -> None:
        if self._block or self.rig is None:
            return
        self.rig.set(name, v)
        self.ed.state_changed.emit()

    def apply_preset(self, name: str) -> None:
        if self.rig is None:
            return
        self.rig.apply_preset(name)
        self._sync_sliders()
        self.ed.state_changed.emit()

    def _eye(self, side: str, axis: str, deg: float) -> None:
        if self._block:
            return
        fc = self.controller()
        if fc is None:
            return
        sides = ("left", "right") if self.link.isChecked() else (side,)
        try:
            for s in sides:
                yaw, pitch = fc.eye_rotation(s)
                if axis == "yaw":
                    yaw = math.radians(deg)
                else:
                    pitch = math.radians(deg)
                fc.set_eye_rotation(s, yaw, pitch)
        except KeyError as ex:
            log.warning("%s", ex)
            return
        mujoco.mj_forward(self.ed.model, self.ed.data)
        self.ed.state_changed.emit()

    def _jaw(self, amount: float) -> None:
        if self._block:
            return
        fc = self.controller()
        try:
            fc.set_jaw_open(amount)
        except (KeyError, AttributeError) as ex:
            log.warning("%s", ex)
            return
        mujoco.mj_forward(self.ed.model, self.ed.data)
        self.ed.state_changed.emit()

    def add_procedural(self) -> None:
        from ..core.faceops import attach_face_rig
        from ..library.face import build_face_rig
        if self.ed.doc.find_named("body", "head") is None:
            QMessageBox.information(self, "얼굴", "모델에 이름이 'head'인 바디가 필요합니다 ('jaw'는 선택).\n"
                                    "추가·편집 › 신체 부품 › 머리 또는 전신 인간형을 먼저 추가하세요.")
            return
        name = self.ed.doc.unique_name("skin", "face")
        try:
            attach_face_rig(self.ed, build_face_rig(name))
        except (ValueError, KeyError) as ex:
            QMessageBox.warning(self, "얼굴", str(ex))
            return
        self.rig_box.setCurrentText(name)

    def import_glb(self) -> None:
        from ..core.faceops import attach_face_rig, rig_from_gltf
        path, _ = QFileDialog.getOpenFileName(self, "Blender 얼굴 가져오기 (glTF/GLB)", os.getcwd(),
                                              "glTF (*.glb *.gltf)")
        if not path:
            return
        bodies = self.ed.doc.names("body")
        head = "head" if "head" in bodies else None
        if head is None:
            QMessageBox.information(self, "얼굴", "모델에 이름이 'head'인 바디가 필요합니다.")
            return
        try:
            rig = rig_from_gltf(path, self.ed.doc.unique_name("skin", os.path.splitext(os.path.basename(path))[0]),
                                head_body=head, jaw_body="jaw" if "jaw" in bodies else None)
            attach_face_rig(self.ed, rig)
        except (OSError, ValueError, KeyError) as ex:
            QMessageBox.warning(self, "얼굴 가져오기", str(ex))
            return
        self.rig_box.setCurrentText(rig.name)
