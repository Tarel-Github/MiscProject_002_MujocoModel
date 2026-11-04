"""Dark theme and small generated icons (no binary resources needed)."""
from __future__ import annotations

from functools import lru_cache

from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QBrush, QColor, QIcon, QPainter, QPainterPath, QPalette, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QApplication

C = {
    "bg": "#1e1f22", "base": "#16171a", "alt": "#212226", "panel": "#25272b", "text": "#d5d7dc",
    "dim": "#7d828c", "border": "#34373d", "accent": "#4c8dff", "hl": "#ff9b3d", "err": "#ff6b6b",
    "warn": "#ffcc66", "ok": "#7bd88f",
}

# element-type colours used for tree icons and labels
TAG_COLORS = {
    "body": "#e0b45a", "geom": "#62a8ff", "joint": "#ff7a6e", "freejoint": "#ff7a6e", "site": "#c38cff",
    "camera": "#9be27c", "light": "#ffe066", "inertial": "#8aa0b8", "mesh": "#5fd1c9", "texture": "#d98fd1",
    "material": "#f0a37a", "default": "#9aa3ad", "key": "#b8c27a", "actuator": "#ff9f43", "sensor": "#4dd0e1",
    "tendon": "#b39ddb", "equality": "#aed581", "contact": "#90a4ae", "section": "#9aa3ad",
}

STYLE = f"""
QWidget {{ font-size: 9.5pt; }}
QMainWindow::separator {{ background: {C['border']}; width: 3px; height: 3px; }}
QDockWidget {{ titlebar-close-icon: none; }}
QDockWidget::title {{ background: {C['panel']}; padding: 4px 6px; border-bottom: 1px solid {C['border']}; }}
QTreeView, QTreeWidget, QTableWidget, QListWidget, QPlainTextEdit {{
    background: {C['base']}; alternate-background-color: {C['alt']}; border: 1px solid {C['border']}; }}
QTreeView::item {{ padding: 2px 0px; }}
QTreeView::item:selected, QTableWidget::item:selected {{ background: #2f4f86; color: white; }}
QHeaderView::section {{ background: {C['panel']}; color: {C['dim']}; border: 0px; border-right: 1px solid {C['border']}; padding: 3px; }}
QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox {{ background: {C['base']}; border: 1px solid {C['border']};
    border-radius: 3px; padding: 2px 4px; selection-background-color: {C['accent']}; }}
QAbstractSpinBox[nobtn="true"] {{ padding-right: 0px; }}
QAbstractSpinBox[nobtn="true"]::up-button, QAbstractSpinBox[nobtn="true"]::down-button {{
    subcontrol-origin: border; width: 0px; border: 0px; }}
QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus {{ border: 1px solid {C['accent']}; }}
QLineEdit[explicit="false"] {{ color: {C['dim']}; }}
QLineEdit[invalid="true"] {{ border: 1px solid {C['err']}; }}
QToolBar {{ background: {C['panel']}; border: 0px; spacing: 2px; padding: 2px; }}
QToolButton {{ padding: 3px 6px; border-radius: 3px; }}
QToolButton:checked {{ background: #34507f; }}
QToolButton:hover {{ background: #33363c; }}
QGroupBox {{ border: 1px solid {C['border']}; border-radius: 4px; margin-top: 10px; padding-top: 6px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; color: {C['dim']}; }}
QPushButton {{ background: #2c2f35; border: 1px solid {C['border']}; border-radius: 3px; padding: 3px 10px; }}
QPushButton:hover {{ background: #353941; }}
QPushButton:checked {{ background: #34507f; }}
QTabBar::tab {{ background: {C['panel']}; padding: 4px 10px; border: 1px solid {C['border']}; border-bottom: 0px; }}
QTabBar::tab:selected {{ background: {C['base']}; color: white; }}
QStatusBar {{ background: {C['panel']}; color: {C['dim']}; }}
QSlider::groove:horizontal {{ height: 4px; background: #3a3d44; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {C['accent']}; width: 12px; margin: -5px 0; border-radius: 6px; }}
QSlider::sub-page:horizontal {{ background: #2f5aa8; border-radius: 2px; }}
QScrollArea {{ border: 0px; }}
QLabel[section="true"] {{ color: {C['dim']}; font-weight: bold; padding-top: 6px; }}
"""


def apply_dark(app: QApplication) -> None:
    app.setStyle("Fusion")
    p = QPalette()
    col = lambda k: QColor(C[k])  # noqa: E731
    p.setColor(QPalette.Window, col("bg"))
    p.setColor(QPalette.WindowText, col("text"))
    p.setColor(QPalette.Base, col("base"))
    p.setColor(QPalette.AlternateBase, col("alt"))
    p.setColor(QPalette.ToolTipBase, col("panel"))
    p.setColor(QPalette.ToolTipText, col("text"))
    p.setColor(QPalette.Text, col("text"))
    p.setColor(QPalette.Button, col("panel"))
    p.setColor(QPalette.ButtonText, col("text"))
    p.setColor(QPalette.BrightText, QColor("#ffffff"))
    p.setColor(QPalette.Link, col("accent"))
    p.setColor(QPalette.Highlight, QColor("#2f5aa8"))
    p.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    p.setColor(QPalette.PlaceholderText, col("dim"))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        p.setColor(QPalette.Disabled, role, QColor("#5b5f66"))
    app.setPalette(p)
    app.setStyleSheet(STYLE)


def tag_color(tag: str, parent_tag: str | None = None) -> str:
    if parent_tag == "actuator":
        return TAG_COLORS["actuator"]
    if parent_tag == "sensor":
        return TAG_COLORS["sensor"]
    if parent_tag == "tendon":
        return TAG_COLORS["tendon"]
    if parent_tag == "equality":
        return TAG_COLORS["equality"]
    if parent_tag == "contact":
        return TAG_COLORS["contact"]
    return TAG_COLORS.get(tag, "#a0a6b0")


@lru_cache(maxsize=128)
def shape_icon(shape: str, color: str, size: int = 14) -> QIcon:
    """Tiny vector icon: circle, square, diamond, triangle, ring, hex."""
    pm = QPixmap(size * 2, size * 2)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    c = QColor(color)
    p.setPen(QPen(c.darker(150), 2))
    p.setBrush(QBrush(c))
    s = size * 2
    r = QRectF(5, 5, s - 10, s - 10)
    if shape == "circle":
        p.drawEllipse(r)
    elif shape == "square":
        p.drawRoundedRect(r, 4, 4)
    elif shape == "diamond":
        p.drawPolygon(QPolygonF([QPointF(s / 2, 3), QPointF(s - 3, s / 2), QPointF(s / 2, s - 3), QPointF(3, s / 2)]))
    elif shape == "triangle":
        p.drawPolygon(QPolygonF([QPointF(s / 2, 4), QPointF(s - 4, s - 5), QPointF(4, s - 5)]))
    elif shape == "ring":
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(c, 5))
        p.drawEllipse(r.adjusted(2, 2, -2, -2))
    elif shape == "hex":
        path = QPainterPath()
        import math
        for i in range(6):
            a = math.pi / 3 * i + math.pi / 6
            pt = QPointF(s / 2 + (s / 2 - 4) * math.cos(a), s / 2 + (s / 2 - 4) * math.sin(a))
            path.moveTo(pt) if i == 0 else path.lineTo(pt)
        path.closeSubpath()
        p.drawPath(path)
    else:
        p.drawRect(r)
    p.end()
    return QIcon(pm)


TAG_SHAPES = {"body": "square", "geom": "circle", "joint": "ring", "freejoint": "ring", "site": "diamond",
              "camera": "triangle", "light": "hex", "inertial": "diamond", "default": "square"}


def icon_for(tag: str, parent_tag: str | None = None) -> QIcon:
    return shape_icon(TAG_SHAPES.get(tag, "circle"), tag_color(tag, parent_tag))


def no_buttons(spin) -> None:
    """Hide spin box arrows (works around a Qt stylesheet quirk that collapses the edit field)."""
    from PySide6.QtWidgets import QAbstractSpinBox
    spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
    spin.setProperty("nobtn", True)
