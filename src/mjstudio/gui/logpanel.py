"""Log view fed by the 'mjstudio' logger (thread safe through a Qt signal)."""
from __future__ import annotations

import html
import logging
import time

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QHBoxLayout, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget

from .theme import C

COLORS = {logging.DEBUG: C["dim"], logging.INFO: C["text"], logging.WARNING: C["warn"],
          logging.ERROR: C["err"], logging.CRITICAL: C["err"]}


class _Bridge(QObject):
    record = Signal(int, str)


class QtLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.bridge = _Bridge()
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, rec: logging.LogRecord) -> None:
        try:
            self.bridge.record.emit(rec.levelno, self.format(rec))
        except RuntimeError:
            pass


class LogPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        self.view = QPlainTextEdit()
        self.view.setReadOnly(True)
        self.view.setMaximumBlockCount(5000)
        lay.addWidget(self.view)
        bar = QHBoxLayout()
        bar.addStretch(1)
        b = QPushButton("Clear")
        b.clicked.connect(self.view.clear)
        bar.addWidget(b)
        lay.addLayout(bar)
        self.handler = QtLogHandler()
        self.handler.bridge.record.connect(self.append)
        lg = logging.getLogger("mjstudio")
        lg.addHandler(self.handler)
        lg.setLevel(logging.INFO)

    def append(self, level: int, msg: str) -> None:
        col = COLORS.get(level, C["text"])
        ts = time.strftime("%H:%M:%S")
        tag = logging.getLevelName(level)[0]
        body = html.escape(msg).replace("\n", "<br>&nbsp;&nbsp;&nbsp;&nbsp;")
        self.view.appendHtml(f"<span style='color:{C['dim']}'>{ts} {tag}</span> "
                             f"<span style='color:{col}'>{body}</span>")

    def detach(self) -> None:
        logging.getLogger("mjstudio").removeHandler(self.handler)
