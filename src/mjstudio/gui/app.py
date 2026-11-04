"""Application entry point."""
from __future__ import annotations

import logging
import os
import sys


def configure_gl() -> None:
    """MuJoCo's renderer needs a compatibility-profile OpenGL context.
    (Requesting a version together with the compatibility profile can yield a
    core context on some drivers, so only the profile is requested.)"""
    from PySide6.QtGui import QSurfaceFormat
    fmt = QSurfaceFormat()
    fmt.setProfile(QSurfaceFormat.CompatibilityProfile)
    fmt.setDepthBufferSize(24)
    fmt.setStencilBufferSize(8)
    QSurfaceFormat.setDefaultFormat(fmt)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    configure_gl()
    from PySide6.QtWidgets import QApplication
    from .theme import apply_dark
    app = QApplication.instance() or QApplication(argv)
    app.setApplicationName("MuJoCo Model Studio")
    apply_dark(app)
    from .mainwin import MainWindow
    w = MainWindow()
    w.show()
    files = [a for a in argv[1:] if not a.startswith("-")]
    if files:
        w.open_path(files[0])
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
