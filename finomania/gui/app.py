"""Start the Finomania desktop application."""
from __future__ import annotations

import os
import sys
import traceback


def _install_excepthook(app):
    from PySide6.QtWidgets import QMessageBox

    from ..appenv import log_path

    def hook(etype, value, tb):
        txt = "".join(traceback.format_exception(etype, value, tb))
        try:
            sys.stderr.write(txt)
        except Exception:
            pass
        try:
            box = QMessageBox(QMessageBox.Critical, "Finomania - unexpected error",
                              f"{etype.__name__}: {value}\n\nThe application will keep running. Details were written "
                              f"to the log file:\n{log_path()}")
            box.setDetailedText(txt)
            box.exec()
        except Exception:
            pass

    sys.excepthook = hook


def _splash(app):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QFont, QPainter, QPixmap
    from PySide6.QtWidgets import QSplashScreen

    from .. import __version__
    from .style import app_icon
    pm = QPixmap(520, 300)
    pm.fill(QColor("#0f2f6e"))
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.drawPixmap(40, 70, app_icon().pixmap(150, 150))
    p.setPen(QColor("#ffffff"))
    f = QFont("Segoe UI", 30)
    f.setBold(True)
    p.setFont(f)
    p.drawText(215, 135, "Finomania")
    p.setFont(QFont("Segoe UI", 11))
    p.setPen(QColor("#c9dbff"))
    p.drawText(217, 165, "Econometrics for financial time series")
    p.drawText(217, 187, "and panel data")
    p.setPen(QColor("#8fb0e8"))
    p.setFont(QFont("Segoe UI", 9))
    p.drawText(217, 255, f"Version {__version__}   ·   open source (MIT)")
    p.end()
    sp = QSplashScreen(pm, Qt.WindowStaysOnTopHint)
    sp.show()
    sp.showMessage("Loading…", Qt.AlignBottom | Qt.AlignRight, QColor("#c9dbff"))
    app.processEvents()
    return sp


def run_gui(path: str | None = None) -> int:
    os.environ.setdefault("QT_API", "pyside6")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Finomania")
    app.setOrganizationName("Finomania")
    if sys.platform.startswith("win"):
        try:  # own taskbar icon instead of the generic python one
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Finomania.Finomania")
        except Exception:
            pass
    from .style import app_icon, apply_theme
    apply_theme(app)
    app.setWindowIcon(app_icon())
    sp = _splash(app)
    import matplotlib
    try:
        matplotlib.use("QtAgg")
    except Exception:  # figures are embedded with FigureCanvasQTAgg directly, so this is optional
        pass
    sp.showMessage("Loading econometrics engine…", 0x0002 | 0x0040)
    app.processEvents()
    from .mainwindow import MainWindow
    _install_excepthook(app)
    w = MainWindow()
    w.show()
    sp.finish(w)
    if path:
        w.open_path(os.path.abspath(path))
    return app.exec()
