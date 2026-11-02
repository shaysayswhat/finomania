"""Look & feel: palette, stylesheet and programmatically drawn object icons."""
from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPainterPath, QPalette, QPen, QPixmap

from ..core.objects import OBJECT_TYPES

ACCENT = "#1f4e9c"

STYLESHEET = """
QMainWindow { background: #eef1f5; }
QMdiArea { background: #dfe4ea; }
QMenuBar { background: #f7f8fa; border-bottom: 1px solid #d5dae1; }
QMenuBar::item { padding: 4px 10px; }
QMenuBar::item:selected { background: #dbe6f6; border-radius: 3px; }
QMenu { background: #ffffff; border: 1px solid #c9d0da; }
QMenu::item { padding: 5px 26px 5px 22px; }
QMenu::item:selected { background: #dbe6f6; color: #0b2e66; }
QMenu::separator { height: 1px; background: #e3e7ed; margin: 3px 8px; }
QToolBar { background: #f4f6f9; border: none; border-bottom: 1px solid #d9dee5; spacing: 2px; padding: 2px; }
QToolButton { padding: 3px 8px; border: 1px solid transparent; border-radius: 3px; }
QToolButton:hover { background: #e3ebf7; border: 1px solid #bcd0ee; }
QToolButton:pressed { background: #cfdcf2; }
QToolButton::menu-indicator { subcontrol-position: right center; right: 2px; }
QStatusBar { background: #f7f8fa; border-top: 1px solid #d5dae1; }
QStatusBar QLabel { padding: 0 8px; color: #333; }
QDockWidget::title { background: #e9edf2; padding: 4px 6px; border-bottom: 1px solid #d5dae1; font-weight: 600; }
QPlainTextEdit#command { background: #ffffff; border: none; font-family: Consolas, 'Courier New', monospace;
                         font-size: 10pt; }
QPlainTextEdit#log { background: #fbfbfc; border: none; font-family: Consolas, 'Courier New', monospace;
                     font-size: 9pt; color: #333; }
QTextBrowser { background: #ffffff; border: none; }
QTableView { gridline-color: #e1e5ea; selection-background-color: #cfe0fa; selection-color: #000;
             alternate-background-color: #fafbfd; }
QHeaderView::section { background: #f1f3f6; border: none; border-right: 1px solid #dde2e8;
                       border-bottom: 1px solid #dde2e8; padding: 3px 6px; font-weight: 600; }
QListWidget#objects { background: #ffffff; border: none; font-size: 9pt; }
QListWidget#objects::item { padding: 1px 2px; }
QListWidget#objects::item:selected { background: #cfe0fa; color: #000; }
QLabel#wfinfo { color: #333; padding: 2px 6px; }
QWidget#wfheader { background: #f7f9fb; border-bottom: 1px solid #e0e4ea; }
QGroupBox { font-weight: 600; border: 1px solid #d5dae1; border-radius: 4px; margin-top: 10px; padding-top: 8px; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
QPushButton { padding: 4px 14px; border: 1px solid #b9c3d0; border-radius: 3px; background: #fdfdfe; }
QPushButton:hover { background: #eef3fb; border-color: #8fb0e0; }
QPushButton:default { border: 1px solid #1f4e9c; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { padding: 3px 5px; border: 1px solid #c3ccd8; border-radius: 3px;
                                                 background: #fff; }
QTabWidget::pane { border: 1px solid #d5dae1; }
QTabBar::tab { padding: 5px 14px; background: #eef1f5; border: 1px solid #d5dae1; border-bottom: none; }
QTabBar::tab:selected { background: #ffffff; }
"""


def apply_theme(app):
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor("#eef1f5"))
    pal.setColor(QPalette.Base, QColor("#ffffff"))
    pal.setColor(QPalette.AlternateBase, QColor("#f6f8fb"))
    pal.setColor(QPalette.Text, QColor("#111111"))
    pal.setColor(QPalette.WindowText, QColor("#111111"))
    pal.setColor(QPalette.Button, QColor("#f7f8fa"))
    pal.setColor(QPalette.ButtonText, QColor("#111111"))
    pal.setColor(QPalette.Highlight, QColor("#2f6fd0"))
    pal.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.ToolTipBase, QColor("#ffffe8"))
    pal.setColor(QPalette.ToolTipText, QColor("#111111"))
    app.setPalette(pal)
    f = QFont("Segoe UI", 9)
    f.setStyleHint(QFont.SansSerif)
    app.setFont(f)
    app.setStyleSheet(STYLESHEET)


_icon_cache: dict[str, QIcon] = {}


def object_icon(kind: str) -> QIcon:
    if kind in _icon_cache:
        return _icon_cache[kind]
    glyph, color = OBJECT_TYPES.get(kind, ("?", "#666"))
    pm = QPixmap(32, 32)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    c = QColor(color)
    p.setBrush(c.lighter(185))
    p.setPen(QPen(c, 2))
    p.drawRoundedRect(QRectF(3, 3, 26, 26), 5, 5)
    p.setPen(QPen(c.darker(130), 2.4))
    if kind == "series":
        path = QPainterPath()
        path.moveTo(7, 20)
        path.cubicTo(11, 6, 15, 6, 16, 16)
        path.cubicTo(17, 26, 22, 26, 25, 11)
        p.setBrush(Qt.NoBrush)
        p.drawPath(path)
    elif kind == "graph":
        p.setBrush(c.darker(110))
        p.setPen(Qt.NoPen)
        for i, hgt in enumerate((8, 14, 11, 18)):
            p.drawRect(QRectF(7 + i * 5, 25 - hgt, 3.5, hgt))
    elif kind == "table":
        p.setPen(QPen(c.darker(130), 1.4))
        for y in (11, 16, 21):
            p.drawLine(7, y, 25, y)
        p.drawLine(14, 7, 14, 25)
    else:
        f = QFont("Segoe UI", 13 if len(glyph) == 1 else 9)
        f.setBold(True)
        p.setFont(f)
        p.drawText(QRectF(3, 3, 26, 26), Qt.AlignCenter, glyph)
    p.end()
    ic = QIcon(pm)
    _icon_cache[kind] = ic
    return ic


def app_icon() -> QIcon:
    import os
    png = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "finomania.png")
    if os.path.exists(png):
        return QIcon(png)
    pm = QPixmap(64, 64)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QColor(ACCENT))
    p.setPen(Qt.NoPen)
    p.drawRoundedRect(QRectF(2, 2, 60, 60), 12, 12)
    p.setPen(QPen(QColor("#ffffff"), 4.5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    path = QPainterPath()
    path.moveTo(12, 44)
    path.lineTo(24, 30)
    path.lineTo(33, 38)
    path.lineTo(52, 16)
    p.drawPath(path)
    p.setPen(QPen(QColor("#9fc3ff"), 3))
    p.drawLine(12, 52, 52, 52)
    p.end()
    return QIcon(pm)
