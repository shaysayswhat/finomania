"""Reusable GUI widgets: spreadsheet model, output viewers, command editor, program editor, option forms."""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QRegularExpression, Qt, Signal
from PySide6.QtGui import (QColor, QFont, QKeySequence, QSyntaxHighlighter, QTextCharFormat, QTextCursor)
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit,
                               QSizePolicy, QSpinBox, QTabWidget, QTextBrowser, QVBoxLayout, QWidget)

from ..models.base import Param
from ..report import FigureOutput, Message, Report


# ============================================================================ spreadsheet
class SeriesTableModel(QAbstractTableModel):
    """Spreadsheet view of one or several workfile series (editable when edit mode is on)."""

    def __init__(self, wf, names: list[str], parent=None):
        super().__init__(parent)
        self.wf = wf
        self.names = list(names)
        self.editable = False
        self._labels = wf.labels()

    def refresh(self):
        self.beginResetModel()
        self._labels = self.wf.labels()
        self.names = [n for n in self.names if n in self.wf.data.columns]
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else self.wf.nobs

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.names)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        r, c = index.row(), index.column()
        name = self.names[c]
        if role in (Qt.DisplayRole, Qt.EditRole):
            v = self.wf.data[name].iloc[r]
            if isinstance(v, str):
                return v
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return "NA"
            try:
                fv = float(v)
            except (TypeError, ValueError):
                return str(v)
            return _fmt_cell(fv) if role == Qt.DisplayRole else repr(fv)
        if role == Qt.TextAlignmentRole:
            return int(Qt.AlignRight | Qt.AlignVCenter)
        if role == Qt.ForegroundRole:
            v = self.wf.data[name].iloc[r]
            if isinstance(v, float) and math.isnan(v):
                return QColor("#9aa3ad")
            if not self.wf.sample[r]:
                return QColor("#8a93a0")
        if role == Qt.BackgroundRole and not self.wf.sample[r]:
            return QColor("#f3f4f6")
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return self.names[section].upper()
        return self._labels[section] if section < len(self._labels) else str(section + 1)

    def flags(self, index):
        f = Qt.ItemIsSelectable | Qt.ItemIsEnabled
        if self.editable:
            f |= Qt.ItemIsEditable
        return f

    def setData(self, index, value, role=Qt.EditRole):
        if role != Qt.EditRole or not self.editable:
            return False
        name = self.names[index.column()]
        col = self.wf.data[name]
        txt = str(value).strip()
        if pd.api.types.is_numeric_dtype(col):
            try:
                v = float("nan") if txt.upper() in ("NA", "") else float(txt)
            except ValueError:
                return False
            vals = col.values.astype(float).copy()
        else:
            v = txt
            vals = col.values.copy()
        vals[index.row()] = v
        self.wf.data[name] = vals
        self.wf.dirty = True
        self.dataChanged.emit(index, index)
        return True


def _fmt_cell(v: float) -> str:
    a = abs(v)
    if a != 0 and (a >= 1e10 or a < 1e-5):
        return f"{v:.6E}"
    if v == int(v):
        return f"{int(v)}"
    if a < 100:
        return f"{v:.6f}"
    if a < 1e5:
        return f"{v:.4f}"
    return f"{v:.2f}"


# ============================================================================ output viewers
class FigureView(QWidget):
    def __init__(self, fig, parent=None):
        super().__init__(parent)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.fig = fig
        if not hasattr(fig, "add_subplot"):   # restored PNG figure
            from PySide6.QtGui import QPixmap
            import base64
            lab = QLabel()
            pm = QPixmap()
            pm.loadFromData(base64.b64decode(fig.b64))
            lab.setPixmap(pm)
            lab.setAlignment(Qt.AlignCenter)
            lay.addWidget(lab)
            self.canvas = None
            return
        self.canvas = FigureCanvasQTAgg(fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        tb = NavigationToolbar2QT(self.canvas, self)
        tb.setIconSize(tb.iconSize() * 0.8)
        lay.addWidget(tb)
        lay.addWidget(self.canvas)
        self.canvas.draw_idle()


class ReportView(QTextBrowser):
    def __init__(self, rep: Report, parent=None):
        super().__init__(parent)
        self.report = rep
        self.setOpenExternalLinks(True)
        self.setHtml(rep.to_html(qt=True))
        self.document().setDocumentMargin(14)


class OutputView(QWidget):
    """Displays a list of outputs (reports / figures / messages); tabs when there are several."""

    def __init__(self, outputs: list, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.outputs = [o for o in outputs if isinstance(o, (Report, FigureOutput))]
        msgs = [o for o in outputs if isinstance(o, Message)]
        if msgs and not self.outputs:
            r = Report("")
            for m in msgs:
                r.text(m.text, "warning" if m.level == "error" else "normal")
            self.outputs = [r]
        if len(self.outputs) == 1:
            lay.addWidget(_make_view(self.outputs[0]))
        else:
            tabs = QTabWidget()
            tabs.setDocumentMode(True)
            for o in self.outputs:
                title = (o.title or "Output")[:40]
                tabs.addTab(_make_view(o), ("Graph: " if isinstance(o, FigureOutput) else "") + title)
            lay.addWidget(tabs)
            self.tabs = tabs

    def current(self):
        if len(self.outputs) == 1:
            return self.outputs[0]
        return self.outputs[self.tabs.currentIndex()] if self.outputs else None


def _make_view(o):
    if isinstance(o, FigureOutput):
        return FigureView(o.figure)
    return ReportView(o)


# ============================================================================ command window
class CommandEdit(QPlainTextEdit):
    """EViews-style command window: Enter executes the line under the cursor."""
    command = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("command")
        self.setPlaceholderText("Type a command and press Enter   (e.g.  wfcreate m 2000m1 2020m12   |   help)")
        self.history: list[str] = []
        self.hpos = 0

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Return, Qt.Key_Enter) and not (e.modifiers() & Qt.ShiftModifier):
            cur = self.textCursor()
            if cur.hasSelection():
                txt = cur.selectedText().replace(" ", "\n")
            else:
                cur.select(QTextCursor.LineUnderCursor)
                txt = cur.selectedText()
            txt = txt.strip()
            if txt:
                self.history.append(txt)
                self.hpos = len(self.history)
                # keep EViews behaviour: the line stays, cursor moves to a fresh line at the end
                doc = self.toPlainText()
                if not doc.endswith("\n"):
                    self.moveCursor(QTextCursor.End)
                    self.insertPlainText("\n")
                else:
                    self.moveCursor(QTextCursor.End)
                self.command.emit(txt)
            return
        if e.key() == Qt.Key_Up and e.modifiers() & Qt.ControlModifier and self.history:
            self.hpos = max(0, self.hpos - 1)
            self._replace_last_line(self.history[self.hpos])
            return
        if e.key() == Qt.Key_Down and e.modifiers() & Qt.ControlModifier and self.history:
            self.hpos = min(len(self.history), self.hpos + 1)
            self._replace_last_line(self.history[self.hpos] if self.hpos < len(self.history) else "")
            return
        super().keyPressEvent(e)

    def _replace_last_line(self, txt):
        cur = self.textCursor()
        cur.movePosition(QTextCursor.End)
        cur.select(QTextCursor.LineUnderCursor)
        cur.insertText(txt)
        self.setTextCursor(cur)

    def append_command(self, txt: str):
        self.moveCursor(QTextCursor.End)
        doc = self.toPlainText()
        cur = self.textCursor()
        cur.select(QTextCursor.LineUnderCursor)
        if cur.selectedText().strip():
            self.moveCursor(QTextCursor.End)
            self.insertPlainText("\n")
        self.insertPlainText(txt + "\n")
        self.ensureCursorVisible()
        _ = doc


# ============================================================================ program editor
class ProgramHighlighter(QSyntaxHighlighter):
    KEYWORDS = r"\b(for|next|if|then|else|endif|while|wend|to|step|exitloop|stop|equation|series|genr|scalar|group|" \
               r"smpl|var|graph|freeze|show|import|wfcreate|wfopen|wfsave|delete|rename|logmsg|matrix)\b"

    def __init__(self, doc):
        super().__init__(doc)
        self.rules = []
        f = QTextCharFormat()
        f.setForeground(QColor("#0b4ea2"))
        f.setFontWeight(QFont.Bold)
        self.rules.append((QRegularExpression(self.KEYWORDS, QRegularExpression.CaseInsensitiveOption), f))
        f2 = QTextCharFormat()
        f2.setForeground(QColor("#7a3e9d"))
        self.rules.append((QRegularExpression(r"@\w+"), f2))
        f3 = QTextCharFormat()
        f3.setForeground(QColor("#b35c00"))
        self.rules.append((QRegularExpression(r"[!%]\w+|\{[!%]\w+\}"), f3))
        f4 = QTextCharFormat()
        f4.setForeground(QColor("#2b7a2b"))
        self.rules.append((QRegularExpression(r'"[^"]*"'), f4))
        self.comment = QTextCharFormat()
        self.comment.setForeground(QColor("#8a8f98"))
        self.comment.setFontItalic(True)

    def highlightBlock(self, text):
        for rx, fmt in self.rules:
            it = rx.globalMatch(text)
            while it.hasNext():
                m = it.next()
                self.setFormat(m.capturedStart(), m.capturedLength(), fmt)
        inq = False
        for i, ch in enumerate(text):
            if ch == '"':
                inq = not inq
            elif ch == "'" and not inq:
                self.setFormat(i, len(text) - i, self.comment)
                break


class ProgramEdit(QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        f = QFont("Consolas", 10)
        f.setStyleHint(QFont.Monospace)
        self.setFont(f)
        self.setTabStopDistance(28)
        self.highlighter = ProgramHighlighter(self.document())
        self.setLineWrapMode(QPlainTextEdit.NoWrap)


# ============================================================================ option forms
class ParamForm(QWidget):
    """Builds input widgets from a list of Param and returns their values."""

    def __init__(self, params: list[Param], parent=None, series_names: list[str] | None = None):
        super().__init__(parent)
        self.params = params
        self.widgets = {}
        lay = QFormLayout(self)
        lay.setLabelAlignment(Qt.AlignRight)
        for p in params:
            w = self._widget(p, series_names)
            self.widgets[p.name] = w
            if p.help:
                w.setToolTip(p.help)
            lay.addRow(p.label + ":", w)

    def _widget(self, p: Param, series_names):
        if p.kind == "bool":
            w = QCheckBox()
            w.setChecked(bool(p.default))
            return w
        if p.kind == "choice":
            w = QComboBox()
            w.addItems([str(c) for c in (p.choices or [])])
            if p.default is not None and str(p.default) in [str(c) for c in p.choices or []]:
                w.setCurrentText(str(p.default))
            return w
        if p.kind == "int":
            w = QSpinBox()
            w.setRange(-1_000_000, 1_000_000)
            w.setValue(int(p.default) if p.default not in (None, "") else 0)
            return w
        w = QLineEdit()
        if p.default is not None:
            w.setText(str(p.default))
        return w

    def values(self) -> dict:
        out = {}
        for p in self.params:
            w = self.widgets[p.name]
            if isinstance(w, QCheckBox):
                out[p.name] = w.isChecked()
            elif isinstance(w, QComboBox):
                out[p.name] = w.currentText()
            elif isinstance(w, QSpinBox):
                out[p.name] = w.value()
            else:
                out[p.name] = w.text().strip()
        return out

    def options_string(self, skip_defaults: bool = True, positional: list[str] | None = None,
                       always: list[str] | None = None) -> str:
        vals = self.values()
        parts = []
        always = always or []
        for p in self.params:
            v = vals[p.name]
            if skip_defaults and p.name not in always and (v == p.default or (v in ("", None) and p.default in ("", None))):
                continue
            if v in ("", None):
                continue
            if isinstance(v, bool):
                v = 1 if v else 0
            sv = str(v)
            if positional and p.name in positional:
                parts.append(sv)
            elif " " in sv or "," in sv:
                parts.append(f'{p.name}="{sv}"')
            else:
                parts.append(f"{p.name}={sv}")
        return ", ".join(parts)


def hline():
    from PySide6.QtWidgets import QFrame
    f = QFrame()
    f.setFrameShape(QFrame.HLine)
    f.setStyleSheet("color: #dde2e8")
    return f
