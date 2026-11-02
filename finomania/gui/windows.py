"""MDI child windows: workfile, series, group, estimated objects, output, graph and program windows."""
from __future__ import annotations

import os
import re

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QAbstractItemView, QFileDialog, QHBoxLayout, QInputDialog, QLabel, QLineEdit,
                               QListView, QListWidget, QListWidgetItem, QMenu, QMessageBox, QTableView, QToolBar,
                               QToolButton, QVBoxLayout, QWidget)

from ..core.objects import Estimation, Graph, Group, Table
from ..report import FigureOutput, Message, Report
from .style import object_icon
from .widgets import OutputView, ProgramEdit, SeriesTableModel


class ChildWindow(QWidget):
    """Base class: a toolbar + a replaceable content area."""
    kind = "window"

    def __init__(self, main, title: str):
        super().__init__()
        self.main = main
        self.setWindowTitle(title)
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(0, 0, 0, 0)
        self.lay.setSpacing(0)
        self.tb = QToolBar()
        self.tb.setIconSize(QSize(14, 14))
        self.tb.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.lay.addWidget(self.tb)
        self.content = None
        self.current_outputs: list = []

    def set_content(self, w):
        if self.content is not None:
            self.lay.removeWidget(self.content)
            self.content.deleteLater()
        self.content = w
        self.lay.addWidget(w, 1)

    def show_outputs(self, outputs):
        outs = [o for o in outputs if isinstance(o, (Report, FigureOutput))]
        if not outs:
            return False
        self.current_outputs = outs
        self.set_content(OutputView(outs))
        return True

    def menu_button(self, text: str) -> QMenu:
        b = QToolButton()
        b.setText(text)
        b.setPopupMode(QToolButton.InstantPopup)
        m = QMenu(b)
        b.setMenu(m)
        self.tb.addWidget(b)
        return m

    def button(self, text: str, slot, tip: str = ""):
        a = QAction(text, self)
        a.triggered.connect(slot)
        if tip:
            a.setToolTip(tip)
        self.tb.addAction(a)
        return a

    def freeze(self):
        if not self.current_outputs:
            QMessageBox.information(self, "Freeze", "Nothing to freeze in this view.")
            return
        self.main.freeze_outputs(self.current_outputs)

    def save_output(self):
        if not self.current_outputs:
            return
        o = self.current_outputs[0] if len(self.current_outputs) == 1 else self.content.current()
        if isinstance(o, FigureOutput):
            p, _ = QFileDialog.getSaveFileName(self, "Save graph", "graph.png", "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
            if p:
                o.figure.savefig(p, dpi=150, bbox_inches="tight")
            return
        p, _ = QFileDialog.getSaveFileName(self, "Save table", "output.html",
                                           "HTML (*.html);;CSV (*.csv);;LaTeX (*.tex);;Text (*.txt)")
        if p:
            ext = os.path.splitext(p)[1].lower()
            txt = o.to_csv() if ext == ".csv" else o.to_latex() if ext == ".tex" else o.to_text() if ext == ".txt" \
                else o.to_html()
            with open(p, "w", encoding="utf-8") as f:
                f.write(txt)
            self.main.log(f"Saved {p}")

    def copy_output(self):
        from PySide6.QtWidgets import QApplication
        o = self.current_outputs[0] if self.current_outputs else None
        if isinstance(o, Report):
            QApplication.clipboard().setText(o.to_text())
            self.main.log("Output copied to clipboard (text)")


# ============================================================================ workfile window
class WorkfileWindow(ChildWindow):
    kind = "workfile"
    changed = Signal(str)

    def __init__(self, main, wf):
        super().__init__(main, f"Workfile: {wf.name.upper()}")
        self.wf = wf
        m = self.menu_button("View")
        m.addAction("Refresh", self.refresh)
        m.addAction("Name filter...", lambda: self.filter.setFocus())
        m = self.menu_button("Proc")
        m.addAction("Set Sample...", self.main.dlg_sample)
        m.addAction("Generate Series...", self.main.dlg_genr)
        m.addAction("Expand range...", self.main.dlg_expand)
        m.addAction("Import data...", self.main.dlg_import)
        m.addAction("Fetch from web...", self.main.dlg_fetch)
        m = self.menu_button("Object")
        m.addAction("Delete", self.delete_sel)
        m.addAction("Rename...", self.rename_sel)
        m.addAction("Copy...", self.copy_sel)
        m.addSeparator()
        m.addAction("New Group from selection", self.open_as_group)
        self.tb.addSeparator()
        self.button("Save", self.main.save_workfile)
        self.button("Show", self.show_sel)
        self.button("Genr", self.main.dlg_genr)
        self.button("Sample", self.main.dlg_sample)
        self.button("Estimate", self.estimate_sel)
        self.button("Delete", self.delete_sel)
        hdr = QWidget()
        hdr.setObjectName("wfheader")
        hl = QVBoxLayout(hdr)
        hl.setContentsMargins(6, 4, 6, 4)
        hl.setSpacing(1)
        self.range_lab = QLabel()
        self.range_lab.setObjectName("wfinfo")
        self.sample_lab = QLabel()
        self.sample_lab.setObjectName("wfinfo")
        self.sample_lab.setCursor(Qt.PointingHandCursor)
        self.sample_lab.mouseDoubleClickEvent = lambda e: self.main.dlg_sample()
        frow = QHBoxLayout()
        frow.addWidget(QLabel("Filter:"))
        self.filter = QLineEdit("*")
        self.filter.setMaximumWidth(160)
        self.filter.textChanged.connect(self.refresh)
        frow.addWidget(self.filter)
        frow.addStretch()
        hl.addWidget(self.range_lab)
        hl.addWidget(self.sample_lab)
        hl.addLayout(frow)
        self.lay.addWidget(hdr)
        self.list = QListWidget()
        self.list.setObjectName("objects")
        self.list.setViewMode(QListView.ListMode)
        self.list.setFlow(QListView.TopToBottom)
        self.list.setWrapping(True)
        self.list.setResizeMode(QListView.Adjust)
        self.list.setGridSize(QSize(150, 20))
        self.list.setIconSize(QSize(16, 16))
        self.list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list.itemDoubleClicked.connect(lambda it: self.open_item(it.data(Qt.UserRole)))
        self.list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self.context_menu)
        self.set_content(self.list)
        self.changed.connect(self.refresh)
        wf.on_change(lambda what: self.changed.emit(what))
        self.refresh()

    def refresh(self, *_):
        wf = self.wf
        self.setWindowTitle(f"Workfile: {wf.name.upper()}" + (f"  —  {wf.path}" if wf.path else ""))
        self.range_lab.setText(f"Range:   {wf.range_text()}")
        self.sample_lab.setText(f"Sample:  {wf.sample_text}   ({int(wf.sample.sum())} obs)      (double-click to change)")
        pat = self.filter.text().strip().lower() or "*"
        rx = re.compile("^" + re.escape(pat).replace(r"\*", ".*").replace(r"\?", ".") + "$")
        sel = {i.data(Qt.UserRole) for i in self.list.selectedItems()}
        self.list.clear()
        for name, typ in wf.listing():
            if not rx.match(name):
                continue
            it = QListWidgetItem(object_icon(typ or "series"), name)
            it.setData(Qt.UserRole, name)
            it.setToolTip(f"{typ}: {name.upper()}")
            self.list.addItem(it)
            if name in sel:
                it.setSelected(True)
        self.main.update_status()

    def selected(self) -> list[str]:
        return [i.data(Qt.UserRole) for i in self.list.selectedItems()]

    def open_item(self, name):
        self.main.activate_workfile(self.wf)
        self.main.run_command(f"show {name}", echo=False)

    def show_sel(self):
        s = self.selected()
        if s:
            self.main.activate_workfile(self.wf)
            self.main.run_command("show " + " ".join(s))

    def open_as_group(self):
        s = self.selected()
        if not s:
            return
        n, ok = QInputDialog.getText(self, "New group", "Group name:", text=self.wf.unique_name("group"))
        if ok and n:
            self.main.run_command(f"group {n} " + " ".join(s))

    def estimate_sel(self):
        s = [x for x in self.selected() if self.wf.obj_type(x) == "series"]
        spec = " ".join([s[0], "c"] + s[1:]) if s else ""
        self.main.dlg_estimate("ls", spec)

    def delete_sel(self):
        s = self.selected()
        if not s:
            return
        if QMessageBox.question(self, "Delete", "Delete " + ", ".join(x.upper() for x in s) + "?") == QMessageBox.Yes:
            self.main.run_command("delete " + " ".join(s))

    def rename_sel(self):
        s = self.selected()
        if len(s) != 1:
            return
        n, ok = QInputDialog.getText(self, "Rename", f"New name for {s[0].upper()}:", text=s[0])
        if ok and n and n != s[0]:
            self.main.run_command(f"rename {s[0]} {n}")

    def copy_sel(self):
        s = self.selected()
        if len(s) != 1:
            return
        n, ok = QInputDialog.getText(self, "Copy", f"Copy {s[0].upper()} to:", text=s[0] + "_copy")
        if ok and n:
            self.main.run_command(f"copy {s[0]} {n}")

    def context_menu(self, pos):
        s = self.selected()
        m = QMenu(self)
        if s:
            m.addAction("Open", lambda: self.open_item(s[0]) if len(s) == 1 else self.show_sel())
            if len(s) > 1:
                m.addAction("Open as Group", self.show_sel)
                m.addAction("Save as named Group...", self.open_as_group)
            series = [x for x in s if self.wf.obj_type(x) == "series"]
            if series:
                m.addAction("Open as Equation...", self.estimate_sel)
                if len(series) > 1:
                    m.addAction("Open as VAR...", lambda: self.main.dlg_estimate("var", " ".join(series)))
                    m.addAction("Estimate DCC-GARCH...", lambda: self.main.dlg_estimate("dcc", " ".join(series)))
                m.addAction("ARCH/GARCH...", lambda: self.main.dlg_estimate("arch", f"{series[0]} c"))
            m.addSeparator()
            m.addAction("Rename...", self.rename_sel)
            m.addAction("Copy...", self.copy_sel)
            m.addAction("Delete", self.delete_sel)
        else:
            m.addAction("Generate Series...", self.main.dlg_genr)
            m.addAction("Import data...", self.main.dlg_import)
        m.exec(self.list.mapToGlobal(pos))


# ============================================================================ series & group windows
class SeriesWindow(ChildWindow):
    kind = "series"

    def __init__(self, main, wf, name: str):
        super().__init__(main, f"Series: {name.upper()}   Workfile: {wf.name.upper()}")
        self.wf, self.name = wf, name
        v = self.menu_button("View")
        v.addAction("Spreadsheet", self.show_sheet)
        g = v.addMenu("Graph")
        g.addAction("Line", lambda: self.cmd(f"line {name}"))
        g.addAction("Bar", lambda: self.cmd(f"bar {name}"))
        g.addAction("Kernel density", lambda: self.cmd(f"kdensity {name}"))
        g.addAction("Quantile-Quantile", lambda: self.cmd(f"qqplot {name}"))
        g.addAction("Box plot", lambda: self.cmd(f"boxplot {name}"))
        v.addSeparator()
        d = v.addMenu("Descriptive Statistics && Tests")
        d.addAction("Histogram and Stats", lambda: self.cmd(f"hist {name}"))
        d.addAction("Stats table", lambda: self.cmd(f"stats {name}"))
        d.addAction("Normality tests", lambda: self.cmd(f"normtest {name}"))
        d.addAction("Simple hypothesis tests...", lambda: self.dlg("ttest"))
        v.addAction("Correlogram...", lambda: self.dlg("correl"))
        v.addAction("Unit Root Test...", lambda: self.dlg("uroot"))
        v.addAction("Variance Ratio Test...", lambda: self.dlg("vratio"))
        v.addAction("BDS Independence Test...", lambda: self.dlg("bds"))
        v.addAction("Ljung-Box Q...", lambda: self.dlg("lbq"))
        p = self.menu_button("Proc")
        p.addAction("Generate by Equation...", self.main.dlg_genr)
        p.addAction("Hodrick-Prescott Filter...", lambda: self.dlg("hpf"))
        p.addSeparator()
        p.addAction("ARIMA...", lambda: self.main.dlg_estimate("arima", f"{name} c"))
        p.addAction("Automatic ARIMA...", lambda: self.main.dlg_estimate("autoarima", f"{name} c"))
        p.addAction("Exponential Smoothing (ETS)...", lambda: self.main.dlg_estimate("ets", name))
        p.addAction("ARCH/GARCH...", lambda: self.main.dlg_estimate("arch", f"{name} c"))
        p.addAction("LSTM forecast...", lambda: self.main.dlg_estimate("lstm", name))
        p.addSeparator()
        p.addAction("Value-at-Risk...", lambda: self.dlg("risk"))
        p.addAction("VaR Backtest...", lambda: self.dlg("backtest"))
        p.addAction("Performance analytics...", lambda: self.dlg("perf"))
        self.tb.addSeparator()
        self.edit_act = self.button("Edit+/-", self.toggle_edit, "Toggle edit mode")
        self.button("Freeze", self.freeze)
        self.button("Save as...", self.save_output)
        self.button("Sample", self.main.dlg_sample)
        self.show_sheet()

    def cmd(self, c):
        self.main.run_command(c, target=self)

    def dlg(self, key):
        self.main.dlg_test(key, self.name, target=self)

    def show_sheet(self):
        self.current_outputs = []
        self.model = SeriesTableModel(self.wf, [self.name])
        tv = QTableView()
        tv.setModel(self.model)
        tv.setAlternatingRowColors(True)
        tv.horizontalHeader().setDefaultSectionSize(130)
        tv.verticalHeader().setDefaultSectionSize(20)
        self.set_content(tv)

    def toggle_edit(self):
        if hasattr(self, "model") and isinstance(self.content, QTableView):
            self.model.editable = not self.model.editable
            self.main.log(f"Edit mode {'on' if self.model.editable else 'off'} for {self.name.upper()}")

    def refresh(self):
        if isinstance(self.content, QTableView):
            self.model.refresh()


class GroupWindow(ChildWindow):
    kind = "group"

    def __init__(self, main, wf, group: Group, name: str = ""):
        super().__init__(main, f"Group: {(name or 'UNTITLED').upper()}   Workfile: {wf.name.upper()}")
        self.wf, self.group, self.gname = wf, group, name
        mem = " ".join(group.members)
        self.mem = mem
        v = self.menu_button("View")
        v.addAction("Spreadsheet", self.show_sheet)
        g = v.addMenu("Graph")
        g.addAction("Line (single graph)", lambda: self.cmd(f"line {mem}"))
        g.addAction("Line (multiple graphs)", lambda: self.cmd(f"line(m) {mem}"))
        g.addAction("Scatter (first two)", lambda: self.cmd(f"scat {' '.join(group.members[:2])}"))
        g.addAction("Bar", lambda: self.cmd(f"bar {mem}"))
        g.addAction("Box plots", lambda: self.cmd(f"boxplot {mem}"))
        v.addSeparator()
        d = v.addMenu("Descriptive Stats")
        d.addAction("Common sample", lambda: self.cmd(f"stats {mem}"))
        d.addAction("Individual samples", lambda: self.cmd(f"stats(i) {mem}"))
        if wf.is_panel:
            d.addAction("Panel overall / between / within", lambda: self.cmd(f"stats(panel) {mem}"))
        v.addAction("Covariance Analysis...", lambda: self.dlg("cor"))
        v.addAction("Covariance matrix", lambda: self.cmd(f"cov {mem}"))
        v.addAction("Equality Tests by Series", lambda: self.cmd(f"eqtest {mem}"))
        v.addAction("Principal Components...", lambda: self.dlg("pca"))
        v.addSeparator()
        v.addAction("Unit Root Summary...", lambda: self.dlg("urootsum"))
        v.addAction("Cointegration Test...", lambda: self.dlg("coint"))
        v.addAction("Granger Causality...", lambda: self.dlg("granger" if not wf.is_panel else "dhcause"))
        p = self.menu_button("Proc")
        p.addAction("Make Equation...", lambda: self.main.dlg_estimate(
            "ls", " ".join([group.members[0], "c"] + group.members[1:])))
        p.addAction("Make VAR...", lambda: self.main.dlg_estimate("var", mem))
        p.addAction("Make VEC...", lambda: self.main.dlg_estimate("vec", mem))
        p.addAction("DCC-GARCH...", lambda: self.main.dlg_estimate("dcc", mem))
        p.addAction("BEKK-GARCH...", lambda: self.main.dlg_estimate("bekk", mem))
        p.addAction("Factor model (Fama-French)...", lambda: self.main.dlg_estimate("ff", mem))
        p.addAction("Performance analytics...", lambda: self.main.dlg_test("perf", mem, target=self))
        self.tb.addSeparator()
        self.button("Edit+/-", self.toggle_edit)
        self.button("Freeze", self.freeze)
        self.button("Save as...", self.save_output)
        if not name:
            self.button("Name", self.name_group)
        self.show_sheet()

    def cmd(self, c):
        self.main.run_command(c, target=self)

    def dlg(self, key):
        self.main.dlg_test(key, self.mem, target=self)

    def show_sheet(self):
        self.current_outputs = []
        self.model = SeriesTableModel(self.wf, self.group.members)
        tv = QTableView()
        tv.setModel(self.model)
        tv.setAlternatingRowColors(True)
        tv.horizontalHeader().setDefaultSectionSize(110)
        tv.verticalHeader().setDefaultSectionSize(20)
        self.set_content(tv)

    def toggle_edit(self):
        if hasattr(self, "model"):
            self.model.editable = not self.model.editable

    def name_group(self):
        n, ok = QInputDialog.getText(self, "Name group", "Group name:", text=self.wf.unique_name("group"))
        if ok and n:
            self.main.run_command(f"group {n} {self.mem}")

    def refresh(self):
        if isinstance(self.content, QTableView):
            self.model.refresh()


# ============================================================================ estimated objects
class EstimationWindow(ChildWindow):
    kind = "estimation"

    def __init__(self, main, wf, est: Estimation):
        super().__init__(main, "")
        self.wf, self.est = wf, est
        self._title()
        self.view_menu = self.menu_button("View")
        self.proc_menu = self.menu_button("Proc")
        self.tb.addSeparator()
        self.button("Estimate", self.reestimate)
        self.fc_act = self.button("Forecast", lambda: self.run_action("forecast"))
        self.button("Stats", lambda: self.run_action("output"))
        self.resid_act = self.button("Resids", lambda: self.run_action("resids"))
        self.button("Freeze", self.freeze)
        self.button("Save as...", self.save_output)
        self.button("Copy", self.copy_output)
        self.name_act = self.button("Name", self.name_object)
        self.build_menus()
        self.show_outputs([est.report()])

    def _title(self):
        from ..core.commands import TYPE_LABEL
        n = self.est.name.upper() if self.est.name else "UNTITLED"
        self.setWindowTitle(f"{TYPE_LABEL.get(self.est.type, self.est.type.capitalize())}: {n}   "
                            f"Workfile: {self.wf.name.upper()}")

    def build_menus(self):
        self.view_menu.clear()
        self.proc_menu.clear()
        try:
            res = self.est.result
        except Exception as e:  # lazily re-estimated objects
            self.view_menu.addAction(f"(re-estimation failed: {e})")
            return
        subs = {}
        for v in res.views():
            menu = self.view_menu
            if v.group:
                if v.group not in subs:
                    subs[v.group] = self.view_menu.addMenu(v.group)
                menu = subs[v.group]
            menu.addAction(v.label, lambda k=v.key: self.run_action(k))
        for v in res.procs():
            self.proc_menu.addAction(v.label, lambda k=v.key: self.run_action(k))
        acts = res.all_actions()
        self.fc_act.setVisible("forecast" in acts)
        self.resid_act.setVisible("resids" in acts)
        self.name_act.setVisible(not self.est.name)

    def run_action(self, key):
        try:
            acts = self.est.result.all_actions()
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))
            return
        if key not in acts:
            return
        v = acts[key]
        opts = ""
        if v.params:
            from .dialogs import ParamDialog
            dlg = ParamDialog(self, v.label.rstrip("."), v.params)
            if not dlg.exec():
                return
            vals = dlg.form.values()
            parts = []
            for p in v.params:
                val = vals[p.name]
                if isinstance(val, bool):
                    val = 1 if val else 0
                sval = str(val)
                parts.append(f'{p.name}="{sval}"' if (" " in sval or "," in sval or "=" in sval) else f"{p.name}={sval}")
            opts = ", ".join(parts)
        if self.est.name:
            self.main.echo(f"{self.est.name}.{key}" + (f"({opts})" if opts else ""))
        est = self.est
        interp = self.main.interp
        self.main.run_task(lambda: interp.call_view(est, key, opts or None, ""), target=self,
                           label=v.label, refresh_after=(v.kind == "proc"))

    def reestimate(self):
        cmd = self.est.command
        m = re.match(r"^([A-Za-z_]\w*)(\((.*?)\))?\s*(.*)$", cmd, flags=re.S)
        method = m.group(1) if m else "ls"
        spec = m.group(4) if m else ""
        from .dialogs import METHODS
        if method in ("garch", "egarch", "tarch", "gjr", "parch", "cgarch", "igarch", "figarch"):
            method = "arch"
        self.main.dlg_estimate(method if method in METHODS else "ls", spec, self.est.name, options=m.group(3) if m else "")

    def name_object(self):
        n, ok = QInputDialog.getText(self, "Name object", "Name to identify object:",
                                     text=self.wf.unique_name("eq" if self.est.type == "equation" else self.est.type))
        if ok and n:
            try:
                self.est.name = n.lower()
                if hasattr(self.est.result, "name"):
                    self.est.result.name = n.lower()
                self.wf.add(self.est)
                if self.main.interp.last is self.est:
                    self.main.interp.last = None
            except Exception as e:
                QMessageBox.warning(self, "Name", str(e))
                return
            self._title()
            self.build_menus()
            self.main.register_object_window(n.lower(), self)


class OutputWindow(ChildWindow):
    kind = "output"

    def __init__(self, main, outputs: list, title: str = "Output", table_obj=None):
        super().__init__(main, title)
        self.table_obj = table_obj
        self.button("Freeze", self.freeze)
        self.button("Save as...", self.save_output)
        self.button("Copy", self.copy_output)
        self.show_outputs(outputs)


class GraphWindow(ChildWindow):
    kind = "graph"

    def __init__(self, main, fig_output: FigureOutput, title: str = "Graph"):
        super().__init__(main, title)
        self.button("Freeze / Name", self.freeze)
        self.button("Save as...", self.save_output)
        self.show_outputs([fig_output])


# ============================================================================ program window
class ProgramWindow(ChildWindow):
    kind = "program"

    def __init__(self, main, path: str | None = None, text: str = ""):
        super().__init__(main, "Program: " + (os.path.basename(path) if path else "UNTITLED"))
        self.path = path
        self.button("Run", self.run)
        self.button("Run selection", self.run_sel)
        self.button("Save", self.save)
        self.button("Save as...", self.save_as)
        self.edit = ProgramEdit()
        if path and os.path.exists(path):
            with open(path, encoding="utf-8-sig") as f:
                self.edit.setPlainText(f.read())
        else:
            self.edit.setPlainText(text or PROGRAM_TEMPLATE)
        self.set_content(self.edit)
        run_sc = QAction(self)
        run_sc.setShortcut(QKeySequence("F5"))
        run_sc.triggered.connect(self.run)
        self.addAction(run_sc)

    def run(self):
        txt = self.edit.toPlainText()
        cwd = os.path.dirname(self.path) if self.path else None
        self.main.run_program(txt, cwd)

    def run_sel(self):
        cur = self.edit.textCursor()
        txt = cur.selectedText().replace(" ", "\n")
        if txt.strip():
            self.main.run_program(txt, os.path.dirname(self.path) if self.path else None)

    def save(self):
        if not self.path:
            return self.save_as()
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(self.edit.toPlainText())
        self.main.log(f"Program saved: {self.path}")

    def save_as(self):
        p, _ = QFileDialog.getSaveFileName(self, "Save program", "program.fprg", "Finomania programs (*.fprg *.prg)")
        if p:
            self.path = p
            self.setWindowTitle("Program: " + os.path.basename(p))
            self.save()


PROGRAM_TEMPLATE = """' Finomania program  —  press F5 or click Run
' Lines starting with an apostrophe are comments.
'
' wfcreate m 2000m1 2024m12
' series y = nrnd
' equation eq1.ls y c y(-1)
' for !i = 1 to 3
'     series x{!i} = nrnd
' next
"""
