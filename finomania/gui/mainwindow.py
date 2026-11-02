"""Main application window: menus, command window, MDI area, background execution."""
from __future__ import annotations

import os
import traceback

from PySide6.QtCore import QObject, QRunnable, QSettings, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QApplication, QDockWidget, QFileDialog, QInputDialog, QLabel, QMainWindow, QMdiArea,
                               QMdiSubWindow, QMessageBox, QPlainTextEdit, QProgressBar)

from .. import __version__
from ..appenv import data_dir, docs_path, examples_dir, is_frozen, log_path, resource_dir, user_dir
from ..core.commands import Interpreter, ObjectOutput
from ..core.objects import Estimation, Graph, Group, Table
from ..models.base import parse_options
from ..report import FigureOutput, Message, Report
from . import dialogs as D
from .style import app_icon
from .widgets import CommandEdit
from .windows import (EstimationWindow, GraphWindow, GroupWindow, OutputWindow, ProgramWindow, SeriesWindow,
                      WorkfileWindow)

try:
    from shiboken6 import isValid as _qt_valid
except Exception:  # pragma: no cover
    def _qt_valid(o):
        return o is not None

PROJECT_URL = "https://github.com/shaysayswhat/finomania"


class _Signals(QObject):
    done = Signal(object, object, object)
    progress = Signal(str, float)


class _Task(QRunnable):
    def __init__(self, fn, target, meta, sig):
        super().__init__()
        self.fn, self.target, self.meta, self.sig = fn, target, meta, sig

    def run(self):
        try:
            out = self.fn()
        except Exception as e:  # noqa: BLE001
            from ..core.commands import _err_text
            out = [Message(_err_text(e), "error")]
            traceback.print_exc()
        self.sig.done.emit(out, self.target, self.meta)


class MainWindow(QMainWindow):
    def __init__(self, show_welcome: bool = True, interactive: bool = True):
        super().__init__()
        self.interactive = interactive      # False in the self test: never block on modal dialogs
        self.setWindowTitle(f"Finomania {__version__} — open-source econometrics")
        self.setWindowIcon(app_icon())
        self.resize(1360, 860)
        self.sig = _Signals()
        self.sig.done.connect(self._done)
        self.sig.progress.connect(self._progress)
        self.interp = Interpreter(progress=lambda msg, frac: self.sig.progress.emit(msg, float(frac)))
        self.pool = QThreadPool()
        self.pool.setMaxThreadCount(1)
        self.busy = 0
        self.wf_windows: dict[int, WorkfileWindow] = {}
        self.obj_windows: dict[tuple, object] = {}
        self.settings = QSettings("Finomania", "Finomania")
        start_dir = self.settings.value("default_dir", "") or (user_dir() if is_frozen() else os.getcwd())
        if os.path.isdir(start_dir):
            self.interp.cwd = start_dir
        # central MDI
        self.mdi = QMdiArea()
        self.mdi.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.mdi.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.mdi.subWindowActivated.connect(self._sub_activated)
        self.setCentralWidget(self.mdi)
        # command window (top, like EViews)
        self.cmdedit = CommandEdit()
        self.cmdedit.command.connect(lambda c: self.run_command(c, echo=False))
        dock = QDockWidget("Command", self)
        dock.setObjectName("commanddock")
        dock.setWidget(self.cmdedit)
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.TopDockWidgetArea, dock)
        self.cmd_dock = dock
        self.cmdedit.setMinimumHeight(70)
        # log / capture window (bottom)
        self.logw = QPlainTextEdit()
        self.logw.setObjectName("log")
        self.logw.setReadOnly(True)
        ld = QDockWidget("Messages", self)
        ld.setObjectName("logdock")
        ld.setWidget(self.logw)
        self.addDockWidget(Qt.BottomDockWidgetArea, ld)
        self.log_dock = ld
        self.resizeDocks([dock, ld], [110, 90], Qt.Vertical)
        # status bar
        self.st_path = QLabel()
        self.st_wf = QLabel()
        self.st_smpl = QLabel()
        self.st_busy = QProgressBar()
        self.st_busy.setMaximumWidth(220)
        self.st_busy.setMaximumHeight(14)
        self.st_busy.setTextVisible(False)
        self.st_busy.hide()
        self.st_msg = QLabel()
        sb = self.statusBar()
        sb.addWidget(self.st_msg, 1)
        sb.addPermanentWidget(self.st_busy)
        sb.addPermanentWidget(self.st_path)
        sb.addPermanentWidget(self.st_wf)
        sb.addPermanentWidget(self.st_smpl)
        self._menus()
        self.setAcceptDrops(True)
        self.update_status()
        geo = self.settings.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        st = self.settings.value("windowState")
        if st is not None:
            self.restoreState(st)
        if show_welcome and str(self.settings.value("show_welcome", "true")).lower() == "true":
            QTimer.singleShot(0, self._welcome)

    # ===================================================================== menus
    def _menus(self):
        mb = self.menuBar()
        f = mb.addMenu("&File")
        new = f.addMenu("New")
        new.addAction("Workfile...", self.dlg_new_workfile, QKeySequence("Ctrl+N"))
        new.addAction("Program", lambda: self.add_window(ProgramWindow(self)))
        op = f.addMenu("Open")
        op.addAction("Workfile or data file...", self.dlg_open, QKeySequence("Ctrl+O"))
        op.addAction("Program...", self.dlg_open_program)
        self.recent_menu = f.addMenu("Recent files")
        self.recent_menu.aboutToShow.connect(self._fill_recent)
        f.addAction("Save workfile", self.save_workfile, QKeySequence("Ctrl+S"))
        f.addAction("Save workfile as...", lambda: self.save_workfile(True))
        f.addSeparator()
        imp = f.addMenu("Import")
        imp.addAction("Import data file (CSV, Excel, Stata, Parquet)...", self.dlg_import)
        imp.addAction("Fetch from Yahoo Finance / FRED...", self.dlg_fetch)
        imp.addAction("Fama-French factors...", lambda: self.dlg_fetch("ff"))
        f.addAction("Export series...", self.dlg_export)
        f.addSeparator()
        f.addAction("Run program...", self.dlg_run_program)
        ex = f.addMenu("Demo programs")
        ex.aboutToShow.connect(lambda m=ex: self._fill_demos(m))
        f.addSeparator()
        f.addAction("Exit", self.close)

        e = mb.addMenu("&Edit")
        e.addAction("Copy output (text)", self._copy_current)
        e.addAction("Clear command window", self.cmdedit.clear)
        e.addAction("Clear messages", self.logw.clear)
        e.addSeparator()
        e.addAction("Preferences...", self.dlg_preferences)

        o = mb.addMenu("&Object")
        nw = o.addMenu("New Object")
        nw.addAction("Series (generate)...", self.dlg_genr)
        nw.addAction("Group...", self.dlg_group)
        nw.addAction("Equation...", lambda: self.dlg_estimate("ls"))
        nw.addAction("Program", lambda: self.add_window(ProgramWindow(self)))
        o.addAction("Freeze current output", self._freeze_current)

        q = mb.addMenu("&Quick")
        q.addAction("Sample...", self.dlg_sample)
        q.addAction("Generate Series...", self.dlg_genr)
        q.addAction("Show...", self.dlg_show)
        q.addAction("Graph...", self.dlg_graph)
        s = q.addMenu("Series Statistics")
        for lab, key in (("Histogram and Stats", "hist"), ("Correlogram", "correl"), ("Unit Root Test", "uroot"),
                         ("Normality tests", "normtest"), ("BDS test", "bds"), ("Variance ratio", "vratio")):
            s.addAction(lab + "...", lambda k=key: self._ask_series_then(k))
        g = q.addMenu("Group Statistics")
        for lab, key in (("Descriptive Statistics", "stats"), ("Correlations", "cor"), ("Covariances", "cov"),
                         ("Granger Causality", "granger"), ("Cointegration Test", "coint"),
                         ("Principal Components", "pca"), ("Unit Root Summary", "urootsum")):
            g.addAction(lab + "...", lambda k=key: self.dlg_test(k, ""))
        q.addSeparator()
        q.addAction("Estimate Equation...", lambda: self.dlg_estimate("ls"), QKeySequence("Ctrl+E"))
        q.addAction("Estimate VAR...", lambda: self.dlg_estimate("var"))

        est = mb.addMenu("E&stimate")
        groups: dict[str, list] = {}
        for k, m in D.METHODS.items():
            groups.setdefault(m["group"], []).append(k)
        for gname, keys in groups.items():
            sm = est.addMenu(gname)
            for k in keys:
                sm.addAction(D.METHODS[k]["label"] + "...", lambda k=k: self.dlg_estimate(k))

        t = mb.addMenu("&Tests")
        for lab, key in (("Unit root test", "uroot"), ("Unit root summary (ADF/PP/KPSS)", "urootsum"),
                         ("Cointegration (Johansen / Engle-Granger)", "coint"), ("Granger causality", "granger"),
                         ("Dumitrescu-Hurlin panel causality", "dhcause"), (None, None),
                         ("Normality", "normtest"), ("BDS independence", "bds"), ("Variance ratio", "vratio"),
                         ("Ljung-Box Q", "lbq"), ("Mean = mu", "ttest"), ("Equality of means/variances", "eqtest"),
                         (None, None), ("Value-at-Risk / ES", "risk"), ("VaR backtest", "backtest"),
                         ("Performance analytics", "perf")):
            if lab is None:
                t.addSeparator()
            else:
                t.addAction(lab + "...", lambda k=key: self.dlg_test(k, ""))

        gr = mb.addMenu("&Graph")
        for lab, c in (("Line", "line"), ("Line (multiple)", "line(m)"), ("Scatter", "scat"), ("Bar", "bar"),
                       ("Histogram", "hist"), ("Kernel density", "kdensity"), ("Q-Q plot", "qqplot"),
                       ("Box plot", "boxplot")):
            gr.addAction(lab + "...", lambda c=c: self.dlg_graph(c))

        w = mb.addMenu("&Window")
        w.addAction("Tile", self.mdi.tileSubWindows)
        w.addAction("Cascade", self.mdi.cascadeSubWindows)
        w.addAction("Close all", self.mdi.closeAllSubWindows)
        w.addSeparator()
        w.addAction(self.cmd_dock.toggleViewAction())
        w.addAction(self.log_dock.toggleViewAction())

        h = mb.addMenu("&Help")
        h.addAction("Quick start", self._welcome)
        h.addAction("User guide", lambda: self._show_doc("README.md", "User guide"), QKeySequence("F1"))
        h.addAction("Command reference", lambda: self._show_doc("COMMANDS.md", "Command reference"))
        h.addAction("Command list (help)", lambda: self.run_command("help", echo=False))
        h.addSeparator()
        h.addAction("Open my Finomania folder", lambda: self._open_folder(self.interp.cwd))
        h.addAction("Open log folder", lambda: self._open_folder(os.path.dirname(log_path())))
        h.addAction("Project website", lambda: self._open_url(PROJECT_URL))
        h.addSeparator()
        h.addAction("About Finomania", self._about)

    # ===================================================================== execution
    def echo(self, cmd: str):
        self.cmdedit.append_command(cmd)

    def run_command(self, cmd: str, target=None, echo: bool = True, sample: str | None = None):
        if echo:
            self.echo(cmd)
        text = cmd
        wf = self.interp.wf
        if sample and wf is not None and sample.strip() and sample.strip() != wf.sample_text:
            text = f"smpl {sample}\n{cmd}\nsmpl {wf.sample_text}"
        self.log("> " + cmd, "cmd")
        self.run_task(lambda: self.interp.run(text), target, label=cmd)

    def run_program(self, text: str, cwd: str | None = None):
        if cwd:
            self.interp.cwd = cwd
        self.log("> [run program]", "cmd")
        self.run_task(lambda: self.interp.run(text), None, label="program")

    def run_task(self, fn, target=None, label: str = "", refresh_after: bool = False):
        self.busy += 1
        self.st_busy.show()
        self.st_busy.setRange(0, 0)
        self.st_msg.setText(f"Running: {label[:80]}")
        QApplication.setOverrideCursor(Qt.BusyCursor)
        self.pool.start(_Task(fn, target, {"label": label, "refresh": refresh_after}, self.sig))

    def _progress(self, msg, frac):
        self.st_msg.setText(msg)
        if frac:
            self.st_busy.setRange(0, 100)
            self.st_busy.setValue(int(100 * frac))

    def _done(self, outputs, target, meta):
        self.busy -= 1
        QApplication.restoreOverrideCursor()
        if self.busy <= 0:
            self.busy = 0
            self.st_busy.hide()
            self.st_msg.setText("Ready")
        if target is not None and not _qt_valid(target):
            target = None
        to_target = []
        first_error = None
        for o in outputs:
            if isinstance(o, Message):
                self.log(o.text, o.level)
                if o.level == "error" and first_error is None:
                    first_error = o.text
            elif isinstance(o, (Report, FigureOutput)):
                if target is not None:
                    to_target.append(o)
                else:
                    self._open_output(o)
            elif isinstance(o, ObjectOutput):
                self._open_object(o)
        if to_target:
            target.show_outputs(to_target)
            sub = self._sub_of(target)
            if sub:
                self.mdi.setActiveSubWindow(sub)
        for w in self.wf_windows.values():
            w.refresh()
        for (wid, name), w in list(self.obj_windows.items()):
            if _qt_valid(w) and hasattr(w, "refresh"):
                try:
                    w.refresh()
                except Exception:
                    pass
        self.update_status()
        if first_error and self.interactive:
            QMessageBox.warning(self, "Finomania", first_error)
        if first_error:
            self.last_error = first_error

    # ===================================================================== windows
    def add_window(self, w, size=None):
        sub = QMdiSubWindow()
        sub.setWidget(w)
        sub.setAttribute(Qt.WA_DeleteOnClose)
        sub.setWindowIcon(app_icon())
        self.mdi.addSubWindow(sub)
        n = len(self.mdi.subWindowList())
        sub.resize(*(size or (760, 520)))
        sub.move(20 + 24 * ((n - 1) % 8), 10 + 24 * ((n - 1) % 8))
        sub.show()
        self.mdi.setActiveSubWindow(sub)
        return sub

    def _sub_of(self, w):
        for s in self.mdi.subWindowList():
            if s.widget() is w:
                return s
        return None

    def _sub_activated(self, sub):
        if sub is None:
            return
        w = sub.widget()
        wf = getattr(w, "wf", None)
        if wf is not None and self.interp.wf is not wf and wf in self.interp.workfiles.values():
            self.interp.wf = wf
            self.update_status()

    def activate_workfile(self, wf):
        if wf in self.interp.workfiles.values():
            self.interp.wf = wf

    def register_object_window(self, name, w):
        wf = getattr(w, "wf", None)
        self.obj_windows[(id(wf), name)] = w

    def _focus_existing(self, key):
        w = self.obj_windows.get(key)
        if w is not None and _qt_valid(w):
            sub = self._sub_of(w)
            if sub:
                self.mdi.setActiveSubWindow(sub)
                return w
        self.obj_windows.pop(key, None)
        return None

    def _open_object(self, o: ObjectOutput):
        wf = self.interp.wf
        if o.kind == "workfile":
            wf = o.obj
            ww = self.wf_windows.get(id(wf))
            if ww is None or not _qt_valid(ww):
                ww = WorkfileWindow(self, wf)
                self.wf_windows[id(wf)] = ww
                self.add_window(ww, (430, 470))
            else:
                ww.refresh()
                self.mdi.setActiveSubWindow(self._sub_of(ww))
            return
        key = (id(wf), o.name) if o.name else None
        if o.kind == "series":
            if key and self._focus_existing(key):
                return
            w = SeriesWindow(self, wf, o.name)
            self.add_window(w, (420, 520))
        elif o.kind == "group":
            if key and self._focus_existing(key):
                return
            w = GroupWindow(self, wf, o.obj, o.name)
            self.add_window(w, (640, 520))
        elif o.kind == "estimation":
            if key:
                ex = self.obj_windows.get(key)
                if ex is not None and _qt_valid(ex) and isinstance(ex, EstimationWindow):
                    ex.est = o.obj
                    ex.build_menus()
                    ex._title()
                    ex.show_outputs([o.obj.report()])
                    self.mdi.setActiveSubWindow(self._sub_of(ex))
                    return
            w = EstimationWindow(self, wf, o.obj)
            self.add_window(w, (780, 600))
        elif o.kind == "table":
            if key and self._focus_existing(key):
                return
            w = OutputWindow(self, [o.obj.report()], o.title, o.obj)
            self.add_window(w)
        elif o.kind == "graph":
            if key and self._focus_existing(key):
                return
            w = GraphWindow(self, FigureOutput(o.obj.figure, o.title), o.title)
            self.add_window(w, (760, 520))
        elif o.kind in ("matrix", "vector"):
            w = OutputWindow(self, [o.obj.report()], o.title)
            self.add_window(w)
        elif o.kind == "scalar":
            self.log(f"{o.name.upper()} = {o.obj.value:.10g}")
            return
        else:
            return
        if key:
            self.obj_windows[key] = w

    def _open_output(self, o):
        if isinstance(o, FigureOutput):
            self.add_window(GraphWindow(self, o, o.title or "Graph"), (760, 520))
        else:
            self.add_window(OutputWindow(self, [o], o.title or "Output"), (780, 560))

    # ===================================================================== logging / status
    def log(self, text: str, level: str = "info"):
        color = {"error": "#b00020", "warning": "#a15c00", "cmd": "#0b4ea2"}.get(level, "#333333")
        import html
        self.logw.appendHtml(f'<span style="color:{color}">{html.escape(text).replace(chr(10), "<br>")}</span>')
        if level != "cmd":
            self.st_msg.setText(text.split("\n")[0][:120])

    def update_status(self):
        wf = self.interp.wf
        self.st_path.setText(f"Path = {self.interp.cwd}")
        if wf is None:
            self.st_wf.setText("WF = none")
            self.st_smpl.setText("")
        else:
            self.st_wf.setText(f"WF = {wf.name}")
            self.st_smpl.setText(f"Sample = {wf.sample_text}")

    # ===================================================================== dialogs
    def _need_wf(self):
        if self.interp.wf is None:
            QMessageBox.information(self, "Finomania", "Open or create a workfile first (File > New / Open / Import).")
            return False
        return True

    def dlg_new_workfile(self):
        d = D.NewWorkfileDialog(self)
        if d.exec():
            self.run_command(d.command())

    def dlg_open(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open", self.interp.cwd,
                                           "Workfiles and data (*.fwf *.csv *.xlsx *.xls *.dta *.parquet *.txt);;"
                                           "All files (*)")
        if p:
            self.open_path(p)

    def open_path(self, p):
        self.add_recent(p)
        self.interp.cwd = os.path.dirname(p)
        if p.lower().endswith((".fprg", ".prg")):
            self.add_window(ProgramWindow(self, p))
        else:
            self.run_command(f'wfopen "{p}"')

    def dlg_open_program(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open program", self.interp.cwd, "Programs (*.fprg *.prg);;All (*)")
        if p:
            self.add_window(ProgramWindow(self, p))

    def dlg_run_program(self):
        p, _ = QFileDialog.getOpenFileName(self, "Run program", self.interp.cwd, "Programs (*.fprg *.prg);;All (*)")
        if p:
            self.echo(f'run "{p}"')
            self.run_task(lambda: self.interp.run_file(p), None, label=os.path.basename(p))

    def dlg_import(self):
        p, _ = QFileDialog.getOpenFileName(self, "Import data", self.interp.cwd,
                                           "Data files (*.csv *.txt *.xlsx *.xls *.dta *.parquet *.json);;All (*)")
        if not p:
            return
        self.interp.cwd = os.path.dirname(p)
        self.add_recent(p)
        opts = []
        if self.interp.wf is not None and self.interp.wf.series_names():
            r = QMessageBox.question(self, "Import", "Merge into the active workfile?\n\n"
                                     "Yes = add series to the active workfile\nNo = open as a new workfile",
                                     QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            if r == QMessageBox.Cancel:
                return
            if r == QMessageBox.No:
                self.run_command(f'wfopen "{p}"')
                return
        self.run_command(f'import{"(" + ", ".join(opts) + ")" if opts else ""} "{p}"')

    def dlg_fetch(self, source="yahoo"):
        d = D.FetchDialog(self, source)
        if d.exec():
            self.run_command(d.command())

    def dlg_export(self):
        if not self._need_wf():
            return
        p, _ = QFileDialog.getSaveFileName(self, "Export series", os.path.join(self.interp.cwd, "export.csv"),
                                           "CSV (*.csv);;Excel (*.xlsx);;Stata (*.dta)")
        if p:
            self.run_command(f'export "{p}"')

    def save_workfile(self, save_as=False):
        if not self._need_wf():
            return
        wf = self.interp.wf
        if wf.path and not save_as:
            self.run_command(f'wfsave "{wf.path}"')
            return
        p, _ = QFileDialog.getSaveFileName(self, "Save workfile", os.path.join(self.interp.cwd, wf.name + ".fwf"),
                                           "Finomania workfile (*.fwf)")
        if p:
            self.add_recent(p)
            self.run_command(f'wfsave "{p}"')

    def dlg_genr(self):
        if not self._need_wf():
            return
        d = D.GenrDialog(self, self.interp.wf.sample_text)
        if d.exec():
            cmd, smpl = d.command()
            self.run_command(cmd, sample=smpl)

    def dlg_sample(self):
        if not self._need_wf():
            return
        d = D.SampleDialog(self, self.interp.wf.sample_text)
        if d.exec():
            self.run_command(d.command())

    def dlg_expand(self):
        if not self._need_wf():
            return
        t, ok = QInputDialog.getText(self, "Expand range", "New end date / last observation:")
        if ok and t.strip():
            self.run_command(f"expand {t.strip()}")

    def dlg_show(self):
        if not self._need_wf():
            return
        t, ok = QInputDialog.getText(self, "Show", "Objects to show (series, groups, expressions):")
        if ok and t.strip():
            self.run_command(f"show {t.strip()}")

    def dlg_group(self):
        if not self._need_wf():
            return
        t, ok = QInputDialog.getText(self, "New group", "Group name followed by series, e.g.  g1 x y z")
        if ok and t.strip():
            self.run_command(f"group {t.strip()}")

    def dlg_graph(self, kind="line"):
        if not self._need_wf():
            return
        t, ok = QInputDialog.getText(self, "Graph", f"Series for {kind} graph:")
        if ok and t.strip():
            self.run_command(f"{kind} {t.strip()}")

    def _ask_series_then(self, key):
        self.dlg_test(key, "")

    def dlg_test(self, key, series="", target=None):
        if not self._need_wf():
            return
        if key in ("hist", "normtest") and series:
            self.run_command(f"{key} {series}", target=target)
            return
        if key == "hist":
            t, ok = QInputDialog.getText(self, "Histogram", "Series:")
            if ok and t.strip():
                self.run_command(f"hist {t.strip()}", target=target)
            return
        d = D.test_dialog(self, key, series)
        if d.exec():
            self.run_command(d.command(), target=target)

    def dlg_estimate(self, method="ls", spec="", name="", options=""):
        if not self._need_wf():
            return
        d = D.EstimationDialog(self, method, spec, name, self.interp.wf.sample_text)
        if options:
            pos, kw = parse_options(options)
            if method == "arch":
                ints = [p for p in pos if p.isdigit()]
                if ints:
                    kw.setdefault("p", ints[0])
                if len(ints) > 1:
                    kw.setdefault("q", ints[1])
                for p in pos:
                    if p in ("tarch", "egarch", "parch", "cgarch", "igarch", "figarch", "harch", "ewma"):
                        kw.setdefault("model", p)
            d.set_options(kw)
        if d.exec():
            cmd, smpl = d.command()
            if not d.spec.toPlainText().strip():
                QMessageBox.warning(self, "Estimation", "Please enter a specification.")
                return
            self.run_command(cmd, sample=smpl)

    # ===================================================================== freeze / misc
    def freeze_outputs(self, outputs):
        if not self._need_wf():
            return
        wf = self.interp.wf
        o = outputs[0]
        is_fig = isinstance(o, FigureOutput)
        n, ok = QInputDialog.getText(self, "Freeze", "Name for the frozen object:",
                                     text=wf.unique_name("graph" if is_fig else "table"))
        if not (ok and n):
            return
        try:
            if is_fig:
                wf.add(Graph(n, o.figure, {"title": o.title}))
                self._open_object(ObjectOutput(n.lower(), wf.objects[n.lower()], "graph", f"Graph: {n.upper()}"))
            else:
                wf.add(Table(n, o))
                self._open_object(ObjectOutput(n.lower(), wf.objects[n.lower()], "table", f"Table: {n.upper()}"))
        except Exception as e:
            QMessageBox.warning(self, "Freeze", str(e))

    def _current_child(self):
        s = self.mdi.activeSubWindow()
        return s.widget() if s else None

    def _freeze_current(self):
        w = self._current_child()
        if w is not None and getattr(w, "current_outputs", None):
            self.freeze_outputs(w.current_outputs)

    def _copy_current(self):
        w = self._current_child()
        if w is not None and hasattr(w, "copy_output"):
            w.copy_output()

    def _welcome(self):
        r = Report("Welcome to Finomania")
        r.text("An open-source econometrics workbench for time-series and panel financial data, in the spirit of EViews.")
        r.section("Getting started")
        r.table(["Do this", "How"], [
            ["Create a workfile", "File ▸ New ▸ Workfile   or type:  wfcreate m 2000m1 2024m12"],
            ["Load your data", "File ▸ Import (CSV / Excel / Stata)  — dates and panel ids are detected automatically"],
            ["Download data", "File ▸ Import ▸ Fetch from Yahoo Finance / FRED  or  Fama-French factors"],
            ["Make new series", "Quick ▸ Generate Series   or type:  series r = 100*dlog(price)"],
            ["Estimate a model", "Estimate menu (LS, ARCH/GARCH, DCC, VAR, panel, GMM, Fama-French, LSTM...)"],
            ["Inspect results", "Every estimated object has View and Proc menus: tests, graphs, forecasts"],
            ["Reproduce", "Every dialog writes its command into the Command window; save them as a program (.fprg)"],
            ["Try the demos", "File ▸ Demo programs ▸ 01_volatility.fprg ... then press Run (F5)"],
        ], ["str", "str"], align=["l", "l"])
        r.section("Command window")
        r.text("Type a command and press Enter. Ctrl+Up/Down recalls history. Type  help  for the full command list.")
        r.text("wfcreate m 1990m1 2020m12\nseries y = nrnd\nequation eq1.ls y c y(-1)\neq1.auto(4)", "mono")
        self.add_window(OutputWindow(self, [r], "Quick start"), (820, 560))

    def _about(self):
        from ..models.ml import torch_available
        QMessageBox.about(self, "About Finomania",
                          f"<b>Finomania {__version__}</b><br>Open-source econometrics for financial time series and "
                          "panel data.<br><br>Built on NumPy, pandas, SciPy, statsmodels, arch, linearmodels, "
                          "scikit-learn, matplotlib and Qt (PySide6).<br>"
                          f"PyTorch backend: {'available' if torch_available() else 'not installed (NumPy LSTM in use)'}"
                          "<br><br>License: MIT")

    # ===================================================================== drag & drop / close
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for u in e.mimeData().urls():
            p = u.toLocalFile()
            if p:
                self.open_path(p)

    def closeEvent(self, e):
        dirty = [wf for wf in self.interp.workfiles.values() if wf.dirty]
        if dirty and self.interactive:
            names = ", ".join(w.name.upper() for w in dirty)
            r = QMessageBox.question(self, "Finomania", f"Workfile(s) {names} have unsaved changes. Quit anyway?",
                                     QMessageBox.Yes | QMessageBox.No)
            if r != QMessageBox.Yes:
                e.ignore()
                return
        if self.interactive:
            self.settings.setValue("geometry", self.saveGeometry())
            self.settings.setValue("windowState", self.saveState())
        self.pool.waitForDone(2000)
        e.accept()

    # ===================================================================== recent files, docs, preferences
    def add_recent(self, path: str):
        path = os.path.abspath(path)
        rec = [p for p in (self.settings.value("recent", []) or []) if p != path]
        if isinstance(rec, str):
            rec = [rec]
        rec.insert(0, path)
        self.settings.setValue("recent", rec[:10])

    def _fill_recent(self):
        m = self.recent_menu
        m.clear()
        rec = self.settings.value("recent", []) or []
        if isinstance(rec, str):
            rec = [rec]
        rec = [p for p in rec if os.path.exists(p)]
        if not rec:
            a = m.addAction("(none)")
            a.setEnabled(False)
            return
        for i, p in enumerate(rec):
            m.addAction(f"&{i + 1}  {os.path.basename(p)}   —   {os.path.dirname(p)}", lambda p=p: self.open_path(p))
        m.addSeparator()
        m.addAction("Clear list", lambda: self.settings.setValue("recent", []))

    def _fill_demos(self, m):
        m.clear()
        d = examples_dir()
        files = sorted(f for f in os.listdir(d) if f.endswith(".fprg")) if os.path.isdir(d) else []
        if not files:
            a = m.addAction("(no demo programs found)")
            a.setEnabled(False)
        for fn in files:
            m.addAction(fn, lambda p=os.path.join(d, fn): self.add_window(ProgramWindow(self, p)))
        m.addSeparator()
        m.addAction("Open demo data folder", lambda: self._open_folder(os.path.join(d, "data")))

    def _show_doc(self, name: str, title: str):
        p = docs_path(name) or (os.path.join(resource_dir(), name) if os.path.exists(os.path.join(resource_dir(), name))
                                else None)
        if p is None:
            QMessageBox.information(self, title, f"{name} was not found in this installation.")
            return
        from PySide6.QtWidgets import QTextBrowser
        from .windows import ChildWindow
        w = ChildWindow(self, title)
        tb = QTextBrowser()
        tb.setOpenExternalLinks(True)
        with open(p, encoding="utf-8") as f:
            txt = f.read()
        tb.document().setDefaultStyleSheet("table{border-collapse:collapse} td,th{padding:3px 8px;border:1px solid #ddd}")
        tb.setSearchPaths([os.path.dirname(p)])
        tb.setMarkdown(txt)
        tb.document().setDocumentMargin(16)
        w.set_content(tb)
        self.add_window(w, (900, 640))

    def _open_folder(self, path):
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        os.makedirs(path, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def _open_url(self, url):
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        QDesktopServices.openUrl(QUrl(url))

    def dlg_preferences(self):
        from PySide6.QtWidgets import QCheckBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QLineEdit, \
            QPushButton, QVBoxLayout, QWidget
        d = QDialog(self)
        d.setWindowTitle("Preferences")
        lay = QVBoxLayout(d)
        fl = QFormLayout()
        dir_edit = QLineEdit(self.settings.value("default_dir", "") or self.interp.cwd)
        browse = QPushButton("Browse...")
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(dir_edit)
        rl.addWidget(browse)
        browse.clicked.connect(lambda: dir_edit.setText(QFileDialog.getExistingDirectory(d, "Default folder",
                                                                                         dir_edit.text()) or
                                                        dir_edit.text()))
        welcome = QCheckBox("Show the quick-start page at startup")
        welcome.setChecked(str(self.settings.value("show_welcome", "true")).lower() == "true")
        fl.addRow("Default data / program folder:", row)
        fl.addRow("", welcome)
        lay.addLayout(fl)
        from ..models.ml import torch_available
        info = QLabel(f"Application data: {data_dir()}\nPyTorch backend: "
                      f"{'available' if torch_available() else 'not installed (built-in NumPy LSTM is used)'}")
        info.setStyleSheet("color:#555")
        lay.addWidget(info)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(d.accept)
        bb.rejected.connect(d.reject)
        lay.addWidget(bb)
        if d.exec():
            p = dir_edit.text().strip()
            if p and os.path.isdir(p):
                self.settings.setValue("default_dir", p)
                self.interp.cwd = p
            self.settings.setValue("show_welcome", "true" if welcome.isChecked() else "false")
            self.update_status()
