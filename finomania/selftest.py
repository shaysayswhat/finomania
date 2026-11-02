"""Built-in self test:  Finomania --selftest   (used by the build pipeline on the frozen app).

Runs every demo program through the interpreter, exercises the main estimators and
opens the main window off-screen.  Exit code 0 = success."""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
import traceback


def run(gui: bool = True, quick: bool = False) -> int:
    import warnings
    warnings.simplefilter("ignore")
    from . import __version__
    from .appenv import resource_dir
    from .core.commands import Interpreter
    from .report import Message

    t0 = time.time()
    fails = []
    print(f"Finomania {__version__} self-test  (python {sys.version.split()[0]}, frozen={getattr(sys, 'frozen', False)})")
    src = os.path.join(resource_dir(), "examples")
    work = tempfile.mkdtemp(prefix="finomania_selftest_")
    try:
        shutil.copytree(src, os.path.join(work, "examples"))
        ex = os.path.join(work, "examples")
        progs = sorted(f for f in os.listdir(ex) if f.endswith(".fprg"))
        if quick:
            progs = progs[:1]
        for p in progs:
            t = time.time()
            it = Interpreter()
            try:
                out = it.run_file(os.path.join(ex, p))
                errs = [o.text for o in out if isinstance(o, Message) and o.level == "error"]
            except Exception:
                errs = [traceback.format_exc()]
            status = "ok" if not errs else "FAILED"
            print(f"  program {p:<22} {status:>6}  ({time.time() - t:.1f}s)")
            if errs:
                fails.append((p, errs[0]))
        # optional components
        try:
            import sklearn  # noqa: F401
            import statsmodels  # noqa: F401
            import arch  # noqa: F401
            import linearmodels  # noqa: F401
            print("  libraries              ok")
        except Exception as e:
            fails.append(("libraries", str(e)))
        if gui:
            t = time.time()
            try:
                os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
                os.environ.setdefault("QT_API", "pyside6")
                import matplotlib
                try:
                    matplotlib.use("QtAgg")
                except Exception:   # figures are embedded with FigureCanvasQTAgg directly
                    pass
                from PySide6.QtWidgets import QApplication

                from .gui.mainwindow import MainWindow
                from .gui.style import apply_theme
                app = QApplication.instance() or QApplication([])
                apply_theme(app)
                w = MainWindow(show_welcome=False, interactive=False)
                w.show()
                w.run_command(f'wfopen "{os.path.join(ex, "data", "macro_quarterly.csv")}"', echo=False)
                w.run_command("equation eq1.ls cons c income", echo=False)
                w.run_command("eq1.resids", echo=False)
                deadline = time.time() + 120
                while (w.busy > 0 or time.time() - t < 1.5) and time.time() < deadline:
                    app.processEvents()
                    time.sleep(0.02)
                nwin = len(w.mdi.subWindowList())
                err = getattr(w, "last_error", None)
                w.close()
                if nwin < 2 or err:
                    fails.append(("gui", err or f"expected windows, got {nwin}"))
                print(f"  main window            {'ok' if nwin >= 2 else 'FAILED'}  ({time.time() - t:.1f}s)")
            except Exception:
                fails.append(("gui", traceback.format_exc()))
                print("  main window            FAILED")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(f"Self-test finished in {time.time() - t0:.1f}s: {'PASSED' if not fails else 'FAILED'}")
    for name, err in fails:
        print(f"--- {name}:\n{err}")
    return 0 if not fails else 1
