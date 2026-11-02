"""Console front-end: interactive command prompt and batch program runner."""
from __future__ import annotations

import base64
import datetime as _dt
import html
import os
import sys

from .core.commands import Interpreter, ObjectOutput
from .report import FigureOutput, Message, Report, fig_to_b64


class ConsoleRenderer:
    def __init__(self, outdir: str | None = None, html_path: str | None = None, quiet: bool = False):
        self.outdir = outdir or os.path.join(os.getcwd(), "finomania_output")
        self.html_path = html_path
        self.quiet = quiet
        self.parts: list[str] = []
        self.nfig = 0

    def emit(self, outs, interp: Interpreter):
        for o in outs:
            if isinstance(o, Message):
                txt = ("ERROR: " if o.level == "error" else "") + o.text
                print(txt, file=sys.stderr if o.level == "error" else sys.stdout)
                self.parts.append(f'<p class="{"err" if o.level == "error" else "msg"}">{html.escape(txt)}</p>')
            elif isinstance(o, Report):
                self._report(o)
            elif isinstance(o, FigureOutput):
                self._figure(o)
            elif isinstance(o, ObjectOutput):
                if o.kind == "estimation":
                    self._report(o.obj.report())
                elif o.kind in ("table",):
                    self._report(o.obj.report())
                elif o.kind == "graph":
                    self._figure(FigureOutput(o.obj.figure, o.title))
                elif o.kind == "series":
                    wf = interp.wf
                    s = wf.series(o.name)
                    r = Report(o.title)
                    labs = wf.labels()
                    rows = [[labs[i], s.iloc[i]] for i in range(min(len(s), 15))]
                    r.table(["obs", o.name.upper()], rows)
                    if len(s) > 15:
                        r.note(f"... {len(s) - 15} more observations")
                    self._report(r)
                elif o.kind == "group":
                    wf = interp.wf
                    labs = wf.labels()
                    r = Report(o.title)
                    rows = [[labs[i]] + [wf.series(m).iloc[i] for m in o.obj.members] for i in range(min(wf.nobs, 15))]
                    r.table(["obs"] + [m.upper() for m in o.obj.members], rows)
                    self._report(r)
                elif o.kind == "workfile":
                    print(o.obj.describe())
                elif o.kind in ("matrix", "vector"):
                    self._report(o.obj.report())
                elif o.kind == "scalar":
                    print(f"{o.name.upper()} = {o.obj.value:.10g}")

    def _report(self, r: Report):
        if not self.quiet:
            print(r.to_text())
        self.parts.append(r.to_html(standalone=False))
        for b in r.blocks:
            if b.__class__.__name__ == "FigureBlock":
                self._save_png(b.figure, r.title)

    def _figure(self, f: FigureOutput):
        p = self._save_png(f.figure, f.title)
        if not self.quiet:
            print(f"[graph saved: {p}]")
        self.parts.append(f'<h3>{html.escape(f.title)}</h3><img style="max-width:100%" '
                          f'src="data:image/png;base64,{fig_to_b64(f.figure)}">')

    def _save_png(self, fig, title):
        os.makedirs(self.outdir, exist_ok=True)
        self.nfig += 1
        safe = "".join(ch if ch.isalnum() else "_" for ch in (title or "graph"))[:40]
        p = os.path.join(self.outdir, f"{self.nfig:03d}_{safe}.png")
        fig.savefig(p, dpi=120, bbox_inches="tight")
        return p

    def write_html(self, title="Finomania output"):
        if not self.html_path:
            return None
        doc = (f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
               "<style>body{max-width:1100px;margin:24px auto;font-family:Segoe UI,Arial,sans-serif}"
               ".err{color:#b00}.msg{color:#555}</style></head><body>"
               f"<h1>{html.escape(title)}</h1><p class='msg'>Generated {_dt.datetime.now():%Y-%m-%d %H:%M}</p>"
               + "\n".join(self.parts) + "</body></html>")
        with open(self.html_path, "w", encoding="utf-8") as f:
            f.write(doc)
        return self.html_path


def _utf8_console():
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def repl():
    _utf8_console()
    it = Interpreter()
    r = ConsoleRenderer()
    print("Finomania command prompt — type 'help' for commands, 'exit' to quit.")
    try:
        import readline  # noqa: F401
    except ImportError:
        pass
    while True:
        try:
            line = input("fino> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line.strip().lower() in ("exit", "quit"):
            break
        r.emit(it.run(line, stop_on_error=False), it)


def run_program(path: str, html_out: str | None = None, quiet: bool = False) -> int:
    _utf8_console()
    it = Interpreter()
    r = ConsoleRenderer(html_path=html_out, quiet=quiet,
                        outdir=os.path.join(os.path.dirname(os.path.abspath(path)), "finomania_output"))
    outs = it.run_file(path)
    r.emit(outs, it)
    p = r.write_html(f"Finomania: {os.path.basename(path)}")
    if p:
        print(f"HTML report written to {p}")
    return 1 if any(isinstance(o, Message) and o.level == "error" for o in outs) else 0
