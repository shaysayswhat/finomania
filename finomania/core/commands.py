"""
The Finomania command language (EViews-compatible where practical).

Every GUI dialog builds a command string and sends it here, so anything done
with the mouse can be reproduced in a program file.

Examples::

    wfcreate m 1990m1 2020m12
    import "C:\\data\\prices.csv"
    series r = 100*dlog(close)
    smpl 2000m1 @last
    equation eq1.ls(cov=hac) r c r(-1)
    eq1.arch(1,1,egarch,dist=t) r c
    var v1.ls 1 2 r1 r2 @ c
    dcc(dist=t) r1 r2 r3
    ff(model=ff5, freq=m) p1 p2 p3
    lstm(lookback=20, epochs=50) r
    for !i = 1 to 3
        series x{!i} = nrnd
    next
"""
from __future__ import annotations

import os
import re
import shlex
import traceback
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..models.base import ModelError, ModelResult, opt_bool, opt_float, opt_int, parse_options, split_at, split_terms
from ..report import FigureOutput, Message, Report
from . import dataio
from .expr import Evaluator, ExprError
from .objects import Estimation, Graph, Group, Matrix, Scalar, Table
from .workfile import Workfile, WorkfileError, sanitize_name


class CommandError(Exception):
    pass


@dataclass
class ObjectOutput:
    """Asks the front-end to open a window for an object."""
    name: str
    obj: Any
    kind: str           # series | group | estimation | table | graph | matrix | scalar
    title: str = ""


class _ExitLoop(Exception):
    pass


class _Stop(Exception):
    pass


TYPE_LABEL = {"equation": "Equation", "var": "VAR", "mgarch": "System", "panel": "Equation", "factor": "Factor Model",
              "ml": "ML Model", "model": "Model"}
ARCH_MODELS = {"garch", "arch", "tarch", "gjr", "egarch", "parch", "aparch", "cgarch", "igarch", "figarch", "harch",
               "ewma", "riskmetrics", "zarch", "tgarch", "component"}
DISTS = {"normal", "n", "t", "tdist", "student", "ged", "skewt", "sst", "skt"}

# --------------------------------------------------------------------------- command catalogue (for help & docs)
HELP: dict[str, tuple[str, str, str]] = {}


def cmd(name, category, syntax, desc, aliases=()):
    def deco(fn):
        fn._cmd = (name, aliases)
        HELP[name] = (category, syntax, desc)
        return fn
    return deco


class Interpreter:
    def __init__(self, wf: Workfile | None = None, progress: Callable | None = None):
        self.workfiles: dict[str, Workfile] = {}
        self.wf: Workfile | None = None
        if wf is not None:
            self.set_workfile(wf)
        self.pvars: dict[str, Any] = {}
        self.last: Estimation | None = None
        self.progress = progress
        self.cwd = os.getcwd()
        self.handlers: dict[str, Callable] = {}
        for attr in dir(type(self)):
            raw = getattr(type(self), attr, None)
            meta = getattr(raw, "_cmd", None)
            if meta:
                fn = getattr(self, attr)
                name, aliases = meta
                self.handlers[name] = fn
                for a in aliases:
                    self.handlers[a] = fn

    # ------------------------------------------------------------------ workfiles
    def set_workfile(self, wf: Workfile):
        base = wf.name.lower()
        name = base
        k = 1
        while name in self.workfiles and self.workfiles[name] is not wf:
            k += 1
            name = f"{base}{k}"
        wf.name = name
        self.workfiles[name] = wf
        self.wf = wf

    def need_wf(self) -> Workfile:
        if self.wf is None:
            raise CommandError("No workfile is open. Create one with wfcreate or open/import a file.")
        return self.wf

    @property
    def ev(self) -> Evaluator:
        return Evaluator(self.need_wf(), {})

    # ------------------------------------------------------------------ program execution
    def run(self, text: str, stop_on_error: bool = True) -> list:
        lines = self._prep(text)
        out: list = []
        try:
            self._block(lines, 0, len(lines), out, stop_on_error)
        except _Stop:
            out.append(Message("Program stopped"))
        except _ExitLoop:
            pass
        return out

    def run_file(self, path: str) -> list:
        with open(path, encoding="utf-8-sig") as f:
            text = f.read()
        old = self.cwd
        self.cwd = os.path.dirname(os.path.abspath(path)) or old
        try:
            return self.run(text)
        finally:
            self.cwd = old

    def _prep(self, text: str) -> list[tuple[int, str]]:
        out = []
        buf, start = "", 0
        for i, raw in enumerate(text.replace("\r", "").split("\n"), start=1):
            ln = _strip_comment(raw).rstrip()
            if buf:
                ln = buf + " " + ln.strip()
                buf = ""
            if ln.endswith(" _"):
                buf = ln[:-2]
                start = start or i
                continue
            if ln.strip():
                out.append((start or i, ln.strip()))
            start = 0
        return out

    def _block(self, lines, a, b, out, stop_on_error):
        i = a
        while i < b:
            lineno, ln = lines[i]
            kw = ln.split(None, 1)[0].lower()
            if kw == "for":
                j = self._match(lines, i, b, ("for",), ("next",))
                self._for(lines, i, j, out, stop_on_error)
                i = j + 1
                continue
            if kw == "while":
                j = self._match(lines, i, b, ("while",), ("wend",))
                cond = ln.split(None, 1)[1] if len(ln.split(None, 1)) > 1 else "0"
                guard = 0
                try:
                    while self._truth(cond):
                        self._block(lines, i + 1, j, out, stop_on_error)
                        guard += 1
                        if guard > 1_000_000:
                            raise CommandError("while loop exceeded 1,000,000 iterations")
                except _ExitLoop:
                    pass
                i = j + 1
                continue
            if kw == "if":
                j = self._match(lines, i, b, ("if",), ("endif",))
                els = None
                depth = 0
                for k in range(i + 1, j):
                    w = lines[k][1].split(None, 1)[0].lower()
                    if w == "if" and not lines[k][1].lower().rstrip().endswith("endif"):
                        depth += 1
                    elif w == "endif":
                        depth -= 1
                    elif w == "else" and depth == 0:
                        els = k
                        break
                m = re.match(r"^if\s+(.*?)\s+then\s*$", ln, flags=re.I) or re.match(r"^if\s+(.*)$", ln, flags=re.I)
                cond = m.group(1)
                if self._truth(cond):
                    self._block(lines, i + 1, els if els is not None else j, out, stop_on_error)
                elif els is not None:
                    self._block(lines, els + 1, j, out, stop_on_error)
                i = j + 1
                continue
            if kw == "exitloop":
                raise _ExitLoop()
            if kw == "stop":
                raise _Stop()
            if kw in ("next", "wend", "endif", "else"):
                raise CommandError(f"Line {lineno}: unexpected '{kw}'")
            try:
                out.extend(self.execute(ln))
            except (_ExitLoop, _Stop):
                raise
            except Exception as e:  # noqa: BLE001
                msg = _err_text(e)
                out.append(Message(f"Error in line {lineno}: {ln}\n{msg}" if len(lines) > 1 else msg, "error"))
                if stop_on_error:
                    raise _Stop() from None
            i += 1

    def _match(self, lines, i, b, opens, closes):
        depth = 0
        for k in range(i, b):
            w = lines[k][1].split(None, 1)[0].lower()
            if w in opens:
                depth += 1
            elif w in closes:
                depth -= 1
                if depth == 0:
                    return k
        raise CommandError(f"Line {lines[i][0]}: missing '{closes[0]}'")

    def _for(self, lines, i, j, out, stop_on_error):
        ln = lines[i][1]
        m = re.match(r"^for\s+!(\w+)\s*=\s*(.+?)\s+to\s+(.+?)(?:\s+step\s+(.+))?$", ln, flags=re.I)
        try:
            if m:
                var = "!" + m.group(1).lower()
                a, bnd = self._num(m.group(2)), self._num(m.group(3))
                st = self._num(m.group(4)) if m.group(4) else 1.0
                v = a
                n = 0
                while (st > 0 and v <= bnd + 1e-12) or (st < 0 and v >= bnd - 1e-12):
                    self.pvars[var] = int(v) if float(v).is_integer() else v
                    self._block(lines, i + 1, j, out, stop_on_error)
                    v += st
                    n += 1
                    if n > 10_000_000:
                        break
                return
            m = re.match(r"^for\s+%(\w+)\s+(.+)$", ln, flags=re.I)
            if m:
                var = "%" + m.group(1).lower()
                items = self._subst(m.group(2)).split()
                for it in items:
                    self.pvars[var] = it
                    self._block(lines, i + 1, j, out, stop_on_error)
                return
        except _ExitLoop:
            return
        raise CommandError(f"Bad for statement: {ln}")

    def _num(self, s):
        v = self.ev.eval(self._subst(s)) if self.wf else float(eval(self._subst(s), {"__builtins__": {}}))  # noqa
        return float(v)

    def _truth(self, cond: str) -> bool:
        cond = self._subst(cond)
        if self.wf is None:
            return bool(eval(cond.replace("<>", "!=").replace("=", "=="), {"__builtins__": {}}))  # noqa: S307
        v = self.ev.eval(cond)
        if isinstance(v, pd.Series):
            raise CommandError("if/while conditions must be scalars")
        return bool(v) and not (isinstance(v, float) and np.isnan(v))

    def _subst(self, s: str) -> str:
        """Program-variable substitution: {!i}/{%s} textual, then !x -> number, %s -> "text"."""
        def brace(m):
            key = m.group(1).lower()
            if key not in self.pvars:
                raise CommandError(f"Program variable {key} is not defined")
            v = self.pvars[key]
            return str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)
        s = re.sub(r"\{([!%]\w+)\}", brace, s)

        def num(m):
            key = "!" + m.group(1).lower()
            if key not in self.pvars:
                raise CommandError(f"Program variable {key} is not defined")
            return repr(float(self.pvars[key])) if not float(self.pvars[key]).is_integer() else str(int(self.pvars[key]))

        def st(m):
            key = "%" + m.group(1).lower()
            if key not in self.pvars:
                raise CommandError(f"Program variable {key} is not defined")
            return '"' + str(self.pvars[key]) + '"'
        s = re.sub(r"(?<![<>=!])!([A-Za-z_]\w*)", num, s)
        s = re.sub(r"(?<!\w)%([A-Za-z_]\w*)", st, s)
        return s

    # ------------------------------------------------------------------ single command
    def execute(self, line: str) -> list:
        line = _strip_comment(line).strip()
        if not line:
            return []
        line = re.sub(r"\{([!%]\w+)\}", lambda m: self._subst(m.group(0)), line)
        # program variable assignment
        m = re.match(r"^([!%])([A-Za-z_]\w*)\s*=(?!=)\s*(.*)$", line)
        if m:
            key = m.group(1) + m.group(2).lower()
            rhs = self._subst(m.group(3))
            if m.group(1) == "!":
                self.pvars[key] = self._num(rhs) if self.wf else float(eval(rhs, {"__builtins__": {}}))  # noqa
            else:
                if self.wf:
                    v = self.ev.eval(rhs)
                else:
                    v = rhs.strip().strip('"')
                self.pvars[key] = str(v) if not isinstance(v, float) else (str(int(v)) if v.is_integer() else str(v))
            return []
        head = re.match(r"^([A-Za-z_][\w]*)(?:\.([A-Za-z_@][\w]*))?", line)
        if not head:
            raise CommandError(f"Cannot parse command: {line}")
        word = head.group(1).lower()
        method = head.group(2).lower() if head.group(2) else None
        rest = line[head.end():]
        opts, rest = _take_parens(rest)
        args = rest.strip()
        # assignment:  [series|genr|scalar] name = expr
        if word in ("series", "genr", "frml", "scalar", "alpha", "vector") and method is None:
            m2 = re.match(r"^([A-Za-z_]\w*)\s*(?:=(?!=)\s*(.*))?$", args, flags=re.S)
            if m2:
                return self._assign(word, m2.group(1), m2.group(2))
        if method is None and word not in self.handlers:
            m2 = re.match(r"^([A-Za-z_]\w*)\s*=(?!=)\s*(.+)$", line, flags=re.S)
            if m2:
                return self._assign("series", m2.group(1), m2.group(2))
        if method is not None:
            return self._object_call(word, method, opts, self._subst(args), line)
        h = self.handlers.get(word)
        if h is None:
            raise CommandError(f"{word.upper()} is not a recognised command or object")
        return h(opts, self._subst(args) if word not in ("series", "genr") else args)

    # ------------------------------------------------------------------ assignment
    def _assign(self, kind, name, rhs):
        wf = self.need_wf()
        name = name.lower()
        if rhs is None:
            if kind in ("series", "genr", "frml"):
                wf.set_series(name, np.full(wf.nobs, np.nan), respect_sample=False)
                return [Message(f"Series {name.upper()} created")]
            if kind == "scalar":
                wf.add(Scalar(name, np.nan))
                return []
            raise CommandError("Missing '=' in assignment")
        rhs = self._subst(rhs)
        if kind == "scalar":
            v = self.ev.eval(rhs)
            if isinstance(v, pd.Series):
                raise CommandError("Scalar assignment needs a scalar expression")
            wf.add(Scalar(name, float(v)))
            return [Message(f"{name.upper()} = {float(v):.8g}")]
        if name in wf.objects and isinstance(wf.objects[name], Scalar) and kind not in ("series", "genr"):
            v = self.ev.eval(rhs)
            wf.objects[name].value = float(v)
            return []
        existed = name in wf.series_names()
        val = self.ev.eval(rhs, as_series=True)
        wf.set_series(name, val, respect_sample=True)
        return [Message(f"{name.upper()} {'modified' if existed else 'created'}")]

    # ------------------------------------------------------------------ objects
    def _object_call(self, objname, method, opts, args, line):
        wf = self.need_wf()
        if objname in ("equation", "var", "system", "model", "panel"):
            raise CommandError("Use 'equation name.method ...'")
        otype = wf.obj_type(objname)
        existing = wf.objects.get(objname)
        if isinstance(existing, Estimation) and method in existing.result.all_actions():
            return self.call_view(existing, method, opts, args)
        if method in ESTIMATION_METHODS and (otype is None or otype not in ("series", "alpha", "group")):
            return self._estimate_named(method, opts, args, objname)
        if otype is None:
            raise CommandError(f"{objname.upper()} does not exist")
        if otype in ("series", "alpha"):
            return self._series_method(objname, method, opts, args)
        if otype == "group":
            members = wf.group_members(objname)
            if method in ("add",):
                wf.objects[objname].members += wf.expand_names(split_terms(args))
                wf.notify()
                return []
            if method in ("drop",):
                dr = [a.lower() for a in split_terms(args)]
                wf.objects[objname].members = [m for m in members if m not in dr]
                wf.notify()
                return []
            if method in ("sheet", "show"):
                return [ObjectOutput(objname, wf.objects[objname], "group", f"Group: {objname.upper()}")]
            h = self.handlers.get(method)
            if h is None:
                raise CommandError(f"Unknown group view/proc '{method}'")
            return h(opts, " ".join(members) + (" " + args if args else ""))
        obj = wf.objects[objname]
        if method in ("display", "show", "output"):
            return [self._obj_output(objname)]
        if method in ("name", "rename"):
            wf.rename(objname, args.split()[0])
            return []
        if isinstance(obj, Estimation):
            return self.call_view(obj, method, opts, args)
        if isinstance(obj, Table):
            if method in ("save", "export"):
                return [Message(self._export_report(obj.report(), args))]
        raise CommandError(f"{objname.upper()} ({otype}) has no method '{method}'")

    def call_view(self, est: Estimation, key: str, opts: str | None, args: str) -> list:
        res = est.result
        acts = res.all_actions()
        if key not in acts:
            raise CommandError(f"{est.name.upper() or 'Object'} has no view or proc '{key}'. Available: "
                               + ", ".join(sorted(acts)))
        v = acts[key]
        pos, kw = parse_options(opts)
        kwargs = {}
        params = list(v.params)
        for p, val in zip(params, pos):
            kwargs[p.name] = val
        for k, val in kw.items():
            for p in params:
                if p.name == k:
                    kwargs[k] = val
        if args:
            for p in params:
                if p.name not in kwargs and p.kind in ("text", "list", "choice"):
                    kwargs[p.name] = args
                    break
        for p in params:
            if p.name in kwargs:
                kwargs[p.name] = _convert(kwargs[p.name], p.kind)
        out = v.func(**kwargs)
        return _as_list(out)

    def _series_method(self, name, method, opts, args):
        wf = self.need_wf()
        if method in ("sheet", "show", "display"):
            return [ObjectOutput(name, name, "series", f"Series: {name.upper()}")]
        alias = {"plot": "line", "graph": "line", "correl": "correl", "statby": "stats", "teststat": "ttest",
                 "hpf": "hpf", "label": None}
        method = alias.get(method, method)
        if method is None:
            return []
        if method == "fill":
            vals = [float(v) for v in re.split(r"[\s,]+", args.strip()) if v]
            s = wf.series(name).copy().values
            s[: len(vals)] = vals
            wf.set_series(name, s, respect_sample=False)
            return []
        h = self.handlers.get(method)
        if h is None:
            raise CommandError(f"Unknown series view/proc '{method}'")
        return h(opts, name + (" " + args if args else ""))

    def _obj_output(self, name):
        wf = self.need_wf()
        o = wf.objects.get(name)
        t = wf.obj_type(name)
        if t in ("series", "alpha"):
            return ObjectOutput(name, name, "series", f"Series: {name.upper()}")
        if isinstance(o, Group):
            return ObjectOutput(name, o, "group", f"Group: {name.upper()}")
        if isinstance(o, Estimation):
            return ObjectOutput(name, o, "estimation", f"{o.type.capitalize()}: {name.upper()}")
        if isinstance(o, Table):
            return ObjectOutput(name, o, "table", f"Table: {name.upper()}")
        if isinstance(o, Graph):
            return ObjectOutput(name, o, "graph", f"Graph: {name.upper()}")
        if isinstance(o, (Matrix, Scalar)):
            return ObjectOutput(name, o, t, f"{t.capitalize()}: {name.upper()}")
        raise CommandError(f"{name.upper()} does not exist")

    # ------------------------------------------------------------------ estimation plumbing
    def _estimate_named(self, method, opts, args, name):
        res = self._estimate(method, opts, args, name)
        cmdtext = f"{method}{'(' + opts + ')' if opts else ''} {args}".strip()
        return self._store(res, name, cmdtext)

    def _store(self, res: ModelResult, name: str | None, cmdtext: str) -> list:
        wf = self.need_wf()
        est = Estimation(name or "", res, cmdtext, wf.sample_text)
        if hasattr(res, "name"):
            res.name = name or ""
        if hasattr(res, "command"):
            try:
                res.command = cmdtext
            except Exception:
                pass
        if name:
            wf.add(est)
        else:
            self.last = est
        # EViews conventions: update RESID and the C coefficient vector
        try:
            if hasattr(res, "resid") and callable(res.resid) and res.object_type in ("equation", "panel"):
                rs = res.resid()
                if isinstance(rs, pd.Series) and len(rs) == wf.nobs:
                    wf.set_series("resid", rs, respect_sample=False, notify=False)
            if hasattr(res, "params"):
                wf.cvec = np.asarray(res.params, dtype=float)
        except Exception:
            pass
        wf.notify()
        return [ObjectOutput(name or "", est, "estimation",
                             f"{TYPE_LABEL.get(est.type, est.type.capitalize())}: {(name or 'untitled').upper()}")]

    def estimate_command(self, command: str, name: str = ""):
        head = re.match(r"^([A-Za-z_]\w*)", command)
        method = head.group(1).lower()
        opts, rest = _take_parens(command[head.end():])
        return self._estimate(method, opts, rest.strip(), name)

    def _estimate(self, method, opts, args, name=""):
        wf = self.need_wf()
        ev = self.ev
        pos, kw = parse_options(opts)
        fn = ESTIMATION_METHODS.get(method)
        if fn is None:
            raise CommandError(f"Unknown estimation method {method}")
        try:
            return fn(self, wf, ev, pos, kw, args, name)
        except (ModelError, WorkfileError, ExprError) as e:
            raise CommandError(str(e)) from e

    # ================================================================== commands
    # ---------------------------------------------------------------- workfile
    @cmd("wfcreate", "Workfile", "wfcreate(name) freq start end  |  wfcreate u n  |  wfcreate(panel=N) freq start end",
         "Create a new workfile. freq: a, s, q, m, w, d5, d7 or u (undated).", aliases=("create",))
    def c_wfcreate(self, opts, args):
        pos, kw = parse_options(opts)
        name = kw.get("wf") or kw.get("name") or (pos[0] if pos else "untitled")
        toks = args.split()
        if not toks:
            raise CommandError("wfcreate freq start end")
        f = toks[0]
        ncross = opt_int(kw, "panel")
        if f.lower() == "u":
            if len(toks) == 2:
                wf = Workfile.create("u", int(toks[1]), name=name, ncross=ncross)
            else:
                wf = Workfile.create("u", int(toks[1]), int(toks[2]), name=name, ncross=ncross)
        else:
            if len(toks) < 3:
                raise CommandError("wfcreate freq start end")
            wf = Workfile.create(f, toks[1], toks[2], name=name, ncross=ncross)
        self.set_workfile(wf)
        return [Message(f"Workfile {wf.name.upper()} created: {wf.range_text()}"), ObjectOutput(wf.name, wf, "workfile")]

    @cmd("wfopen", "Workfile", 'wfopen "path"', "Open a Finomania workfile (.fwf) or a data file as a new workfile.",
         aliases=("open", "load"))
    def c_wfopen(self, opts, args):
        path = self._path(args)
        if path.lower().endswith(".fwf"):
            wf = Workfile.load(path)
        else:
            pos, kw = parse_options(opts)
            wf, names, _ = dataio.import_file(path, None, date=kw.get("date"), ident=kw.get("id"), freq=kw.get("freq"),
                                              sheet=kw.get("sheet"))
        self.set_workfile(wf)
        return [Message(f"Opened {os.path.basename(path)}: {wf.describe()}"), ObjectOutput(wf.name, wf, "workfile")]

    @cmd("wfsave", "Workfile", 'wfsave ["path"]', "Save the active workfile (.fwf).", aliases=("save",))
    def c_wfsave(self, opts, args):
        wf = self.need_wf()
        path = self._path(args) if args.strip() else (wf.path or os.path.join(self.cwd, wf.name + ".fwf"))
        wf.save(path)
        return [Message(f"Workfile saved to {wf.path}")]

    @cmd("wfselect", "Workfile", "wfselect name", "Make another open workfile active.")
    def c_wfselect(self, opts, args):
        n = args.strip().lower()
        if n not in self.workfiles:
            raise CommandError(f"No open workfile named {n}")
        self.wf = self.workfiles[n]
        return [Message(f"Active workfile: {n.upper()}"), ObjectOutput(n, self.wf, "workfile")]

    @cmd("wfclose", "Workfile", "wfclose [name]", "Close a workfile.", aliases=("close",))
    def c_wfclose(self, opts, args):
        n = args.strip().lower() or (self.wf.name if self.wf else "")
        self.workfiles.pop(n, None)
        self.wf = next(iter(self.workfiles.values()), None)
        return [Message(f"Closed {n.upper()}")]

    @cmd("import", "Workfile", 'import(date=col, id=col, freq=m, sheet=) "path"',
         "Import CSV / Excel / Stata / Parquet. Dates and panel structure are detected automatically.",
         aliases=("read",))
    def c_import(self, opts, args):
        pos, kw = parse_options(opts)
        path = self._path(args)
        wf, names, new = dataio.import_file(path, self.wf, date=kw.get("date"), ident=kw.get("id"),
                                            freq=kw.get("freq"), sheet=kw.get("sheet"),
                                            start=kw.get("start"), undated=opt_bool(kw, "undated"))
        if new:
            self.set_workfile(wf)
        msg = f"Imported {len(names)} series from {os.path.basename(path)}" + (f" — {wf.describe()}" if new else "")
        out = [Message(msg)]
        if new:
            out.append(ObjectOutput(wf.name, wf, "workfile"))
        return out

    @cmd("export", "Workfile", 'export "path" [series...]', "Export series to CSV / Excel / Stata / Parquet.",
         aliases=("write",))
    def c_export(self, opts, args):
        wf = self.need_wf()
        toks = _split_args(args)
        if not toks:
            raise CommandError('export "path" [series ...]')
        path = self._path(toks[0])
        dataio.export_workfile(wf, path, toks[1:] or None)
        return [Message(f"Exported to {path}")]

    @cmd("fetch", "Workfile", 'fetch(source=yahoo|fred, start=2010-01-01, end=, field=adjclose) symbols...',
         "Download data from Yahoo Finance or FRED into the workfile (creates one if needed).")
    def c_fetch(self, opts, args):
        pos, kw = parse_options(opts)
        src = (kw.get("source") or (pos[0] if pos else "yahoo")).lower()
        syms = args.split()
        if not syms:
            raise CommandError("List symbols / series codes to fetch")
        if src in ("yahoo", "yf", "y"):
            df = dataio.fetch_yahoo(syms, kw.get("start"), kw.get("end"), kw.get("interval", "1d"),
                                    kw.get("field", "adjclose"))
            freq = "d5"
        elif src in ("fred", "f"):
            df = dataio.fetch_fred(syms, kw.get("start"), kw.get("end"))
            freq = None
        else:
            raise CommandError("source must be yahoo or fred")
        return self._absorb(df, freq, f"{src.upper()} data")

    def _absorb(self, df: pd.DataFrame, freq, label):
        df = df.sort_index()
        if self.wf is None or (not self.wf.series_names() and not self.wf.objects):
            wf = Workfile.from_dataframe(df, name=sanitize_name(label.split()[0].lower()), freq=freq)
            self.set_workfile(wf)
            return [Message(f"Created workfile from {label}: {wf.describe()}"), ObjectOutput(wf.name, wf, "workfile")]
        new = Workfile.from_dataframe(df, freq=freq)
        dataio.merge_into(self.wf, new)
        return [Message(f"Added {', '.join(c.upper() for c in new.series_names())} from {label}")]

    @cmd("ffget", "Factor models", "ffget(model=ff5, freq=m, units=decimal, file=, refresh=0)",
         "Import Kenneth French factors (MKT_RF, SMB, HML, RMW, CMA, MOM, RF) into the workfile.")
    def c_ffget(self, opts, args):
        from ..models.factors import load_french
        pos, kw = parse_options(opts)
        model = kw.get("model", pos[0] if pos else "ff5")
        freq = kw.get("freq") or ({"m": "m", "d5": "d", "d7": "d", "w": "w"}.get(self.wf.freq, "m") if self.wf else "m")
        F = load_french(model, freq, self._path(kw["file"]) if kw.get("file") else None, opt_bool(kw, "refresh"),
                        kw.get("units", "decimal"))
        return self._absorb(F, {"m": "m", "d": "d5", "w": "w"}[freq if freq in ("m", "d", "w") else "m"],
                            f"Fama-French {model.upper()}")

    @cmd("smpl", "Workfile", "smpl start end [start end ...] [if condition]  |  smpl @all",
         "Set the current sample.", aliases=("sample",))
    def c_smpl(self, opts, args):
        wf = self.need_wf()
        wf.set_sample(self._subst(args or "@all"), self.ev)
        return [Message(f"Sample: {wf.sample_text}  ({int(wf.sample.sum())} obs)")]

    @cmd("expand", "Workfile", "expand end_date", "Extend the workfile range (for out-of-sample forecasting).",
         aliases=("wfrange", "range"))
    def c_expand(self, opts, args):
        wf = self.need_wf()
        tok = args.split()[-1]
        n = wf.expand(tok)
        return [Message(f"Range extended by {n} observations: {wf.range_text()}")]

    @cmd("rndseed", "Workfile", "rndseed n", "Seed the random number generator (nrnd, rnd).")
    def c_rndseed(self, opts, args):
        wf = self.need_wf()
        wf.rng = np.random.default_rng(int(float(args.strip())))
        return []

    # ---------------------------------------------------------------- objects
    @cmd("group", "Objects", "group name ser1 ser2 ...", "Create a group of series.")
    def c_group(self, opts, args):
        wf = self.need_wf()
        toks = split_terms(args)
        if not toks:
            raise CommandError("group name series...")
        name, members = toks[0].lower(), toks[1:]
        mem = []
        for m in wf.expand_names(members):
            if not wf.exists(m):
                # allow expressions: create an auxiliary series
                s = self.ev.eval(m, ignore_sample=True, as_series=True)
                nm = sanitize_name(m)
                wf.set_series(nm, s, respect_sample=False, notify=False)
                m = nm
            mem.append(m.lower())
        wf.add(Group(name, mem))
        return [ObjectOutput(name, wf.objects[name], "group", f"Group: {name.upper()}")]

    @cmd("equation", "Objects", "equation name.method(options) specification",
         "Declare and estimate a named equation, e.g. equation eq1.ls y c x")
    def c_equation(self, opts, args):
        m = re.match(r"^([A-Za-z_]\w*)\.([A-Za-z_]\w*)", args)
        if not m:
            name = args.strip().split()[0].lower()
            return [Message(f"Empty equation {name.upper()} declared — estimate with {name}.ls ...")]
        name, method = m.group(1).lower(), m.group(2).lower()
        o2, rest = _take_parens(args[m.end():])
        return self._estimate_named(method, o2, rest.strip(), name)

    @cmd("var", "Objects", "var name.ls lag1 lag2 y1 y2 [@ c x]  |  var(lags=2) y1 y2 [@ exog]",
         "Estimate a vector autoregression.")
    def c_var(self, opts, args):
        m = re.match(r"^([A-Za-z_]\w*)\.(ls|ec|vec)\b", args)
        if m:
            name = m.group(1).lower()
            o2, rest = _take_parens(args[m.end():])
            method = "var" if m.group(2) == "ls" else "vec"
            if method == "var":
                return self._estimate_named("var", o2, "@lagpairs " + rest.strip(), name)
            return self._estimate_named("vec", o2, rest.strip(), name)
        res = self._estimate("var", opts, args)
        return self._store(res, None, f"var{'(' + opts + ')' if opts else ''} {args}")

    @cmd("graph", "Graphs", "graph name.type series...", "Create a named graph object (type: line, scat, hist, bar...).")
    def c_graph(self, opts, args):
        m = re.match(r"^([A-Za-z_]\w*)\.([A-Za-z_]\w*)", args)
        if not m:
            raise CommandError("graph name.line series ...")
        name, typ = m.group(1).lower(), m.group(2).lower()
        o2, rest = _take_parens(args[m.end():])
        h = self.handlers.get(typ)
        if h is None:
            raise CommandError(f"Unknown graph type {typ}")
        out = h(o2, rest.strip())
        fig = next((o for o in out if isinstance(o, FigureOutput)), None)
        if fig is None:
            raise CommandError("No graph produced")
        self.need_wf().add(Graph(name, fig.figure, {"title": fig.title, "cmd": typ + " " + rest}))
        return [ObjectOutput(name, self.wf.objects[name], "graph", f"Graph: {name.upper()}")]

    @cmd("matrix", "Objects", "matrix name = @cov(group) | @cor(group)", "Create a matrix from a group statistic.")
    def c_matrix(self, opts, args):
        wf = self.need_wf()
        m = re.match(r"^([A-Za-z_]\w*)\s*=\s*@(cov|cor)\((\w+)\)", args, flags=re.I)
        if not m:
            raise CommandError("matrix name = @cov(group) or @cor(group)")
        name, fn, g = m.group(1).lower(), m.group(2).lower(), m.group(3).lower()
        mem = wf.group_members(g)
        df = pd.DataFrame({x: wf.series(x) for x in mem})[wf.sample].dropna()
        M = df.cov(ddof=0) if fn == "cov" else df.corr()
        wf.add(Matrix(name, M.values, [x.upper() for x in mem], [x.upper() for x in mem]))
        return [self._obj_output(name)]

    @cmd("delete", "Objects", "delete name [name...] (wildcards * ? allowed)", "Delete objects.", aliases=("del",))
    def c_delete(self, opts, args):
        gone = self.need_wf().delete(args.split())
        return [Message("Deleted " + ", ".join(g.upper() for g in gone))]

    @cmd("rename", "Objects", "rename old new", "Rename an object.")
    def c_rename(self, opts, args):
        a, b = args.split()[:2]
        self.need_wf().rename(a, b)
        return [Message(f"{a.upper()} renamed to {b.upper()}")]

    @cmd("copy", "Objects", "copy source destination", "Copy an object.")
    def c_copy(self, opts, args):
        a, b = args.split()[:2]
        self.need_wf().copy_object(a, b)
        return []

    @cmd("name", "Objects", "name newname", "Name the last untitled estimation so it is stored in the workfile.")
    def c_name(self, opts, args):
        if self.last is None:
            raise CommandError("There is no untitled object to name")
        nm = args.strip().split()[0].lower()
        self.last.name = nm
        if hasattr(self.last.result, "name"):
            self.last.result.name = nm
        self.need_wf().add(self.last)
        self.last = None
        return [self._obj_output(nm)]

    @cmd("show", "Objects", "show name | show ser1 ser2 ...", "Open an object (several series open as a group).",
         aliases=("display",))
    def c_show(self, opts, args):
        wf = self.need_wf()
        toks = split_terms(args)
        if len(toks) == 1 and wf.exists(toks[0]):
            return [self._obj_output(toks[0].lower())]
        names = []
        for t in toks:
            if wf.exists(t):
                names += wf.expand_names([t])
            else:
                s = self.ev.eval(t, ignore_sample=True, as_series=True)
                nm = sanitize_name(t)
                wf.set_series(nm, s, respect_sample=False, notify=False)
                names.append(nm)
        g = Group("untitled", names)
        return [ObjectOutput("", g, "group", "Group: UNTITLED")]

    @cmd("freeze", "Objects", "freeze(name) command_or_view", "Store the output of a view as a table / graph object.")
    def c_freeze(self, opts, args):
        wf = self.need_wf()
        name = (opts or "").strip().lower() or wf.unique_name("table")
        out = self.execute(args)
        for o in out:
            rep = None
            if isinstance(o, Report):
                rep = o
            elif isinstance(o, ObjectOutput) and o.kind == "estimation":
                rep = o.obj.report()
            if rep is not None:
                wf.add(Table(name, rep))
                return [self._obj_output(name)]
            if isinstance(o, FigureOutput):
                wf.add(Graph(name, o.figure, {"title": o.title}))
                return [self._obj_output(name)]
        raise CommandError("Nothing to freeze")

    @cmd("print", "Objects", "print name", "Print an object's output.")
    def c_print(self, opts, args):
        o = self._obj_output(args.strip().lower())
        if o.kind == "estimation":
            return [o.obj.report()]
        if o.kind == "table":
            return [o.obj.report()]
        return [o]

    # ---------------------------------------------------------------- statistics on series
    def _series_list(self, args) -> list[str]:
        wf = self.need_wf()
        out = []
        for t in split_terms(args):
            if wf.obj_type(t) == "group" or any(ch in t for ch in "*?"):
                out += wf.expand_names([t])
            else:
                out.append(t.lower())
        if not out:
            raise CommandError("Specify one or more series")
        return out

    def _values(self, names) -> dict[str, np.ndarray]:
        wf, ev = self.need_wf(), self.ev
        return {n: ev.eval(n, ignore_sample=True, as_series=True).values[wf.sample].astype(float) for n in names}

    @cmd("stats", "Statistics", "stats(i) ser1 ser2 ...", "Descriptive statistics (common sample; (i) = individual).",
         aliases=("describe", "summary"))
    def c_stats(self, opts, args):
        from ..models.stats import stats_report
        from ..models.panel import panel_describe
        names = self._series_list(args)
        pos, kw = parse_options(opts)
        if "panel" in pos and self.wf.is_panel:
            return [panel_describe(self.wf, self.ev, names)]
        return [stats_report(self._values(names), common="i" not in pos)]

    @cmd("hist", "Statistics", "hist series", "Histogram with descriptive statistics and Jarque-Bera test.")
    def c_hist(self, opts, args):
        from ..models.stats import histogram_output
        pos, kw = parse_options(opts)
        n = self._series_list(args)[0]
        return [histogram_output(self._values([n])[n], n, self.wf.sample_label(), opt_int(kw, "bins"))]

    @cmd("correl", "Statistics", "correl(lags) series", "Correlogram (ACF, PACF, Ljung-Box Q).", aliases=("ac", "acf"))
    def c_correl(self, opts, args):
        from ..models.stats import correlogram_report
        pos, kw = parse_options(opts)
        lags = int(pos[0]) if pos else opt_int(kw, "lags", 24)
        n = self._series_list(args)[0]
        v = self._values([n])[n]
        if "d" in kw or "dif" in kw:
            for _ in range(opt_int(kw, "d", opt_int(kw, "dif", 1))):
                v = np.diff(v)
        return [correlogram_report(v, n, lags)]

    @cmd("cor", "Statistics", "cor(method=pearson|spearman|kendall, tests=0) ser1 ser2 ...", "Correlation matrix.",
         aliases=("corr", "correlation"))
    def c_cor(self, opts, args):
        from ..models.stats import corr_report
        pos, kw = parse_options(opts)
        names = self._series_list(args)
        df = pd.DataFrame(self._values(names))
        return [corr_report(df, "corr", opt_bool(kw, "tests") or "tests" in pos, kw.get("method", "pearson"))]

    @cmd("cov", "Statistics", "cov ser1 ser2 ...", "Covariance matrix.")
    def c_cov(self, opts, args):
        from ..models.stats import corr_report
        names = self._series_list(args)
        return [corr_report(pd.DataFrame(self._values(names)), "cov")]

    @cmd("ttest", "Statistics", "ttest(mu=0) series", "Test that the mean equals mu (t and Wilcoxon).")
    def c_ttest(self, opts, args):
        from ..models.stats import mean_test
        pos, kw = parse_options(opts)
        n = self._series_list(args)[0]
        return [mean_test(self._values([n])[n], opt_float(kw, "mu", float(pos[0]) if pos else 0.0), n)]

    @cmd("eqtest", "Statistics", "eqtest ser1 ser2 ...", "Equality of means / medians / variances across series.")
    def c_eqtest(self, opts, args):
        from ..models.stats import equality_tests
        return [equality_tests(self._values(self._series_list(args)))]

    @cmd("normtest", "Statistics", "normtest series", "Normality tests (JB, Shapiro-Wilk, Anderson-Darling, ...) + Q-Q.",
         aliases=("jb", "normality"))
    def c_normtest(self, opts, args):
        from ..models.stats import normality_report
        n = self._series_list(args)[0]
        return [normality_report(self._values([n])[n], n)]

    @cmd("bds", "Statistics", "bds(dim=6) series", "BDS test for independence.")
    def c_bds(self, opts, args):
        from ..models.stats import bds_report
        pos, kw = parse_options(opts)
        n = self._series_list(args)[0]
        return [bds_report(self._values([n])[n], n, opt_int(kw, "dim", 6))]

    @cmd("vratio", "Statistics", 'vratio(lags="2 4 8 16", log=0) series', "Lo-MacKinlay variance ratio tests.")
    def c_vratio(self, opts, args):
        from ..models.stats import variance_ratio_report
        pos, kw = parse_options(opts)
        n = self._series_list(args)[0]
        return [variance_ratio_report(self._values([n])[n], n, kw.get("lags", "2 4 8 16"), opt_bool(kw, "log"))]

    @cmd("lbq", "Statistics", "lbq(lags=10) series", "Ljung-Box and Box-Pierce Q tests.", aliases=("ljungbox",))
    def c_lbq(self, opts, args):
        from ..models.stats import autocorr_test_report
        pos, kw = parse_options(opts)
        n = self._series_list(args)[0]
        return [autocorr_test_report(self._values([n])[n], n, opt_int(kw, "lags", int(pos[0]) if pos else 10))]

    @cmd("pca", "Statistics", "pca(cov=0, scores=prefix) ser1 ser2 ...", "Principal components.")
    def c_pca(self, opts, args):
        from ..models.stats import pca_report
        pos, kw = parse_options(opts)
        names = self._series_list(args)
        wf = self.need_wf()
        df = pd.DataFrame({n: self.ev.eval(n, ignore_sample=True, as_series=True) for n in names})[wf.sample]
        rep, sc = pca_report(df, not opt_bool(kw, "cov"))
        pre = kw.get("scores")
        if pre:
            for c in sc.columns:
                wf.set_series(f"{pre}{c[2:]}", sc[c].reindex(wf.data.index), respect_sample=False)
            rep.note(f"Scores saved as {pre.upper()}1..{pre.upper()}{sc.shape[1]}")
        return [rep]

    # ---------------------------------------------------------------- unit roots, cointegration, causality
    @cmd("uroot", "Time series tests", "uroot(adf|pp|kpss|dfgls|za, trend=c|ct|n, lags=, maxlag=, ic=sic, dif=0) series",
         "Unit-root / stationarity tests. In a panel workfile runs LLC, IPS, Fisher and Hadri tests.",
         aliases=("unitroot", "adf"))
    def c_uroot(self, opts, args):
        from ..models.panel import panel_unit_root
        from ..models.timeseries import unit_root_test
        pos, kw = parse_options(opts)
        wf = self.need_wf()
        n = self._series_list(args)[0]
        test = kw.get("test") or next((p for p in pos if p.lower() in ("adf", "pp", "kpss", "dfgls", "za", "ers",
                                                                       "llc", "ips", "fisher", "hadri", "sum")), "adf")
        trend = kw.get("trend") or kw.get("exog") or next((p for p in pos if p.lower() in ("c", "ct", "n", "const",
                                                                                          "trend", "none")), "c")
        if wf.is_panel:
            return [panel_unit_root(wf, self.ev, n, trend, opt_int(kw, "maxlag"), opt_int(kw, "lags"))]
        return [unit_root_test(wf, self.ev, n, test, trend, opt_int(kw, "lags"), opt_int(kw, "maxlag"),
                               kw.get("ic", "sic"), opt_int(kw, "dif", 0))]

    @cmd("urootsum", "Time series tests", "urootsum(trend=c) ser1 ser2 ...",
         "ADF / PP / KPSS summary on levels and first differences.")
    def c_urootsum(self, opts, args):
        from ..models.timeseries import unit_root_summary
        pos, kw = parse_options(opts)
        return [unit_root_summary(self.need_wf(), self.ev, self._series_list(args), kw.get("trend", "c"))]

    @cmd("coint", "Time series tests", "coint(johansen|eg|po, lags=1, det=c) ser1 ser2 ...",
         "Cointegration tests: Johansen trace/max-eigenvalue, Engle-Granger, Phillips-Ouliaris.")
    def c_coint(self, opts, args):
        from ..models.timeseries import engle_granger_test, johansen_test
        pos, kw = parse_options(opts)
        names = self._series_list(args)
        method = kw.get("method") or next((p for p in pos if p.lower() in ("johansen", "eg", "po")), "johansen")
        if method == "johansen":
            return [johansen_test(self.need_wf(), self.ev, names, opt_int(kw, "lags", 1), kw.get("det", "c"))]
        return [engle_granger_test(self.need_wf(), self.ev, names, kw.get("trend", "c"), method)]

    @cmd("granger", "Time series tests", "granger(lags=2) ser1 ser2 ...", "Pairwise Granger causality tests.",
         aliases=("gcause",))
    def c_granger(self, opts, args):
        from ..models.panel import dumitrescu_hurlin
        from ..models.timeseries import granger_pairwise
        pos, kw = parse_options(opts)
        lags = opt_int(kw, "lags", int(pos[0]) if pos else 2)
        names = self._series_list(args)
        if self.need_wf().is_panel:
            return [dumitrescu_hurlin(self.wf, self.ev, names, lags)]
        return [granger_pairwise(self.wf, self.ev, names, lags)]

    @cmd("dhcause", "Panel", "dhcause(lags=2) ser1 ser2", "Dumitrescu-Hurlin panel Granger causality.")
    def c_dhcause(self, opts, args):
        from ..models.panel import dumitrescu_hurlin
        pos, kw = parse_options(opts)
        return [dumitrescu_hurlin(self.need_wf(), self.ev, self._series_list(args), opt_int(kw, "lags", 2))]

    # ---------------------------------------------------------------- filters
    @cmd("hpf", "Time series tests", "hpf(lambda=1600) series [trend_name] [cycle_name]", "Hodrick-Prescott filter.")
    def c_hpf(self, opts, args):
        from ..models.timeseries import hp_filter
        pos, kw = parse_options(opts)
        toks = args.split()
        lam = opt_float(kw, "lambda", float(pos[0]) if pos else None)
        return hp_filter(self.need_wf(), self.ev, toks[0], lam, toks[1] if len(toks) > 1 else None,
                         toks[2] if len(toks) > 2 else None)

    @cmd("bpf", "Time series tests", "bpf(type=bk|cf, low=6, high=32) series [name]", "Band-pass filters.")
    def c_bpf(self, opts, args):
        from ..models.timeseries import bp_filter
        pos, kw = parse_options(opts)
        toks = args.split()
        return bp_filter(self.need_wf(), self.ev, toks[0], kw.get("type", "bk"), opt_float(kw, "low", 6),
                         opt_float(kw, "high", 32), opt_int(kw, "k", 12), toks[1] if len(toks) > 1 else None)

    # ---------------------------------------------------------------- risk
    @cmd("risk", "Risk", 'risk(levels="0.95 0.99", methods="historical gaussian t garch", value=) returns',
         "Value-at-Risk and Expected Shortfall by several methods.", aliases=("valueatrisk",))
    def c_risk(self, opts, args):
        from ..models.risk import var_report
        pos, kw = parse_options(opts)
        n = self._series_list(args)[0]
        return [var_report(self.need_wf(), self.ev, n, kw.get("levels", "0.95 0.99"),
                           kw.get("methods", "historical gaussian cornish t ewma garch fhs"), opt_float(kw, "value"))]

    @cmd("backtest", "Risk", "backtest(level=0.99, window=250, method=historical|gaussian|t|ewma|garch|fhs) returns",
         "Rolling VaR backtest with Kupiec, Christoffersen and Basel traffic-light.")
    def c_backtest(self, opts, args):
        from ..models.risk import backtest_report
        pos, kw = parse_options(opts)
        n = self._series_list(args)[0]
        return backtest_report(self.need_wf(), self.ev, n, opt_float(kw, "level", 0.99), opt_int(kw, "window", 250),
                               kw.get("method", "historical"), opt_int(kw, "refit", 20))

    @cmd("perf", "Risk", "perf(rf=, bench=, periods=) ret1 ret2 ...",
         "Performance analytics: Sharpe, Sortino, drawdowns, VaR, beta, information ratio.", aliases=("performance",))
    def c_perf(self, opts, args):
        from ..models.risk import performance_report
        pos, kw = parse_options(opts)
        return performance_report(self.need_wf(), self.ev, self._series_list(args), kw.get("rf"),
                                  opt_int(kw, "periods"), kw.get("bench"))

    # ---------------------------------------------------------------- graphs
    def _graph_data(self, args):
        wf, ev = self.need_wf(), self.ev
        names = self._series_list(args)
        data = {}
        for n in names:
            s = ev.eval(n, ignore_sample=True, as_series=True)[wf.sample]
            if wf.is_panel:
                s = s.reset_index(drop=True)
            data[n] = s
        return data

    @cmd("line", "Graphs", "line(m) ser1 ser2 ...", "Line graph ((m) = multiple graphs).", aliases=("plot",))
    def c_line(self, opts, args):
        from ..models import plots
        pos, kw = parse_options(opts)
        data = self._graph_data(args)
        title = kw.get("title", " ".join(n.upper() for n in data) if len(data) <= 4 else "")
        fig = plots.line_plot(data, title, multiple="m" in pos or opt_bool(kw, "multiple"))
        return [FigureOutput(fig, "Line: " + " ".join(n.upper() for n in data))]

    @cmd("bar", "Graphs", "bar ser1 ser2 ...", "Bar graph.")
    def c_bar(self, opts, args):
        from ..models import plots
        data = self._graph_data(args)
        return [FigureOutput(plots.bar_plot(data, " ".join(n.upper() for n in data)), "Bar graph")]

    @cmd("scat", "Graphs", "scat x y", "Scatter plot with regression line.", aliases=("scatter", "xy"))
    def c_scat(self, opts, args):
        from ..models import plots
        data = self._graph_data(args)
        ks = list(data)
        if len(ks) < 2:
            raise CommandError("scat needs two series")
        return [FigureOutput(plots.scatter_plot(data[ks[0]], data[ks[1]], "", True, ks[0], ks[1]),
                             f"Scatter {ks[1].upper()} vs {ks[0].upper()}")]

    @cmd("boxplot", "Graphs", "boxplot ser1 ser2 ...", "Box plots.")
    def c_boxplot(self, opts, args):
        from ..models import plots
        return [FigureOutput(plots.boxplot(self._values(self._series_list(args)), "Box plots"), "Box plot")]

    @cmd("kdensity", "Graphs", "kdensity series", "Kernel density estimate.")
    def c_kdensity(self, opts, args):
        from ..models import plots
        n = self._series_list(args)[0]
        return [FigureOutput(plots.kde_plot(self._values([n])[n], n), "Kernel density")]

    @cmd("qqplot", "Graphs", "qqplot series", "Normal quantile-quantile plot.")
    def c_qqplot(self, opts, args):
        from ..models import plots
        n = self._series_list(args)[0]
        return [FigureOutput(plots.qq_plot(self._values([n])[n], n), "Q-Q plot")]

    # ---------------------------------------------------------------- generic estimation command entry points
    def _est_cmd(method):
        def f(self, opts, args):
            res = self._estimate(method, opts, args)
            return self._store(res, None, f"{method}{'(' + opts + ')' if opts else ''} {args}".strip())
        return f

    c_ls = cmd("ls", "Estimation", "ls(cov=white|hac|cluster, w=weights) y c x1 x2 [ar(1) ma(1)]  |  ls y = c(1)+c(2)*x",
               "Least squares (with ARMA errors), NLS for explicit formulas; panel LS in panel workfiles "
               "(cx=f|r, per=f).")(_est_cmd("ls"))
    c_tsls = cmd("tsls", "Estimation", "tsls(cov=) y c x1 x2 @ c z1 z2", "Two-stage least squares / IV.")(_est_cmd("tsls"))
    c_gmm = cmd("gmm", "Estimation", "gmm(w=robust|hac) y c x1 x2 @ c z1 z2", "GMM estimation.")(_est_cmd("gmm"))
    c_nls = cmd("nls", "Estimation", "nls y = c(1) + c(2)*x^c(3)", "Nonlinear least squares.")(_est_cmd("nls"))
    c_qreg = cmd("qreg", "Estimation", "qreg(quant=0.5) y c x", "Quantile regression.")(_est_cmd("qreg"))
    c_logit = cmd("logit", "Estimation", "logit y c x", "Binary logit.")(_est_cmd("logit"))
    c_probit = cmd("probit", "Estimation", "probit y c x", "Binary probit.")(_est_cmd("probit"))
    c_binary = cmd("binary", "Estimation", "binary(d=l|p) y c x", "Binary choice (EViews syntax).")(_est_cmd("binary"))
    c_count = cmd("count", "Estimation", "count(d=p|n) y c x", "Poisson / negative binomial count models.",
                  aliases=("poisson", "negbin"))(_est_cmd("count"))
    c_arch = cmd("arch", "Volatility (ARCH)",
                 "arch(p,q, model=garch|tarch|egarch|parch|cgarch|igarch|figarch|harch|ewma, dist=normal|t|ged|skewt, "
                 "archm=sd|var|log, thrsh=1, cov=bw|opg|hessian) y c x [ar(1) ma(1)] [@ variance_regressors]",
                 "Univariate ARCH/GARCH family.", aliases=("garch",))(_est_cmd("arch"))
    c_egarch = cmd("egarch", "Volatility (ARCH)", "egarch(p,o,q, dist=) y c", "EGARCH shortcut.")(_est_cmd("egarch"))
    c_tarch = cmd("tarch", "Volatility (ARCH)", "tarch(p,o,q, dist=) y c", "TARCH / GJR-GARCH shortcut.",
                  aliases=("gjr",))(_est_cmd("tarch"))
    c_parch = cmd("parch", "Volatility (ARCH)", "parch(p,o,q, power=, dist=) y c", "Power ARCH / APARCH shortcut.",
                  aliases=("aparch",))(_est_cmd("parch"))
    c_cgarch = cmd("cgarch", "Volatility (ARCH)", "cgarch(dist=) y c", "Component GARCH shortcut.")(_est_cmd("cgarch"))
    c_igarch = cmd("igarch", "Volatility (ARCH)", "igarch(p,q) y c", "Integrated GARCH shortcut.")(_est_cmd("igarch"))
    c_figarch = cmd("figarch", "Volatility (ARCH)", "figarch(dist=) y c", "Fractionally integrated GARCH.")(_est_cmd("figarch"))
    c_dcc = cmd("dcc", "Volatility (multivariate)",
                "dcc(model=garch, p=1, q=1, dist=normal, mvdist=normal|t, mean=c|zero|ar1) r1 r2 ...",
                "DCC-GARCH (Engle 2002).")(_est_cmd("dcc"))
    c_adcc = cmd("adcc", "Volatility (multivariate)", "adcc(...) r1 r2 ...", "Asymmetric DCC-GARCH.")(_est_cmd("adcc"))
    c_ccc = cmd("ccc", "Volatility (multivariate)", "ccc(...) r1 r2 ...", "Constant conditional correlation GARCH.")(
        _est_cmd("ccc"))
    c_bekk = cmd("bekk", "Volatility (multivariate)", "bekk(type=diagonal|scalar) r1 r2 ...", "BEKK(1,1)-GARCH.")(
        _est_cmd("bekk"))
    c_arima = cmd("arima", "Time series models", "arima(p,d,q[,P,D,Q,s]) y [c] [exog...]", "ARIMA / SARIMA.",
                  aliases=("sarima",))(_est_cmd("arima"))
    c_autoarima = cmd("autoarima", "Time series models", "autoarima(d=0, maxp=4, maxq=4, ic=aic) y [c]",
                      "Automatic ARMA order selection.")(_est_cmd("autoarima"))
    c_ets = cmd("ets", "Time series models", "ets(error=add, trend=none|add|mul, seasonal=none|add|mul, damped=0, periods=) y",
                "Exponential smoothing (ETS state space).", aliases=("smooth",))(_est_cmd("ets"))
    c_vec = cmd("vec", "Time series models", "vec(rank=1, lags=1, det=co|ci|lo|li|n) y1 y2 ...", "Vector error correction model.",
                aliases=("vecm",))(_est_cmd("vec"))
    c_ardl = cmd("ardl", "Time series models", 'ardl(maxlag=4, maxdl=4, ic=aic, fixed="p q1 q2", case=3) y x1 x2',
                 "ARDL with automatic lag selection and Pesaran-Shin-Smith bounds test.")(_est_cmd("ardl"))
    c_switchreg = cmd("switchreg", "Time series models", "switchreg(k=2, ar=0, switchvar=1) y c [x]",
                      "Markov-switching regression.", aliases=("markov",))(_est_cmd("switchreg"))
    c_panel = cmd("panel", "Panel", "panel(method=pooled|fe|te|twoway|re|be|fd|fm, cov=cluster|white|dk) y c x1 x2",
                  "Static panel regression.", aliases=("pls",))(_est_cmd("panel"))
    c_dpd = cmd("dpd", "Panel", 'dpd(lags=1, sys=0, twostep=1, endog="x1", predet="x2", maxlag=99, collapse=0, timedum=1) '
                'y x1 x2', "Dynamic panel GMM: Arellano-Bond (sys=0) or Blundell-Bond system GMM (sys=1).",
                aliases=("abond", "gmmpanel"))(_est_cmd("dpd"))
    c_ff = cmd("ff", "Factor models", "ff(model=capm|ff3|carhart|ff5|ff6, freq=m|d|w, cov=ordinary|white|hac, file=) "
               "asset1 asset2 ...", "Fama-French factor regressions with GRS test (downloads factors).",
               aliases=("famafrench",))(_est_cmd("ff"))
    c_factor = cmd("factor", "Factor models", 'factor(factors="mkt_rf smb hml", rf=rf, excess=0, cov=) asset1 ...',
                   "Factor regressions on factors already in the workfile.")(_est_cmd("factor"))
    c_fmb = cmd("fmb", "Factor models", 'fmb(model=ff3 | factors="...", nw=0, window=0) assets...  |  fmb y c x (panel)',
                "Fama-MacBeth two-pass regressions (Shanken-corrected); panel FMB in panel workfiles.")(_est_cmd("fmb"))
    c_lstm = cmd("lstm", "Machine learning", "lstm(lookback=20, horizon=1, hidden=32, layers=1, epochs=100, lr=0.001, "
                 "batch=32, transform=none|diff|dlog, test=0.2, backend=auto|numpy|torch) y [features...]",
                 "LSTM neural-network forecasting.")(_est_cmd("lstm"))
    c_gru = cmd("gru", "Machine learning", "gru(...) y", "GRU network (requires PyTorch).")(_est_cmd("gru"))
    c_ml = cmd("ml", "Machine learning", "ml(method=mlp|rf|gbm|svr|ridge|lstm, lookback=20, ...) y [features...]",
               "Machine-learning forecasting comparators.")(_est_cmd("ml"))

    @cmd("help", "General", "help [command]", "List commands or show syntax for one command.", aliases=("?",))
    def c_help(self, opts, args):
        a = args.strip().lower()
        r = Report("Finomania command reference")
        if a:
            k = a if a in HELP else next((n for n, fn in self.handlers.items() if n == a), None)
            fn = self.handlers.get(a)
            if fn is not None and getattr(fn, "_cmd", None):
                k = fn._cmd[0]
            if k not in HELP:
                raise CommandError(f"No help for {a}")
            cat, syn, desc = HELP[k]
            r.table(["Command", "Syntax", "Description"], [[k, syn, desc]], ["str", "str", "str"], align=["l", "l", "l"])
            return [r]
        cats: dict[str, list] = {}
        for k, (cat, syn, desc) in HELP.items():
            cats.setdefault(cat, []).append([k, syn, desc])
        for cat, rows in cats.items():
            r.table(["Command", "Syntax", "Description"], sorted(rows), ["str", "str", "str"], align=["l", "l", "l"],
                    title=cat)
        return [r]

    @cmd("run", "General", 'run "program.fprg"', "Run a program file.", aliases=("exec",))
    def c_run(self, opts, args):
        return self.run_file(self._path(args))

    @cmd("logmsg", "General", "logmsg text", "Write a message to the log.", aliases=("statusline",))
    def c_logmsg(self, opts, args):
        return [Message(self._subst(args).strip().strip('"'))]

    @cmd("cd", "General", 'cd "folder"', "Change the default directory.")
    def c_cd(self, opts, args):
        p = self._path(args)
        if not os.path.isdir(p):
            raise CommandError(f"No such folder: {p}")
        self.cwd = p
        return [Message(f"Default directory: {p}")]

    # ---------------------------------------------------------------- helpers
    def _path(self, s: str) -> str:
        s = s.strip().strip('"').strip("'")
        if not s:
            raise CommandError("A file path is required")
        s = os.path.expanduser(s)
        return s if os.path.isabs(s) else os.path.join(self.cwd, s)

    def _export_report(self, rep: Report, path: str) -> str:
        p = self._path(path)
        ext = os.path.splitext(p)[1].lower()
        with open(p, "w", encoding="utf-8") as f:
            f.write(rep.to_csv() if ext == ".csv" else rep.to_latex() if ext == ".tex" else
                    rep.to_text() if ext == ".txt" else rep.to_html())
        return f"Saved {p}"


# ======================================================================= estimation builders

def _cov(pos, kw):
    c = kw.get("cov")
    if c:
        return c
    if "h" in pos or "white" in pos:
        return "white"
    if "n" in pos or "hac" in pos:
        return "hac"
    return "ordinary"


def _b_ls(it, wf, ev, pos, kw, args, name):
    from ..models.regression import LSResult, NLSResult, _split_eq
    if re.search(r"(?<![<>!=])=(?!=)", args) and re.search(r"\bc\s*\(\s*\d+\s*\)", args.lower()):
        return NLSResult(wf, ev, args, _cov(pos, kw), kw.get("c") or kw.get("start"), name, opt_int(kw, "lag"))
    if wf.is_panel:
        cx, per = kw.get("cx", "").lower(), kw.get("per", "").lower()
        method = ("twoway" if cx == "f" and per == "f" else "fe" if cx == "f" else "te" if per == "f" else
                  "re" if cx == "r" else "pooled")
        return _panel(wf, ev, args, method, kw, pos, name)
    return LSResult(wf, ev, args, _cov(pos, kw), opt_int(kw, "lag", opt_int(kw, "hac")), kw.get("cluster"), name,
                    wls=kw.get("w"))


def _panel(wf, ev, args, method, kw, pos, name):
    from ..models.panel import PanelResult
    cov = kw.get("cov") or ("white" if "h" in pos else "ordinary")
    return PanelResult(wf, ev, args, method, cov, opt_int(kw, "lag"), name)


def _b_panel(it, wf, ev, pos, kw, args, name):
    return _panel(wf, ev, args, kw.get("method", pos[0] if pos else "fe"), kw, pos, name)


def _b_tsls(it, wf, ev, pos, kw, args, name):
    from ..models.regression import IVResult
    return IVResult(wf, ev, args, "tsls", _cov(pos, kw), opt_int(kw, "lag"), name)


def _b_gmm(it, wf, ev, pos, kw, args, name):
    from ..models.regression import IVResult
    return IVResult(wf, ev, args, "gmm", _cov(pos, kw), opt_int(kw, "lag"), name, kw.get("w", "robust"))


def _b_nls(it, wf, ev, pos, kw, args, name):
    from ..models.regression import NLSResult
    return NLSResult(wf, ev, args, _cov(pos, kw), kw.get("c") or kw.get("start"), name, opt_int(kw, "lag"))


def _b_qreg(it, wf, ev, pos, kw, args, name):
    from ..models.regression import QuantResult
    return QuantResult(wf, ev, args, opt_float(kw, "quant", opt_float(kw, "tau", 0.5)), kw.get("cov", "robust"), name)


def _b_binary(link):
    def f(it, wf, ev, pos, kw, args, name):
        from ..models.regression import BinaryResult
        lk = link or {"l": "logit", "p": "probit", "x": "probit"}.get(kw.get("d", "p")[0].lower(), "probit")
        return BinaryResult(wf, ev, args, lk, _cov(pos, kw), name)
    return f


def _b_count(it, wf, ev, pos, kw, args, name):
    from ..models.regression import CountResult
    d = kw.get("d", "p").lower()
    return CountResult(wf, ev, args, "negbin" if d.startswith("n") else "poisson", _cov(pos, kw), name)


def _b_arch(default_model):
    def f(it, wf, ev, pos, kw, args, name):
        from ..models.garch import GarchResult
        model = kw.get("model", default_model)
        dist = kw.get("dist", kw.get("tdist", None))
        ints = []
        archm = kw.get("archm")
        for p_ in pos:
            pl = p_.lower()
            if re.fullmatch(r"\d+", pl):
                ints.append(int(pl))
            elif pl in ARCH_MODELS:
                model = pl
            elif pl in DISTS:
                dist = pl
            elif pl in ("archm", "m"):
                archm = archm or "var"
            elif pl in ("b", "bw", "h"):
                kw.setdefault("cov", "bw")
        model = {"garch": "garch"}.get(model, model)
        mdl = model.lower()
        if mdl in ("tarch", "gjr", "egarch", "parch", "aparch"):
            # p, o, q   (EViews: arch(p,q) with thrsh=o)
            if len(ints) >= 3:
                p, o, q = ints[:3]
            elif len(ints) == 2:
                p, q = ints
                o = opt_int(kw, "thrsh", opt_int(kw, "o", 1))
            elif len(ints) == 1:
                p, q, o = ints[0], 1, opt_int(kw, "thrsh", 1)
            else:
                p, o, q = 1, opt_int(kw, "thrsh", 1), 1
        else:
            p = ints[0] if ints else opt_int(kw, "p", 1)
            q = ints[1] if len(ints) > 1 else opt_int(kw, "q", 1 if mdl != "arch" else 0)
            o = opt_int(kw, "thrsh", opt_int(kw, "o", 0))
            if o and mdl == "garch":
                mdl = "tarch"
        return GarchResult(wf, ev, args, mdl, p, q, o, dist or "normal", archm, kw.get("power"), kw.get("cov", "bw"),
                           kw.get("engine", "auto"), name)
    return f


def _b_arima(it, wf, ev, pos, kw, args, name, auto=False):
    from ..models.timeseries import ARIMAResult
    ints = [int(p) for p in pos if re.fullmatch(r"\d+", p)]
    order = tuple((ints + [0, 0, 0])[:3]) if ints else (opt_int(kw, "p", 1), opt_int(kw, "d", 0), opt_int(kw, "q", 0))
    seas = tuple(ints[3:7]) if len(ints) >= 7 else (opt_int(kw, "sp", 0), opt_int(kw, "sd", 0), opt_int(kw, "sq", 0),
                                                    opt_int(kw, "s", 0))
    if auto:
        order = (0, opt_int(kw, "d", 0), 0)
    return ARIMAResult(wf, ev, args, order, seas, kw.get("trend"), name, auto, opt_int(kw, "maxp", 4),
                       opt_int(kw, "maxq", 4), kw.get("ic", "aic"))


def _b_ets(it, wf, ev, pos, kw, args, name):
    from ..models.timeseries import ETSResult
    return ETSResult(wf, ev, args.split()[0], kw.get("error", "add"), kw.get("trend"), kw.get("seasonal"),
                     opt_int(kw, "periods"), opt_bool(kw, "damped"), name)


def _b_var(it, wf, ev, pos, kw, args, name):
    from ..models.timeseries import VARResult
    lagpairs = args.startswith("@lagpairs")
    if lagpairs:
        args = args[len("@lagpairs"):].strip()
    parts = split_at(args)
    toks = split_terms(parts[0])
    lags = None
    if lagpairs:
        ints = []
        while toks and re.fullmatch(r"\d+", toks[0]):
            ints.append(int(toks.pop(0)))
        if len(ints) % 2:
            raise CommandError("VAR lag specification needs pairs, e.g. var v1.ls 1 2 y1 y2")
        lags = list(range(ints[0], ints[-1] + 1)) if ints else [1]
    if lags is None:
        lags = opt_int(kw, "lags", int(pos[0]) if pos and re.fullmatch(r"\d+", pos[0]) else 1)
    exog = split_terms(parts[1]) if len(parts) > 1 else []
    if lagpairs:
        trend = "ct" if "@trend" in [e.lower() for e in exog] else "c" if "c" in [e.lower() for e in exog] else "n"
    else:
        trend = kw.get("trend", "c")
    exog = [e for e in exog if e.lower() not in ("c", "@trend")]
    return VARResult(wf, ev, wf.expand_names(toks), lags, exog, trend, name)


def _b_vec(it, wf, ev, pos, kw, args, name):
    from ..models.timeseries import VECMResult
    toks = split_terms(split_at(args)[0])
    return VECMResult(wf, ev, wf.expand_names(toks), opt_int(kw, "lags", 1), opt_int(kw, "rank", 1),
                      kw.get("det", "co"), name)


def _b_ardl(it, wf, ev, pos, kw, args, name):
    from ..models.timeseries import ARDLResult
    return ARDLResult(wf, ev, args, opt_int(kw, "maxlag", 4), opt_int(kw, "maxdl", 4), kw.get("ic", "aic"),
                      kw.get("fixed"), kw.get("trend", "c"), opt_int(kw, "case", 3), name)


def _b_switch(it, wf, ev, pos, kw, args, name):
    from ..models.timeseries import MarkovResult
    return MarkovResult(wf, ev, args, opt_int(kw, "k", opt_int(kw, "regimes", 2)), opt_int(kw, "ar", 0),
                        opt_bool(kw, "switchvar", True), name)


def _b_mgarch(kind):
    def f(it, wf, ev, pos, kw, args, name):
        from ..models.mgarch import BEKKResult, DCCResult
        names = wf.expand_names(split_terms(args))
        if kind == "bekk":
            return BEKKResult(wf, ev, names, kw.get("type", "diagonal"), kw.get("cov", "bw"), name)
        ints = [int(p) for p in pos if re.fullmatch(r"\d+", p)]
        return DCCResult(wf, ev, names, kind, kw.get("model", "garch"), ints[0] if ints else opt_int(kw, "p", 1),
                         ints[1] if len(ints) > 1 else opt_int(kw, "q", 1), opt_int(kw, "o", 0), kw.get("dist", "normal"),
                         kw.get("mean", "c"), kw.get("mvdist", "normal"), kw.get("cov", "bw"), name)
    return f


def _b_dpd(it, wf, ev, pos, kw, args, name):
    from ..models.panel import DPDResult
    return DPDResult(wf, ev, args, opt_int(kw, "lags", 1), opt_bool(kw, "sys", opt_bool(kw, "system", False)),
                     opt_bool(kw, "twostep", True), (kw.get("endog") or "").split(), (kw.get("predet") or "").split(),
                     opt_int(kw, "maxlag", 99), opt_bool(kw, "collapse"), opt_bool(kw, "timedum", True), name)


def _ensure_factors(it, wf, kw, pos):
    from ..models.factors import MODELS, load_french
    model = (kw.get("model") or next((p for p in pos if p.lower() in MODELS), "ff3")).lower()
    facs = MODELS[model]
    have = all(wf.exists(f) for f in facs + ["rf"])
    if not have or opt_bool(kw, "refresh") or kw.get("file"):
        if not wf.is_dated:
            raise CommandError("Fama-French factors require a dated (daily/weekly/monthly) workfile")
        freq = kw.get("freq") or {"m": "m", "d5": "d", "d7": "d", "w": "w"}.get(wf.freq)
        if freq is None:
            raise CommandError(f"Factor data are not available at the workfile frequency ({wf.freq})")
        F = load_french(model, freq, it._path(kw["file"]) if kw.get("file") else None, opt_bool(kw, "refresh"),
                        kw.get("units", "decimal"))
        from .workfile import Workfile as _W
        dataio.merge_into(wf, _W.from_dataframe(F, freq={"m": "m", "d": "d5", "w": "w"}[freq]))
    return model, facs


def _b_ff(it, wf, ev, pos, kw, args, name):
    from ..models.factors import FactorResult
    model, facs = _ensure_factors(it, wf, kw, pos)
    assets = wf.expand_names(split_terms(args))
    return FactorResult(wf, ev, assets, facs, "rf", opt_bool(kw, "excess"), kw.get("cov", "ordinary"),
                        opt_int(kw, "hac", opt_int(kw, "lag")), model, name)


def _b_factor(it, wf, ev, pos, kw, args, name):
    from ..models.factors import FactorResult
    facs = wf.expand_names((kw.get("factors") or "").split())
    if not facs:
        raise CommandError('factor(factors="f1 f2 ...") assets...')
    return FactorResult(wf, ev, wf.expand_names(split_terms(args)), facs, kw.get("rf"), opt_bool(kw, "excess"),
                        kw.get("cov", "ordinary"), opt_int(kw, "hac"), kw.get("label", "custom"), name)


class _FMBWrap(ModelResult):
    object_type = "factor"
    method_label = "Fama-MacBeth"

    def __init__(self, fr, nw, window):
        self.fr, self.nw, self.window = fr, nw, window
        self.name = ""

    def report(self):
        return self.fr.view_fmb(self.nw, self.window)

    def views(self):
        return [View_("output", "Fama-MacBeth Output", self.report)] + self.fr.views()[1:]


def View_(*a, **k):
    from ..models.base import View
    return View(*a, **k)


def _b_fmb(it, wf, ev, pos, kw, args, name):
    from ..models.factors import FactorResult
    if wf.is_panel:
        return _panel(wf, ev, args, "fm", kw, pos, name)
    if kw.get("factors"):
        facs = wf.expand_names(kw["factors"].split())
        model = "custom"
        rf = kw.get("rf")
    else:
        model, facs = _ensure_factors(it, wf, kw, pos)
        rf = "rf"
    fr = FactorResult(wf, ev, wf.expand_names(split_terms(args)), facs, rf, opt_bool(kw, "excess"), "ordinary", None,
                      model, name)
    return _FMBWrap(fr, opt_int(kw, "nw", 0), opt_int(kw, "window", 0))


def _b_ml(default):
    def f(it, wf, ev, pos, kw, args, name):
        from ..models.ml import MLResult
        toks = split_terms(args)
        if not toks:
            raise CommandError("Specify the target series")
        method = kw.get("method", default)
        prog = None
        if it.progress:
            prog = lambda ep, tot, tl, vl: it.progress(f"Epoch {ep}/{tot}  train MSE {tl:.4g}  val MSE {vl:.4g}",  # noqa
                                                       ep / max(int(tot), 1))
        return MLResult(wf, ev, toks[0], toks[1:], method, opt_int(kw, "lookback", 20), opt_int(kw, "horizon", 1),
                        kw.get("transform", "none"), opt_float(kw, "test", 0.2), opt_float(kw, "val", 0.15),
                        opt_int(kw, "hidden", 32), opt_int(kw, "layers", 1), opt_int(kw, "epochs", 100),
                        opt_int(kw, "batch", 32), opt_float(kw, "lr", 1e-3), opt_int(kw, "patience", 15),
                        opt_int(kw, "seed", 0), kw.get("backend", "auto"), name, prog)
    return f


ESTIMATION_METHODS: dict[str, Callable] = {
    "ls": _b_ls, "tsls": _b_tsls, "gmm": _b_gmm, "nls": _b_nls, "qreg": _b_qreg,
    "logit": _b_binary("logit"), "probit": _b_binary("probit"), "binary": _b_binary(None), "count": _b_count,
    "poisson": _b_count, "negbin": lambda *a: _b_count(*a[:4], {**a[4], "d": "n"}, *a[5:]),
    "arch": _b_arch("garch"), "garch": _b_arch("garch"), "egarch": _b_arch("egarch"), "tarch": _b_arch("tarch"),
    "gjr": _b_arch("tarch"), "parch": _b_arch("parch"), "aparch": _b_arch("parch"), "cgarch": _b_arch("cgarch"),
    "igarch": _b_arch("igarch"), "figarch": _b_arch("figarch"),
    "arima": _b_arima, "sarima": _b_arima, "autoarima": lambda *a: _b_arima(*a, auto=True), "ets": _b_ets,
    "smooth": _b_ets, "var": _b_var, "vec": _b_vec, "vecm": _b_vec, "ardl": _b_ardl, "switchreg": _b_switch,
    "markov": _b_switch, "dcc": _b_mgarch("dcc"), "adcc": _b_mgarch("adcc"), "ccc": _b_mgarch("ccc"),
    "bekk": _b_mgarch("bekk"), "panel": _b_panel, "pls": _b_panel, "dpd": _b_dpd, "abond": _b_dpd,
    "gmmpanel": _b_dpd, "ff": _b_ff, "famafrench": _b_ff, "factor": _b_factor, "fmb": _b_fmb,
    "lstm": _b_ml("lstm"), "gru": _b_ml("gru"), "ml": _b_ml("rf"),
}


# ======================================================================= small utilities

def _strip_comment(line: str) -> str:
    inq = False
    for i, ch in enumerate(line):
        if ch == '"':
            inq = not inq
        elif ch == "'" and not inq:
            return line[:i]
    s = line.lstrip()
    if s.startswith("#") or s.startswith("//"):
        return ""
    return line


def _take_parens(s: str) -> tuple[str | None, str]:
    """If s starts with '(...)' return (inside, remainder)."""
    if not s.startswith("("):
        return None, s
    depth = 0
    inq = False
    for i, ch in enumerate(s):
        if ch == '"':
            inq = not inq
        if inq:
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return s[1:i], s[i + 1:]
    raise CommandError("Unbalanced parentheses in options")


def _split_args(s: str) -> list[str]:
    try:
        return shlex.split(s, posix=False)
    except ValueError:
        return s.split()


def _convert(v, kind):
    if kind == "int":
        return int(float(v))
    if kind == "float":
        return float(v)
    if kind == "bool":
        return str(v).lower() in ("1", "true", "yes", "y", "t", "on")
    if isinstance(v, str):
        return v.strip().strip('"')
    return v


def _as_list(out) -> list:
    if out is None:
        return []
    if isinstance(out, (list, tuple)):
        res = []
        for o in out:
            res.extend(_as_list(o))
        return res
    if isinstance(out, str):
        return [Message(out)]
    return [out]


def _err_text(e: Exception) -> str:
    if isinstance(e, (CommandError, ModelError, WorkfileError, ExprError)):
        return str(e)
    tb = traceback.extract_tb(e.__traceback__)
    where = f" ({tb[-1].name}:{tb[-1].lineno})" if tb else ""
    return f"{type(e).__name__}: {e}{where}"
