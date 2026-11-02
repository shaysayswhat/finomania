"""
The Workfile: Finomania's equivalent of an EViews workfile.

A workfile holds

* a data frame of series (rows = observations, columns = series) whose index is
  a DatetimeIndex (dated), a RangeIndex (undated) or a MultiIndex
  (cross-section id, period) for panels;
* named non-series objects (groups, equations, tables, graphs, ...);
* the current sample (``smpl``), a boolean mask over the observations.
"""
from __future__ import annotations

import json
import os
import re
import zipfile
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from ..report import Report
from . import dates as D
from .objects import (OBJECT_TYPES, Estimation, Graph, Group, Matrix, Scalar, Table,
                      WFObject)

RESERVED = {"c", "na", "and", "or", "not", "if", "then", "else", "to", "step", "next", "endif",
            "for", "while", "wend", "nan", "true", "false"}
NAME_RX = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class WorkfileError(Exception):
    pass


class Workfile:
    def __init__(self, name: str = "untitled", data: pd.DataFrame | None = None, freq: str = "u"):
        self.name = name
        self.freq = D.norm_freq(freq)
        self.data = data if data is not None else pd.DataFrame(index=pd.RangeIndex(1, 1))
        self.objects: dict[str, WFObject] = {}
        self.sample_text = "@all"
        self._sample = np.ones(len(self.data), dtype=bool)
        self.path: str | None = None
        self.dirty = False
        self._listeners: list[Callable[[str], None]] = []
        self.seed: int | None = None
        self.rng = np.random.default_rng()

    # ------------------------------------------------------------------ creation
    @classmethod
    def create(cls, freq: str, start: str | int | None = None, end: str | int | None = None,
               name: str = "untitled", ncross: int | None = None) -> "Workfile":
        freq = D.norm_freq(freq)
        if freq == "u":
            n = int(start if end is None else int(end) - int(start) + 1)
            idx = pd.RangeIndex(1, n + 1, name="obs")
        else:
            idx = D.make_range(freq, str(start), str(end))
            idx.name = "date"
        if ncross:
            ids = [f"{i + 1}" for i in range(int(ncross))]
            idx = pd.MultiIndex.from_product([ids, idx], names=["crossid", "dateid" if freq != "u" else "period"])
        wf = cls(name, pd.DataFrame(index=idx), freq)
        return wf

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, name: str = "untitled", freq: str | None = None,
                       panel: tuple[str, str] | None = None) -> "Workfile":
        df = df.copy()
        if panel:
            idc, tc = panel
            df[tc] = _maybe_dates(df[tc])
            df = df.set_index([idc, tc]).sort_index()
            df.index.names = [str(idc).lower(), str(tc).lower()]
            tlev = df.index.get_level_values(1)
            if isinstance(tlev, pd.DatetimeIndex):
                f = freq or D.infer_freq(pd.DatetimeIndex(sorted(set(tlev))))
                if f in ("a", "s", "q", "m"):
                    per = {"a": "Y", "s": "6M", "q": "Q", "m": "M"}[f]
                    newt = tlev.to_period(per if f != "s" else "M").to_timestamp()
                    df.index = pd.MultiIndex.from_arrays([df.index.get_level_values(0), newt], names=df.index.names)
            else:
                f = "u"
            wf = cls(name, pd.DataFrame(index=df.index), f)
        else:
            if isinstance(df.index, pd.DatetimeIndex):
                idx = df.index
            else:
                idx = None
            if idx is not None:
                f = D.norm_freq(freq) if freq else D.infer_freq(idx)
                df = df[~df.index.isna()].sort_index()
                if f in ("a", "s", "q", "m"):
                    per = {"a": "Y", "q": "Q", "m": "M", "s": "M"}[f]
                    df.index = df.index.to_period(per).to_timestamp()
                    df = df[~df.index.duplicated(keep="last")]
                df.index.name = "date"
                wf = cls(name, pd.DataFrame(index=df.index), f)
            else:
                df.index = pd.RangeIndex(1, len(df) + 1, name="obs")
                wf = cls(name, pd.DataFrame(index=df.index), "u")
        wf._reset_sample()
        for c in df.columns:
            nm = sanitize_name(str(c))
            col = df[c]
            if not pd.api.types.is_numeric_dtype(col):
                conv = pd.to_numeric(col, errors="coerce")
                if conv.notna().sum() >= 0.8 * col.notna().sum() and col.notna().sum() > 0:
                    col = conv
            if pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col):
                wf.data[nm] = col.astype(float).values
            else:
                wf.data[nm] = col.astype(object).values
        return wf

    # ------------------------------------------------------------------ structure
    @property
    def index(self):
        return self.data.index

    @property
    def nobs(self) -> int:
        return len(self.data)

    @property
    def is_panel(self) -> bool:
        return isinstance(self.data.index, pd.MultiIndex)

    @property
    def is_dated(self) -> bool:
        return self.freq != "u"

    def time_index(self):
        return self.data.index.get_level_values(1) if self.is_panel else self.data.index

    def cross_ids(self):
        return self.data.index.get_level_values(0) if self.is_panel else None

    def labels(self) -> list[str]:
        return D.obs_labels(self.data.index, self.freq)

    def range_text(self) -> str:
        if self.nobs == 0:
            return "(empty)"
        if self.is_panel:
            t = sorted(set(self.time_index()))
            ncs = len(set(self.cross_ids()))
            a, b = (D.obs_label(t[0], self.freq), D.obs_label(t[-1], self.freq)) if self.is_dated else (t[0], t[-1])
            return f"{a} {b}  x {ncs} cross-sections  ({self.nobs} obs)"
        lab = self.labels()
        return f"{lab[0]} {lab[-1]}  ({self.nobs} obs)"

    def describe(self) -> str:
        kind = "Panel" if self.is_panel else D.FREQ_NAMES.get(self.freq, self.freq)
        return f"Workfile {self.name.upper()} — {kind} — Range: {self.range_text()} — Sample: {self.sample_text}"

    # ------------------------------------------------------------------ listeners
    def on_change(self, cb: Callable[[str], None]) -> None:
        self._listeners.append(cb)

    def notify(self, what: str = "objects") -> None:
        self.dirty = True
        for cb in list(self._listeners):
            try:
                cb(what)
            except Exception:  # pragma: no cover - GUI callbacks must never break the engine
                pass

    # ------------------------------------------------------------------ names
    def series_names(self) -> list[str]:
        return [c for c in self.data.columns]

    def all_names(self) -> list[str]:
        return sorted(set(self.series_names()) | set(self.objects))

    def obj_type(self, name: str) -> str | None:
        name = name.lower()
        if name in self.data.columns:
            return "series" if pd.api.types.is_numeric_dtype(self.data[name]) else "alpha"
        o = self.objects.get(name)
        return o.type if o is not None else None

    def exists(self, name: str) -> bool:
        return self.obj_type(name) is not None

    def listing(self) -> list[tuple[str, str]]:
        out = [(n, self.obj_type(n)) for n in self.all_names()]
        return out

    def check_name(self, name: str) -> str:
        name = name.lower()
        if not NAME_RX.match(name):
            raise WorkfileError(f"Illegal object name '{name}'.")
        if name in RESERVED:
            raise WorkfileError(f"'{name}' is a reserved name.")
        return name

    # ------------------------------------------------------------------ series
    def series(self, name: str) -> pd.Series:
        name = name.lower()
        if name not in self.data.columns:
            raise WorkfileError(f"{name.upper()} is not defined (no such series).")
        return self.data[name]

    def __getitem__(self, name: str) -> pd.Series:
        return self.series(name)

    def __contains__(self, name: str) -> bool:
        return self.exists(name)

    def set_series(self, name: str, values, respect_sample: bool = True, notify: bool = True) -> None:
        name = self.check_name(name)
        if name in self.objects:
            raise WorkfileError(f"{name.upper()} already exists as a {self.objects[name].type}.")
        if isinstance(values, pd.Series):
            if values.index.equals(self.data.index):
                vals = values.values
            else:
                vals = values.reindex(self.data.index).values
        else:
            vals = np.asarray(values)
            if vals.ndim == 0:
                vals = np.full(self.nobs, vals.item() if hasattr(vals, "item") else vals)
            if len(vals) != self.nobs:
                raise WorkfileError(f"Length mismatch assigning {name}: {len(vals)} vs {self.nobs} obs.")
        is_alpha = vals.dtype == object and not _all_numeric(vals)
        if not is_alpha:
            vals = np.asarray(vals, dtype=float)
        if respect_sample and not self._sample.all():
            if name in self.data.columns:
                old = self.data[name].values.copy()
            else:
                old = np.full(self.nobs, np.nan if not is_alpha else None, dtype=float if not is_alpha else object)
            if not is_alpha and old.dtype != float:
                old = pd.to_numeric(pd.Series(old), errors="coerce").values
            old[self._sample] = vals[self._sample]
            vals = old
        self.data[name] = vals
        if notify:
            self.notify("series")

    def delete(self, names: Iterable[str]) -> list[str]:
        gone = []
        for n in names:
            n = n.lower()
            if any(ch in n for ch in "*?"):
                rx = re.compile("^" + n.replace("*", ".*").replace("?", ".") + "$")
                targets = [x for x in self.all_names() if rx.match(x)]
            else:
                targets = [n]
            for t in targets:
                if t in self.data.columns:
                    self.data = self.data.drop(columns=[t])
                    gone.append(t)
                elif t in self.objects:
                    del self.objects[t]
                    gone.append(t)
                else:
                    raise WorkfileError(f"{t.upper()} does not exist.")
        self.notify()
        return gone

    def rename(self, old: str, new: str) -> None:
        old, new = old.lower(), self.check_name(new)
        if self.exists(new):
            raise WorkfileError(f"{new.upper()} already exists.")
        if old in self.data.columns:
            self.data = self.data.rename(columns={old: new})
        elif old in self.objects:
            o = self.objects.pop(old)
            o.name = new
            self.objects[new] = o
        else:
            raise WorkfileError(f"{old.upper()} does not exist.")
        for o in self.objects.values():
            if isinstance(o, Group):
                o.members = [new if m == old else m for m in o.members]
        self.notify()

    def copy_object(self, src: str, dst: str) -> None:
        src, dst = src.lower(), self.check_name(dst)
        if src in self.data.columns:
            self.data[dst] = self.data[src].copy()
        elif src in self.objects:
            import copy
            o = copy.copy(self.objects[src])
            o.name = dst
            self.objects[dst] = o
        else:
            raise WorkfileError(f"{src.upper()} does not exist.")
        self.notify()

    # ------------------------------------------------------------------ objects
    def add(self, obj: WFObject, replace: bool = True) -> WFObject:
        name = self.check_name(obj.name)
        obj.name = name
        if name in self.data.columns:
            raise WorkfileError(f"{name.upper()} already exists as a series.")
        if name in self.objects and not replace:
            raise WorkfileError(f"{name.upper()} already exists.")
        self.objects[name] = obj
        self.notify()
        return obj

    def get(self, name: str) -> WFObject:
        name = name.lower()
        if name in self.objects:
            return self.objects[name]
        if name in self.data.columns:
            return name  # type: ignore[return-value]
        raise WorkfileError(f"{name.upper()} does not exist.")

    def group_members(self, name: str) -> list[str]:
        o = self.objects.get(name.lower())
        if isinstance(o, Group):
            return list(o.members)
        raise WorkfileError(f"{name.upper()} is not a group.")

    def expand_names(self, names: Iterable[str]) -> list[str]:
        """Expand group names and wildcards into series names (order kept)."""
        out: list[str] = []
        for n in names:
            n = n.lower()
            if any(ch in n for ch in "*?"):
                rx = re.compile("^" + n.replace("*", ".*").replace("?", ".") + "$")
                out.extend(x for x in self.series_names() if rx.match(x))
            elif isinstance(self.objects.get(n), Group):
                out.extend(self.objects[n].members)
            else:
                out.append(n)
        return out

    def unique_name(self, prefix: str) -> str:
        prefix = prefix.lower()
        i = 1
        while self.exists(f"{prefix}{i:02d}"):
            i += 1
        return f"{prefix}{i:02d}"

    # ------------------------------------------------------------------ sample
    def _reset_sample(self) -> None:
        self._sample = np.ones(self.nobs, dtype=bool)
        self.sample_text = "@all"

    @property
    def sample(self) -> np.ndarray:
        if len(self._sample) != self.nobs:
            self._reset_sample()
        return self._sample

    def set_sample(self, text: str, evaluator=None) -> None:
        self._sample = self.parse_sample(text, evaluator)
        self.sample_text = " ".join(text.split()) if text.strip() else "@all"
        self.notify("sample")

    def parse_sample(self, text: str, evaluator=None) -> np.ndarray:
        """Parse 'smpl' syntax: pairs of dates (or @first/@last/@all) optionally
        followed by 'if <condition>'."""
        text = (text or "").strip()
        if not text or text.lower() == "@all":
            return np.ones(self.nobs, dtype=bool)
        cond = None
        m = re.search(r"\bif\b", text, flags=re.I)
        if m:
            cond = text[m.end():].strip()
            text = text[:m.start()].strip()
        toks = text.split()
        if not toks or (len(toks) == 1 and toks[0].lower() == "@all"):
            mask = np.ones(self.nobs, dtype=bool)
        else:
            if len(toks) % 2:
                raise WorkfileError("smpl requires pairs of start/end dates.")
            mask = np.zeros(self.nobs, dtype=bool)
            for a, b in zip(toks[0::2], toks[1::2]):
                mask |= self._range_mask(a, b)
        if cond:
            if evaluator is None:
                from .expr import Evaluator
                evaluator = Evaluator(self)
            v = evaluator.eval(cond, ignore_sample=True)
            v = np.asarray(pd.Series(v).reindex(self.data.index).values if isinstance(v, pd.Series) else v,
                           dtype=float)
            mask &= np.nan_to_num(v, nan=0.0) != 0
        return mask

    def _range_mask(self, a: str, b: str) -> np.ndarray:
        t = self.time_index()
        if self.is_dated:
            tt = pd.DatetimeIndex(t)
            uniq = tt.unique().sort_values()
            lo = _offset_token(a, uniq, self.freq, end=False)
            hi = _offset_token(b, uniq, self.freq, end=True)
            return np.asarray((tt >= lo) & (tt <= hi))
        pos = np.asarray(t, dtype=float) if self.is_panel else np.arange(1, self.nobs + 1, dtype=float)
        vals = np.unique(pos)
        lo = vals[0] if a.lower() == "@first" else _offset_num(a, vals)
        hi = vals[-1] if b.lower() == "@last" else _offset_num(b, vals)
        return (pos >= lo) & (pos <= hi)

    def sample_label(self, mask: np.ndarray | None = None) -> str:
        mask = self.sample if mask is None else mask
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            return "(empty)"
        if self.is_panel:
            t = self.time_index()[idx]
            ts = sorted(set(t))
            a, b = ts[0], ts[-1]
            if self.is_dated:
                a, b = D.obs_label(a, self.freq), D.obs_label(b, self.freq)
            return f"{a} {b}"
        lab = self.labels()
        return f"{lab[idx[0]]} {lab[idx[-1]]}"

    # ------------------------------------------------------------------ range
    def expand(self, end: str) -> int:
        """Extend the range of a dated/undated (non-panel) workfile."""
        if self.is_panel:
            raise WorkfileError("Range expansion of panel workfiles is not supported.")
        if self.is_dated:
            e = D.parse_date(end, self.freq, end=True)
            idx = pd.DatetimeIndex(self.data.index)
            n = 0
            new = idx
            while new[-1] < e - pd.Timedelta(days=0) and n < 100000:
                new = D.extend_index(idx, self.freq, n + 1)
                if new[-1] > e:
                    new = new[:-1]
                    break
                n += 1
            added = len(new) - len(idx)
        else:
            n = int(end) - self.nobs
            new = pd.RangeIndex(1, self.nobs + max(n, 0) + 1, name="obs")
            added = len(new) - self.nobs
        if added > 0:
            self.data = self.data.reindex(new)
            if self.is_dated:
                self.data.index.name = "date"
            self._sample = np.concatenate([self._sample, np.zeros(added, dtype=bool)])
            if self.sample_text == "@all":
                self._sample[:] = True
            self.notify("range")
        return added

    # ------------------------------------------------------------------ persistence
    def save(self, path: str) -> None:
        if not path.lower().endswith(".fwf"):
            path += ".fwf"
        meta: dict[str, Any] = {
            "format": "finomania-workfile", "version": 1, "name": self.name, "freq": self.freq,
            "panel": self.is_panel, "sample": self.sample_text,
            "index_names": list(self.data.index.names),
            "alpha": [c for c in self.data.columns if not pd.api.types.is_numeric_dtype(self.data[c])],
            "objects": [],
        }
        for o in self.objects.values():
            try:
                meta["objects"].append(o.to_dict())
            except Exception as e:  # pragma: no cover
                meta["objects"].append({"type": "table", "name": o.name,
                                        "report": Report(f"{o.name}: could not be saved ({e})").to_dict()})
        df = self.data.reset_index()
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("meta.json", json.dumps(meta, indent=1, default=str))
            z.writestr("data.csv", df.to_csv(index=False, float_format="%.17g"))
        self.path = path
        self.dirty = False

    @classmethod
    def load(cls, path: str) -> "Workfile":
        with zipfile.ZipFile(path) as z:
            meta = json.loads(z.read("meta.json"))
            import io
            df = pd.read_csv(io.BytesIO(z.read("data.csv")), low_memory=False)
        names = meta["index_names"]
        freq = meta["freq"]
        for n in names:
            if freq != "u" and n in ("date", "dateid") or (meta["panel"] and n == names[-1] and freq != "u"):
                df[n] = pd.to_datetime(df[n])
        df = df.set_index(names if meta["panel"] else names[0])
        wf = cls(meta.get("name", "untitled"), pd.DataFrame(index=df.index), freq)
        for c in df.columns:
            if c in meta.get("alpha", []):
                wf.data[c] = df[c].astype(object).values
            else:
                wf.data[c] = pd.to_numeric(df[c], errors="coerce").astype(float).values
        wf._reset_sample()
        for od in meta.get("objects", []):
            t = od.get("type")
            nm = od["name"]
            if t == "group":
                wf.objects[nm] = Group(nm, od["members"])
            elif t == "scalar":
                wf.objects[nm] = Scalar(nm, od["value"])
            elif t in ("matrix", "vector"):
                m = Matrix(nm, od["values"], od.get("rownames"), od.get("colnames"))
                m.type = od.get("mtype", t)
                wf.objects[nm] = m
            elif t == "table":
                wf.objects[nm] = Table(nm, Report.from_dict(od["report"]))
            elif t == "graph":
                from ..report import _PngFigure
                wf.objects[nm] = Graph(nm, _PngFigure(od["png"]), od.get("spec"))
            else:  # estimated objects -> lazily re-estimated from their command
                wf.objects[nm] = LazyEstimation(nm, od, wf)
        try:
            wf.set_sample(meta.get("sample", "@all"))
        except Exception:
            wf._reset_sample()
        wf.path = path
        wf.dirty = False
        return wf


class LazyEstimation(Estimation):
    """An estimated object restored from disk.  The stored output is shown
    immediately; the model is re-estimated from its command on first use."""

    def __init__(self, name: str, d: dict, wf: Workfile):
        WFObject.__init__(self, name)
        self.command = d.get("command", "")
        self.sample = d.get("sample", "@all")
        self.type = d.get("otype", "equation")
        self._stored = Report.from_dict(d["report"]) if d.get("report") else None
        self._wf = wf
        self._result = None

    @property
    def result(self):
        if self._result is None:
            from .commands import Interpreter
            it = Interpreter(self._wf)
            saved = (self._wf.sample_text, self._wf.sample.copy())
            try:
                self._wf._sample = self._wf.parse_sample(self.sample)
                self._result = it.estimate_command(self.command, self.name)
            finally:
                self._wf.sample_text, self._wf._sample = saved
        return self._result

    @result.setter
    def result(self, v):
        self._result = v

    def report(self) -> Report:
        if self._result is None and self._stored is not None:
            return self._stored
        return self.result.report()


# ---------------------------------------------------------------------- helpers

def sanitize_name(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]", "_", s.strip()).lower()
    s = re.sub(r"_+", "_", s).strip("_") or "x"
    if s[0].isdigit():
        s = "x" + s
    if s in RESERVED:
        s = s + "_"
    return s


def _all_numeric(vals) -> bool:
    try:
        np.asarray(vals, dtype=float)
        return True
    except (TypeError, ValueError):
        return False


def _maybe_dates(col: pd.Series):
    if pd.api.types.is_numeric_dtype(col):
        v = col.dropna()
        if len(v) and (v.between(1000, 2999)).all() and (v == v.round()).all():
            return pd.to_datetime(col.astype("Int64").astype(str), format="%Y", errors="coerce")
        return col
    try:
        return pd.to_datetime(col)
    except (ValueError, TypeError):
        return col


_RX_OFF = re.compile(r"^(.*?)([+-]\d+)$")


def _offset_token(tok: str, uniq: pd.DatetimeIndex, freq: str, end: bool):
    """Resolve @first/@last/date tokens with optional +k/-k observation offsets."""
    tok = tok.strip()
    off = 0
    m = _RX_OFF.match(tok)
    if m and (m.group(1).lower().startswith("@") or re.match(r"^\d{4}([qQmMsS:]\d+)?$", m.group(1))):
        tok, off = m.group(1), int(m.group(2))
    low = tok.lower()
    if low == "@first":
        pos = 0
    elif low == "@last":
        pos = len(uniq) - 1
    else:
        d = D.parse_date(tok, freq, end=end)
        if off == 0:
            return d
        pos = int(np.searchsorted(uniq, d, side="right" if end else "left"))
        pos = pos - 1 if end else pos
    pos = int(np.clip(pos + off, 0, len(uniq) - 1))
    if end:
        return uniq[pos] + pd.Timedelta(hours=23, minutes=59)
    return uniq[pos]


def _offset_num(tok: str, vals: np.ndarray) -> float:
    tok = tok.strip()
    m = _RX_OFF.match(tok)
    if m and m.group(1).lower() in ("@first", "@last"):
        base = vals[0] if m.group(1).lower() == "@first" else vals[-1]
        return base + int(m.group(2))
    return float(tok)
