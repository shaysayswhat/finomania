"""
EViews-style expression evaluator.

Supports the familiar syntax, e.g.::

    log(x) - log(x(-1))          lags / leads with x(-k) / x(+k)
    d(y) , d(y,2) , dlog(p)       differences
    @mean(x) , @stdev(x)          sample statistics (respect the current smpl)
    @movav(x,20) , @pch(x)        moving / percentage-change functions
    x>0 and y<>3                  logical operators, '=' tests equality
    @recode(x>0, 1, 0) , @trend   recoding, deterministic terms
    eq1.@r2 , eq1.@coefs(2)       results of estimated objects

Expressions are translated to a Python AST (never ``eval``-ed as raw text),
operating on pandas Series aligned to the workfile index.  Within panels,
lags and differences are taken within each cross-section.
"""
from __future__ import annotations

import ast
import math
import re
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sps

from .dates import PERIODS_PER_YEAR

_STR_RX = re.compile(r'"[^"]*"')
_ID = r"[a-z_][a-z0-9_]*"


class ExprError(Exception):
    pass


class _EstProxy:
    """Lets expressions use ``eq1.@r2`` / ``eq1.@coefs(2)``."""

    def __init__(self, obj):
        self._obj = obj

    def __getattr__(self, item):
        if not item.startswith("_at_"):
            raise ExprError(f"Unknown member '{item}'")
        key = item[4:]
        res = self._obj.result
        if not hasattr(res, "stat"):
            raise ExprError(f"Object {self._obj.name} has no statistics")
        val = res.stat(key)
        if callable(val):
            return val
        return val


class _NS(dict):
    def __init__(self, ev: "Evaluator", extra: dict | None = None):
        super().__init__()
        self.ev = ev
        if extra:
            self.update(extra)

    def __missing__(self, key):
        wf = self.ev.wf
        if key in wf.data.columns:
            col = wf.data[key]
            if pd.api.types.is_numeric_dtype(col):
                return col.astype(float)
            return col
        o = wf.objects.get(key)
        if o is not None:
            t = getattr(o, "type", "")
            if t == "scalar":
                return o.value
            if t in ("matrix", "vector"):
                return o.values
            if hasattr(o, "result") or hasattr(o, "command"):
                return _EstProxy(o)
        if key in self.ev.program_vars:
            return self.ev.program_vars[key]
        raise ExprError(f"{key.upper()} is not defined")


class _Rewriter(ast.NodeTransformer):
    """Turn comparisons / boolean operators into NA-aware helper calls."""

    OPS = {ast.Gt: ">", ast.GtE: ">=", ast.Lt: "<", ast.LtE: "<=", ast.Eq: "==", ast.NotEq: "!="}

    def visit_Compare(self, node):
        self.generic_visit(node)
        left = node.left
        out = None
        for op, right in zip(node.ops, node.comparators):
            sym = self.OPS.get(type(op))
            if sym is None:
                raise ExprError("Unsupported comparison")
            call = ast.Call(func=ast.Name("_cmp", ast.Load()),
                            args=[ast.Constant(sym), left, right], keywords=[])
            out = call if out is None else ast.Call(func=ast.Name("_and", ast.Load()), args=[out, call],
                                                    keywords=[])
            left = right
        return out

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        fn = "_and" if isinstance(node.op, ast.And) else "_or"
        expr = node.values[0]
        for v in node.values[1:]:
            expr = ast.Call(func=ast.Name(fn, ast.Load()), args=[expr, v], keywords=[])
        return expr

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Not):
            return ast.Call(func=ast.Name("_not", ast.Load()), args=[node.operand], keywords=[])
        return node


ALLOWED_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load, ast.Constant,
                 ast.Attribute, ast.Subscript, ast.Tuple, ast.List, ast.Add, ast.Sub, ast.Mult, ast.Div,
                 ast.Pow, ast.Mod, ast.USub, ast.UAdd, ast.FloorDiv, ast.keyword, ast.Compare, ast.BoolOp,
                 ast.And, ast.Or, ast.Not, ast.Gt, ast.GtE, ast.Lt, ast.LtE, ast.Eq, ast.NotEq, ast.Slice)


class Evaluator:
    def __init__(self, wf, program_vars: dict | None = None):
        self.wf = wf
        self.program_vars = program_vars or {}
        self.ignore_sample = False
        self.coef_vector: np.ndarray | None = None   # for NLS c(k)

    # -------------------------------------------------------------- translation
    def translate(self, expr: str, coef_mode: bool = False) -> str:
        strings: list[str] = []

        def _keep(m):
            strings.append(m.group(0))
            return f"__str{len(strings) - 1}__"

        s = _STR_RX.sub(_keep, expr.strip())
        s = s.lower()
        # @functions / data members
        s = re.sub(r"\.@(" + _ID + r")", r"._at_\1", s)
        s = re.sub(r"@(" + _ID + r")\b(?!\s*\()", r"_at_\1()", s)
        s = re.sub(r"@(" + _ID + r")\s*\(", r"_at_\1(", s)
        # restore "._at_x()" data members created by the bare-@ rule
        s = re.sub(r"\._at_(" + _ID + r")\(\)", r"._at_\1", s)
        s = re.sub(r"(?<![\w.])(nrnd|rnd)\b(?!\s*\()", r"\1()", s)
        s = s.replace("<>", "!=")
        s = re.sub(r"(?<![<>!=])=(?!=)", "==", s)
        s = s.replace("^", "**")
        # coefficient references c(k)
        if coef_mode:
            s = re.sub(r"\bc\s*\(\s*(\d+)\s*\)", lambda m: f"_c[{int(m.group(1)) - 1}]", s)
        else:
            s = re.sub(r"\bc\s*\(\s*(\d+)\s*\)", lambda m: f"_cvec({int(m.group(1))})", s)
        # lags / leads: name(-k) where name is not a function
        funcs = set(FUNCTIONS) | {"_lag", "_cvec"}

        def _lagrep(m):
            nm, k = m.group(1), m.group(2).replace(" ", "")
            if nm in funcs or nm.startswith("_at_") or nm.startswith("_c"):
                return m.group(0)
            return f"_lag({nm}, {int(k)})"

        s = re.sub(r"(?<![.\w])(" + _ID + r")\s*\(\s*([+-]?\s*\d+)\s*\)", _lagrep, s)
        for i, st in enumerate(strings):
            s = s.replace(f"__str{i}__", st)
        return s

    def compile(self, expr: str, coef_mode: bool = False):
        py = self.translate(expr, coef_mode)
        try:
            tree = ast.parse(py, mode="eval")
        except SyntaxError as e:
            raise ExprError(f"Syntax error in expression: {expr}") from e
        tree = _Rewriter().visit(tree)
        ast.fix_missing_locations(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ALLOWED_NODES):
                raise ExprError(f"Illegal construct in expression: {type(node).__name__}")
            if isinstance(node, ast.Attribute) and not node.attr.startswith("_at_"):
                raise ExprError(f"Illegal attribute access '{node.attr}'")
            if isinstance(node, ast.Name) and node.id.startswith("__"):
                raise ExprError("Illegal name")
        return compile(tree, "<expr>", "eval")

    # -------------------------------------------------------------- evaluation
    def namespace(self, coef=None) -> _NS:
        ns = _NS(self)
        ns.update(self._functions())
        if coef is not None:
            ns["_c"] = coef
        return ns

    def eval(self, expr: str, ignore_sample: bool = False, coef=None, as_series: bool = False):
        code = self.compile(expr, coef_mode=coef is not None)
        old = self.ignore_sample
        self.ignore_sample = ignore_sample
        try:
            ns = self.namespace(coef)
            val = eval(code, {"__builtins__": {}}, ns)  # noqa: S307 - AST validated above
        except ExprError:
            raise
        except ZeroDivisionError:
            val = np.nan
        except Exception as e:
            raise ExprError(f"Error evaluating '{expr}': {e}") from e
        finally:
            self.ignore_sample = old
        if isinstance(val, pd.Series) and val.dtype == bool:
            val = val.astype(float)
        if as_series:
            return self.to_series(val)
        return val

    def to_series(self, val) -> pd.Series:
        idx = self.wf.data.index
        if isinstance(val, pd.Series):
            if not val.index.equals(idx):
                val = val.reindex(idx)
            return val.astype(float) if pd.api.types.is_numeric_dtype(val) or val.dtype == bool else val
        if isinstance(val, np.ndarray) and val.ndim == 1 and len(val) == len(idx):
            return pd.Series(val, index=idx, dtype=float)
        if isinstance(val, (int, float, np.floating, np.integer, bool, np.bool_)):
            return pd.Series(float(val), index=idx)
        if isinstance(val, str):
            return pd.Series(val, index=idx, dtype=object)
        raise ExprError("Expression does not evaluate to a series or a number")

    # -------------------------------------------------------------- helpers
    def _mask(self):
        if self.ignore_sample:
            return np.ones(self.wf.nobs, dtype=bool)
        return self.wf.sample

    def _ser(self, x) -> pd.Series:
        if isinstance(x, pd.Series):
            return x.astype(float) if x.dtype != object else x
        return pd.Series(float(x), index=self.wf.data.index)

    def _in_sample(self, x) -> np.ndarray:
        s = self._ser(x).values.astype(float)
        m = self._mask()
        v = s[m]
        return v[~np.isnan(v)]

    def _grouped(self, x: pd.Series):
        if self.wf.is_panel:
            return x.groupby(level=0, sort=False, group_keys=False)
        return None

    def _shift(self, x, k: int) -> pd.Series:
        """Lag (k<0) or lead (k>0) in EViews convention x(-1) = previous obs."""
        x = self._ser(x)
        g = self._grouped(x)
        if g is not None:
            return g.shift(-k)
        return x.shift(-k)

    def _functions(self) -> dict[str, Any]:
        ev = self
        wf = self.wf

        def lag(x, k):
            return ev._shift(x, int(k))

        def d(x, n=1, s=0):
            x = ev._ser(x)
            for _ in range(int(n)):
                x = x - ev._shift(x, -1)
            if s:
                x = x - ev._shift(x, -int(s))
            return x

        def dlog(x, n=1, s=0):
            return d(np.log(ev._ser(x).where(ev._ser(x) > 0)), n, s)

        def _rolling(x, n, fn, minp=None):
            x = ev._ser(x)
            n = int(n)
            g = ev._grouped(x)
            if g is not None:
                return g.apply(lambda s: getattr(s.rolling(n, min_periods=minp or n), fn)())
            return getattr(x.rolling(n, min_periods=minp or n), fn)()

        def _cum(x, fn):
            x = ev._ser(x)
            g = ev._grouped(x)
            if g is not None:
                return getattr(g, fn)()
            return getattr(x, fn)()

        def recode(cond, a, b):
            c = ev._ser(cond)
            av, bv = ev._ser(a), ev._ser(b)
            out = pd.Series(np.where(c.values != 0, av.values, bv.values), index=c.index)
            out[c.isna()] = np.nan
            return out

        def nan_replace(x, y):
            x = ev._ser(x)
            return x.fillna(ev._ser(y))

        def scalar_stat(fn):
            def f(x, *args):
                v = ev._in_sample(x)
                if len(v) == 0:
                    return np.nan
                return float(fn(v, *args))
            return f

        def cor(x, y):
            m = ev._mask()
            a, b = ev._ser(x).values[m], ev._ser(y).values[m]
            ok = ~(np.isnan(a) | np.isnan(b))
            return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 1 else np.nan

        def cov(x, y, ddof=0):
            m = ev._mask()
            a, b = ev._ser(x).values[m], ev._ser(y).values[m]
            ok = ~(np.isnan(a) | np.isnan(b))
            return float(np.cov(a[ok], b[ok], ddof=ddof)[0, 1]) if ok.sum() > 1 else np.nan

        def tindex():
            t = wf.time_index()
            return pd.DatetimeIndex(t) if wf.is_dated else None

        def trend(*args):
            if wf.is_panel:
                t = wf.time_index()
                u = {v: i for i, v in enumerate(sorted(set(t)))}
                return pd.Series([float(u[v]) for v in t], index=wf.data.index)
            return pd.Series(np.arange(wf.nobs, dtype=float), index=wf.data.index)

        def datepart(attr):
            def f():
                t = tindex()
                if t is None:
                    return pd.Series(np.nan, index=wf.data.index)
                v = getattr(t, attr)
                if attr == "dayofweek":
                    v = v + 1
                return pd.Series(np.asarray(v, dtype=float), index=wf.data.index)
            return f

        def seas(k):
            t = tindex()
            if t is None:
                raise ExprError("@seas requires a dated workfile")
            v = t.quarter if wf.freq == "q" else t.month
            return pd.Series((np.asarray(v) == int(k)).astype(float), index=wf.data.index)

        def pch(x, k=1):
            x = ev._ser(x)
            return x / ev._shift(x, -int(k)) - 1.0

        def pchy(x):
            return pch(x, PERIODS_PER_YEAR.get(wf.freq, 1))

        def pcha(x):
            p = PERIODS_PER_YEAR.get(wf.freq, 1)
            return (1 + pch(x)) ** p - 1

        def elem(x, date):
            x = ev._ser(x)
            labs = wf.labels()
            date = str(date)
            if date in labs:
                return float(x.iloc[labs.index(date)])
            from .dates import parse_date
            t = parse_date(date, wf.freq)
            ti = wf.time_index()
            pos = np.flatnonzero(pd.DatetimeIndex(ti) == t)
            if len(pos) == 0:
                raise ExprError(f"@elem: date {date} not in workfile")
            return float(x.iloc[pos[0]])

        def demean(x):
            x = ev._ser(x)
            return x - np.nanmean(ev._in_sample(x))

        def standardize(x):
            x = ev._ser(x)
            v = ev._in_sample(x)
            return (x - v.mean()) / v.std(ddof=1)

        def detrend(x):
            x = ev._ser(x)
            t = np.arange(len(x), dtype=float)
            ok = ~np.isnan(x.values) & ev._mask()
            b = np.polyfit(t[ok], x.values[ok], 1)
            return x - (b[0] * t + b[1])

        def meansby(x, g):
            x = ev._ser(x)
            return x.groupby(ev._ser(g).values).transform("mean")

        def crossid():
            if not wf.is_panel:
                return pd.Series(1.0, index=wf.data.index)
            ids = wf.cross_ids()
            u = {v: i + 1 for i, v in enumerate(dict.fromkeys(ids))}
            return pd.Series([float(u[v]) for v in ids], index=wf.data.index)

        def rnd(*_):
            return pd.Series(wf.rng.random(wf.nobs), index=wf.data.index)

        def nrnd(*_):
            return pd.Series(wf.rng.standard_normal(wf.nobs), index=wf.data.index)

        def rtdist(df):
            return pd.Series(wf.rng.standard_t(float(df), wf.nobs), index=wf.data.index)

        def rchisq(df):
            return pd.Series(wf.rng.chisquare(float(df), wf.nobs), index=wf.data.index)

        def cvec(k):
            cv = getattr(wf, "cvec", None)
            if cv is None or k > len(cv):
                return 0.0
            return float(cv[k - 1])

        def ew(fn):
            def f(x, *a):
                if isinstance(x, pd.Series):
                    return fn(x.astype(float), *a)
                return fn(x, *a)
            return f

        def _log(x):
            if isinstance(x, pd.Series):
                x = x.astype(float)
                return np.log(x.where(x > 0))
            return math.log(x) if x > 0 else np.nan

        def _sqrt(x):
            if isinstance(x, pd.Series):
                x = x.astype(float)
                return np.sqrt(x.where(x >= 0))
            return math.sqrt(x) if x >= 0 else np.nan

        def _cnorm(x):
            return _apply(x, sps.norm.cdf)

        def _apply(x, fn):
            if isinstance(x, pd.Series):
                return pd.Series(fn(x.values.astype(float)), index=x.index)
            return float(fn(x))

        f: dict[str, Any] = {
            "_lag": lag, "_cvec": cvec,
            "_cmp": _cmp, "_and": _and, "_or": _or, "_not": _not,
            "log": _log, "ln": _log, "exp": ew(np.exp), "abs": ew(np.abs), "sqrt": _sqrt,
            "sin": ew(np.sin), "cos": ew(np.cos), "tan": ew(np.tan), "atan": ew(np.arctan),
            "d": d, "dlog": dlog, "lag": lag,
            "rnd": rnd, "nrnd": nrnd,
            "_at_log": _log, "_at_exp": ew(np.exp), "_at_abs": ew(np.abs), "_at_sqrt": _sqrt,
            "_at_log10": ew(lambda x: np.log10(x)), "_at_log2": ew(lambda x: np.log2(x)),
            "_at_floor": ew(np.floor), "_at_ceil": ew(np.ceil), "_at_ceiling": ew(np.ceil),
            "_at_round": ew(lambda x, k=0: np.round(x, int(k))), "_at_inv": ew(lambda x: 1.0 / x),
            "_at_logit": ew(lambda x: 1.0 / (1.0 + np.exp(-x))), "_at_sign": ew(np.sign),
            "_at_cnorm": _cnorm, "_at_dnorm": lambda x: _apply(x, sps.norm.pdf),
            "_at_qnorm": lambda x: _apply(x, sps.norm.ppf),
            "_at_isna": lambda x: ev._ser(x).isna().astype(float), "_at_nan": nan_replace,
            "_at_recode": recode, "_at_iif": recode,
            "_at_d": d, "_at_dlog": dlog, "_at_lag": lag,
            "_at_pch": pch, "_at_pchy": pchy, "_at_pcha": pcha,
            "_at_mean": scalar_stat(np.mean), "_at_sum": scalar_stat(np.sum),
            "_at_stdev": scalar_stat(lambda v: v.std(ddof=1)), "_at_stdevp": scalar_stat(lambda v: v.std(ddof=0)),
            "_at_stdevs": scalar_stat(lambda v: v.std(ddof=1)),
            "_at_var": scalar_stat(lambda v: v.var(ddof=0)), "_at_varp": scalar_stat(lambda v: v.var(ddof=0)),
            "_at_vars": scalar_stat(lambda v: v.var(ddof=1)),
            "_at_min": scalar_stat(np.min), "_at_max": scalar_stat(np.max), "_at_median": scalar_stat(np.median),
            "_at_quantile": scalar_stat(lambda v, q: np.quantile(v, float(q))),
            "_at_obs": scalar_stat(len), "_at_sumsq": scalar_stat(lambda v: np.sum(v ** 2)),
            "_at_skew": scalar_stat(lambda v: sps.skew(v)), "_at_kurt": scalar_stat(lambda v: sps.kurtosis(v, fisher=False)),
            "_at_nas": lambda x: float(np.isnan(ev._ser(x).values[ev._mask()].astype(float)).sum()),
            "_at_cor": cor, "_at_cov": cov, "_at_covs": lambda x, y: cov(x, y, 1),
            "_at_movav": lambda x, n: _rolling(x, n, "mean"), "_at_movsum": lambda x, n: _rolling(x, n, "sum"),
            "_at_movstdev": lambda x, n: _rolling(x, n, "std"), "_at_movvar": lambda x, n: _rolling(x, n, "var"),
            "_at_movmin": lambda x, n: _rolling(x, n, "min"), "_at_movmax": lambda x, n: _rolling(x, n, "max"),
            "_at_mav": lambda x, n: _rolling(x, n, "mean"), "_at_msum": lambda x, n: _rolling(x, n, "sum"),
            "_at_mstdev": lambda x, n: _rolling(x, n, "std"),
            "_at_ewma": lambda x, lam=0.94: ev._ser(x).ewm(alpha=1 - float(lam), adjust=False).mean(),
            "_at_cumsum": lambda x: _cum(x, "cumsum"), "_at_cumprod": lambda x: _cum(x, "cumprod"),
            "_at_cummax": lambda x: _cum(x, "cummax"), "_at_cummin": lambda x: _cum(x, "cummin"),
            "_at_trend": trend, "_at_obsnum": lambda: pd.Series(np.arange(1, wf.nobs + 1, dtype=float),
                                                                index=wf.data.index),
            "_at_year": datepart("year"), "_at_month": datepart("month"), "_at_quarter": datepart("quarter"),
            "_at_day": datepart("day"), "_at_weekday": datepart("dayofweek"), "_at_seas": seas,
            "_at_elem": elem, "_at_demean": demean, "_at_detrend": detrend,
            "_at_standardize": standardize, "_at_zscore": standardize, "_at_meansby": meansby,
            "_at_crossid": crossid, "_at_rnd": rnd, "_at_nrnd": nrnd, "_at_rtdist": rtdist, "_at_rchisq": rchisq,
            "_at_str": lambda x, *a: (str(int(x)) if isinstance(x, (int, float)) and float(x).is_integer()
                                      else str(x)),
            "_at_val": lambda x: float(x), "_at_upper": lambda x: str(x).upper(), "_at_lower": lambda x: str(x).lower(),
            "_at_len": lambda x: float(len(str(x))),
            "_at_pi": lambda: math.pi, "_at_e": lambda: math.e, "_at_na": lambda: np.nan, "na": np.nan,
            "nan": np.nan,
        }
        return f


def _isnan(x):
    if isinstance(x, pd.Series):
        return x.isna().values
    if isinstance(x, np.ndarray):
        return np.isnan(x)
    try:
        return math.isnan(x)
    except TypeError:
        return False


def _vals(x):
    if isinstance(x, pd.Series):
        return x.values
    return x


def _wrap(res, a, b):
    idx = a.index if isinstance(a, pd.Series) else b.index if isinstance(b, pd.Series) else None
    if idx is None:
        return res
    return pd.Series(res, index=idx)


def _cmp(op, a, b):
    av, bv = _vals(a), _vals(b)
    if isinstance(av, np.ndarray) and av.dtype == object or isinstance(bv, np.ndarray) and bv.dtype == object \
            or isinstance(av, str) or isinstance(bv, str):
        av_ = np.asarray(av, dtype=object)
        bv_ = np.asarray(bv, dtype=object)
        r = {"==": av_ == bv_, "!=": av_ != bv_, ">": av_ > bv_, ">=": av_ >= bv_, "<": av_ < bv_,
             "<=": av_ <= bv_}[op]
        return _wrap(np.asarray(r, dtype=float), a, b)
    with np.errstate(invalid="ignore"):
        av = np.asarray(av, dtype=float)
        bv = np.asarray(bv, dtype=float)
        r = {"==": av == bv, "!=": av != bv, ">": av > bv, ">=": av >= bv, "<": av < bv, "<=": av <= bv}[op]
        r = np.asarray(r, dtype=float)
        nanm = np.isnan(av) | np.isnan(bv)
        if np.ndim(r):
            r = np.where(nanm, np.nan, r)
        elif nanm:
            r = np.nan
    return _wrap(r, a, b)


def _and(a, b):
    av, bv = np.asarray(_vals(a), dtype=float), np.asarray(_vals(b), dtype=float)
    r = ((av != 0) & (bv != 0)).astype(float)
    r = np.where(np.isnan(av) | np.isnan(bv), np.nan, r)
    return _wrap(r, a, b) if np.ndim(r) else float(r)


def _or(a, b):
    av, bv = np.asarray(_vals(a), dtype=float), np.asarray(_vals(b), dtype=float)
    r = (((av != 0) & ~np.isnan(av)) | ((bv != 0) & ~np.isnan(bv))).astype(float)
    return _wrap(r, a, b) if np.ndim(r) else float(r)


def _not(a):
    av = np.asarray(_vals(a), dtype=float)
    r = np.where(np.isnan(av), np.nan, (av == 0).astype(float))
    return _wrap(r, a, a) if np.ndim(r) else float(r)


FUNCTIONS = {"log", "ln", "exp", "abs", "sqrt", "sin", "cos", "tan", "atan", "d", "dlog", "lag", "rnd", "nrnd",
             "_cmp", "_and", "_or", "_not"}
