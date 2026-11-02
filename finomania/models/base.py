"""Shared machinery for estimated models: views, parameters, design matrices."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import stats

from ..report import FigureOutput, Report


class ModelError(Exception):
    pass


@dataclass
class Param:
    """A user-settable option of a view / proc / estimation dialog."""
    name: str
    label: str
    kind: str = "text"            # text | int | float | bool | choice | series | list
    default: Any = None
    choices: list[str] | None = None
    help: str = ""


@dataclass
class View:
    key: str                      # command name, e.g. 'auto' for eq.auto(4)
    label: str                    # menu label
    func: Callable[..., Any]
    params: list[Param] = field(default_factory=list)
    group: str = ""               # sub-menu
    kind: str = "view"            # view | proc


class ModelResult:
    """Base class.  Subclasses set object_type / method_label and implement
    report(); they may add views() and procs()."""

    object_type = "equation"
    method_label = ""

    # The estimation mask is padded automatically when the workfile range is
    # extended (e.g. by an out-of-sample forecast).
    @property
    def mask(self):
        m = self.__dict__.get("_mask")
        wf = self.__dict__.get("wf")
        if m is not None and wf is not None and len(m) < wf.nobs:
            m = np.concatenate([m, np.zeros(wf.nobs - len(m), dtype=bool)])
            self.__dict__["_mask"] = m
        return m

    @mask.setter
    def mask(self, v):
        self.__dict__["_mask"] = v

    def report(self) -> Report:  # pragma: no cover - abstract
        raise NotImplementedError

    def views(self) -> list[View]:
        return []

    def procs(self) -> list[View]:
        return []

    def all_actions(self) -> dict[str, View]:
        d = {}
        for v in self.views() + self.procs():
            d[v.key] = v
        return d

    def stat(self, key: str):
        raise ModelError(f"@{key} is not available for this object")


# ---------------------------------------------------------------------------
# spec parsing
# ---------------------------------------------------------------------------

def split_terms(spec: str) -> list[str]:
    """Split a list-form specification on whitespace (and commas) at paren depth 0."""
    out, cur, depth, inq = [], [], 0, False
    for ch in spec:
        if ch == '"':
            inq = not inq
        if not inq:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if depth == 0 and (ch.isspace() or ch == ","):
                if cur:
                    out.append("".join(cur))
                    cur = []
                continue
        cur.append(ch)
    if cur:
        out.append("".join(cur))
    return out


def split_at(spec: str) -> list[str]:
    """Split a specification on a standalone '@' (instrument / variance-regressor separator)."""
    parts = re.split(r"(?:(?<=\s)|^)@(?=\s|$)", spec)
    return [p.strip() for p in parts]


_LAGRANGE = re.compile(r"^(.+?)\(\s*([+-]?\d+)\s+to\s+([+-]?\d+)\s*\)$", re.I)
_ARMA = re.compile(r"^(ar|ma|sar|sma)\(\s*(\d+)(?:\s+to\s+(\d+))?\s*\)$", re.I)


def expand_terms(terms: list[str]) -> list[str]:
    out = []
    for t in terms:
        m = _LAGRANGE.match(t)
        if m and not _ARMA.match(t):
            base, a, b = m.group(1), int(m.group(2)), int(m.group(3))
            step = 1 if b >= a else -1
            for k in range(a, b + step, step):
                out.append(f"{base}({k})" if k else base)
        else:
            out.append(t)
    return out


@dataclass
class Design:
    y: pd.Series
    X: pd.DataFrame
    yname: str
    xnames: list[str]
    mask: np.ndarray               # observations used (full-length boolean)
    ar: list[int] = field(default_factory=list)
    ma: list[int] = field(default_factory=list)
    sar: list[int] = field(default_factory=list)
    sma: list[int] = field(default_factory=list)
    has_const: bool = False
    terms: list[str] = field(default_factory=list)

    @property
    def nobs(self) -> int:
        return len(self.y)


def build_design(wf, ev, spec: str, mask: np.ndarray | None = None, extra_series: list[pd.Series] | None = None,
                 allow_arma: bool = True, require_y: bool = True) -> Design:
    terms = expand_terms(split_terms(spec))
    if require_y and not terms:
        raise ModelError("Empty specification")
    ar, ma, sar, sma = [], [], [], []
    regs: list[str] = []
    for t in (terms[1:] if require_y else terms):
        m = _ARMA.match(t)
        if m:
            if not allow_arma:
                raise ModelError(f"{t.upper()} terms are not allowed with this estimator")
            a = int(m.group(2))
            b = int(m.group(3) or a)
            lst = {"ar": ar, "ma": ma, "sar": sar, "sma": sma}[m.group(1).lower()]
            lst.extend(range(a, b + 1))
        else:
            regs.append(t)
    cols: dict[str, pd.Series] = {}
    has_const = False
    for t in regs:
        tl = t.lower()
        if tl == "c":
            cols["C"] = pd.Series(1.0, index=wf.data.index)
            has_const = True
            continue
        m = re.match(r"^@expand\((.+)\)$", tl)
        if m:
            args = [a.strip() for a in m.group(1).split(",")]
            s = ev.eval(args[0], ignore_sample=True, as_series=True)
            levels = sorted(pd.Series(s).dropna().unique(), key=lambda v: (str(type(v)), v))
            drop = any(a in ("@dropfirst", "@droplast") for a in args[1:])
            if "@dropfirst" in args[1:]:
                levels = levels[1:]
            elif "@droplast" in args[1:]:
                levels = levels[:-1]
            for lv in levels:
                nm = f"{args[0].upper()}={lv:g}" if isinstance(lv, (float, int)) else f"{args[0].upper()}={lv}"
                cols[nm] = (s == lv).astype(float).where(s.notna())
            _ = drop
            continue
        cols[t.upper()] = ev.eval(t, ignore_sample=True, as_series=True).astype(float)
    y = None
    yname = ""
    if require_y:
        yname = terms[0].upper()
        y = ev.eval(terms[0], ignore_sample=True, as_series=True).astype(float)
    X = pd.DataFrame(cols, index=wf.data.index)
    base = wf.sample.copy() if mask is None else mask.copy()
    ok = base.copy()
    if y is not None:
        ok &= y.notna().values
    if X.shape[1]:
        ok &= X.notna().all(axis=1).values
    for s in extra_series or []:
        ok &= s.notna().values
    if ok.sum() == 0:
        raise ModelError("No valid observations in the estimation sample")
    d = Design(y=y[ok] if y is not None else None, X=X[ok], yname=yname, xnames=list(X.columns), mask=ok,
               ar=sorted(set(ar)), ma=sorted(set(ma)), sar=sorted(set(sar)), sma=sorted(set(sma)),
               has_const=has_const, terms=terms)
    return d


def contiguous_check(mask: np.ndarray) -> bool:
    idx = np.flatnonzero(mask)
    return len(idx) == 0 or (idx[-1] - idx[0] + 1 == len(idx))


# ---------------------------------------------------------------------------
# option parsing  (e.g. "1,1,egarch,dist=t,archm=var")
# ---------------------------------------------------------------------------

def parse_options(s: str | None) -> tuple[list[str], dict[str, str]]:
    pos: list[str] = []
    kw: dict[str, str] = {}
    if not s:
        return pos, kw
    for part in split_options(s):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
            kw[k.strip().lower()] = v.strip().strip('"')
        else:
            pos.append(part.strip().strip('"'))
    return pos, kw


def split_options(s: str) -> list[str]:
    out, cur, depth = [], [], 0
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return out


def opt_int(kw, key, default=None):
    v = kw.get(key)
    if v is None or v == "":
        return default
    try:
        return int(float(v))
    except ValueError:
        raise ModelError(f"Option {key}= expects an integer, got '{v}'")


def opt_float(kw, key, default=None):
    v = kw.get(key)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except ValueError:
        raise ModelError(f"Option {key}= expects a number, got '{v}'")


def opt_bool(kw, key, default=False):
    v = kw.get(key)
    if v is None:
        return default
    return str(v).lower() in ("1", "true", "yes", "y", "on", "t")


# ---------------------------------------------------------------------------
# statistics helpers
# ---------------------------------------------------------------------------

def info_criteria(logl: float, k: int, n: int) -> tuple[float, float, float]:
    """EViews-style (per observation) Akaike, Schwarz and Hannan-Quinn criteria."""
    aic = -2 * logl / n + 2 * k / n
    sc = -2 * logl / n + k * math.log(n) / n
    hq = -2 * logl / n + 2 * k * math.log(math.log(n)) / n if n > 2 else np.nan
    return aic, sc, hq


def pvalue_t(t, df):
    return 2 * stats.t.sf(np.abs(t), df)


def pvalue_z(z):
    return 2 * stats.norm.sf(np.abs(z))


def numeric_hessian(f: Callable[[np.ndarray], float], x: np.ndarray, rel: float = 1e-4) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    k = len(x)
    h = rel * np.maximum(np.abs(x), 1e-2)
    H = np.zeros((k, k))
    f0 = f(x)
    for i in range(k):
        ei = np.zeros(k)
        ei[i] = h[i]
        fp, fm = f(x + ei), f(x - ei)
        H[i, i] = (fp - 2 * f0 + fm) / h[i] ** 2
        for j in range(i + 1, k):
            ej = np.zeros(k)
            ej[j] = h[j]
            v = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h[i] * h[j])
            H[i, j] = H[j, i] = v
    return H


def numeric_jacobian(f: Callable[[np.ndarray], np.ndarray], x: np.ndarray, rel: float = 1e-5) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    f0 = np.asarray(f(x))
    J = np.zeros((f0.size, len(x)))
    for i in range(len(x)):
        h = rel * max(abs(x[i]), 1e-2)
        e = np.zeros(len(x))
        e[i] = h
        J[:, i] = (np.asarray(f(x + e)) - np.asarray(f(x - e))).ravel() / (2 * h)
    return J


def safe_inv(A: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A)


def jarque_bera(v: np.ndarray) -> tuple[float, float]:
    v = v[~np.isnan(v)]
    n = len(v)
    if n < 3:
        return np.nan, np.nan
    s = stats.skew(v)
    k = stats.kurtosis(v, fisher=False)
    jb = n / 6 * (s ** 2 + (k - 3) ** 2 / 4)
    return float(jb), float(stats.chi2.sf(jb, 2))


def descriptive_block(v: np.ndarray) -> list[tuple[str, Any]]:
    v = v[~np.isnan(v)]
    jb, p = jarque_bera(v)
    return [("Mean", float(np.mean(v))), ("Median", float(np.median(v))), ("Maximum", float(np.max(v))),
            ("Minimum", float(np.min(v))), ("Std. Dev.", float(np.std(v, ddof=1))),
            ("Skewness", float(stats.skew(v))), ("Kurtosis", float(stats.kurtosis(v, fisher=False))),
            ("Jarque-Bera", jb), ("Probability", p), ("Sum", float(np.sum(v))),
            ("Sum Sq. Dev.", float(np.sum((v - v.mean()) ** 2))), ("Observations", int(len(v)))]


def figure_output(fig, title: str) -> FigureOutput:
    return FigureOutput(fig, title)


def series_full(wf, values: np.ndarray | pd.Series, mask: np.ndarray) -> pd.Series:
    """Embed values computed on `mask` observations into a full-length series."""
    out = pd.Series(np.nan, index=wf.data.index)
    v = values.values if isinstance(values, pd.Series) else np.asarray(values)
    if len(mask) < wf.nobs:
        mask = np.concatenate([mask, np.zeros(wf.nobs - len(mask), dtype=bool)])
    out.iloc[np.flatnonzero(mask)] = v
    return out


def sample_header(wf, mask: np.ndarray) -> list[str]:
    lines = [f"Sample (adjusted): {wf.sample_label(mask)}" if not np.array_equal(mask, wf.sample)
             else f"Sample: {wf.sample_label(mask)}",
             f"Included observations: {int(mask.sum())}"
             + ("" if np.array_equal(mask, wf.sample) else " after adjustments")]
    return lines


def wald_from_restrictions(text: str, b: np.ndarray, V: np.ndarray, df_resid: int | None = None,
                           names: list[str] | None = None) -> Report:
    """Wald test of (possibly nonlinear) restrictions like 'c(2)=0, c(3)=2*c(4)'."""
    restr = [r.strip() for r in split_options(text) if r.strip()]
    if not restr:
        raise ModelError("No restrictions given")

    def g(beta):
        vals = []
        for r in restr:
            lhs, rhs = r.split("=", 1) if "=" in r else (r, "0")
            vals.append(_eval_coef_expr(lhs, beta) - _eval_coef_expr(rhs, beta))
        return np.array(vals)

    gv = g(b)
    G = numeric_jacobian(g, b)
    Vg = G @ V @ G.T
    W = float(gv @ safe_inv(Vg) @ gv)
    q = len(restr)
    rep = Report("Wald Test")
    rows = []
    if df_resid:
        rows.append(["F-statistic", W / q, f"({q}, {df_resid})", stats.f.sf(W / q, q, df_resid)])
    rows.append(["Chi-square", W, str(q), stats.chi2.sf(W, q)])
    rep.table(["Test Statistic", "Value", "df", "Probability"], rows, ["str", "num", "str", "prob"])
    rep.text("Null Hypothesis: " + ", ".join(restr).upper())
    se = np.sqrt(np.clip(np.diag(Vg), 0, None))
    rep.table(["Normalized Restriction (= 0)", "Value", "Std. Err."],
              [[restr[i].upper(), gv[i], se[i]] for i in range(q)], ["str", "num", "num"],
              title="Null Hypothesis Summary")
    rep.note("Restrictions are linear in coefficients." if np.allclose(G, numeric_jacobian(g, b * 1.1 + 0.1))
             else "Restrictions are nonlinear in coefficients (delta method).")
    return rep


def _eval_coef_expr(s: str, beta: np.ndarray) -> float:
    s = s.strip().lower().replace("^", "**")
    s = re.sub(r"c\s*\(\s*(\d+)\s*\)", lambda m: f"_b[{int(m.group(1)) - 1}]", s)
    import ast
    tree = ast.parse(s, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name, ast.Load,
                                 ast.Subscript, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub,
                                 ast.UAdd, ast.Call, ast.Index if hasattr(ast, "Index") else ast.Constant)):
            raise ModelError(f"Illegal restriction '{s}'")
    ns = {"_b": beta, "log": np.log, "exp": np.exp, "sqrt": np.sqrt, "abs": abs}
    return float(eval(compile(tree, "<r>", "eval"), {"__builtins__": {}}, ns))  # noqa: S307
