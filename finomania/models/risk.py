"""Risk management: Value-at-Risk / Expected Shortfall, VaR backtesting
(Kupiec, Christoffersen, Basel traffic light) and performance analytics."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import optimize, stats

from ..core.dates import PERIODS_PER_YEAR
from ..report import FigureOutput, Report, fmt_num
from . import plots
from .base import ModelError


def _ret(wf, ev, expr):
    s = ev.eval(expr, ignore_sample=True, as_series=True)
    m = wf.sample & s.notna().values
    return s[m], m


def _var_es(r: np.ndarray, a: float, method: str, ewma_lambda=0.94):
    """One-period VaR and ES (positive numbers = losses) at tail probability a."""
    if method == "historical":
        q = np.quantile(r, a)
        es = -r[r <= q].mean() if (r <= q).any() else -q
        return -q, es
    mu, sd = r.mean(), r.std(ddof=1)
    if method == "gaussian":
        z = stats.norm.ppf(a)
        return -(mu + z * sd), -(mu - sd * stats.norm.pdf(z) / a)
    if method == "cornish":
        z = stats.norm.ppf(a)
        s, k = stats.skew(r), stats.kurtosis(r)
        zc = z + (z ** 2 - 1) * s / 6 + (z ** 3 - 3 * z) * k / 24 - (2 * z ** 3 - 5 * z) * s ** 2 / 36
        var = -(mu + zc * sd)
        sim = np.sort(r)
        es = max(var, -sim[: max(int(a * len(r)), 1)].mean())
        return var, es
    if method == "t":
        nu, loc, scale = stats.t.fit(r)
        nu = max(nu, 2.05)
        q = stats.t.ppf(a, nu)
        var = -(loc + scale * q)
        es = -(loc - scale * (nu + q ** 2) / (nu - 1) * stats.t.pdf(q, nu) / a)
        return var, es
    if method == "ewma":
        w = ewma_lambda ** np.arange(len(r))[::-1]
        s2 = np.sum(w * (r - 0) ** 2) * (1 - ewma_lambda)
        sd_t = math.sqrt(s2)
        z = stats.norm.ppf(a)
        return -z * sd_t, sd_t * stats.norm.pdf(z) / a
    if method in ("garch", "fhs"):
        from arch import arch_model
        sc = 100.0 / max(np.std(r), 1e-12) * 0.01 if np.std(r) < 0.5 else 1.0
        am = arch_model(r * sc, mean="Constant", vol="GARCH", p=1, q=1, dist="t" if method == "garch" else "normal")
        res = am.fit(disp="off")
        f = res.forecast(horizon=1, reindex=False)
        m1 = float(np.asarray(f.mean)[-1, 0]) / sc
        s1 = math.sqrt(float(np.asarray(f.variance)[-1, 0])) / sc
        if method == "fhs":
            z = np.asarray(res.std_resid)
            z = z[~np.isnan(z)]
            q = np.quantile(z, a)
            es = -(m1 + s1 * z[z <= q].mean())
            return -(m1 + s1 * q), es
        nu = float(res.params["nu"])
        qt = stats.t.ppf(a, nu) * math.sqrt((nu - 2) / nu)
        tq = stats.t.ppf(a, nu)
        es_std = (nu + tq ** 2) / (nu - 1) * stats.t.pdf(tq, nu) / a * math.sqrt((nu - 2) / nu)
        return -(m1 + s1 * qt), -(m1 - s1 * es_std)
    raise ModelError(f"Unknown VaR method '{method}'")


METHOD_NAMES = {"historical": "Historical simulation", "gaussian": "Parametric normal", "cornish":
                "Cornish-Fisher (modified)", "t": "Parametric Student-t", "ewma": "RiskMetrics EWMA (λ=0.94)",
                "garch": "GARCH(1,1)-t", "fhs": "Filtered historical simulation (GARCH)"}


def var_report(wf, ev, expr: str, levels="0.95 0.99", methods="historical gaussian cornish t ewma garch fhs",
               value: float | None = None) -> Report:
    r, m = _ret(wf, ev, expr)
    v = r.values.astype(float)
    if len(v) < 30:
        raise ModelError("Need at least 30 observations for VaR")
    lv = [float(x) for x in str(levels).replace(",", " ").split()]
    meths = str(methods).split()
    rows = []
    for me in meths:
        row = [METHOD_NAMES.get(me, me)]
        for l in lv:
            try:
                var, es = _var_es(v, 1 - l, me)
            except Exception:
                var, es = np.nan, np.nan
            if value:
                var, es = var * value, es * value
            row += [var, es]
        rows.append(row)
    rep = Report(f"Value-at-Risk and Expected Shortfall: {expr.upper()}")
    rep.header([f"Sample: {wf.sample_label(m)}", f"Observations: {len(v)}",
                "One-period horizon; figures are losses (positive = loss)"
                + (f" on a position of {fmt_num(value)}" if value else " as return fractions")], stamp=True)
    cols = ["Method"]
    for l in lv:
        cols += [f"VaR {l:.1%}", f"ES {l:.1%}"]
    rep.table(cols, rows)
    rep.note("Scale to an h-period horizon by sqrt(h) under i.i.d. returns (square-root-of-time rule).")
    fig, ax = plots.new_figure(6.8, 4)
    ax.hist(v, bins=60, color="#9fb8e3", edgecolor="white", density=True)
    for k, me in enumerate(meths[:4]):
        try:
            var, _ = _var_es(v, 1 - lv[-1], me)
            ax.axvline(-var, color=plots.PALETTE[k + 1], lw=1.3, label=f"{METHOD_NAMES.get(me, me)} {lv[-1]:.0%}")
        except Exception:
            pass
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Return distribution and VaR thresholds")
    fig.tight_layout()
    rep.figure(fig, "VaR")
    return rep


def backtest_report(wf, ev, expr: str, level: float = 0.99, window: int = 250, method: str = "historical",
                    refit: int = 20):
    r, m = _ret(wf, ev, expr)
    v = r.values.astype(float)
    a = 1 - float(level)
    W = int(window)
    if len(v) <= W + 20:
        raise ModelError("Sample too short for the chosen window")
    VaR = np.full(len(v), np.nan)
    if method in ("garch", "fhs"):
        from arch import arch_model
        sc = 100.0 if np.std(v) < 0.5 else 1.0
        params = None
        for t in range(W, len(v)):
            if params is None or (t - W) % int(refit) == 0:
                am = arch_model(v[t - W:t] * sc, mean="Constant", vol="GARCH", p=1, q=1,
                                dist="t" if method == "garch" else "normal")
                res = am.fit(disp="off")
                params = res.params
            am = arch_model(v[t - W:t] * sc, mean="Constant", vol="GARCH", p=1, q=1,
                            dist="t" if method == "garch" else "normal")
            rf = am.fix(params)
            f = rf.forecast(horizon=1, reindex=False)
            mu1 = float(np.asarray(f.mean)[-1, 0]) / sc
            s1 = math.sqrt(float(np.asarray(f.variance)[-1, 0])) / sc
            if method == "garch":
                nu = float(params["nu"])
                q = stats.t.ppf(a, nu) * math.sqrt((nu - 2) / nu)
            else:
                z = np.asarray(rf.std_resid)
                q = np.nanquantile(z, a)
            VaR[t] = -(mu1 + s1 * q)
    else:
        for t in range(W, len(v)):
            VaR[t], _ = _var_es(v[t - W:t], a, method)
    test = slice(W, len(v))
    hit = (v[test] < -VaR[test]).astype(int)
    n, x = len(hit), int(hit.sum())
    pi = x / n
    LRpof = -2 * (x * math.log(a) + (n - x) * math.log(1 - a)
                  - (x * math.log(pi) if x else 0) - ((n - x) * math.log(1 - pi) if n - x else 0))
    n00 = n01 = n10 = n11 = 0
    for i in range(1, n):
        p, c = hit[i - 1], hit[i]
        if p == 0 and c == 0:
            n00 += 1
        elif p == 0 and c == 1:
            n01 += 1
        elif p == 1 and c == 0:
            n10 += 1
        else:
            n11 += 1
    pi0 = n01 / max(n00 + n01, 1)
    pi1 = n11 / max(n10 + n11, 1)
    pi_ = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    def ll(p, k0, k1):
        return (k0 * math.log(1 - p) if k0 and p < 1 else 0) + (k1 * math.log(p) if k1 and p > 0 else 0)

    LRind = -2 * (ll(pi_, n00 + n10, n01 + n11) - ll(pi0, n00, n01) - ll(pi1, n10, n11))
    LRcc = LRpof + LRind
    zone = ""
    if abs(level - 0.99) < 1e-9:
        x250 = x * 250 / n
        zone = "Green" if x250 <= 4 else "Yellow" if x250 <= 9 else "Red"
    rep = Report(f"VaR Backtest: {expr.upper()}")
    rep.header([f"Method: {METHOD_NAMES.get(method, method)}; rolling window {W}"
                + (f", parameters re-estimated every {refit} obs" if method in ("garch", "fhs") else ""),
                f"VaR level: {level:.1%}", f"Backtest observations: {n}"], stamp=True)
    rep.stats([("Violations (actual)", x), ("Violations (expected)", a * n), ("Violation rate", pi),
               ("Basel traffic light (scaled to 250 obs)", zone or "n/a")],
              [("Kupiec POF LR", LRpof), ("Prob. Chi-Square(1)", stats.chi2.sf(LRpof, 1)),
               ("Christoffersen independence LR", LRind), ("Prob. Chi-Square(1) ", stats.chi2.sf(LRind, 1)),
               ("Conditional coverage LR", LRcc), ("Prob. Chi-Square(2)", stats.chi2.sf(LRcc, 2))])
    fig, ax = plots.new_figure(7.5, 4.2)
    idx = r.index[test] if not isinstance(r.index, pd.MultiIndex) else np.arange(n)
    ax.plot(idx, v[test], color="#777", lw=0.6, label="Return")
    ax.plot(idx, -VaR[test], color=plots.PALETTE[1], lw=1.2, label=f"-VaR {level:.0%}")
    viol = hit.astype(bool)
    ax.scatter(np.asarray(idx)[viol], v[test][viol], color=plots.PALETTE[1], s=14, zorder=3, label="Violation")
    ax.legend(frameon=False)
    ax.set_title(f"VaR backtest ({x} violations, {a * n:.1f} expected)")
    fig.tight_layout()
    return [rep, FigureOutput(fig, "VaR backtest")]


def performance_report(wf, ev, names: list[str], rf: str | float | None = None, periods: int | None = None,
                       benchmark: str | None = None):
    ann = periods or (PERIODS_PER_YEAR.get(wf.freq, 252) if wf.freq != "u" else 252)
    rows, curves, dds = [], {}, {}
    rfv = None
    if rf not in (None, "", 0, "0"):
        try:
            rfv = float(rf)
        except (TypeError, ValueError):
            rfv = ev.eval(str(rf), ignore_sample=True, as_series=True)
    bench = ev.eval(benchmark, ignore_sample=True, as_series=True) if benchmark else None
    for n in names:
        r, m = _ret(wf, ev, n)
        x = r.values.astype(float)
        if rfv is None:
            ex = x
        elif isinstance(rfv, float):
            ex = x - rfv
        else:
            ex = x - rfv[m].fillna(0).values
        mu, sd = x.mean() * ann, x.std(ddof=1) * math.sqrt(ann)
        sharpe = ex.mean() / ex.std(ddof=1) * math.sqrt(ann) if ex.std() > 0 else np.nan
        dn = ex[ex < 0]
        sortino = ex.mean() * ann / (math.sqrt((dn ** 2).sum() / len(ex)) * math.sqrt(ann)) if len(dn) else np.nan
        wealth = np.cumprod(1 + x)
        peak = np.maximum.accumulate(wealth)
        dd = wealth / peak - 1
        mdd = dd.min()
        cagr = wealth[-1] ** (ann / len(x)) - 1 if wealth[-1] > 0 else np.nan
        calmar = cagr / abs(mdd) if mdd < 0 else np.nan
        q = np.quantile(x, 0.05)
        row = [n.upper(), mu, cagr, sd, sharpe, sortino, mdd, calmar, stats.skew(x), stats.kurtosis(x, fisher=False),
               -q, -x[x <= q].mean(), (x > 0).mean(), x.max(), x.min(), len(x)]
        if bench is not None:
            b = bench[m].values
            ok = ~np.isnan(b)
            beta = np.cov(x[ok], b[ok])[0, 1] / np.var(b[ok], ddof=1)
            te = (x[ok] - b[ok]).std(ddof=1) * math.sqrt(ann)
            ir = (x[ok] - b[ok]).mean() * ann / te if te > 0 else np.nan
            row += [beta, te, ir]
        rows.append(row)
        idx = r.index if not isinstance(r.index, pd.MultiIndex) else np.arange(len(r))
        curves[n.upper()] = pd.Series(wealth, index=idx)
        dds[n.upper()] = pd.Series(dd, index=idx)
    cols = ["Series", "Mean (ann.)", "CAGR", "Vol (ann.)", "Sharpe", "Sortino", "Max drawdown", "Calmar", "Skewness",
            "Kurtosis", "VaR 95% (hist)", "ES 95%", "% positive", "Best", "Worst", "Obs"]
    fm = ["str", "pct", "pct", "pct", "f3", "f3", "pct", "f3", "f3", "f3", "pct", "pct", "pct", "pct", "pct", "int"]
    if bench is not None:
        cols += ["Beta", "Tracking error", "Information ratio"]
        fm += ["f3", "pct", "f3"]
    rep = Report("Performance Analytics")
    rep.header([f"Sample: {wf.sample_label()}", f"Annualisation: {ann} periods per year; returns as simple returns",
                f"Risk-free: {rf if rf not in (None, '') else 'none'}" + (f"; benchmark: {benchmark.upper()}" if benchmark
                                                                         else "")], stamp=True)
    T = list(map(list, zip(*([cols] + rows))))
    rep.table(["Statistic"] + [r_[0] for r_ in rows], [[T[i][0]] + T[i][1:] for i in range(1, len(cols))],
              ["str"] + ["str"] * len(rows))
    # nicer formatting row-wise
    tb = rep.blocks[-1]
    for i, fmt_key in enumerate(fm[1:]):
        from ..report import FORMATTERS
        f = FORMATTERS[fmt_key]
        tb.rows[i] = [tb.rows[i][0]] + [f(v) for v in tb.rows[i][1:]]
    fig, axes = plots.new_figure(7.5, 5.2, 2, 1, sharex=True)
    for k, (n, c) in enumerate(curves.items()):
        axes[0].plot(c.index, c.values, color=plots.PALETTE[k % 10], lw=1.2, label=n)
        axes[1].fill_between(dds[n].index, dds[n].values, 0, color=plots.PALETTE[k % 10], alpha=0.35)
    axes[0].set_title("Growth of 1")
    axes[0].legend(frameon=False)
    axes[1].set_title("Drawdown")
    fig.tight_layout()
    return [rep, FigureOutput(fig, "Performance")]
