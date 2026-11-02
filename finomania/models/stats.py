"""Descriptive statistics and single/multiple-series tests (series and group views)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from ..report import FigureOutput, Report, fmt_num
from . import plots
from .base import ModelError, descriptive_block, jarque_bera


def _clean(v):
    v = np.asarray(v, dtype=float)
    return v[~np.isnan(v)]


def histogram_output(v, name: str, sample_label: str = "", bins: int | None = None):
    v = _clean(v)
    if len(v) < 2:
        raise ModelError("Not enough observations")
    blk = descriptive_block(v)
    lines = [f"Series: {name.upper()}", f"Sample {sample_label}", f"Observations {len(v)}", ""]
    for k, val in blk:
        if k == "Observations":
            continue
        lines.append(f"{k:<13}{fmt_num(val):>12}")
    fig = plots.hist_plot(v, name, "\n".join(lines), bins)
    return FigureOutput(fig, f"Histogram and Stats: {name.upper()}")


def stats_report(data: dict[str, np.ndarray], common: bool = True, title="Descriptive Statistics") -> Report:
    names = list(data)
    if common:
        M = np.column_stack([np.asarray(data[n], dtype=float) for n in names])
        ok = ~np.isnan(M).any(axis=1)
        cols = {n: M[ok, j] for j, n in enumerate(names)}
    else:
        cols = {n: _clean(data[n]) for n in names}
    blocks = {n: descriptive_block(v) if len(v) > 2 else None for n, v in cols.items()}
    labels = [k for k, _ in next(b for b in blocks.values() if b)]
    rows = []
    for i, lab in enumerate(labels):
        row = [lab]
        for n in names:
            b = blocks[n]
            row.append(b[i][1] if b else np.nan)
        rows.append(row)
    fm = ["str"] + ["num"] * len(names)
    r = Report(title)
    r.header([f"Sample: {'common' if common else 'individual samples'}"])
    r.table([""] + [n.upper() for n in names], rows, fm)
    return r


def acf_pacf(v: np.ndarray, lags: int):
    from statsmodels.tsa.stattools import acf, pacf
    v = _clean(v)
    n = len(v)
    lags = min(lags, n // 2 - 1)
    a, q, p = acf(v, nlags=lags, qstat=True, fft=False)
    pa = pacf(v, nlags=lags, method="ywm")
    return a[1:], pa[1:], q, p, n


def correlogram_report(v, name: str, lags: int = 24, title: str | None = None, dof_adjust: int = 0) -> Report:
    a, pa, q, p, n = acf_pacf(v, lags)
    if dof_adjust:
        p = np.array([stats.chi2.sf(q[i], i + 1 - dof_adjust) if i + 1 > dof_adjust else np.nan
                      for i in range(len(q))])
    r = Report(title or f"Correlogram of {name.upper()}")
    r.header([f"Included observations: {n}"])
    band = 2 / math.sqrt(n)
    rows = []
    for i in range(len(a)):
        rows.append([_bar(a[i], band), _bar(pa[i], band), i + 1, a[i], pa[i], q[i], p[i]])
    r.table(["Autocorrelation", "Partial Correlation", "", "AC", "PAC", "Q-Stat", "Prob"], rows,
            ["str", "str", "int", "f3", "f3", "f3", "f3"], align=["l", "l", "r", "r", "r", "r", "r"])
    if dof_adjust:
        r.note(f"Q-statistic probabilities adjusted for {dof_adjust} ARMA term(s)")
    r.figure(plots.correlogram_plot(a, pa, n, name), "Correlogram")
    return r


def _bar(x, band):
    w = 10
    k = int(round(min(abs(x), 1) * w))
    s = "*" * k
    left = (s.rjust(w) if x < 0 else " " * w)
    right = (s.ljust(w) if x >= 0 else " " * w)
    return f"{left}|{right}"


def corr_report(df: pd.DataFrame, kind: str = "corr", tests: bool = False, method: str = "pearson") -> Report:
    df = df.dropna()
    names = [c.upper() for c in df.columns]
    n = len(df)
    if kind == "cov":
        M = df.cov(ddof=0).values
        title = "Covariance Matrix"
    else:
        M = df.corr(method=method).values
        title = {"pearson": "Correlation Matrix", "spearman": "Spearman Rank Correlation",
                 "kendall": "Kendall's tau"}.get(method, "Correlation")
    r = Report(title)
    r.header(["Sample: common", f"Included observations: {n}"])
    rows = []
    for i in range(len(names)):
        rows.append([names[i]] + [M[i, j] for j in range(len(names))])
        if tests and kind == "corr":
            tr, pr = ["  t-Statistic"], ["  Probability"]
            for j in range(len(names)):
                if i == j:
                    tr.append("-----")
                    pr.append("-----")
                else:
                    rho = M[i, j]
                    t = rho * math.sqrt((n - 2) / max(1 - rho ** 2, 1e-300))
                    tr.append(t)
                    pr.append(f"{2 * stats.t.sf(abs(t), n - 2):.4f}")
            rows += [tr, pr]
    r.table([""] + names, rows)
    if kind == "corr" and len(names) > 1:
        r.figure(plots.heatmap(M, list(df.columns), title), "Heatmap")
    return r


def mean_test(v, mu: float = 0.0, name="") -> Report:
    v = _clean(v)
    n = len(v)
    t = (v.mean() - mu) / (v.std(ddof=1) / math.sqrt(n))
    r = Report(f"Hypothesis Testing for {name.upper()}")
    r.header([f"Included observations: {n}", f"Test of Hypothesis: Mean = {mu:g}",
              f"Sample Mean = {fmt_num(v.mean())}", f"Sample Std. Dev. = {fmt_num(v.std(ddof=1))}"])
    w = stats.wilcoxon(v - mu) if n > 10 else None
    rows = [["t-statistic", t, 2 * stats.t.sf(abs(t), n - 1)]]
    if w is not None:
        rows.append(["Wilcoxon signed rank", float(w.statistic), float(w.pvalue)])
    r.table(["Method", "Value", "Probability"], rows, ["str", "num", "prob"])
    return r


def equality_tests(data: dict[str, np.ndarray]) -> Report:
    vals = [_clean(v) for v in data.values()]
    names = [n.upper() for n in data]
    r = Report("Test for Equality of Means / Medians / Variances Between Series")
    r.header([f"Series: {' '.join(names)}"])
    rows = []
    if len(vals) == 2:
        t = stats.ttest_ind(vals[0], vals[1], equal_var=True)
        rows.append(["t-test", f"{len(vals[0]) + len(vals[1]) - 2}", float(t.statistic), float(t.pvalue)])
        tw = stats.ttest_ind(vals[0], vals[1], equal_var=False)
        rows.append(["Welch t-test", "", float(tw.statistic), float(tw.pvalue)])
    a = stats.f_oneway(*vals)
    rows.append(["Anova F-test (means)", f"({len(vals) - 1}, {sum(map(len, vals)) - len(vals)})", float(a.statistic),
                 float(a.pvalue)])
    kw = stats.kruskal(*vals)
    rows.append(["Kruskal-Wallis (medians)", str(len(vals) - 1), float(kw.statistic), float(kw.pvalue)])
    b = stats.bartlett(*vals)
    rows.append(["Bartlett (variances)", str(len(vals) - 1), float(b.statistic), float(b.pvalue)])
    lv = stats.levene(*vals)
    rows.append(["Levene (variances)", "", float(lv.statistic), float(lv.pvalue)])
    bf = stats.levene(*vals, center="median")
    rows.append(["Brown-Forsythe (variances)", "", float(bf.statistic), float(bf.pvalue)])
    r.table(["Method", "df", "Value", "Probability"], rows, ["str", "str", "num", "prob"])
    return r


def normality_report(v, name="") -> Report:
    from statsmodels.stats.diagnostic import lilliefors
    v = _clean(v)
    r = Report(f"Normality Tests: {name.upper()}")
    jb, jp = jarque_bera(v)
    rows = [["Jarque-Bera", jb, jp]]
    if len(v) <= 5000:
        sw = stats.shapiro(v)
        rows.append(["Shapiro-Wilk", float(sw.statistic), float(sw.pvalue)])
    ad = stats.anderson(v, dist="norm")
    rows.append(["Anderson-Darling (A²)", float(ad.statistic), _ad_p(float(ad.statistic), len(v))])
    ks, kp = lilliefors(v, dist="norm")
    rows.append(["Lilliefors (D)", float(ks), float(kp)])
    dg = stats.normaltest(v)
    rows.append(["D'Agostino-Pearson K²", float(dg.statistic), float(dg.pvalue)])
    r.table(["Test", "Statistic", "Probability"], rows, ["str", "num", "prob"])
    r.figure(plots.qq_plot(v, name), "Q-Q plot")
    return r


def _ad_p(a2, n):
    a = a2 * (1 + 0.75 / n + 2.25 / n ** 2)
    if a >= 0.6:
        return math.exp(1.2937 - 5.709 * a + 0.0186 * a ** 2)
    if a >= 0.34:
        return math.exp(0.9177 - 4.279 * a - 1.38 * a ** 2)
    if a >= 0.2:
        return 1 - math.exp(-8.318 + 42.796 * a - 59.938 * a ** 2)
    return 1 - math.exp(-13.436 + 101.14 * a - 223.73 * a ** 2)


def bds_report(v, name="", maxdim: int = 6) -> Report:
    from statsmodels.tsa.stattools import bds
    v = _clean(v)
    stat, p = bds(v, max_dim=int(maxdim))
    r = Report(f"BDS Test for {name.upper()}")
    r.header([f"Included observations: {len(v)}", "Distance: 1.5 x standard deviation"])
    rows = [[m + 2, float(stat[m]), float(stat[m]) , float(p[m])] for m in range(len(np.atleast_1d(stat)))]
    rows = [[row[0], row[1], row[3]] for row in rows]
    r.table(["Dimension", "BDS z-Statistic", "Prob."], rows, ["int", "num", "prob"])
    r.note("H0: the series is i.i.d.")
    return r


def variance_ratio_report(v, name="", lags="2 4 8 16", log: bool = False, robust: bool = True,
                          overlap: bool = True) -> Report:
    from arch.unitroot import VarianceRatio
    v = _clean(v)
    if log:
        v = np.log(v)
    r = Report(f"Variance Ratio Test for {name.upper()}")
    r.header(["Null Hypothesis: series is a random walk",
              f"Heteroskedasticity robust: {'Yes' if robust else 'No'}; overlapping: {'Yes' if overlap else 'No'}",
              f"Observations: {len(v)}"])
    rows = []
    zs = []
    for q in [int(x) for x in str(lags).replace(",", " ").split()]:
        vr = VarianceRatio(v, lags=q, robust=robust, overlap=overlap)
        rows.append([q, float(vr.vr), float(vr.stat), float(vr.pvalue)])
        zs.append(abs(float(vr.stat)))
    r.table(["Period", "Var. Ratio", "z-Statistic", "Probability"], rows, ["int", "num", "num", "prob"],
            title="Individual Tests")
    m = len(zs)
    zmax = max(zs)
    pj = 1 - (1 - 2 * stats.norm.sf(zmax)) ** m
    r.stats([("Max |z| (Chow-Denning)", zmax)], [("Probability (Sidak)", min(pj, 1.0))])
    return r


def autocorr_test_report(v, name, lags=10) -> Report:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    v = _clean(v)
    lb = acorr_ljungbox(v, lags=list(range(1, int(lags) + 1)), boxpierce=True)
    r = Report(f"Ljung-Box / Box-Pierce Q tests: {name.upper()}")
    rows = [[int(i), float(row["lb_stat"]), float(row["lb_pvalue"]), float(row["bp_stat"]), float(row["bp_pvalue"])]
            for i, row in lb.iterrows()]
    r.table(["Lag", "Ljung-Box Q", "Prob.", "Box-Pierce Q", "Prob."], rows, ["int", "num", "prob", "num", "prob"])
    return r


def pca_report(df: pd.DataFrame, use_corr: bool = True):
    df = df.dropna()
    X = df.values
    Z = (X - X.mean(0)) / (X.std(0, ddof=1) if use_corr else 1)
    C = np.cov(Z, rowvar=False)
    w, V = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    V = V * np.sign(V[np.argmax(np.abs(V), axis=0), range(V.shape[1])])
    scores = Z @ V
    names = [c.upper() for c in df.columns]
    r = Report("Principal Components Analysis")
    r.header([f"Sample: common ({len(df)} obs)", f"Computed using: {'Ordinary correlations' if use_corr else 'Covariances'}",
              f"Extracting {len(w)} of {len(w)} possible components"])
    cum = np.cumsum(w) / w.sum()
    r.table(["Number", "Value", "Difference", "Proportion", "Cumulative Value", "Cumulative Proportion"],
            [[f"Comp {i + 1}", w[i], (w[i] - w[i + 1]) if i + 1 < len(w) else np.nan, w[i] / w.sum(),
              np.sum(w[:i + 1]), cum[i]] for i in range(len(w))], title="Eigenvalues")
    r.table(["Variable"] + [f"PC {i + 1}" for i in range(len(w))],
            [[names[j]] + list(V[j]) for j in range(len(names))], title="Eigenvectors (loadings)")
    fig, ax = plots.new_figure(6, 3.6)
    ax.bar(np.arange(1, len(w) + 1), w, color=plots.PALETTE[0])
    ax.plot(np.arange(1, len(w) + 1), w, color=plots.PALETTE[1], marker="o")
    ax.set_title("Scree plot")
    ax.set_xlabel("Component")
    fig.tight_layout()
    r.figure(fig, "Scree plot")
    sc = pd.DataFrame(scores, index=df.index, columns=[f"pc{i + 1}" for i in range(len(w))])
    return r, sc
