"""
Asset-pricing factor models.

* Kenneth French data library download (cached) for CAPM, FF3, Carhart 4,
  FF5 and FF5 + momentum factors at daily / weekly / monthly frequency.
* Time-series factor regressions for many assets with OLS / White / HAC
  standard errors, annualised alphas, information ratios.
* Gibbons-Ross-Shanken (1989) test of joint zero alphas.
* Fama-MacBeth (1973) two-pass cross-sectional regressions with
  Shanken (1992) errors-in-variables correction and optional Newey-West.
* Rolling factor betas, factor summary statistics.
"""
from __future__ import annotations

import io
import math
import os
import re
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from scipy import stats

from ..core.dates import PERIODS_PER_YEAR
from ..report import FigureOutput, Report, fmt_num
from . import plots
from .base import ModelError, ModelResult, Param, View, pvalue_t, safe_inv

FF_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FF_FILES = {
    ("ff3", "m"): "F-F_Research_Data_Factors_CSV.zip",
    ("ff3", "d"): "F-F_Research_Data_Factors_daily_CSV.zip",
    ("ff3", "w"): "F-F_Research_Data_Factors_weekly_CSV.zip",
    ("ff5", "m"): "F-F_Research_Data_5_Factors_2x3_CSV.zip",
    ("ff5", "d"): "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    ("mom", "m"): "F-F_Momentum_Factor_CSV.zip",
    ("mom", "d"): "F-F_Momentum_Factor_daily_CSV.zip",
}
MODELS = {
    "capm": ["mkt_rf"],
    "ff3": ["mkt_rf", "smb", "hml"],
    "carhart": ["mkt_rf", "smb", "hml", "mom"],
    "ff4": ["mkt_rf", "smb", "hml", "mom"],
    "ff5": ["mkt_rf", "smb", "hml", "rmw", "cma"],
    "ff6": ["mkt_rf", "smb", "hml", "rmw", "cma", "mom"],
}
COLMAP = {"mkt-rf": "mkt_rf", "smb": "smb", "hml": "hml", "rmw": "rmw", "cma": "cma", "rf": "rf", "mom": "mom",
          "umd": "mom", "wml": "mom"}


def cache_dir() -> str:
    d = os.environ.get("FINOMANIA_CACHE") or os.path.join(os.path.expanduser("~"), ".finomania", "cache")
    os.makedirs(d, exist_ok=True)
    return d


def parse_french_csv(text: str, freq: str) -> pd.DataFrame:
    """Parse a Kenneth French data-library CSV into a DataFrame (percent units).
    Picks the first table whose dates match the frequency (YYYYMM monthly, YYYYMMDD daily/weekly)."""
    want = {"m": 6, "d": 8, "w": 8, "a": 4}[freq]
    lines = text.replace("\r", "").split("\n")
    blocks = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        cells = [c.strip() for c in ln.split(",")]
        if len(cells) >= 2 and any(re.search(r"[A-Za-z]", c) for c in cells[1:]) and not re.match(r"^\d", cells[0]):
            header = cells
            rows = []
            j = i + 1
            while j < len(lines):
                c2 = [c.strip() for c in lines[j].split(",")]
                if len(c2) >= 2 and re.fullmatch(r"\d{4,8}", c2[0] or ""):
                    rows.append(c2)
                    j += 1
                else:
                    break
            if rows:
                blocks.append((header, rows))
            i = j
        else:
            i += 1
    for header, rows in blocks:
        if len(rows[0][0]) != want:
            continue
        cols = [COLMAP.get(h.strip().lower(), h.strip().lower().replace("-", "_").replace(" ", "_")) for h in header[1:]]
        data = []
        idx = []
        for rw in rows:
            d = rw[0]
            if want == 6:
                idx.append(pd.Timestamp(year=int(d[:4]), month=int(d[4:6]), day=1))
            elif want == 8:
                idx.append(pd.Timestamp(year=int(d[:4]), month=int(d[4:6]), day=int(d[6:8])))
            else:
                idx.append(pd.Timestamp(year=int(d), month=1, day=1))
            vals = []
            for v in rw[1:len(cols) + 1]:
                try:
                    f = float(v)
                    vals.append(np.nan if f <= -99.99 else f)
                except ValueError:
                    vals.append(np.nan)
            data.append(vals)
        return pd.DataFrame(data, index=pd.DatetimeIndex(idx), columns=cols)
    raise ModelError(f"Could not find a {freq}-frequency table in the Fama-French file")


def _fetch(fname: str, refresh: bool = False) -> str:
    path = os.path.join(cache_dir(), fname)
    if refresh or not os.path.exists(path):
        url = FF_BASE + fname
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 Finomania"})
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = r.read()
        except Exception as e:
            raise ModelError(f"Could not download {url}: {e}.  You can download the file manually and pass "
                             f"file=<path> or place it in {cache_dir()}") from e
        with open(path, "wb") as f:
            f.write(raw)
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            name = [n for n in z.namelist() if n.lower().endswith((".csv", ".txt"))][0]
            return z.read(name).decode("latin-1")
    return raw.decode("latin-1")


def load_french(model: str = "ff3", freq: str = "m", file: str | None = None, refresh: bool = False,
                units: str = "decimal") -> pd.DataFrame:
    """Return factor returns (+ RF) for the requested model/frequency."""
    freq = {"monthly": "m", "m": "m", "daily": "d", "d": "d", "d5": "d", "weekly": "w", "w": "w"}.get(freq, freq)
    model = model.lower()
    if model not in MODELS:
        raise ModelError(f"Unknown factor model '{model}' (capm, ff3, carhart, ff5, ff6)")
    if file:
        if file.lower().endswith(".zip"):
            with zipfile.ZipFile(file) as z:
                text = z.read([n for n in z.namelist() if n.lower().endswith((".csv", ".txt"))][0]).decode("latin-1")
        else:
            text = open(file, encoding="latin-1").read()
        df = parse_french_csv(text, freq)
    else:
        base = "ff5" if model in ("ff5", "ff6") else "ff3"
        if (base, freq) not in FF_FILES:
            raise ModelError(f"{base.upper()} factors are not available at frequency '{freq}' (use m or d)")
        df = parse_french_csv(_fetch(FF_FILES[(base, freq)], refresh), freq)
        if "mom" in MODELS[model]:
            if ("mom", freq) not in FF_FILES:
                raise ModelError("Momentum factor available only monthly or daily")
            mom = parse_french_csv(_fetch(FF_FILES[("mom", freq)], refresh), freq)
            df = df.join(mom[["mom"]], how="inner")
    if units == "decimal":
        df = df / 100.0
    return df


# ============================================================================
# factor regressions
# ============================================================================

class FactorResult(ModelResult):
    object_type = "factor"
    method_label = "Factor model time-series regressions"

    def __init__(self, wf, ev, assets: list[str], factors: list[str], rf: str | None = None, excess: bool = False,
                 cov: str = "ordinary", hac_lags: int | None = None, model_label: str = "", name: str = ""):
        self.wf, self.ev, self.name = wf, ev, name
        self.assets = [a.lower() for a in assets]
        self.factors = [f.lower() for f in factors]
        self.rf = rf.lower() if rf else None
        self.excess = excess
        self.model_label = model_label or "custom"
        self.cov = (cov or "ordinary").lower()
        self.hac_lags = hac_lags
        if not self.assets:
            raise ModelError("List at least one asset return series")
        F = pd.DataFrame({f: ev.eval(f, ignore_sample=True, as_series=True) for f in self.factors},
                         index=wf.data.index)
        self.F = F
        self.R = pd.DataFrame({a: ev.eval(a, ignore_sample=True, as_series=True) for a in self.assets},
                              index=wf.data.index)
        if self.rf and not excess:
            rfv = ev.eval(self.rf, ignore_sample=True, as_series=True)
            self.Re = self.R.sub(rfv, axis=0)
        else:
            self.Re = self.R.copy()
        self.ann = PERIODS_PER_YEAR.get(wf.freq, 12) if wf.freq != "u" else 1
        self.fits = {}
        smp = wf.sample
        for a in self.assets:
            ok = smp & self.Re[a].notna().values & F.notna().all(axis=1).values
            if ok.sum() <= len(self.factors) + 2:
                raise ModelError(f"Too few observations for {a.upper()}")
            self.fits[a] = self._ols(self.Re[a].values[ok], F.values[ok], ok)
        self._warn_units()

    def _ols(self, y, Fm, ok):
        import statsmodels.api as sm
        X = np.column_stack([np.ones(len(y)), Fm])
        if self.cov in ("hac", "nw"):
            L = self.hac_lags if self.hac_lags is not None else int(4 * (len(y) / 100) ** (2 / 9))
            res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": L, "use_correction": True})
        elif self.cov in ("white", "robust", "hc1"):
            res = sm.OLS(y, X).fit(cov_type="HC1")
        else:
            res = sm.OLS(y, X).fit()
        e = np.asarray(res.resid)
        return {"b": np.asarray(res.params), "se": np.asarray(res.bse), "t": np.asarray(res.tvalues),
                "p": np.asarray(res.pvalues), "r2": float(res.rsquared), "r2a": float(res.rsquared_adj), "n": len(y),
                "sde": float(e.std(ddof=X.shape[1])), "resid": e, "mask": ok, "y": y}

    def _warn_units(self):
        mf = np.nanmean(np.abs(self.F.values))
        mr = np.nanmean(np.abs(self.Re.values))
        self.unit_warning = None
        if mf > 0 and (mr / mf > 30 or mf / mr > 30):
            self.unit_warning = ("Asset returns and factors appear to be in different units (percent vs decimal). "
                                 "Use units=pct when fetching factors, or rescale your returns.")

    def report(self) -> Report:
        r = Report(f"Factor Model: {self.name.upper()}" if self.name else "Factor Model Regressions")
        fl = [f.upper() for f in self.factors]
        cov_lab = {"hac": "Newey-West HAC", "nw": "Newey-West HAC", "white": "White", "robust": "White",
                   "hc1": "White"}.get(self.cov, "OLS")
        r.header([f"Model: {self.model_label.upper()}  (factors: {' '.join(fl)})",
                  f"Dependent variables: excess returns{' (R - ' + self.rf.upper() + ')' if self.rf and not self.excess else ''}",
                  f"Sample: {self.wf.sample_label()}", f"Standard errors: {cov_lab}",
                  f"Alpha annualised with {self.ann} periods per year"], stamp=True)
        cols = ["Asset", "Alpha", "t(Alpha)", "Alpha (ann.)"]
        for f in fl:
            cols += [f"β {f}", "t"]
        cols += ["R²", "Adj. R²", "Info. ratio", "Obs"]
        rows = []
        for a in self.assets:
            ft = self.fits[a]
            row = [a.upper(), ft["b"][0], ft["t"][0], ft["b"][0] * self.ann]
            for j in range(len(fl)):
                row += [ft["b"][j + 1], ft["t"][j + 1]]
            ir = ft["b"][0] / ft["sde"] * math.sqrt(self.ann) if ft["sde"] > 0 else np.nan
            row += [ft["r2"], ft["r2a"], ir, ft["n"]]
            rows.append(row)
        fm = ["str", "num", "f3", "num"] + ["num", "f3"] * len(fl) + ["f4", "f4", "f3", "int"]
        r.table(cols, rows, fm)
        sig = [a.upper() for a in self.assets if self.fits[a]["p"][0] < 0.05]
        r.note(f"Assets with alpha significant at 5%: {', '.join(sig) if sig else 'none'}")
        if len(self.assets) > 1:
            try:
                g = self.grs()
                r.stats([("GRS F-statistic", g["F"]), ("Mean |alpha|", g["mabs"])],
                        [(f"Prob. F({g['N']},{g['T'] - g['N'] - g['L']})", g["p"]), ("Common obs", g["T"])])
            except ModelError as e:
                r.note(str(e))
        if self.unit_warning:
            r.warn(self.unit_warning)
        return r

    def grs(self) -> dict:
        ok = self.wf.sample & self.Re.notna().all(axis=1).values & self.F.notna().all(axis=1).values
        R, F = self.Re.values[ok], self.F.values[ok]
        T, N = R.shape
        L = F.shape[1]
        if T <= N + L:
            raise ModelError("GRS test needs more observations than assets + factors")
        X = np.column_stack([np.ones(T), F])
        B = np.linalg.lstsq(X, R, rcond=None)[0]
        E = R - X @ B
        alpha = B[0]
        Sigma = E.T @ E / (T - L - 1)
        mu = F.mean(0)
        Om = np.cov(F, rowvar=False, ddof=1).reshape(L, L)
        Fs = (T / N) * ((T - N - L) / (T - L - 1)) * float(alpha @ safe_inv(Sigma) @ alpha) / \
            (1 + float(mu @ safe_inv(Om) @ mu))
        return {"F": Fs, "p": float(stats.f.sf(Fs, N, T - N - L)), "T": T, "N": N, "L": L,
                "mabs": float(np.mean(np.abs(alpha))), "alpha": alpha}

    def stat(self, key):
        if key == "grs":
            return self.grs()["F"]
        if key == "alphas":
            return np.array([self.fits[a]["b"][0] for a in self.assets])
        raise ModelError(f"@{key} not available")

    # ---------------------------------------------------------------- views
    def views(self):
        return [
            View("output", "Factor Regression Summary", self.report),
            View("detail", "Asset Regression Output...", self.view_detail,
                 [Param("asset", "Asset", "choice", self.assets[0], self.assets)]),
            View("grs", "GRS Test of Zero Alphas", self.view_grs),
            View("fmb", "Fama-MacBeth Two-Pass Regression", self.view_fmb,
                 [Param("nw", "Newey-West lags (0 = none)", "int", 0),
                  Param("window", "Rolling beta window (0 = full sample)", "int", 0)]),
            View("rolling", "Rolling Betas", self.view_rolling,
                 [Param("asset", "Asset", "choice", self.assets[0], self.assets),
                  Param("window", "Window", "int", min(60, max(12, len(self.F) // 5)))]),
            View("alphaplot", "Alphas with 95% Confidence Intervals", self.view_alphas),
            View("factorstats", "Factor Summary Statistics", self.view_factor_stats),
            View("cumret", "Cumulative Factor Returns", self.view_cumret),
        ]

    def procs(self):
        return [View("makeresids", "Make Residual Series", self.proc_makeresids,
                     [Param("prefix", "Prefix", "text", "e_")], kind="proc")]

    def view_detail(self, asset=None):
        from .regression import LSResult
        a = (asset or self.assets[0]).lower()
        dep = f"({a}-{self.rf})" if self.rf and not self.excess else a
        cov = "hac" if self.cov in ("hac", "nw") else "white" if self.cov in ("white", "robust", "hc1") else "ordinary"
        eq = LSResult(self.wf, self.ev, f"{dep} c " + " ".join(self.factors), cov=cov, hac_lags=self.hac_lags)
        return eq.report()

    def view_grs(self):
        g = self.grs()
        r = Report("Gibbons-Ross-Shanken Test")
        r.text("H0: all alphas are jointly zero (factors price the test assets)")
        r.stats([("GRS F-statistic", g["F"]), ("Assets (N)", g["N"]), ("Factors (L)", g["L"])],
                [(f"Prob. F({g['N']},{g['T'] - g['N'] - g['L']})", g["p"]), ("Observations (T)", g["T"]),
                 ("Mean |alpha|", g["mabs"])])
        r.table(["Asset", "Alpha"], [[a.upper(), v] for a, v in zip(self.assets, g["alpha"])])
        return r

    def view_fmb(self, nw=0, window=0):
        return fama_macbeth(self.wf, self.Re, self.F, self.factors, nw=int(nw), window=int(window))

    def view_rolling(self, asset=None, window=60):
        a = (asset or self.assets[0]).lower()
        w = int(window)
        ok = self.wf.sample & self.Re[a].notna().values & self.F.notna().all(axis=1).values
        y = self.Re[a].values[ok]
        Fm = self.F.values[ok]
        idx = self.wf.data.index[ok]
        T = len(y)
        if T <= w + 2:
            raise ModelError("Window too large for the sample")
        B = np.full((T, Fm.shape[1] + 1), np.nan)
        for t in range(w, T + 1):
            X = np.column_stack([np.ones(w), Fm[t - w:t]])
            B[t - 1] = np.linalg.lstsq(X, y[t - w:t], rcond=None)[0]
        data = {f"Beta {f.upper()}": pd.Series(B[:, j + 1], index=idx) for j, f in enumerate(self.factors)}
        data["Alpha"] = pd.Series(B[:, 0], index=idx)
        return FigureOutput(plots.line_plot(data, f"Rolling {w}-period factor loadings: {a.upper()}", multiple=True),
                            "Rolling betas")

    def view_alphas(self):
        a = np.array([self.fits[x]["b"][0] for x in self.assets]) * self.ann
        se = np.array([self.fits[x]["se"][0] for x in self.assets]) * self.ann
        fig, ax = plots.new_figure(max(5, 0.6 * len(a) + 2), 4)
        xs = np.arange(len(a))
        ax.bar(xs, a, color=[plots.PALETTE[1] if abs(v / s) > 1.96 else plots.PALETTE[0] for v, s in zip(a, se)])
        ax.errorbar(xs, a, yerr=1.96 * se, fmt="none", ecolor="#333", capsize=3, lw=1)
        ax.axhline(0, color="#333", lw=0.7)
        ax.set_xticks(xs)
        ax.set_xticklabels([x.upper() for x in self.assets], rotation=45, ha="right")
        ax.set_title("Annualised alphas (95% CI; red = significant)")
        fig.tight_layout()
        return FigureOutput(fig, "Alphas")

    def view_factor_stats(self):
        F = self.F[self.wf.sample].dropna()
        rows = []
        for f in self.factors:
            v = F[f].values
            m, s = v.mean(), v.std(ddof=1)
            rows.append([f.upper(), m, m * self.ann, s * math.sqrt(self.ann), m / s * math.sqrt(self.ann),
                         m / (s / math.sqrt(len(v))), stats.skew(v), stats.kurtosis(v, fisher=False), len(v)])
        r = Report("Factor Summary Statistics")
        r.table(["Factor", "Mean", "Mean (ann.)", "Vol (ann.)", "Sharpe (ann.)", "t(mean)", "Skew", "Kurt", "Obs"], rows,
                ["str", "num", "num", "num", "f3", "f3", "f3", "f3", "int"])
        from .stats import corr_report
        r.extend(corr_report(F[self.factors]), with_title=True)
        return r

    def view_cumret(self):
        F = self.F[self.wf.sample].dropna()
        cum = (1 + F).cumprod()
        return FigureOutput(plots.line_plot({c.upper(): cum[c] for c in F.columns}, "Cumulative factor returns "
                                            "(growth of 1)"), "Cumulative factor returns")

    def proc_makeresids(self, prefix="e_"):
        for a in self.assets:
            ft = self.fits[a]
            s = pd.Series(np.nan, index=self.wf.data.index)
            s.iloc[np.flatnonzero(ft["mask"])] = ft["resid"]
            self.wf.set_series(prefix + a, s, respect_sample=False)
        return f"Residuals saved with prefix {prefix.upper()}"


def fama_macbeth(wf, Re: pd.DataFrame, F: pd.DataFrame, factors: list[str], nw: int = 0, window: int = 0) -> Report:
    ok = wf.sample & Re.notna().all(axis=1).values & F.notna().all(axis=1).values
    R, Fm = Re.values[ok], F.values[ok]
    T, N = R.shape
    L = Fm.shape[1]
    if N <= L + 1:
        raise ModelError("Fama-MacBeth needs more assets than factors + 1")
    lam = []
    r2s = []
    if window and window > L + 2:
        start = window
        for t in range(start, T):
            X = np.column_stack([np.ones(window), Fm[t - window:t]])
            B = np.linalg.lstsq(X, R[t - window:t], rcond=None)[0][1:].T  # N x L
            Z = np.column_stack([np.ones(N), B])
            g = np.linalg.lstsq(Z, R[t], rcond=None)[0]
            lam.append(g)
        betas = B
        Tn = T - start
    else:
        X = np.column_stack([np.ones(T), Fm])
        betas = np.linalg.lstsq(X, R, rcond=None)[0][1:].T
        Z = np.column_stack([np.ones(N), betas])
        for t in range(T):
            g = np.linalg.lstsq(Z, R[t], rcond=None)[0]
            lam.append(g)
            e = R[t] - Z @ g
            tss = ((R[t] - R[t].mean()) ** 2).sum()
            r2s.append(1 - e @ e / tss if tss > 0 else np.nan)
        Tn = T
    lam = np.array(lam)
    lbar = lam.mean(0)
    dev = lam - lbar
    S = dev.T @ dev / Tn
    if nw:
        for l in range(1, nw + 1):
            w = 1 - l / (nw + 1)
            G = dev[l:].T @ dev[:-l] / Tn
            S += w * (G + G.T)
    V_fm = S / Tn
    Sf = np.cov(Fm, rowvar=False, ddof=1).reshape(L, L)
    lf = lbar[1:]
    c = float(lf @ safe_inv(Sf) @ lf)
    V_sh = (1 + c) * V_fm
    V_sh[1:, 1:] += Sf / Tn
    se_fm, se_sh = np.sqrt(np.diag(V_fm)), np.sqrt(np.diag(V_sh))
    names = ["Intercept (zero-beta excess)"] + [f.upper() for f in factors]
    fmeans = [np.nan] + list(Fm.mean(0))
    rows = []
    for i in range(L + 1):
        rows.append([names[i], lbar[i], se_fm[i], lbar[i] / se_fm[i], 2 * stats.t.sf(abs(lbar[i] / se_fm[i]), Tn - 1),
                     se_sh[i], lbar[i] / se_sh[i], 2 * stats.t.sf(abs(lbar[i] / se_sh[i]), Tn - 1), fmeans[i]])
    r = Report("Fama-MacBeth Two-Pass Cross-Sectional Regression")
    r.header([f"Test assets: {N}", f"Time periods: {Tn}",
              "First pass: " + (f"rolling {window}-period betas" if window else "full-sample time-series betas"),
              "Second pass: period-by-period OLS of excess returns on betas",
              f"Standard errors: Fama-MacBeth{' with Newey-West (' + str(nw) + ' lags)' if nw else ''}; "
              "Shanken (1992) errors-in-variables correction"], stamp=True)
    r.table(["Risk premium", "Lambda", "S.E. (FM)", "t (FM)", "Prob.", "S.E. (Shanken)", "t (Shanken)", "Prob.",
             "Factor mean"], rows, ["str", "num", "num", "f3", "prob", "num", "f3", "prob", "num"])
    avg = R.mean(0)
    Zb = np.column_stack([np.ones(N), betas])
    gb = np.linalg.lstsq(Zb, avg, rcond=None)[0]
    eb = avg - Zb @ gb
    r2cs = 1 - eb @ eb / ((avg - avg.mean()) ** 2).sum()
    # pricing error test (chi-square on average pricing errors, FM covariance)
    r.stats([("Cross-sectional R² (avg. returns)", r2cs), ("Mean period R²", np.nanmean(r2s) if r2s else np.nan)],
            [("Shanken c = λ'Σf⁻¹λ", c), ("Mean |pricing error|", float(np.mean(np.abs(eb))))])
    r.note("For traded factors the risk premium should equal the factor mean.")
    fig, ax = plots.new_figure(5.5, 4.5)
    ax.scatter(Zb @ gb, avg, color=plots.PALETTE[0])
    lo, hi = min((Zb @ gb).min(), avg.min()), max((Zb @ gb).max(), avg.max())
    ax.plot([lo, hi], [lo, hi], color=plots.PALETTE[1], lw=1)
    for k, nm in enumerate(Re.columns):
        ax.annotate(nm.upper(), ((Zb @ gb)[k], avg[k]), fontsize=7, alpha=0.7)
    ax.set_xlabel("Predicted mean excess return")
    ax.set_ylabel("Realised mean excess return")
    ax.set_title("Pricing errors")
    fig.tight_layout()
    r.figure(fig, "Predicted vs realised")
    return r
