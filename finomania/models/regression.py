"""Single-equation estimators: LS (with ARMA errors), TSLS, GMM, NLS, quantile
regression, binary choice and count models, plus the residual / stability /
coefficient diagnostics found under an EViews equation's View menu."""
from __future__ import annotations

import math
import re
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from ..report import FigureOutput, Report, fmt_num
from . import plots
from .base import (Design, ModelError, ModelResult, Param, View, build_design, contiguous_check,
                   descriptive_block, info_criteria, jarque_bera, pvalue_t, pvalue_z, safe_inv,
                   sample_header, series_full, split_at, split_terms, wald_from_restrictions)

COV_LABELS = {
    "nonrobust": "Ordinary", "HC0": "White (HC0)", "HC1": "White (HC1, d.f. adjusted)",
    "HC2": "White (HC2)", "HC3": "White (HC3)", "HAC": "Newey-West HAC", "cluster": "Cluster-robust",
}


def resolve_cov(cov: str | None, n: int, hac_lags: int | None = None, cluster=None):
    cov = (cov or "ordinary").lower()
    if cov in ("ordinary", "ols", "nonrobust", "default", "conventional"):
        return "nonrobust", {}
    if cov in ("white", "hc1", "robust", "huber"):
        return "HC1", {}
    if cov in ("hc0", "hc2", "hc3"):
        return cov.upper(), {}
    if cov in ("hac", "nw", "newey", "neweywest", "newey-west"):
        L = hac_lags if hac_lags is not None else int(math.floor(4 * (n / 100) ** (2 / 9)))
        return "HAC", {"maxlags": max(int(L), 0), "use_correction": True}
    if cov in ("cluster", "clustered"):
        if cluster is None:
            raise ModelError("cov=cluster requires cluster=<series>")
        return "cluster", {"groups": cluster}
    raise ModelError(f"Unknown covariance option '{cov}'")


def _dw(e: np.ndarray) -> float:
    return float(np.sum(np.diff(e) ** 2) / np.sum(e ** 2))


def aux_ols(y: np.ndarray, X: np.ndarray):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    ssr = float(e @ e)
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ssr / tss if tss > 0 else np.nan
    return b, e, ssr, r2


# ============================================================================
# common base for single equations
# ============================================================================

class EquationBase(ModelResult):
    object_type = "equation"

    def __init__(self, wf, ev, spec: str, name: str = ""):
        self.wf, self.ev, self.spec, self.name = wf, ev, spec, name
        self.command = ""

    # subclasses fill these
    names: list[str]
    params: np.ndarray
    bse: np.ndarray
    cov: np.ndarray
    resid_: np.ndarray
    fitted_: np.ndarray
    yv: np.ndarray
    mask: np.ndarray
    df_resid: int = 0
    logl: float = np.nan

    # ---- series accessors
    def resid(self) -> pd.Series:
        return series_full(self.wf, self.resid_, self.mask)

    def fitted(self) -> pd.Series:
        return series_full(self.wf, self.fitted_, self.mask)

    def actual(self) -> pd.Series:
        return series_full(self.wf, self.yv, self.mask)

    # ---- generic stat() access used by expressions, e.g. eq1.@r2
    def stat(self, key: str):
        key = key.lower()
        table = {
            "coefs": lambda i=None: self.params if i is None else float(self.params[int(i) - 1]),
            "stderrs": lambda i=None: self.bse if i is None else float(self.bse[int(i) - 1]),
            "tstats": lambda i=None: (self.params / self.bse) if i is None else float(self.params[int(i) - 1] / self.bse[int(i) - 1]),
            "pvals": lambda i=None: getattr(self, "pvalues", None) if i is None else float(self.pvalues[int(i) - 1]),
            "coefcov": lambda i=None, j=None: self.cov if i is None else float(self.cov[int(i) - 1, int(j) - 1]),
        }
        if key in table:
            return table[key]
        simple = getattr(self, "stats_dict", {})
        if key in simple:
            return simple[key]
        raise ModelError(f"@{key} is not available for this equation")

    # ---- common views
    def base_views(self) -> list[View]:
        return [
            View("output", "Estimation Output", lambda: self.report()),
            View("representations", "Representations", self.view_representations),
            View("resids", "Actual, Fitted, Residual Graph", self.view_afr_graph, group="Actual, Fitted, Residual"),
            View("afrtable", "Actual, Fitted, Residual Table", self.view_afr_table, group="Actual, Fitted, Residual"),
            View("correl", "Correlogram - Q-statistics", self.view_correlogram,
                 [Param("lags", "Lags to include", "int", 24)], group="Residual Diagnostics"),
            View("correlsq", "Correlogram Squared Residuals", self.view_correlogram_sq,
                 [Param("lags", "Lags to include", "int", 24)], group="Residual Diagnostics"),
            View("hist", "Histogram - Normality Test", self.view_hist, group="Residual Diagnostics"),
            View("archtest", "ARCH LM Test", self.view_archtest, [Param("lags", "Lags", "int", 1)],
                 group="Residual Diagnostics"),
            View("wald", "Wald Test - Coefficient Restrictions", self.view_wald,
                 [Param("restrictions", "Restrictions (e.g. c(2)=0, c(3)=c(4))", "text", "c(2)=0")],
                 group="Coefficient Diagnostics"),
            View("cinterval", "Confidence Intervals", self.view_cinterval,
                 [Param("levels", "Levels", "text", "0.90 0.95 0.99")], group="Coefficient Diagnostics"),
            View("coefcov", "Covariance Matrix", self.view_coefcov, group="Coefficient Diagnostics"),
        ]

    def base_procs(self) -> list[View]:
        return [
            View("makeresids", "Make Residual Series...", self.proc_makeresids,
                 [Param("name", "Series name", "text", "resid01")], kind="proc"),
            View("makefitted", "Make Fitted Series...", self.proc_makefitted,
                 [Param("name", "Series name", "text", "fitted01")], kind="proc"),
        ]

    def views(self):
        return self.base_views()

    def procs(self):
        return self.base_procs()

    def view_representations(self) -> Report:
        r = Report("Representations")
        r.text("Estimation Command:", "title")
        r.text(self.command or self.spec, "mono")
        r.text("Estimation Equation:", "title")
        terms = [n for n in self.names]
        r.text(f"{getattr(self, 'yname', 'Y')} = " + " + ".join(f"C({i + 1})*{t}" if t != "C" else f"C({i + 1})"
                                                        for i, t in enumerate(terms)), "mono")
        r.text("Substituted Coefficients:", "title")
        r.text(f"{getattr(self, 'yname', 'Y')} = " + " + ".join(
            (f"{fmt_num(self.params[i])}*{t}" if t != "C" else fmt_num(self.params[i]))
            for i, t in enumerate(terms)).replace("+ -", "- "), "mono")
        return r

    def view_afr_graph(self):
        return FigureOutput(plots.afr_plot(self.actual()[self.mask], self.fitted()[self.mask],
                                           self.resid()[self.mask]), "Actual, Fitted, Residual")

    def view_afr_table(self) -> Report:
        r = Report("Actual, Fitted, Residual Table")
        labs = np.array(self.wf.labels())[self.mask]
        sd = np.std(self.resid_)
        rows = []
        for lab, a, f, e in zip(labs, self.yv, self.fitted_, self.resid_):
            z = e / sd if sd > 0 else 0
            bar = ("*" * min(int(abs(z) * 4), 12))
            plot = (bar.rjust(12) + "|" + " " * 12) if z < 0 else (" " * 12 + "|" + bar.ljust(12))
            rows.append([lab, a, f, e, plot])
        r.table(["obs", "Actual", "Fitted", "Residual", "Residual Plot"], rows,
                ["str", "num", "num", "num", "str"], align=["l", "r", "r", "r", "l"])
        return r

    def view_correlogram(self, lags: int = 24) -> Report:
        from .stats import correlogram_report
        return correlogram_report(self.resid_, "RESID", int(lags), title="Correlogram of Residuals",
                                  dof_adjust=len(getattr(self, "arma_terms", [])))

    def view_correlogram_sq(self, lags: int = 24) -> Report:
        from .stats import correlogram_report
        return correlogram_report(self.resid_ ** 2, "RESID^2", int(lags), title="Correlogram of Residuals Squared")

    def view_hist(self):
        from .stats import histogram_output
        return histogram_output(self.resid_, "Residuals", self.wf.sample_label(self.mask))

    def view_archtest(self, lags: int = 1) -> Report:
        return arch_lm_test(self.resid_, int(lags), self.wf, self.mask)

    def view_wald(self, restrictions: str = "c(2)=0") -> Report:
        return wald_from_restrictions(restrictions, self.params, self.cov, self.df_resid or None, self.names)

    def view_cinterval(self, levels="0.90 0.95 0.99") -> Report:
        lv = [float(x) for x in re.split(r"[\s,]+", str(levels).strip()) if x]
        df = self.df_resid or None
        cols = ["Variable", "Coefficient"]
        for l in lv:
            cols += [f"{int(l * 100)}% Low", f"{int(l * 100)}% High"]
        rows = []
        for i, n in enumerate(self.names):
            row = [n, self.params[i]]
            for l in lv:
                q = stats.t.ppf(0.5 + l / 2, df) if df else stats.norm.ppf(0.5 + l / 2)
                row += [self.params[i] - q * self.bse[i], self.params[i] + q * self.bse[i]]
            rows.append(row)
        return Report("Coefficient Confidence Intervals").table(cols, rows)

    def view_coefcov(self) -> Report:
        r = Report("Coefficient Covariance Matrix")
        r.table([""] + self.names, [[n] + list(self.cov[i]) for i, n in enumerate(self.names)])
        return r

    def proc_makeresids(self, name: str = "resid01"):
        self.wf.set_series(name, self.resid(), respect_sample=False)
        return f"Residuals saved to series {name.upper()}"

    def proc_makefitted(self, name: str = "fitted01"):
        self.wf.set_series(name, self.fitted(), respect_sample=False)
        return f"Fitted values saved to series {name.upper()}"


# ============================================================================
# Least squares (optionally with ARMA errors)
# ============================================================================

class LSResult(EquationBase):
    method_label = "Least Squares"

    def __init__(self, wf, ev, spec: str, cov: str = "ordinary", hac_lags: int | None = None,
                 cluster: str | None = None, name: str = "", wls: str | None = None, method: str = "ml"):
        super().__init__(wf, ev, spec, name)
        self.d: Design = build_design(wf, ev, spec)
        d = self.d
        self.yname = d.yname
        self.mask = d.mask
        self.weights = None
        if wls:
            w = ev.eval(wls, ignore_sample=True, as_series=True)
            self.mask = self.mask & w.notna().values
            d = self.d = build_design(wf, ev, spec, mask=self.mask)
            self.weights = w[self.mask].values
        self.arma = bool(d.ar or d.ma or d.sar or d.sma)
        self.cov_label = ""
        self.arma_terms: list[str] = []
        if self.arma:
            self._fit_arma(cov)
        else:
            self._fit_ols(cov, hac_lags, cluster)
        self._stats()

    # -------------------------------------------------------------- OLS
    def _fit_ols(self, cov, hac_lags, cluster):
        import statsmodels.api as sm
        d = self.d
        y, X = d.y.values, d.X.values
        if X.shape[1] == 0:
            raise ModelError("No regressors specified (did you forget C?)")
        if np.linalg.matrix_rank(X) < X.shape[1]:
            raise ModelError("Near singular matrix: regressors are perfectly collinear")
        groups = None
        if cluster:
            groups = self.ev.eval(cluster, ignore_sample=True, as_series=True)[self.mask].values
        ctype, kw = resolve_cov(cov, len(y), hac_lags, groups)
        if self.weights is not None:
            mod = sm.WLS(y, X, weights=self.weights)
        else:
            mod = sm.OLS(y, X)
        res = mod.fit(cov_type=ctype, cov_kwds=kw or None)
        self.sm = res
        self.params = np.asarray(res.params)
        self.bse = np.asarray(res.bse)
        self.cov = np.asarray(res.cov_params())
        self.tvalues = self.params / self.bse
        self.df_resid = int(res.df_resid)
        self.pvalues = pvalue_t(self.tvalues, self.df_resid) if ctype != "cluster" else np.asarray(res.pvalues)
        self.names = list(d.xnames)
        self.yv = y
        self.fitted_ = X @ self.params
        self.resid_ = y - self.fitted_
        self.cov_type = ctype
        if ctype == "HAC":
            self.cov_label = (f"HAC standard errors & covariance (Bartlett kernel, Newey-West fixed bandwidth = "
                              f"{kw['maxlags'] + 1:.4f})")
        elif ctype != "nonrobust":
            self.cov_label = f"{COV_LABELS[ctype]} standard errors & covariance"
        self.k = X.shape[1]
        self.logl_n = None

    # -------------------------------------------------------------- ARMA
    def _fit_arma(self, cov):
        from statsmodels.tsa.arima.model import ARIMA
        d = self.d
        if not contiguous_check(d.mask):
            raise ModelError("ARMA estimation requires a contiguous sample")
        exog = d.X.reset_index(drop=True) if d.X.shape[1] else None
        order = (d.ar if d.ar else 0, 0, d.ma if d.ma else 0)
        seas = (0, 0, 0, 0)
        if d.sar or d.sma:
            s = (d.sar or d.sma)[0]
            seas = (1 if d.sar else 0, 0, 1 if d.sma else 0, s)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = ARIMA(d.y.values, exog=exog, order=order, seasonal_order=seas, trend="n")
            ctype = "robust" if str(cov).lower() in ("white", "robust", "hc1", "hc0") else "opg"
            res = mod.fit(cov_type=ctype)
        self.sm = res
        names = list(d.xnames)
        pn = list(res.param_names)
        mp = {}
        for p in pn:
            if p in names:
                mp[p] = p
            elif p.startswith("ar.S.L"):
                mp[p] = f"SAR({p[6:]})"
            elif p.startswith("ma.S.L"):
                mp[p] = f"SMA({p[6:]})"
            elif p.startswith("ar.L"):
                mp[p] = f"AR({p[4:]})"
            elif p.startswith("ma.L"):
                mp[p] = f"MA({p[4:]})"
            elif p == "sigma2":
                mp[p] = "SIGMASQ"
            else:
                mp[p] = p.upper()
        self.names = [mp[p] for p in pn]
        self.arma_terms = [n for n in self.names if re.match(r"^S?(AR|MA)\(", n)]
        self.params = np.asarray(res.params)
        # statsmodels ARIMA uses phi(L) with sign convention y = phi y(-1) ...; MA likewise (EViews-compatible)
        self.bse = np.asarray(res.bse)
        self.cov = np.asarray(res.cov_params())
        self.tvalues = self.params / self.bse
        self.yv = d.y.values
        self.resid_ = np.asarray(res.resid)
        self.fitted_ = self.yv - self.resid_
        self.df_resid = len(self.yv) - len(self.params)
        self.pvalues = pvalue_t(self.tvalues, self.df_resid)
        self.k = len(self.params)
        self.cov_type = ctype
        self.cov_label = ("Coefficient covariance computed using outer product of gradients" if ctype == "opg"
                          else "Huber-White robust covariance")
        self.method_label = "ARMA Maximum Likelihood (BFGS)"

    # -------------------------------------------------------------- statistics
    def _stats(self):
        y, e = self.yv, self.resid_
        n = len(y)
        w = self.weights
        if w is not None:
            sw = np.sqrt(w)
            ew, yw = e * sw, y * sw
            ssr = float(ew @ ew)
            tss = float(((yw - np.average(yw)) ** 2).sum())
        else:
            ssr = float(e @ e)
            tss = float(((y - y.mean()) ** 2).sum())
        k = self.k
        r2 = 1 - ssr / tss if tss > 0 else np.nan
        adj = 1 - (1 - r2) * (n - 1) / (n - k) if n > k else np.nan
        if self.arma:
            logl = float(self.sm.llf)
        else:
            logl = -n / 2 * (1 + math.log(2 * math.pi) + math.log(ssr / n))
        aic, sc, hq = info_criteria(logl, k, n)
        se_reg = math.sqrt(ssr / (n - k)) if n > k else np.nan
        kslope = k - (1 if self.d.has_const else 0) - len(self.arma_terms) - (1 if "SIGMASQ" in self.names else 0)
        if self.d.has_const and kslope > 0 and n > k:
            F = (r2 / (k - 1)) / ((1 - r2) / (n - k)) if not self.arma else (r2 / (k - 1)) / ((1 - r2) / (n - k))
            Fp = float(stats.f.sf(F, k - 1, n - k))
        else:
            F, Fp = np.nan, np.nan
        waldF = waldp = np.nan
        if not self.arma and self.cov_type != "nonrobust" and self.d.has_const and kslope > 0:
            idx = [i for i, nm in enumerate(self.names) if nm != "C"]
            R = np.zeros((len(idx), k))
            for r_, i in enumerate(idx):
                R[r_, i] = 1
            rb = R @ self.params
            W = float(rb @ safe_inv(R @ self.cov @ R.T) @ rb)
            waldF = W / len(idx)
            waldp = float(stats.f.sf(waldF, len(idx), n - k))
        self.stats_dict = {
            "r2": r2, "rbar2": adj, "se": se_reg, "ssr": ssr, "logl": logl, "aic": aic, "sc": sc, "schwarz": sc,
            "hq": hq, "dw": _dw(e), "f": F, "fprob": Fp, "meandep": float(np.mean(y)),
            "sddep": float(np.std(y, ddof=1)), "regobs": n, "ncoef": k, "waldf": waldF, "waldfprob": waldp,
        }
        self.logl = logl
        self.n = n

    # -------------------------------------------------------------- output
    def report(self) -> Report:
        s = self.stats_dict
        r = Report(f"Equation: {self.name.upper()}" if self.name else "Equation")
        hdr = [f"Dependent Variable: {self.yname}", f"Method: {self.method_label}"]
        if self.weights is not None:
            hdr[1] = "Method: Weighted Least Squares"
        hdr += sample_header(self.wf, self.mask)
        if self.arma:
            hdr.append(f"Convergence achieved; log-likelihood = {fmt_num(self.logl)}")
        if self.cov_label:
            hdr.append(self.cov_label)
        r.header(hdr, stamp=True)
        r.coef_table(self.names, self.params, self.bse, self.tvalues, self.pvalues)
        left = [("R-squared", s["r2"]), ("Adjusted R-squared", s["rbar2"]), ("S.E. of regression", s["se"]),
                ("Sum squared resid", s["ssr"]), ("Log likelihood", s["logl"])]
        if not math.isnan(s["f"]):
            left += [("F-statistic", s["f"]), ("Prob(F-statistic)", s["fprob"])]
        if not math.isnan(s.get("waldf", np.nan)):
            left += [("Wald F-statistic", s["waldf"]), ("Prob(Wald F-statistic)", s["waldfprob"])]
        right = [("Mean dependent var", s["meandep"]), ("S.D. dependent var", s["sddep"]),
                 ("Akaike info criterion", s["aic"]), ("Schwarz criterion", s["sc"]),
                 ("Hannan-Quinn criter.", s["hq"]), ("Durbin-Watson stat", s["dw"])]
        r.stats(left, right)
        if self.arma:
            roots = []
            try:
                ar_r = 1 / np.asarray(self.sm.arroots) if len(self.sm.arroots) else []
                ma_r = 1 / np.asarray(self.sm.maroots) if len(self.sm.maroots) else []
                if len(ar_r):
                    roots.append(["Inverted AR Roots", "  ".join(_cfmt(z) for z in ar_r)])
                if len(ma_r):
                    roots.append(["Inverted MA Roots", "  ".join(_cfmt(z) for z in ma_r)])
                if roots:
                    r.table(["", "Roots"], roots, ["str", "str"], align=["l", "l"])
                    if len(ar_r) and np.max(np.abs(ar_r)) >= 1:
                        r.warn("Estimated AR process is nonstationary")
            except Exception:
                pass
        return r

    # -------------------------------------------------------------- views
    def views(self):
        v = self.base_views()
        if not self.arma:
            v += [
                View("auto", "Serial Correlation LM Test", self.view_auto, [Param("lags", "Lags to include", "int", 2)],
                     group="Residual Diagnostics"),
                View("hettest", "Heteroskedasticity Tests", self.view_hettest,
                     [Param("type", "Test type", "choice", "bpg", ["bpg", "harvey", "glejser", "white", "arch"]),
                      Param("cross", "White: include cross terms", "bool", True),
                      Param("lags", "ARCH lags", "int", 1)], group="Residual Diagnostics"),
                View("white", "White Heteroskedasticity Test", self.view_white,
                     [Param("cross", "Include cross terms", "bool", True)], group="Residual Diagnostics"),
                View("reset", "Ramsey RESET Test", self.view_reset, [Param("terms", "Number of fitted terms", "int", 1)],
                     group="Stability Diagnostics"),
                View("chow", "Chow Breakpoint Test", self.view_chow,
                     [Param("dates", "Breakpoint date(s)", "text", "")], group="Stability Diagnostics"),
                View("cusum", "Recursive Estimates: CUSUM", lambda: self.view_cusum(False), group="Stability Diagnostics"),
                View("cusumsq", "Recursive Estimates: CUSUM of Squares", lambda: self.view_cusum(True),
                     group="Stability Diagnostics"),
                View("rls", "Recursive Coefficients", self.view_rls, group="Stability Diagnostics"),
                View("testdrop", "Redundant Variables Test", self.view_testdrop,
                     [Param("vars", "Variables to drop", "text", "")], group="Coefficient Diagnostics"),
                View("testadd", "Omitted Variables Test", self.view_testadd,
                     [Param("vars", "Variables to add", "text", "")], group="Coefficient Diagnostics"),
                View("vif", "Variance Inflation Factors", self.view_vif, group="Coefficient Diagnostics"),
            ]
        return v

    def procs(self):
        return self.base_procs() + [
            View("forecast", "Forecast...", self.proc_forecast,
                 [Param("name", "Forecast name", "text", f"{self.yname.lower()}f"),
                  Param("method", "Method", "choice", "dynamic", ["dynamic", "static"]),
                  Param("smpl", "Forecast sample (blank = current sample)", "text", ""),
                  Param("se", "S.E. series name (optional)", "text", "")], kind="proc"),
        ]

    # ---- LM serial correlation (Breusch-Godfrey)
    def view_auto(self, lags: int = 2) -> Report:
        p = int(lags)
        e = self.resid_
        X = self.d.X.values
        n = len(e)
        E = np.column_stack([np.concatenate([np.zeros(j), e[:-j]]) for j in range(1, p + 1)])
        Z = np.column_stack([X, E])
        b, u, ssr_u, r2 = aux_ols(e, Z)
        ssr_r = float(e @ e)
        k = X.shape[1]
        F = ((ssr_r - ssr_u) / p) / (ssr_u / (n - k - p))
        LM = n * r2
        r = Report("Breusch-Godfrey Serial Correlation LM Test")
        r.text(f"Null hypothesis: No serial correlation at up to {p} lag{'s' if p > 1 else ''}")
        r.stats([("F-statistic", F), ("Obs*R-squared", LM)],
                [(f"Prob. F({p},{n - k - p})", stats.f.sf(F, p, n - k - p)),
                 (f"Prob. Chi-Square({p})", stats.chi2.sf(LM, p))])
        names = self.names + [f"RESID(-{j})" for j in range(1, p + 1)]
        se = np.sqrt(np.diag((u @ u / (n - Z.shape[1])) * safe_inv(Z.T @ Z)))
        r.section("Test Equation: Dependent Variable: RESID")
        r.note("Presample missing value lagged residuals set to zero.")
        r.coef_table(names, b, se, b / se, pvalue_t(b / se, n - Z.shape[1]))
        r.stats([("R-squared", r2)], [("Durbin-Watson stat", _dw(u))])
        return r

    def _het_aux(self, z: np.ndarray, Z: np.ndarray, title: str, names: list[str], extra=None) -> Report:
        n = len(z)
        b, u, ssr, r2 = aux_ols(z, Z)
        q = Z.shape[1] - 1
        F = (r2 / q) / ((1 - r2) / (n - q - 1))
        r = Report(title)
        r.text("Null hypothesis: Homoskedasticity")
        left = [("F-statistic", F), ("Obs*R-squared", n * r2)]
        right = [(f"Prob. F({q},{n - q - 1})", stats.f.sf(F, q, n - q - 1)),
                 (f"Prob. Chi-Square({q})", stats.chi2.sf(n * r2, q))]
        if extra:
            left.append(extra[0])
            right.append(extra[1])
        r.stats(left, right)
        se = np.sqrt(np.diag((u @ u / (n - Z.shape[1])) * safe_inv(Z.T @ Z)))
        r.section("Test Equation")
        r.coef_table(names, b, se, b / se, pvalue_t(b / se, n - Z.shape[1]))
        r.stats([("R-squared", r2), ("Adjusted R-squared", 1 - (1 - r2) * (n - 1) / (n - Z.shape[1]))], [])
        return r

    def view_hettest(self, type: str = "bpg", cross: bool = True, lags: int = 1) -> Report:  # noqa: A002
        e = self.resid_
        X = self.d.X.values
        if not self.d.has_const:
            X = np.column_stack([np.ones(len(e)), X])
            nm = ["C"] + self.names
        else:
            nm = self.names
        t = (type or "bpg").lower()
        if t in ("bpg", "bp", "breusch"):
            s2 = e @ e / len(e)
            z = e ** 2 / s2
            b, u, ssr, r2 = aux_ols(z, X)
            ess = float(((X @ b - z.mean()) ** 2).sum())
            q = X.shape[1] - 1
            return self._het_aux(e ** 2, X, "Heteroskedasticity Test: Breusch-Pagan-Godfrey", nm,
                                 (("Scaled explained SS", ess / 2), (f"Prob. Chi-Square({q})", stats.chi2.sf(ess / 2, q))))
        if t == "harvey":
            return self._het_aux(np.log(e ** 2 + 1e-300), X, "Heteroskedasticity Test: Harvey", nm)
        if t == "glejser":
            return self._het_aux(np.abs(e), X, "Heteroskedasticity Test: Glejser", nm)
        if t == "white":
            return self.view_white(cross)
        if t == "arch":
            return self.view_archtest(lags)
        raise ModelError(f"Unknown heteroskedasticity test '{type}'")

    def view_white(self, cross: bool = True) -> Report:
        e = self.resid_
        X = self.d.X
        cols = [c for c in X.columns if c != "C"]
        Z = {"C": np.ones(len(e))}
        for i, a in enumerate(cols):
            Z[a] = X[a].values
        for i, a in enumerate(cols):
            for j, b in enumerate(cols):
                if j < i:
                    continue
                if i == j:
                    Z[f"{a}^2"] = X[a].values ** 2
                elif cross:
                    Z[f"{a}*{b}"] = X[a].values * X[b].values
        names = list(Z)
        M = np.column_stack(list(Z.values()))
        keep = [0]
        for j in range(1, M.shape[1]):
            if np.linalg.matrix_rank(M[:, keep + [j]]) == len(keep) + 1:
                keep.append(j)
        M = M[:, keep]
        names = [names[j] for j in keep]
        return self._het_aux(e ** 2, M, "Heteroskedasticity Test: White" + ("" if cross else " (no cross terms)"), names)

    def view_reset(self, terms: int = 1) -> Report:
        p = int(terms)
        y, X = self.yv, self.d.X.values
        f = self.fitted_
        Z = np.column_stack([X] + [f ** (j + 2) for j in range(p)])
        _, _, ssr_u, _ = aux_ols(y, Z)
        ssr_r = float(self.resid_ @ self.resid_)
        n, k = len(y), X.shape[1]
        F = ((ssr_r - ssr_u) / p) / (ssr_u / (n - k - p))
        LR = n * math.log(ssr_r / ssr_u)
        r = Report("Ramsey RESET Test")
        r.text(f"Omitted variables: Powers of fitted values from 2 to {p + 1}")
        r.table(["", "Value", "df", "Probability"],
                [["t-statistic", math.sqrt(F) if p == 1 else np.nan, str(n - k - p),
                  stats.t.sf(math.sqrt(F), n - k - p) * 2 if p == 1 else np.nan],
                 ["F-statistic", F, f"({p}, {n - k - p})", stats.f.sf(F, p, n - k - p)],
                 ["Likelihood ratio", LR, str(p), stats.chi2.sf(LR, p)]], ["str", "num", "str", "prob"])
        return r

    def view_chow(self, dates: str = "") -> Report:
        from ..core.dates import parse_date
        if not str(dates).strip():
            raise ModelError("Specify breakpoint date(s), e.g. eq.chow 2008m9")
        idx = np.flatnonzero(self.mask)
        tt = self.wf.time_index()[idx]
        cuts = []
        for tok in str(dates).replace(",", " ").split():
            if self.wf.is_dated:
                t = parse_date(tok, self.wf.freq)
                cuts.append(int(np.searchsorted(pd.DatetimeIndex(tt), t)))
            else:
                cuts.append(int(np.searchsorted(np.arange(1, self.wf.nobs + 1)[idx], float(tok))))
        cuts = sorted(set(c for c in cuts if 0 < c < len(idx)))
        y, X = self.yv, self.d.X.values
        n, k = X.shape
        bounds = [0] + cuts + [n]
        ssr_u = 0.0
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b - a <= k:
                raise ModelError("Subsample too small for Chow test")
            ssr_u += aux_ols(y[a:b], X[a:b])[2]
        ssr_r = float(self.resid_ @ self.resid_)
        m = len(cuts)
        q = m * k
        F = ((ssr_r - ssr_u) / q) / (ssr_u / (n - (m + 1) * k))
        LR = n * math.log(ssr_r / ssr_u)
        r = Report("Chow Breakpoint Test")
        r.text("Breakpoints: " + " ".join(self.wf.labels()[idx[c]] for c in cuts))
        r.text("Null Hypothesis: No breaks at specified breakpoints")
        r.stats([("F-statistic", F), ("Log likelihood ratio", LR)],
                [(f"Prob. F({q},{n - (m + 1) * k})", stats.f.sf(F, q, n - (m + 1) * k)),
                 (f"Prob. Chi-Square({q})", stats.chi2.sf(LR, q))])
        return r

    def _recursive(self):
        y, X = self.yv, self.d.X.values
        n, k = X.shape
        w = np.full(n, np.nan)
        B = np.full((n, k), np.nan)
        SE = np.full((n, k), np.nan)
        for t in range(k, n):
            Xt, yt = X[:t], y[:t]
            XtXi = safe_inv(Xt.T @ Xt)
            b = XtXi @ Xt.T @ yt
            x = X[t]
            f = 1 + x @ XtXi @ x
            w[t] = (y[t] - x @ b) / math.sqrt(f)
            bt = safe_inv(X[:t + 1].T @ X[:t + 1]) @ X[:t + 1].T @ y[:t + 1]
            et = y[:t + 1] - X[:t + 1] @ bt
            s2 = et @ et / max(t + 1 - k, 1)
            B[t] = bt
            SE[t] = np.sqrt(np.diag(s2 * safe_inv(X[:t + 1].T @ X[:t + 1])))
        return w, B, SE

    def view_cusum(self, squares: bool = False):
        w, _, _ = self._recursive()
        n, k = self.d.X.shape
        ww = w[k:]
        idx = self.d.y.index[k:]
        fig, ax = plots.new_figure()
        x = idx if isinstance(idx, pd.DatetimeIndex) else np.arange(len(ww))
        if not squares:
            s = np.nanstd(ww, ddof=1)
            cus = np.cumsum(ww) / s
            a = 0.948
            T = len(ww)
            tt = np.arange(1, T + 1)
            bound = a * math.sqrt(T) + 2 * a * tt / math.sqrt(T)
            ax.plot(x, cus, color=plots.PALETTE[0], lw=1.3, label="CUSUM")
            ax.plot(x, bound, color=plots.PALETTE[1], ls="--", lw=1, label="5% Significance")
            ax.plot(x, -bound, color=plots.PALETTE[1], ls="--", lw=1)
            ax.set_title("CUSUM test")
        else:
            cs = np.cumsum(ww ** 2) / np.sum(ww ** 2)
            T = len(ww)
            tt = np.arange(1, T + 1)
            c0 = 1.358 / math.sqrt(T / 2) / 1.0 if T > 60 else 0.2
            ax.plot(x, cs, color=plots.PALETTE[0], lw=1.3, label="CUSUM of Squares")
            ax.plot(x, tt / T + c0, color=plots.PALETTE[1], ls="--", lw=1, label="5% Significance")
            ax.plot(x, tt / T - c0, color=plots.PALETTE[1], ls="--", lw=1)
            ax.set_title("CUSUM of squares test")
        ax.legend(frameon=False)
        fig.tight_layout()
        return FigureOutput(fig, "CUSUM of Squares" if squares else "CUSUM")

    def view_rls(self):
        _, B, SE = self._recursive()
        k = B.shape[1]
        panels = [(f"Recursive {self.names[j]} Estimates", B[k:, j], B[k:, j] - 2 * SE[k:, j],
                   B[k:, j] + 2 * SE[k:, j]) for j in range(k)]
        return FigureOutput(plots.grid_plot(panels, min(k, 2), "Recursive coefficients ± 2 S.E.", zero=False),
                            "Recursive Coefficients")

    def _restricted_compare(self, other: "LSResult", added: bool) -> Report:
        big, small = (other, self) if added else (self, other)
        q = big.k - small.k
        n = big.n
        F = ((small.stats_dict["ssr"] - big.stats_dict["ssr"]) / q) / (big.stats_dict["ssr"] / (n - big.k))
        LR = 2 * (big.logl - small.logl)
        title = "Omitted Variables Test" if added else "Redundant Variables Test"
        r = Report(title)
        r.table(["", "Value", "df", "Probability"],
                [["F-statistic", F, f"({q}, {n - big.k})", stats.f.sf(F, q, n - big.k)],
                 ["Likelihood ratio", LR, str(q), stats.chi2.sf(LR, q)]], ["str", "num", "str", "prob"])
        r.extend(other.report(), with_title=False)
        return r

    def view_testdrop(self, vars: str = "") -> Report:  # noqa: A002
        drop = {v.upper() for v in split_terms(vars)}
        terms = [t for t in self.d.terms[1:] if t.upper() not in drop]
        if len(terms) == len(self.d.terms) - 1:
            raise ModelError("None of the listed variables are in the equation")
        other = _ls_on_mask(self, " ".join([self.d.terms[0]] + terms))
        return self._restricted_compare(other, added=False)

    def view_testadd(self, vars: str = "") -> Report:  # noqa: A002
        other = _ls_on_mask(self, self.spec + " " + vars)
        if other.n != self.n:
            raise ModelError("Added variables change the sample; test requires a common sample")
        return self._restricted_compare(other, added=True)

    def view_vif(self) -> Report:
        X = self.d.X
        rows = []
        XtXi = safe_inv(X.values.T @ X.values)
        for j, c in enumerate(X.columns):
            if c == "C":
                continue
            others = [k for k in range(X.shape[1]) if k != j]
            _, _, _, r2 = aux_ols(X.values[:, j], X.values[:, others])
            unc = XtXi[j, j] * (X.values[:, j] @ X.values[:, j])
            rows.append([c, self.cov[j, j], unc, 1 / (1 - r2) if r2 < 1 else np.inf])
        return Report("Variance Inflation Factors").table(
            ["Variable", "Coefficient Variance", "Uncentered VIF", "Centered VIF"], rows)

    # ---- forecasting
    def proc_forecast(self, name: str = "", method: str = "dynamic", smpl: str = "", se: str = ""):
        name = name or f"{self.yname.lower()}f"
        fmask = self.wf.parse_sample(smpl) if smpl else self.wf.sample.copy()
        if self.arma:
            f, fse = self._forecast_arma(fmask, method)
        else:
            f, fse = self._forecast_ls(fmask, method)
        self.wf.set_series(name, f, respect_sample=False)
        if se:
            self.wf.set_series(se, fse, respect_sample=False)
        return forecast_report(self.wf, name, self.yname, self.ev, f, fse, fmask, method)

    def _forecast_ls(self, fmask, method):
        wf, ev = self.wf, self.ev
        dep = self.d.terms[0]
        is_simple = re.match(r"^[a-z_][a-z0-9_]*$", dep.lower()) is not None
        regs = self.d.terms[1:]
        dyn_terms = [t for t in regs if is_simple and re.search(rf"\b{dep.lower()}\s*\(\s*-\d+\s*\)", t.lower())]
        idx = np.flatnonzero(fmask)
        XtXi = safe_inv(self.d.X.values.T @ self.d.X.values)
        s = self.stats_dict["se"]
        f = pd.Series(np.nan, index=wf.data.index)
        fse = pd.Series(np.nan, index=wf.data.index)
        if method == "static" or not dyn_terms:
            X = self._xmat(regs, ev)
            fv = X.values @ self.params
            f.iloc[idx] = fv[idx]
            for i in idx:
                x = X.values[i]
                if not np.isnan(x).any():
                    fse.iloc[i] = s * math.sqrt(1 + x @ XtXi @ x)
            return f, fse
        # dynamic: recursively replace the dependent variable by its forecast
        ycur = wf.series(dep.lower()).astype(float).copy()
        for i in idx:
            ov = {dep.lower(): ycur}
            row = []
            for t in regs:
                if t.lower() == "c":
                    row.append(1.0)
                else:
                    v = _eval_override(ev, t, ov)
                    row.append(float(v.iloc[i]) if isinstance(v, pd.Series) else float(v))
            x = np.array(row)
            val = float(x @ self.params) if not np.isnan(x).any() else np.nan
            f.iloc[i] = val
            ycur.iloc[i] = val
        # approximate S.E. (ignores accumulation of dynamic error)
        fse.iloc[idx] = s
        return f, fse

    def _xmat(self, regs, ev):
        cols = {}
        for t in regs:
            if t.lower() == "c":
                cols["C"] = pd.Series(1.0, index=self.wf.data.index)
            else:
                cols[t.upper()] = ev.eval(t, ignore_sample=True, as_series=True)
        return pd.DataFrame(cols, index=self.wf.data.index)

    def _forecast_arma(self, fmask, method):
        est = np.flatnonzero(self.mask)
        a, b = est[0], est[-1]
        idx = np.flatnonzero(fmask)
        idx = idx[idx >= a]
        f = pd.Series(np.nan, index=self.wf.data.index)
        fse = pd.Series(np.nan, index=self.wf.data.index)
        if len(idx) == 0:
            return f, fse
        start, end = idx[0] - a, idx[-1] - a
        exog_f = None
        if self.d.X.shape[1] and idx[-1] > b:
            X = self._xmat(self.d.terms[1:] and [t for t in self.d.terms[1:] if not re.match(r"^s?(ar|ma)\(", t.lower())],
                           self.ev)
            exog_f = X.values[b + 1: idx[-1] + 1]
            if np.isnan(exog_f).any():
                raise ModelError("Missing regressor values in the forecast period")
        dyn = (start if method == "dynamic" and start <= (b - a) else False)
        pred = self.sm.get_prediction(start=start, end=end, exog=exog_f, dynamic=dyn)
        pm = np.asarray(pred.predicted_mean)
        ps = np.sqrt(np.asarray(pred.var_pred_mean))
        pos = np.arange(idx[0], idx[-1] + 1)
        f.iloc[pos] = pm
        fse.iloc[pos] = ps
        f[~fmask] = np.nan
        fse[~fmask] = np.nan
        return f, fse


def _ls_on_mask(eq: LSResult, spec: str) -> LSResult:
    old = eq.wf._sample
    try:
        eq.wf._sample = eq.mask.copy()
        return LSResult(eq.wf, eq.ev, spec)
    finally:
        eq.wf._sample = old


def _eval_override(ev, expr, overrides: dict):
    code = ev.compile(expr)
    ns = ev.namespace()
    ns.update(overrides)
    old = ev.ignore_sample
    ev.ignore_sample = True
    try:
        return eval(code, {"__builtins__": {}}, ns)  # noqa: S307
    finally:
        ev.ignore_sample = old


def _cfmt(z) -> str:
    z = complex(z)
    if abs(z.imag) < 1e-10:
        return f"{z.real:.2f}"
    return f"{z.real:.2f}{'+' if z.imag >= 0 else '-'}{abs(z.imag):.2f}i"


def forecast_report(wf, name, yname, ev, f: pd.Series, fse: pd.Series, fmask, method) -> list:
    out = []
    r = Report(f"Forecast: {name.upper()}")
    r.header([f"Actual: {yname}", f"Forecast sample: {wf.sample_label(fmask)}",
              f"Method: {method.capitalize()} forecast"])
    try:
        y = ev.eval(yname.lower(), ignore_sample=True, as_series=True)
        ok = fmask & y.notna().values & f.notna().values
        if ok.sum() > 1:
            a, p = y.values[ok], f.values[ok]
            e = a - p
            rmse = math.sqrt(np.mean(e ** 2))
            mae = np.mean(np.abs(e))
            mape = np.mean(np.abs(e / a)) * 100 if np.all(a != 0) else np.nan
            theil = rmse / (math.sqrt(np.mean(a ** 2)) + math.sqrt(np.mean(p ** 2)))
            mse = np.mean(e ** 2)
            bias = (p.mean() - a.mean()) ** 2 / mse
            var = (p.std() - a.std()) ** 2 / mse
            r.stats([("Root Mean Squared Error", rmse), ("Mean Absolute Error", mae),
                     ("Mean Abs. Percent Error", mape), ("Theil Inequality Coefficient", theil)],
                    [("     Bias Proportion", bias), ("     Variance Proportion", var),
                     ("     Covariance Proportion", max(0.0, 1 - bias - var)), ("Included observations", int(ok.sum()))])
    except Exception:
        pass
    out.append(r)
    fig, ax = plots.new_figure()
    fi = f[fmask]
    x = fi.index if not isinstance(fi.index, pd.MultiIndex) else np.arange(len(fi))
    ax.plot(x, fi.values, color=plots.PALETTE[0], lw=1.4, label=name.upper())
    if fse.notna().any():
        lo, hi = (fi - 2 * fse[fmask]).values, (fi + 2 * fse[fmask]).values
        ax.plot(x, lo, color=plots.PALETTE[1], ls="--", lw=0.9, label="± 2 S.E.")
        ax.plot(x, hi, color=plots.PALETTE[1], ls="--", lw=0.9)
    try:
        y = ev.eval(yname.lower(), ignore_sample=True, as_series=True)[fmask]
        if y.notna().any():
            ax.plot(x, y.values, color="#333", lw=1, alpha=0.7, label=yname)
    except Exception:
        pass
    ax.legend(frameon=False)
    ax.set_title(f"Forecast: {name.upper()}")
    fig.tight_layout()
    out.append(FigureOutput(fig, f"Forecast {name.upper()}"))
    return out


# ============================================================================
# ARCH LM test (shared)
# ============================================================================

def arch_lm_test(e: np.ndarray, p: int, wf=None, mask=None, title="Heteroskedasticity Test: ARCH") -> Report:
    e2 = e ** 2
    n = len(e2)
    Y = e2[p:]
    Z = np.column_stack([np.ones(n - p)] + [e2[p - j:n - j] for j in range(1, p + 1)])
    b, u, ssr, r2 = aux_ols(Y, Z)
    m = len(Y)
    F = (r2 / p) / ((1 - r2) / (m - p - 1))
    r = Report(title)
    r.text(f"Null hypothesis: no ARCH effects up to order {p}")
    r.stats([("F-statistic", F), ("Obs*R-squared", m * r2)],
            [(f"Prob. F({p},{m - p - 1})", stats.f.sf(F, p, m - p - 1)),
             (f"Prob. Chi-Square({p})", stats.chi2.sf(m * r2, p))])
    se = np.sqrt(np.diag((u @ u / (m - p - 1)) * safe_inv(Z.T @ Z)))
    r.section("Test Equation: Dependent Variable: RESID^2")
    names = ["C"] + [f"RESID^2(-{j})" for j in range(1, p + 1)]
    r.coef_table(names, b, se, b / se, pvalue_t(b / se, m - p - 1))
    return r


# ============================================================================
# TSLS / GMM  (linearmodels)
# ============================================================================

class IVResult(EquationBase):
    def __init__(self, wf, ev, spec: str, method: str = "tsls", cov: str = "ordinary", hac_lags=None, name="",
                 wtype: str = "robust"):
        super().__init__(wf, ev, spec, name)
        parts = split_at(spec)
        if len(parts) < 2 or not parts[1]:
            raise ModelError("TSLS/GMM specification needs instruments after '@', e.g. tsls y c x @ c z1 z2")
        inst_terms = split_terms(parts[1])
        d = build_design(wf, ev, parts[0], allow_arma=False)
        Zd = build_design(wf, ev, parts[1], require_y=False, allow_arma=False)
        mask = d.mask & Zd.mask
        d = build_design(wf, ev, parts[0], mask=mask, allow_arma=False)
        Zd = build_design(wf, ev, parts[1], mask=d.mask, require_y=False, allow_arma=False)
        self.d, self.mask, self.yname = d, d.mask, d.yname
        inst_upper = [t.upper() for t in inst_terms]
        exog_cols = [c for c in d.xnames if c in inst_upper or (c == "C" and "C" in inst_upper)]
        endog_cols = [c for c in d.xnames if c not in exog_cols]
        instr_cols = [c for c in Zd.xnames if c not in exog_cols]
        if len(instr_cols) < len(endog_cols):
            raise ModelError("Order condition violated: fewer instruments than endogenous regressors")
        from linearmodels.iv import IV2SLS, IVGMM
        y = d.y
        ex = d.X[exog_cols] if exog_cols else None
        en = d.X[endog_cols] if endog_cols else None
        ins = Zd.X[instr_cols] if instr_cols else None
        n = len(y)
        cv = (cov or "ordinary").lower()
        if method == "gmm":
            if wtype in ("hac", "kernel"):
                L = hac_lags if hac_lags is not None else int(4 * (n / 100) ** (2 / 9))
                mod = IVGMM(y, ex, en, ins, weight_type="kernel", bandwidth=L)
                res = mod.fit(cov_type="kernel", bandwidth=L)
            else:
                mod = IVGMM(y, ex, en, ins, weight_type="robust")
                res = mod.fit(cov_type="robust")
            self.method_label = "Generalized Method of Moments"
        else:
            mod = IV2SLS(y, ex, en, ins)
            if cv in ("white", "robust", "hc1", "hc0"):
                res = mod.fit(cov_type="robust")
            elif cv in ("hac", "nw", "kernel"):
                L = hac_lags if hac_lags is not None else int(4 * (n / 100) ** (2 / 9))
                res = mod.fit(cov_type="kernel", bandwidth=L)
            else:
                res = mod.fit(cov_type="unadjusted", debiased=True)
            self.method_label = "Two-Stage Least Squares"
        self.lm = res
        order = list(d.xnames)
        self.names = order
        self.params = np.asarray(res.params[order])
        self.bse = np.asarray(res.std_errors[order])
        self.cov = np.asarray(res.cov.loc[order, order])
        self.tvalues = self.params / self.bse
        self.df_resid = n - len(order)
        self.pvalues = pvalue_t(self.tvalues, self.df_resid)
        self.yv = y.values
        self.fitted_ = d.X[order].values @ self.params
        self.resid_ = self.yv - self.fitted_
        self.inst_list = " ".join(Zd.xnames)
        self.endog_cols, self.instr_cols, self.exog_cols = endog_cols, instr_cols, exog_cols
        e = self.resid_
        ssr = float(e @ e)
        tss = float(((self.yv - self.yv.mean()) ** 2).sum())
        k = len(order)
        r2 = 1 - ssr / tss
        self.stats_dict = {"r2": r2, "rbar2": 1 - (1 - r2) * (n - 1) / (n - k), "se": math.sqrt(ssr / (n - k)),
                           "ssr": ssr, "dw": _dw(e), "meandep": float(self.yv.mean()),
                           "sddep": float(self.yv.std(ddof=1)), "regobs": n, "ncoef": k,
                           "instrank": len(Zd.xnames)}
        try:
            self.stats_dict["jstat"] = float(res.j_stat.stat) if method == "gmm" else float(res.sargan.stat)
            self.stats_dict["jprob"] = float(res.j_stat.pval) if method == "gmm" else float(res.sargan.pval)
        except Exception:
            self.stats_dict["jstat"] = self.stats_dict["jprob"] = np.nan
        self.method = method

    def report(self) -> Report:
        s = self.stats_dict
        r = Report(f"Equation: {self.name.upper()}" if self.name else "Equation")
        hdr = [f"Dependent Variable: {self.yname}", f"Method: {self.method_label}"]
        hdr += sample_header(self.wf, self.mask)
        hdr.append(f"Instrument specification: {self.inst_list}")
        if self.method == "gmm":
            hdr.append("Estimation weighting matrix: " + str(getattr(self.lm, "weight_type", "robust")).upper())
        r.header(hdr, stamp=True)
        r.coef_table(self.names, self.params, self.bse, self.tvalues, self.pvalues)
        jl = "J-statistic" if self.method == "gmm" else "Sargan statistic"
        r.stats([("R-squared", s["r2"]), ("Adjusted R-squared", s["rbar2"]), ("S.E. of regression", s["se"]),
                 ("Durbin-Watson stat", s["dw"]), (jl, s["jstat"]), (f"Prob({jl})", s["jprob"])],
                [("Mean dependent var", s["meandep"]), ("S.D. dependent var", s["sddep"]),
                 ("Sum squared resid", s["ssr"]), ("Instrument rank", s["instrank"])])
        return r

    def views(self):
        return self.base_views() + [
            View("firststage", "First-stage diagnostics", self.view_first, group="IV Diagnostics"),
            View("endogtest", "Endogeneity test (Durbin-Wu-Hausman)", self.view_endog, group="IV Diagnostics"),
        ]

    def view_first(self) -> Report:
        r = Report("First-stage diagnostics")
        try:
            fs = self.lm.first_stage.diagnostics
            rows = [[idx, float(row["rsquared"]), float(row["partial.rsquared"]), float(row["f.stat"]),
                     float(row["f.pval"])] for idx, row in fs.iterrows()]
            r.table(["Endogenous", "R-squared", "Partial R-sq.", "F-statistic", "Prob."], rows,
                    ["str", "num", "num", "num", "prob"])
            r.note("Rule of thumb: first-stage F below 10 indicates weak instruments.")
        except Exception as e:
            r.text(f"Not available: {e}")
        return r

    def view_endog(self) -> Report:
        r = Report("Endogeneity Test")
        try:
            t = self.lm.wu_hausman()
            r.stats([("Wu-Hausman F-statistic", float(t.stat))], [("Probability", float(t.pval))])
            t2 = self.lm.durbin()
            r.stats([("Durbin chi-square", float(t2.stat))], [("Probability", float(t2.pval))])
            r.text("Null hypothesis: regressors " + ", ".join(self.endog_cols) + " are exogenous")
        except Exception as e:
            r.text(f"Not available: {e}")
        return r


# ============================================================================
# Nonlinear least squares   y = c(1) + c(2)*x^c(3)
# ============================================================================

class NLSResult(EquationBase):
    method_label = "Least Squares (Gauss-Newton / Marquardt steps)"

    def __init__(self, wf, ev, spec: str, cov="ordinary", start=None, name="", hac_lags=None):
        super().__init__(wf, ev, spec, name)
        from scipy.optimize import least_squares
        lhs, rhs = _split_eq(spec)
        ks = [int(m) for m in re.findall(r"\bc\s*\(\s*(\d+)\s*\)", rhs.lower() + " " + lhs.lower())]
        if not ks:
            raise ModelError("NLS specification must contain coefficients c(1), c(2), ...")
        K = max(ks)
        if start is not None:
            x0 = np.array([float(v) for v in re.split(r"[\s,]+", str(start).strip()) if v], dtype=float)
            if len(x0) < K:
                x0 = np.concatenate([x0, np.full(K - len(x0), 0.1)])
        else:
            cv = getattr(wf, "cvec", None)
            x0 = np.full(K, 0.1)
            if cv is not None:
                x0[:min(K, len(cv))] = np.where(np.asarray(cv[:K]) != 0, cv[:K], 0.1)
        y = ev.eval(lhs, ignore_sample=True, as_series=True)

        def rhs_vals(c):
            v = ev.eval(rhs, ignore_sample=True, coef=c)
            return v.values if isinstance(v, pd.Series) else np.full(wf.nobs, float(v))

        f0 = rhs_vals(x0)
        mask = wf.sample & y.notna().values & ~np.isnan(f0)
        if mask.sum() <= K:
            raise ModelError("Insufficient observations for NLS")
        yv = y.values[mask]

        def resid(c):
            r = yv - rhs_vals(c)[mask]
            return np.where(np.isfinite(r), r, 1e10)

        sol = least_squares(resid, x0, method="lm", max_nfev=5000 * K, xtol=1e-12, ftol=1e-12)
        b = sol.x
        e = resid(b)
        J = -sol.jac
        n = len(e)
        s2 = e @ e / (n - K)
        JtJi = safe_inv(J.T @ J)
        cv_ = (cov or "ordinary").lower()
        if cv_ in ("white", "hc1", "robust"):
            meat = (J * e[:, None] ** 2).T @ J * n / (n - K)
            V = JtJi @ meat @ JtJi
            self.cov_label = "White heteroskedasticity-consistent standard errors & covariance"
        elif cv_ in ("hac", "nw"):
            L = hac_lags if hac_lags is not None else int(4 * (n / 100) ** (2 / 9))
            g = J * e[:, None]
            S = g.T @ g
            for l in range(1, L + 1):
                w = 1 - l / (L + 1)
                G = g[l:].T @ g[:-l]
                S += w * (G + G.T)
            V = JtJi @ S @ JtJi * n / (n - K)
            self.cov_label = f"HAC standard errors (Bartlett kernel, bandwidth {L + 1})"
        else:
            V = s2 * JtJi
            self.cov_label = ""
        self.names = [f"C({i + 1})" for i in range(K)]
        self.params, self.cov = b, V
        self.bse = np.sqrt(np.diag(V))
        self.tvalues = b / self.bse
        self.df_resid = n - K
        self.pvalues = pvalue_t(self.tvalues, n - K)
        self.yv, self.resid_ = yv, e
        self.fitted_ = yv - e
        self.mask = mask
        self.yname = lhs.upper()
        self.rhs = rhs
        self.converged = sol.success
        self.nfev = sol.nfev
        ssr = float(e @ e)
        tss = float(((yv - yv.mean()) ** 2).sum())
        logl = -n / 2 * (1 + math.log(2 * math.pi) + math.log(ssr / n))
        aic, sc, hq = info_criteria(logl, K, n)
        r2 = 1 - ssr / tss
        self.logl = logl
        self.stats_dict = {"r2": r2, "rbar2": 1 - (1 - r2) * (n - 1) / (n - K), "se": math.sqrt(ssr / (n - K)),
                           "ssr": ssr, "logl": logl, "aic": aic, "sc": sc, "hq": hq, "dw": _dw(e),
                           "meandep": float(yv.mean()), "sddep": float(yv.std(ddof=1)), "regobs": n, "ncoef": K}
        wf.cvec = b.copy()

    def report(self) -> Report:
        s = self.stats_dict
        r = Report(f"Equation: {self.name.upper()}" if self.name else "Equation")
        hdr = [f"Dependent Variable: {self.yname}", f"Method: {self.method_label}"] + sample_header(self.wf, self.mask)
        hdr.append(("Convergence achieved" if self.converged else "Convergence NOT achieved") +
                   f" after {self.nfev} function evaluations")
        if self.cov_label:
            hdr.append(self.cov_label)
        hdr.append(f"{self.yname} = {self.rhs.upper()}")
        r.header(hdr, stamp=True)
        r.coef_table(self.names, self.params, self.bse, self.tvalues, self.pvalues)
        r.stats([("R-squared", s["r2"]), ("Adjusted R-squared", s["rbar2"]), ("S.E. of regression", s["se"]),
                 ("Sum squared resid", s["ssr"]), ("Log likelihood", s["logl"])],
                [("Mean dependent var", s["meandep"]), ("S.D. dependent var", s["sddep"]),
                 ("Akaike info criterion", s["aic"]), ("Schwarz criterion", s["sc"]),
                 ("Hannan-Quinn criter.", s["hq"]), ("Durbin-Watson stat", s["dw"])])
        return r

    def view_representations(self) -> Report:
        r = Report("Representations")
        r.text(self.command or self.spec, "mono")
        sub = self.rhs.lower()
        for i in range(len(self.params), 0, -1):
            sub = re.sub(rf"c\s*\(\s*{i}\s*\)", f"({fmt_num(self.params[i - 1])})", sub)
        r.text(f"{self.yname} = {sub.upper()}", "mono")
        return r


def _split_eq(spec: str) -> tuple[str, str]:
    depth = 0
    for i, ch in enumerate(spec):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "=" and depth == 0:
            prev = spec[i - 1] if i else ""
            nxt = spec[i + 1] if i + 1 < len(spec) else ""
            if prev not in "<>!=" and nxt != "=":
                return spec[:i].strip(), spec[i + 1:].strip()
    raise ModelError("Expected an explicit equation 'y = f(c(1), c(2), ...)'")


# ============================================================================
# Quantile regression
# ============================================================================

class QuantResult(EquationBase):
    method_label = "Quantile Regression"

    def __init__(self, wf, ev, spec: str, tau: float = 0.5, cov: str = "robust", name=""):
        super().__init__(wf, ev, spec, name)
        from statsmodels.regression.quantile_regression import QuantReg
        d = build_design(wf, ev, spec, allow_arma=False)
        self.d, self.mask, self.yname = d, d.mask, d.yname
        self.tau = float(tau)
        vcov = "iid" if str(cov).lower() in ("iid", "ordinary") else "robust"
        res = QuantReg(d.y.values, d.X.values).fit(q=self.tau, vcov=vcov, max_iter=5000)
        self.sm = res
        self.names = d.xnames
        self.params = np.asarray(res.params)
        self.bse = np.asarray(res.bse)
        self.cov = np.asarray(res.cov_params())
        self.tvalues = self.params / self.bse
        n, k = d.X.shape
        self.df_resid = n - k
        self.pvalues = pvalue_t(self.tvalues, n - k)
        self.yv = d.y.values
        self.fitted_ = d.X.values @ self.params
        self.resid_ = self.yv - self.fitted_

        def rho(u):
            return np.sum(u * (self.tau - (u < 0)))

        V = rho(self.resid_)
        qy = np.quantile(self.yv, self.tau)
        V0 = rho(self.yv - qy)
        self.stats_dict = {"r2": 1 - V / V0, "pseudor2": 1 - V / V0, "objective": V, "restrobj": V0,
                           "quantdep": qy, "meandep": float(self.yv.mean()), "sddep": float(self.yv.std(ddof=1)),
                           "se": math.sqrt(self.resid_ @ self.resid_ / (n - k)), "regobs": n}
        self.vcov = vcov

    def report(self) -> Report:
        s = self.stats_dict
        r = Report(f"Equation: {self.name.upper()}" if self.name else "Equation")
        hdr = [f"Dependent Variable: {self.yname}", f"Method: Quantile Regression (tau = {self.tau:g})"]
        hdr += sample_header(self.wf, self.mask)
        hdr.append("Huber Sandwich Standard Errors & Covariance" if self.vcov == "robust" else "IID standard errors")
        r.header(hdr, stamp=True)
        r.coef_table(self.names, self.params, self.bse, self.tvalues, self.pvalues)
        r.stats([("Pseudo R-squared", s["pseudor2"]), ("S.E. of regression", s["se"]),
                 ("Quantile dependent var", s["quantdep"]), ("Objective", s["objective"])],
                [("Mean dependent var", s["meandep"]), ("S.D. dependent var", s["sddep"]),
                 ("Restr. objective", s["restrobj"]), ("Observations", s["regobs"])])
        return r

    def views(self):
        return self.base_views() + [View("qprocess", "Quantile Process Estimates", self.view_process,
                                         [Param("quantiles", "Quantiles", "text", "0.1 0.25 0.5 0.75 0.9")],
                                         group="Quantile Process")]

    def view_process(self, quantiles="0.1 0.25 0.5 0.75 0.9"):
        from statsmodels.regression.quantile_regression import QuantReg
        qs = [float(q) for q in re.split(r"[\s,]+", str(quantiles).strip()) if q]
        rows, est = [], []
        for q in qs:
            res = QuantReg(self.yv, self.d.X.values).fit(q=q, max_iter=5000)
            est.append((q, np.asarray(res.params), np.asarray(res.bse)))
            for i, n in enumerate(self.names):
                rows.append([f"{q:g}", n, res.params[i], res.bse[i], res.params[i] / res.bse[i],
                             pvalue_t(res.params[i] / res.bse[i], self.df_resid)])
        r = Report("Quantile Process Estimates")
        r.table(["Quantile", "Variable", "Coefficient", "Std. Error", "t-Statistic", "Prob."], rows,
                ["str", "str", "num", "num", "num", "prob"])
        panels = []
        for i, n in enumerate(self.names):
            b = np.array([e[1][i] for e in est])
            s = np.array([e[2][i] for e in est])
            panels.append((n, b, b - 2 * s, b + 2 * s))
        fig = plots.grid_plot(panels, min(len(panels), 2), "Quantile process (± 2 S.E.)", zero=False, x=np.array(qs))
        return [r, FigureOutput(fig, "Quantile Process")]


# ============================================================================
# Binary choice (logit / probit) and count (Poisson / negative binomial)
# ============================================================================

class BinaryResult(EquationBase):
    def __init__(self, wf, ev, spec: str, link: str = "logit", cov="ordinary", name=""):
        super().__init__(wf, ev, spec, name)
        import statsmodels.api as sm
        d = build_design(wf, ev, spec, allow_arma=False)
        self.d, self.mask, self.yname = d, d.mask, d.yname
        yv = d.y.values
        if not set(np.unique(yv)) <= {0.0, 1.0}:
            raise ModelError("Dependent variable must be 0/1 for binary choice models")
        Model = sm.Logit if link == "logit" else sm.Probit
        ctype = "HC0" if str(cov).lower() in ("white", "robust", "huber") else "nonrobust"
        res = Model(yv, d.X.values).fit(disp=0, cov_type=ctype, maxiter=200)
        self.sm, self.link = res, link
        self.method_label = f"ML - Binary {'Logit' if link == 'logit' else 'Probit'} (Newton-Raphson)"
        self.names = d.xnames
        self.params = np.asarray(res.params)
        self.bse = np.asarray(res.bse)
        self.cov = np.asarray(res.cov_params())
        self.tvalues = self.params / self.bse
        self.pvalues = pvalue_z(self.tvalues)
        n, k = d.X.shape
        self.df_resid = 0
        self.yv = yv
        self.fitted_ = np.asarray(res.predict())
        self.resid_ = yv - self.fitted_
        logl, l0 = float(res.llf), float(res.llnull)
        aic, sc, hq = info_criteria(logl, k, n)
        LR = 2 * (logl - l0)
        self.logl = logl
        self.stats_dict = {"r2": 1 - logl / l0, "mcfadden": 1 - logl / l0, "logl": logl, "restrlogl": l0,
                           "lr": LR, "lrprob": float(stats.chi2.sf(LR, k - 1)), "aic": aic, "sc": sc, "hq": hq,
                           "meandep": float(yv.mean()), "sddep": float(yv.std(ddof=1)), "avglogl": logl / n,
                           "se": math.sqrt(self.resid_ @ self.resid_ / (n - k)),
                           "ssr": float(self.resid_ @ self.resid_), "regobs": n,
                           "n0": int((yv == 0).sum()), "n1": int((yv == 1).sum())}
        self.z_label = "z-Statistic"

    def report(self) -> Report:
        s = self.stats_dict
        r = Report(f"Equation: {self.name.upper()}" if self.name else "Equation")
        r.header([f"Dependent Variable: {self.yname}", f"Method: {self.method_label}"]
                 + sample_header(self.wf, self.mask), stamp=True)
        r.coef_table(self.names, self.params, self.bse, self.tvalues, self.pvalues, stat_label="z-Statistic")
        r.stats([("McFadden R-squared", s["mcfadden"]), ("S.D. dependent var", s["sddep"]),
                 ("Akaike info criterion", s["aic"]), ("Schwarz criterion", s["sc"]),
                 ("LR statistic", s["lr"]), ("Prob(LR statistic)", s["lrprob"]),
                 ("Obs with Dep=0", s["n0"]), ("Obs with Dep=1", s["n1"])],
                [("Mean dependent var", s["meandep"]), ("S.E. of regression", s["se"]),
                 ("Sum squared resid", s["ssr"]), ("Log likelihood", s["logl"]),
                 ("Restr. log likelihood", s["restrlogl"]), ("Avg. log likelihood", s["avglogl"]),
                 ("Total obs", s["regobs"]), ("Hannan-Quinn criter.", s["hq"])])
        return r

    def views(self):
        v = [x for x in self.base_views() if x.key not in ("archtest",)]
        return v + [
            View("predict", "Expectation-Prediction Evaluation", self.view_predict,
                 [Param("cutoff", "Success cutoff", "float", 0.5)], group="Binary Diagnostics"),
            View("margeff", "Marginal Effects (at mean)", self.view_margeff, group="Binary Diagnostics"),
        ]

    def view_predict(self, cutoff: float = 0.5) -> Report:
        p = self.fitted_
        yhat = (p > float(cutoff)).astype(int)
        y = self.yv.astype(int)
        tab = np.zeros((2, 2), dtype=int)
        for a, b in zip(y, yhat):
            tab[b, a] += 1
        tot = tab.sum(axis=0)
        correct = [tab[0, 0], tab[1, 1]]
        r = Report("Expectation-Prediction Evaluation for Binary Specification")
        r.text(f"Success cutoff: C = {float(cutoff):g}")
        r.table(["", "Dep=0", "Dep=1", "Total"],
                [["P(Dep=1)<=C", tab[0, 0], tab[0, 1], tab[0].sum()], ["P(Dep=1)>C", tab[1, 0], tab[1, 1], tab[1].sum()],
                 ["Total", tot[0], tot[1], tot.sum()], ["Correct", correct[0], correct[1], sum(correct)],
                 ["% Correct", 100 * correct[0] / max(tot[0], 1), 100 * correct[1] / max(tot[1], 1),
                  100 * sum(correct) / tot.sum()]], ["str", "int", "int", "int"], rules_after=[2])
        return r

    def view_margeff(self) -> Report:
        m = self.sm.get_margeff(at="mean")
        sf = m.summary_frame()
        rows = [[self.names[i + (1 if self.names[0] == "C" else 0)] if i + 1 < len(self.names) else str(i),
                 float(sf.iloc[i, 0]), float(sf.iloc[i, 1]), float(sf.iloc[i, 2]), float(sf.iloc[i, 3])]
                for i in range(len(sf))]
        return Report("Marginal Effects at Means").table(["Variable", "dy/dx", "Std. Error", "z", "Prob."], rows,
                                                         ["str", "num", "num", "num", "prob"])


class CountResult(EquationBase):
    def __init__(self, wf, ev, spec: str, dist: str = "poisson", cov="ordinary", name=""):
        super().__init__(wf, ev, spec, name)
        import statsmodels.api as sm
        d = build_design(wf, ev, spec, allow_arma=False)
        self.d, self.mask, self.yname = d, d.mask, d.yname
        yv = d.y.values
        if (yv < 0).any():
            raise ModelError("Count models require a non-negative dependent variable")
        ctype = "HC0" if str(cov).lower() in ("white", "robust", "huber") else "nonrobust"
        if dist in ("negbin", "nb"):
            res = sm.NegativeBinomial(yv, d.X.values).fit(disp=0, cov_type=ctype, maxiter=300)
            self.names = d.xnames + ["ALPHA"]
            self.method_label = "Negative Binomial Count (Newton-Raphson)"
        else:
            res = sm.Poisson(yv, d.X.values).fit(disp=0, cov_type=ctype, maxiter=300)
            self.names = d.xnames
            self.method_label = "ML/QML - Poisson Count (Newton-Raphson)"
        self.sm = res
        self.params = np.asarray(res.params)
        self.bse = np.asarray(res.bse)
        self.cov = np.asarray(res.cov_params())
        self.tvalues = self.params / self.bse
        self.pvalues = pvalue_z(self.tvalues)
        self.yv = yv
        self.fitted_ = np.asarray(res.predict())
        self.resid_ = yv - self.fitted_
        n, k = len(yv), len(self.params)
        logl = float(res.llf)
        aic, sc, hq = info_criteria(logl, k, n)
        self.logl = logl
        l0 = float(res.llnull)
        self.stats_dict = {"logl": logl, "restrlogl": l0, "r2": 1 - logl / l0, "aic": aic, "sc": sc, "hq": hq,
                           "lr": 2 * (logl - l0), "lrprob": float(stats.chi2.sf(2 * (logl - l0), k - 1)),
                           "meandep": float(yv.mean()), "sddep": float(yv.std(ddof=1)), "regobs": n}
        self.df_resid = 0

    def report(self) -> Report:
        s = self.stats_dict
        r = Report(f"Equation: {self.name.upper()}" if self.name else "Equation")
        r.header([f"Dependent Variable: {self.yname}", f"Method: {self.method_label}"]
                 + sample_header(self.wf, self.mask), stamp=True)
        r.coef_table(self.names, self.params, self.bse, self.tvalues, self.pvalues, stat_label="z-Statistic")
        r.stats([("McFadden R-squared", s["r2"]), ("LR statistic", s["lr"]), ("Prob(LR statistic)", s["lrprob"]),
                 ("Log likelihood", s["logl"])],
                [("Mean dependent var", s["meandep"]), ("S.D. dependent var", s["sddep"]),
                 ("Akaike info criterion", s["aic"]), ("Schwarz criterion", s["sc"])])
        return r
