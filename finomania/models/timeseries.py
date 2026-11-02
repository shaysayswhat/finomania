"""Time-series models and tests: ARIMA / SARIMA / auto-ARIMA, ETS, unit-root
tests, cointegration, Granger causality, VAR, VECM, ARDL bounds testing,
Markov switching and trend/cycle filters."""
from __future__ import annotations

import itertools
import math
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from ..core.workfile import sanitize_name
from ..report import FigureOutput, Report, fmt_num
from . import plots
from .base import (ModelError, ModelResult, Param, View, build_design, contiguous_check, info_criteria,
                   pvalue_t, pvalue_z, sample_header, series_full, split_terms)
from .regression import EquationBase, _dw, forecast_report


def _clean_series(wf, ev, expr, mask=None):
    s = ev.eval(expr, ignore_sample=True, as_series=True).astype(float)
    m = (wf.sample if mask is None else mask) & s.notna().values
    return s, m


def auto_expand(wf, h: int, mask: np.ndarray) -> np.ndarray:
    """Make sure h observations exist after the last obs of mask; returns forecast mask."""
    last = np.flatnonzero(mask)[-1]
    need = last + h - (wf.nobs - 1)
    if need > 0:
        if wf.is_panel:
            raise ModelError("Cannot extend a panel workfile for out-of-sample forecasting")
        from ..core.dates import extend_index
        if wf.is_dated:
            new = extend_index(pd.DatetimeIndex(wf.data.index), wf.freq, need)
            wf.data = wf.data.reindex(new)
            wf.data.index.name = "date"
        else:
            wf.data = wf.data.reindex(pd.RangeIndex(1, wf.nobs + need + 1, name="obs"))
        wf._sample = np.concatenate([wf._sample, np.zeros(need, dtype=bool)])
        wf.notify("range")
    fm = np.zeros(wf.nobs, dtype=bool)
    fm[last + 1: last + 1 + h] = True
    return fm


# ============================================================================
# ARIMA / SARIMA
# ============================================================================

class ARIMAResult(EquationBase):
    method_label = "ARIMA Maximum Likelihood"

    def __init__(self, wf, ev, spec: str, order=(1, 0, 0), seasonal=(0, 0, 0, 0), trend: str | None = None,
                 name: str = "", auto: bool = False, maxp: int = 4, maxq: int = 4, ic: str = "aic"):
        super().__init__(wf, ev, spec, name)
        from statsmodels.tsa.arima.model import ARIMA
        terms = split_terms(spec)
        d = build_design(wf, ev, spec, allow_arma=False)
        if not contiguous_check(d.mask):
            raise ModelError("ARIMA requires a contiguous sample (no gaps)")
        self.d, self.mask, self.yname = d, d.mask, d.yname
        exog = d.X.reset_index(drop=True) if d.X.shape[1] else None
        has_c = "C" in d.xnames
        if has_c and exog is not None:
            exog = exog.drop(columns=["C"]) if exog.shape[1] > 1 else None
        y = d.y.values
        D = order[1] + seasonal[1]
        if trend is None:
            trend = "c" if (has_c and D == 0) else ("t" if has_c and D == 1 and False else "n")
        self.search = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if auto:
                best, table = None, []
                for p, q in itertools.product(range(maxp + 1), range(maxq + 1)):
                    try:
                        r = ARIMA(y, exog=exog, order=(p, order[1], q), seasonal_order=seasonal,
                                  trend=trend).fit()
                        crit = {"aic": r.aic, "bic": r.bic, "sc": r.bic, "hq": r.hqic}[ic]
                        table.append((p, q, r.llf, r.aic / len(y), r.bic / len(y), r.hqic / len(y)))
                        if best is None or crit < best[0]:
                            best = (crit, p, q, r)
                    except Exception:
                        continue
                if best is None:
                    raise ModelError("Automatic ARIMA selection failed")
                order = (best[1], order[1], best[2])
                res = best[3]
                self.search = sorted(table, key=lambda t: {"aic": t[3], "bic": t[4], "sc": t[4], "hq": t[5]}[ic])
                self.ic = ic
            else:
                res = ARIMA(y, exog=exog, order=order, seasonal_order=seasonal, trend=trend).fit()
        self.sm, self.order, self.seasonal, self.trend = res, order, seasonal, trend
        mp = {"const": "C", "x1": "@TREND", "sigma2": "SIGMASQ", "drift": "@TREND"}
        names = []
        for p in res.param_names:
            if p in mp:
                names.append(mp[p])
            elif p.startswith("ar.S.L"):
                names.append(f"SAR({p[6:]})")
            elif p.startswith("ma.S.L"):
                names.append(f"SMA({p[6:]})")
            elif p.startswith("ar.L"):
                names.append(f"AR({p[4:]})")
            elif p.startswith("ma.L"):
                names.append(f"MA({p[4:]})")
            else:
                names.append(p.upper())
        self.names = names
        self.arma_terms = [n for n in names if "AR(" in n or "MA(" in n]
        self.params = np.asarray(res.params)
        self.bse = np.asarray(res.bse)
        self.cov = np.asarray(res.cov_params())
        self.tvalues = self.params / self.bse
        self.pvalues = pvalue_z(self.tvalues)
        self.yv = y
        self.resid_ = np.asarray(res.resid)
        self.fitted_ = y - self.resid_
        n, k = len(y), len(self.params)
        self.df_resid = n - k
        aic, sc, hq = info_criteria(res.llf, k, n)
        e = self.resid_[order[1] + seasonal[1] * seasonal[3]:]
        self.stats_dict = {"logl": float(res.llf), "aic": aic, "sc": sc, "hq": hq, "regobs": n, "ncoef": k,
                           "se": math.sqrt(np.mean(e ** 2)), "dw": _dw(e), "ssr": float(e @ e),
                           "meandep": float(y.mean()), "sddep": float(y.std(ddof=1))}
        self.logl = float(res.llf)

    def report(self) -> Report:
        s = self.stats_dict
        r = Report(f"Equation: {self.name.upper()}" if self.name else "Equation")
        p, d, q = self.order
        lab = f"ARIMA({_o(p)},{d},{_o(q)})"
        if self.seasonal[3]:
            P, Dd, Q, S = self.seasonal
            lab += f"({P},{Dd},{Q})[{S}]"
        hdr = [f"Dependent Variable: {self.yname}", f"Method: {lab} - Maximum Likelihood (state space)"]
        hdr += sample_header(self.wf, self.mask)
        hdr.append("Coefficient covariance computed using outer product of gradients")
        r.header(hdr, stamp=True)
        r.coef_table(self.names, self.params, self.bse, self.tvalues, self.pvalues, stat_label="z-Statistic")
        r.stats([("Log likelihood", s["logl"]), ("S.E. of regression", s["se"]), ("Sum squared resid", s["ssr"]),
                 ("Durbin-Watson stat", s["dw"])],
                [("Akaike info criterion", s["aic"]), ("Schwarz criterion", s["sc"]),
                 ("Hannan-Quinn criter.", s["hq"]), ("Observations", s["regobs"])])
        try:
            rows = []
            if len(self.sm.arroots):
                rows.append(["Inverted AR Roots", "  ".join(_c(1 / z) for z in self.sm.arroots)])
            if len(self.sm.maroots):
                rows.append(["Inverted MA Roots", "  ".join(_c(1 / z) for z in self.sm.maroots)])
            if rows:
                r.table(["", "Roots"], rows, ["str", "str"], align=["l", "l"])
        except Exception:
            pass
        if self.search:
            r.section(f"Model selection ({self.ic.upper()}), best 10 of {len(self.search)} models")
            r.table(["Model", "LogL", "AIC*", "BIC", "HQ"],
                    [[f"({p},{q})", ll, a, b, h] for p, q, ll, a, b, h in self.search[:10]])
        return r

    def views(self):
        v = self.base_views()
        v.append(View("arroots", "ARMA Structure: Roots", self.view_roots, group="ARMA Structure"))
        v.append(View("diag", "Standardized Residual Diagnostics", self.view_diag, group="Residual Diagnostics"))
        return v

    def procs(self):
        return self.base_procs() + [View("forecast", "Forecast...", self.proc_forecast,
                                         [Param("h", "Horizon (periods after estimation end)", "int", 12),
                                          Param("name", "Forecast name", "text", f"{self.yname.lower()}f")],
                                         kind="proc")]

    def view_roots(self):
        fig, ax = plots.new_figure(5, 5)
        t = np.linspace(0, 2 * np.pi, 300)
        ax.plot(np.cos(t), np.sin(t), color="#888", lw=0.8)
        for roots, lab, c in ((self.sm.arroots, "AR roots", plots.PALETTE[0]), (self.sm.maroots, "MA roots",
                                                                                 plots.PALETTE[1])):
            if len(roots):
                inv = 1 / np.asarray(roots)
                ax.scatter(inv.real, inv.imag, color=c, label=lab, zorder=3)
        ax.set_aspect("equal")
        ax.legend(frameon=False)
        ax.set_title("Inverse Roots of AR/MA Polynomial(s)")
        fig.tight_layout()
        return FigureOutput(fig, "Inverse roots")

    def view_diag(self):
        res = self.sm
        z = self.resid_ / np.std(self.resid_)
        fig, axes = plots.new_figure(8, 5.5, 2, 2, squeeze=False)
        axes[0, 0].plot(z, color=plots.PALETTE[0], lw=0.8)
        axes[0, 0].set_title("Standardized residuals")
        axes[0, 1].hist(z, bins=30, density=True, color="#6f93cf", edgecolor="white")
        xs = np.linspace(z.min(), z.max(), 100)
        axes[0, 1].plot(xs, stats.norm.pdf(xs), color=plots.PALETTE[1])
        axes[0, 1].set_title("Histogram vs N(0,1)")
        zz = np.sort(z)
        q = stats.norm.ppf((np.arange(1, len(zz) + 1) - 0.5) / len(zz))
        axes[1, 0].scatter(q, zz, s=6, color=plots.PALETTE[0])
        axes[1, 0].plot(q, q, color=plots.PALETTE[1])
        axes[1, 0].set_title("Normal Q-Q")
        from statsmodels.tsa.stattools import acf
        a = acf(z, nlags=20, fft=False)[1:]
        axes[1, 1].bar(range(1, 21), a, color=plots.PALETTE[0])
        axes[1, 1].axhline(2 / math.sqrt(len(z)), color=plots.PALETTE[1], ls="--")
        axes[1, 1].axhline(-2 / math.sqrt(len(z)), color=plots.PALETTE[1], ls="--")
        axes[1, 1].set_title("Residual ACF")
        fig.tight_layout()
        _ = res
        return FigureOutput(fig, "Residual diagnostics")

    def proc_forecast(self, h: int = 12, name: str = ""):
        name = name or f"{self.yname.lower()}f"
        h = int(h)
        fm = auto_expand(self.wf, h, self.mask)
        exog = None
        if self.d.X.shape[1] and [c for c in self.d.xnames if c != "C"]:
            cols = [c for c in self.d.xnames if c != "C"]
            Xall = pd.DataFrame({c: self.ev.eval(c.lower(), ignore_sample=True, as_series=True) for c in cols})
            exog = Xall.values[fm]
            if np.isnan(exog).any():
                raise ModelError("Regressors have missing values in the forecast period")
        fc = self.sm.get_forecast(steps=h, exog=exog)
        f = pd.Series(np.nan, index=self.wf.data.index)
        fse = pd.Series(np.nan, index=self.wf.data.index)
        f.iloc[np.flatnonzero(fm)] = np.asarray(fc.predicted_mean)
        fse.iloc[np.flatnonzero(fm)] = np.sqrt(np.asarray(fc.var_pred_mean))
        # join the last actual to make the graph continuous
        self.wf.set_series(name, f, respect_sample=False)
        out = forecast_report(self.wf, name, self.yname, self.ev, f, fse, fm, "dynamic")
        r = Report(f"Forecast table: {name.upper()}")
        labs = np.array(self.wf.labels())[fm]
        r.table(["Period", "Forecast", "S.E.", "Lower 95%", "Upper 95%"],
                [[l, a, b, a - 1.96 * b, a + 1.96 * b] for l, a, b in zip(labs, f[fm], fse[fm])])
        return out + [r]


def _o(x):
    return ",".join(map(str, x)) if isinstance(x, (list, tuple)) else str(x)


def _c(z):
    z = complex(z)
    return f"{z.real:.2f}" if abs(z.imag) < 1e-9 else f"{z.real:.2f}{'+' if z.imag >= 0 else '-'}{abs(z.imag):.2f}i"


# ============================================================================
# Exponential smoothing (ETS)
# ============================================================================

class ETSResult(ModelResult):
    object_type = "equation"
    method_label = "Exponential smoothing (ETS)"

    def __init__(self, wf, ev, expr: str, error="add", trend=None, seasonal=None, periods=None, damped=False,
                 name=""):
        from statsmodels.tsa.exponential_smoothing.ets import ETSModel
        self.wf, self.ev, self.name = wf, ev, name
        s, m = _clean_series(wf, ev, expr)
        if not contiguous_check(m):
            raise ModelError("ETS requires a contiguous sample")
        self.mask, self.yname = m, expr.upper()
        y = s[m].values
        trend = None if trend in (None, "", "none", "n") else ("add" if trend in ("a", "add") else "mul")
        seasonal = None if seasonal in (None, "", "none", "n") else ("add" if seasonal in ("a", "add") else "mul")
        error = "add" if error in ("a", "add") else "mul"
        if seasonal and not periods:
            from ..core.dates import PERIODS_PER_YEAR
            periods = {"q": 4, "m": 12, "w": 52, "d5": 5, "d7": 7}.get(wf.freq)
            if not periods:
                raise ModelError("Seasonal ETS needs periods=")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.sm = ETSModel(pd.Series(y), error=error, trend=trend, damped_trend=bool(damped) and trend is not None,
                               seasonal=seasonal, seasonal_periods=periods).fit(disp=False)
        self.spec = (error, trend, seasonal, periods, damped)
        self.y = y

    def report(self) -> Report:
        res = self.sm
        e, t, s, p, dmp = self.spec
        lab = f"ETS({e[0].upper()},{(t or 'N')[0].upper()}{'d' if dmp and t else ''},{(s or 'N')[0].upper()})"
        r = Report(f"Exponential Smoothing: {self.name.upper() or self.yname}")
        r.header([f"Series: {self.yname}", f"Method: {lab}" + (f" seasonal period {p}" if s else ""),
                  f"Sample: {self.wf.sample_label(self.mask)}", f"Included observations: {len(self.y)}"], stamp=True)
        r.table(["Parameter", "Value"], [[n.upper(), v] for n, v in zip(res.param_names, res.params)])
        k = len(res.params)
        aic, sc, hq = info_criteria(res.llf, k, len(self.y))
        resid = self.y - np.asarray(res.fittedvalues)
        r.stats([("Log likelihood", res.llf), ("Akaike info criterion", aic), ("Schwarz criterion", sc)],
                [("Root Mean Squared Error", math.sqrt(np.mean(resid ** 2))),
                 ("Mean Absolute Error", float(np.mean(np.abs(resid)))), ("Observations", len(self.y))])
        return r

    def views(self):
        return [View("output", "Estimation Output", self.report), View("plot", "Fitted vs actual", self.view_plot)]

    def procs(self):
        return [View("forecast", "Forecast...", self.proc_forecast,
                     [Param("h", "Horizon", "int", 12), Param("name", "Name", "text", "etsf")], kind="proc")]

    def view_plot(self):
        idx = self.wf.data.index[self.mask]
        return FigureOutput(plots.line_plot({"actual": pd.Series(self.y, idx),
                                             "fitted": pd.Series(np.asarray(self.sm.fittedvalues), idx)},
                                            "ETS fit"), "ETS fit")

    def proc_forecast(self, h=12, name="etsf"):
        h = int(h)
        fm = auto_expand(self.wf, h, self.mask)
        pr = self.sm.get_prediction(start=len(self.y), end=len(self.y) + h - 1)
        sf = pr.summary_frame(alpha=0.05)
        f = pd.Series(np.nan, index=self.wf.data.index)
        f.iloc[np.flatnonzero(fm)] = sf["mean"].values
        self.wf.set_series(name, f, respect_sample=False)
        fig, ax = plots.new_figure()
        hist_idx = self.wf.data.index[self.mask]
        fi = self.wf.data.index[fm]
        ax.plot(hist_idx, self.y, color="#333", lw=1, label=self.yname)
        ax.plot(fi, sf["mean"].values, color=plots.PALETTE[0], lw=1.5, label="Forecast")
        ax.fill_between(fi, sf["pi_lower"].values, sf["pi_upper"].values, color="#9fb8e3", alpha=0.4, label="95% PI")
        ax.legend(frameon=False)
        ax.set_title(f"ETS forecast: {name.upper()}")
        fig.tight_layout()
        r = Report(f"Forecast: {name.upper()}")
        labs = np.array(self.wf.labels())[fm]
        r.table(["Period", "Forecast", "Lower 95%", "Upper 95%"],
                [[l, a, b, c] for l, a, b, c in zip(labs, sf["mean"], sf["pi_lower"], sf["pi_upper"])])
        return [r, FigureOutput(fig, "ETS forecast")]


# ============================================================================
# Unit-root tests
# ============================================================================

TREND_LABEL = {"n": "None", "c": "Constant", "ct": "Constant, Linear Trend", "ctt": "Constant, Linear & Quad. Trend",
               "t": "Trend"}


def unit_root_test(wf, ev, expr: str, test: str = "adf", trend: str = "c", lags: int | None = None,
                   maxlag: int | None = None, ic: str = "sic", dif: int = 0) -> Report:
    from arch import unitroot as ur
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="arch")
    s, m = _clean_series(wf, ev, expr)
    y = s[m]
    name = expr.upper()
    for _ in range(int(dif)):
        y = y.diff()
        name = f"D({name})"
    y = y.dropna()
    v = y.values
    trend = {"none": "n", "const": "c", "intercept": "c", "trend": "ct", "c": "c", "ct": "ct", "n": "n",
             "t": "ct"}.get(str(trend).lower(), "c")
    method = {"sic": "bic", "bic": "bic", "aic": "aic", "hq": "aic", "tstat": "t-stat", "t": "t-stat"}.get(ic, "bic")
    test = test.lower()
    r = Report({"adf": "Augmented Dickey-Fuller Unit Root Test", "pp": "Phillips-Perron Unit Root Test",
                "kpss": "Kwiatkowski-Phillips-Schmidt-Shin Test", "dfgls": "Elliott-Rothenberg-Stock DF-GLS Test",
                "za": "Zivot-Andrews Unit Root Test with a Structural Break",
                "ers": "Elliott-Rothenberg-Stock DF-GLS Test"}.get(test, "Unit Root Test") + f" on {name}")
    T_ = len(v)
    if maxlag is None and lags is None:
        maxlag = int(math.floor(min(T_ / 3, 12) * (T_ / 100) ** 0.25))
    if test == "adf":
        res = ur.ADF(v, lags=lags, max_lags=maxlag, trend=trend, method=method)
        null = f"{name} has a unit root"
        laginfo = (f"Lag Length: {res.lags} (Automatic - based on {ic.upper()}, maxlag={maxlag})"
                   if lags is None else f"Lag Length: {res.lags} (Fixed)")
        label = "Augmented Dickey-Fuller test statistic"
    elif test == "pp":
        res = ur.PhillipsPerron(v, lags=lags, trend=trend if trend != "n" else "n")
        null = f"{name} has a unit root"
        laginfo = f"Bandwidth: {res.lags} (Newey-West automatic) using Bartlett kernel"
        label = "Phillips-Perron test statistic"
    elif test == "kpss":
        res = ur.KPSS(v, lags=lags, trend="c" if trend in ("c", "n") else "ct")
        null = f"{name} is stationary"
        laginfo = f"Bandwidth: {res.lags} (Newey-West automatic) using Bartlett kernel"
        label = "Kwiatkowski-Phillips-Schmidt-Shin test statistic"
    elif test in ("dfgls", "ers"):
        res = ur.DFGLS(v, lags=lags, max_lags=maxlag, trend="c" if trend in ("c", "n") else "ct", method=method)
        null = f"{name} has a unit root"
        laginfo = f"Lag Length: {res.lags} (based on {ic.upper()})"
        label = "Elliott-Rothenberg-Stock DF-GLS test statistic"
    elif test == "za":
        res = ur.ZivotAndrews(v, lags=lags, max_lags=maxlag, trend={"c": "c", "ct": "ct", "n": "c", "t": "t"}[trend],
                              method=method)
        null = f"{name} has a unit root with a structural break"
        laginfo = f"Lag Length: {res.lags}"
        label = "Zivot-Andrews test statistic"
    else:
        raise ModelError(f"Unknown unit root test '{test}' (adf, pp, kpss, dfgls, za)")
    r.header([f"Null Hypothesis: {null}", f"Exogenous: {TREND_LABEL.get(trend, trend)}", laginfo])
    cv = res.critical_values
    stat_lab = "LM-Stat." if test == "kpss" else "t-Statistic"
    rows = [[label, float(res.stat), float(res.pvalue)]]
    for k in ("1%", "5%", "10%"):
        rows.append([("Test critical values:" if k == "1%" else "") + f"   {k} level", float(cv[k]), ""])
    r.table(["", stat_lab, "Prob.*"], rows, ["str", "num", "prob"], rules_after=[0])
    r.note("*MacKinnon (1996) one-sided p-values." if test in ("adf", "pp", "dfgls") else
           "*Asymptotic critical values." if test == "kpss" else "*Critical values from Zivot & Andrews (1992).")
    if test == "za":
        try:
            pos = int(np.argmin(np.abs(np.asarray(getattr(res, "_all_stats", [np.nan])) - res.stat)))
        except Exception:
            pos = None
        _ = pos
    if test in ("adf", "dfgls") and hasattr(res, "regression"):
        try:
            reg = res.regression
            rn = list(reg.model.exog_names)
            nm = []
            for x in rn:
                if x.lower().startswith("level"):
                    nm.append(f"{name}(-1)")
                elif x.lower().startswith("delta.l") or x.lower().startswith("diff.l"):
                    nm.append(f"D({name}(-{x.split('L')[-1]}))")
                elif x.lower() == "const":
                    nm.append("C")
                elif x.lower().startswith("trend"):
                    nm.append("@TREND")
                else:
                    nm.append(x.upper())
            r.section(f"{'Augmented Dickey-Fuller' if test == 'adf' else 'DF-GLS'} Test Equation")
            r.header([f"Dependent Variable: D({name})", "Method: Least Squares",
                      f"Included observations: {int(reg.nobs)} after adjustments"])
            r.coef_table(nm, reg.params, reg.bse, reg.tvalues, reg.pvalues)
            n = reg.nobs
            aic, sc, hq = info_criteria(reg.llf, len(reg.params), n)
            r.stats([("R-squared", reg.rsquared), ("Adjusted R-squared", reg.rsquared_adj),
                     ("S.E. of regression", math.sqrt(reg.scale)), ("Sum squared resid", reg.ssr),
                     ("Log likelihood", reg.llf)],
                    [("Akaike info criterion", aic), ("Schwarz criterion", sc), ("Hannan-Quinn criter.", hq),
                     ("Durbin-Watson stat", _dw(np.asarray(reg.resid)))])
        except Exception:
            pass
    return r


def unit_root_summary(wf, ev, names: list[str], trend="c") -> Report:
    """ADF / PP / KPSS on levels and first differences for a list of series."""
    from arch import unitroot as ur
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="arch")
    rows = []
    for nm in names:
        s, m = _clean_series(wf, ev, nm)
        for dlab, v in (("Level", s[m].values), ("1st diff.", np.diff(s[m].values))):
            v = v[~np.isnan(v)]
            a = ur.ADF(v, trend=trend, method="bic")
            p = ur.PhillipsPerron(v, trend=trend)
            k = ur.KPSS(v, trend="c" if trend != "ct" else "ct")
            rows.append([nm.upper(), dlab, a.stat, a.pvalue, p.stat, p.pvalue, k.stat,
                         "I(0)" if a.pvalue < 0.05 and k.pvalue > 0.05 else ("I(1)?" if dlab == "Level" else "")])
    r = Report("Unit Root Summary")
    r.header([f"Exogenous: {TREND_LABEL.get(trend, trend)}", "ADF lag length by SIC; PP/KPSS Newey-West bandwidth"])
    r.table(["Series", "Form", "ADF t", "Prob.", "PP t", "Prob.", "KPSS LM", "Verdict"], rows,
            ["str", "str", "num", "prob", "num", "prob", "num", "str"])
    r.note("KPSS 5% critical value: 0.463 (constant) / 0.146 (trend). Verdict I(0) when ADF rejects and KPSS does not.")
    return r


# ============================================================================
# Cointegration
# ============================================================================

def johansen_test(wf, ev, names: list[str], lags: int = 1, det: str = "c") -> Report:
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    df = _frame(wf, ev, names)
    det_order = {"n": -1, "none": -1, "c": 0, "const": 0, "ct": 1, "trend": 1}.get(det, 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        jr = coint_johansen(df.values, det_order, int(lags))
    from types import SimpleNamespace
    res = SimpleNamespace(eig=np.real(jr.eig), evec=np.real(jr.evec), lr1=np.real(jr.lr1), lr2=np.real(jr.lr2),
                          cvt=jr.cvt, cvm=jr.cvm)
    k = df.shape[1]
    r = Report("Johansen Cointegration Test")
    r.header([f"Sample (adjusted): {wf.sample_label(_frame_mask(wf, df))}", f"Included observations: {len(df) - lags - 1}"
              " after adjustments", f"Series: {' '.join(n.upper() for n in names)}",
              f"Lags interval (in first differences): 1 to {lags}",
              {-1: "Trend assumption: No deterministic trend", 0: "Trend assumption: Linear deterministic trend (constant)",
               1: "Trend assumption: Linear deterministic trend (restricted)"}[det_order]])
    hyp = ["None"] + [f"At most {i}" for i in range(1, k)]
    rows = []
    ntrace = 0
    for i in range(k):
        rej = res.lr1[i] > res.cvt[i, 1]
        if rej and ntrace == i:
            ntrace += 1
        rows.append([hyp[i] + (" *" if rej else ""), res.eig[i], res.lr1[i], res.cvt[i, 0], res.cvt[i, 1], res.cvt[i, 2]])
    r.table(["Hypothesized No. of CE(s)", "Eigenvalue", "Trace Statistic", "0.10 Crit.", "0.05 Crit.", "0.01 Crit."], rows,
            title="Unrestricted Cointegration Rank Test (Trace)")
    r.note(f"Trace test indicates {ntrace} cointegrating eqn(s) at the 0.05 level;  * denotes rejection at 0.05")
    rows = []
    nmax = 0
    for i in range(k):
        rej = res.lr2[i] > res.cvm[i, 1]
        if rej and nmax == i:
            nmax += 1
        rows.append([hyp[i] + (" *" if rej else ""), res.eig[i], res.lr2[i], res.cvm[i, 0], res.cvm[i, 1], res.cvm[i, 2]])
    r.table(["Hypothesized No. of CE(s)", "Eigenvalue", "Max-Eigen Statistic", "0.10 Crit.", "0.05 Crit.", "0.01 Crit."],
            rows, title="Unrestricted Cointegration Rank Test (Maximum Eigenvalue)")
    r.note(f"Max-eigenvalue test indicates {nmax} cointegrating eqn(s) at the 0.05 level")
    beta = res.evec[:, : max(1, min(ntrace, k - 1))]
    rows = []
    for j in range(beta.shape[1]):
        b = beta[:, j] / beta[0, j]
        rows.append([f"CE {j + 1}"] + list(b))
    r.table([""] + [n.upper() for n in names], rows, title="Normalized cointegrating coefficients")
    return r


def engle_granger_test(wf, ev, names: list[str], trend: str = "c", method: str = "eg") -> Report:
    from arch.unitroot.cointegration import engle_granger, phillips_ouliaris
    df = _frame(wf, ev, names)
    y, x = df.iloc[:, 0], df.iloc[:, 1:]
    trend = {"n": "n", "c": "c", "ct": "ct", "none": "n"}.get(trend, "c")
    r = Report(("Engle-Granger" if method == "eg" else "Phillips-Ouliaris") + " Cointegration Test")
    r.header([f"Series: {' '.join(n.upper() for n in names)}", "Null hypothesis: Series are not cointegrated",
              f"Cointegrating equation deterministics: {TREND_LABEL.get(trend, trend)}", f"Observations: {len(df)}"])
    if method == "eg":
        res = engle_granger(y, x, trend=trend, method="bic")
        rows = [["Engle-Granger tau-statistic", float(res.stat), float(res.pvalue)]]
        for k, v in res.critical_values.items():
            rows.append([f"  {k} critical value", float(v), ""])
        r.table(["", "Value", "Prob.*"], rows, ["str", "num", "prob"])
        r.text(f"ADF lags on residuals: {res.lags}")
    else:
        rows = []
        for tt in ("Zt", "Za", "Pu", "Pz"):
            res = phillips_ouliaris(y, x, trend=trend, test_type=tt)
            rows.append([f"Phillips-Ouliaris {tt}", float(res.stat), float(res.pvalue)])
        r.table(["Test", "Value", "Prob.*"], rows, ["str", "num", "prob"])
    import statsmodels.api as sm
    X = sm.add_constant(x.values) if trend != "n" else x.values
    if trend == "ct":
        X = np.column_stack([X, np.arange(len(x))])
    reg = sm.OLS(y.values, X).fit()
    nm = (["C"] if trend != "n" else []) + [c.upper() for c in x.columns] + (["@TREND"] if trend == "ct" else [])
    r.section(f"Cointegrating regression: Dependent Variable {names[0].upper()}")
    r.coef_table(nm, reg.params, reg.bse, reg.tvalues, reg.pvalues)
    r.note("*MacKinnon (1996) p-values. Standard errors of the static regression are not valid for inference.")
    return r


def _frame(wf, ev, names) -> pd.DataFrame:
    cols = {n: ev.eval(n, ignore_sample=True, as_series=True) for n in names}
    df = pd.DataFrame(cols, index=wf.data.index)
    df = df[wf.sample]
    return df.dropna()


def _frame_mask(wf, df) -> np.ndarray:
    return wf.data.index.isin(df.index)


def granger_pairwise(wf, ev, names: list[str], lags: int = 2) -> Report:
    from statsmodels.tsa.stattools import grangercausalitytests
    df = _frame(wf, ev, names)
    rows = []
    for a, b in itertools.permutations(names, 2):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = grangercausalitytests(df[[b, a]].values, maxlag=[int(lags)])
        F, p, _, _ = res[int(lags)][0]["ssr_ftest"]
        rows.append([f"{a.upper()} does not Granger Cause {b.upper()}", len(df) - int(lags), F, p])
    r = Report("Pairwise Granger Causality Tests")
    r.header([f"Sample: {wf.sample_label()}", f"Lags: {lags}"])
    r.table(["Null Hypothesis:", "Obs", "F-Statistic", "Prob."], rows, ["str", "int", "num", "prob"])
    return r


# ============================================================================
# VAR
# ============================================================================

class VARResult(ModelResult):
    object_type = "var"
    method_label = "Vector Autoregression"

    def __init__(self, wf, ev, endog: list[str], lags: list[int] | int = 1, exog: list[str] | None = None,
                 trend: str = "c", name: str = ""):
        from statsmodels.tsa.api import VAR
        self.wf, self.ev, self.name = wf, ev, name
        self.endog_names = [e.lower() for e in endog]
        exog = [x for x in (exog or []) if x.lower() != "c"]
        df = pd.DataFrame({e: ev.eval(e, ignore_sample=True, as_series=True) for e in self.endog_names},
                          index=wf.data.index)
        ex = pd.DataFrame({x: ev.eval(x, ignore_sample=True, as_series=True) for x in exog},
                          index=wf.data.index) if exog else None
        mask = wf.sample & df.notna().all(axis=1).values
        if ex is not None:
            mask &= ex.notna().all(axis=1).values
        if not contiguous_check(mask):
            raise ModelError("VAR estimation requires a contiguous sample")
        self.mask = mask
        self.df = df[mask]
        self.exdf = ex[mask] if ex is not None else None
        self.p = max(lags) if isinstance(lags, (list, tuple)) else int(lags)
        self.trend = trend
        mod = VAR(self.df.values, exog=self.exdf.values if self.exdf is not None else None)
        self.mod = mod
        self.sm = mod.fit(self.p, trend=trend)
        self.exog_names = exog

    def _reg_names(self):
        names = []
        if self.trend in ("c", "ct"):
            names.append("C")
        if self.trend == "ct":
            names.append("@TREND")
        names += [x.upper() for x in self.exog_names]
        for l in range(1, self.p + 1):
            for e in self.endog_names:
                names.append(_lagname(e, l))
        return names

    def _reorder(self, M):
        """statsmodels orders [trend, exog, lags]; keep that (matches _reg_names)."""
        return M

    def report(self) -> Report:
        res = self.sm
        names = self._reg_names()
        k = len(self.endog_names)
        r = Report(f"VAR: {self.name.upper()}" if self.name else "Vector Autoregression Estimates")
        first = np.flatnonzero(self.mask)[self.p] if len(np.flatnonzero(self.mask)) > self.p else 0
        m2 = self.mask.copy()
        m2[: first] = False
        r.header(["Vector Autoregression Estimates", f"Sample (adjusted): {self.wf.sample_label(m2)}",
                  f"Included observations: {res.nobs} after adjustments",
                  "Standard errors in ( ) & t-statistics in [ ]"], stamp=True)
        P, S, T = np.asarray(res.params), np.asarray(res.stderr), np.asarray(res.tvalues)
        rows, rules = [], []
        for i, n in enumerate(names):
            rows.append([n] + [P[i, j] for j in range(k)])
            rows.append([""] + [f"({fmt_num(S[i, j])})" for j in range(k)])
            rows.append([""] + [f"[{T[i, j]:.5f}]" for j in range(k)])
            rules.append(len(rows) - 1) if i == len(names) - 1 else None
        r.table([""] + [e.upper() for e in self.endog_names], rows, rules_after=[])
        resid = np.asarray(res.resid)
        y = np.asarray(res.endog)[self.p:]
        srows = {lab: [] for lab in ["R-squared", "Adj. R-squared", "Sum sq. resids", "S.E. equation", "F-statistic",
                                     "Log likelihood", "Akaike AIC", "Schwarz SC", "Mean dependent", "S.D. dependent"]}
        n = resid.shape[0]
        kk = P.shape[0]
        for j in range(k):
            e = resid[:, j]
            yy = y[:, j]
            ssr = e @ e
            tss = ((yy - yy.mean()) ** 2).sum()
            r2 = 1 - ssr / tss
            ll = -n / 2 * (1 + math.log(2 * math.pi) + math.log(ssr / n))
            srows["R-squared"].append(r2)
            srows["Adj. R-squared"].append(1 - (1 - r2) * (n - 1) / (n - kk))
            srows["Sum sq. resids"].append(ssr)
            srows["S.E. equation"].append(math.sqrt(ssr / (n - kk)))
            srows["F-statistic"].append((r2 / (kk - 1)) / ((1 - r2) / (n - kk)) if kk > 1 else np.nan)
            srows["Log likelihood"].append(ll)
            srows["Akaike AIC"].append(-2 * ll / n + 2 * kk / n)
            srows["Schwarz SC"].append(-2 * ll / n + kk * math.log(n) / n)
            srows["Mean dependent"].append(yy.mean())
            srows["S.D. dependent"].append(yy.std(ddof=1))
        r.table([""] + [e.upper() for e in self.endog_names], [[lab] + v for lab, v in srows.items()])
        Sig = np.asarray(res.sigma_u)
        Sml = resid.T @ resid / n
        ll = float(res.llf)
        ncoef = kk * k
        r.stats([("Determinant resid covariance (dof adj.)", np.linalg.det(Sig)),
                 ("Determinant resid covariance", np.linalg.det(Sml)), ("Log likelihood", ll),
                 ("Akaike information criterion", -2 * ll / n + 2 * ncoef / n),
                 ("Schwarz criterion", -2 * ll / n + ncoef * math.log(n) / n),
                 ("Number of coefficients", ncoef)], [])
        return r

    def stat(self, key):
        d = {"logl": float(self.sm.llf), "aic": float(self.sm.aic), "sc": float(self.sm.bic), "hq": float(self.sm.hqic),
             "neqn": len(self.endog_names), "regobs": int(self.sm.nobs)}
        if key in d:
            return d[key]
        raise ModelError(f"@{key} not available for VAR")

    def views(self):
        return [
            View("output", "Estimation Output", self.report),
            View("laglen", "Lag Length Criteria", self.view_laglen, [Param("maxlag", "Maximum lag", "int", 8)],
                 group="Lag Structure"),
            View("arroots", "AR Roots Table & Graph", self.view_roots, group="Lag Structure"),
            View("testexog", "Granger Causality / Block Exogeneity Wald Tests", self.view_granger, group="Lag Structure"),
            View("impulse", "Impulse Responses", self.view_impulse,
                 [Param("h", "Periods", "int", 10), Param("type", "Type", "choice", "cholesky",
                                                          ["cholesky", "generalized", "residual"]),
                  Param("combine", "Combined graphs", "bool", False)]),
            View("decomp", "Variance Decomposition", self.view_decomp, [Param("h", "Periods", "int", 10)]),
            View("portmanteau", "Portmanteau Autocorrelation Test", self.view_portmanteau,
                 [Param("lags", "Lags", "int", 12)], group="Residual Tests"),
            View("jbera", "Normality Test", self.view_normality, group="Residual Tests"),
            View("residcor", "Residual Correlation Matrix", self.view_residcor, group="Residual Tests"),
            View("resids", "Residual Graphs", self.view_resids, group="Residual Tests"),
        ]

    def procs(self):
        return [View("forecast", "Forecast...", self.proc_forecast,
                     [Param("h", "Horizon", "int", 12), Param("suffix", "Name suffix", "text", "_f")], kind="proc"),
                View("makeresids", "Make Residuals", self.proc_makeresids,
                     [Param("prefix", "Prefix", "text", "resid_")], kind="proc")]

    def view_laglen(self, maxlag=8) -> Report:
        from statsmodels.tsa.api import VAR
        maxlag = int(maxlag)
        y = self.df.values
        ex = self.exdf.values if self.exdf is not None else None
        rows = []
        lls = []
        T = len(y) - maxlag
        k = y.shape[1]
        S_prev = None
        for p in range(0, maxlag + 1):
            if p == 0:
                e = y[maxlag:] - y[maxlag:].mean(0)
                S = e.T @ e / T
                ll = -T / 2 * (k * (1 + math.log(2 * math.pi)) + math.log(np.linalg.det(S)))
                npar = k
            else:
                yy = y[maxlag - p:]
                exx = ex[maxlag - p:] if ex is not None else None
                res = VAR(yy, exog=exx).fit(p, trend=self.trend)
                S = np.asarray(res.resid).T @ np.asarray(res.resid) / T
                ll = -T / 2 * (k * (1 + math.log(2 * math.pi)) + math.log(np.linalg.det(S)))
                npar = res.params.size
            lls.append(ll)
            aic = -2 * ll / T + 2 * npar / T
            sc = -2 * ll / T + npar * math.log(T) / T
            hq = -2 * ll / T + 2 * npar * math.log(math.log(T)) / T
            fpe = np.linalg.det(S) * ((T + npar / k) / (T - npar / k)) ** k
            LR = np.nan if p == 0 else (T - (npar / k)) * (math.log(np.linalg.det(S_prev)) - math.log(np.linalg.det(S)))
            rows.append([p, ll, LR, fpe, aic, sc, hq])
            S_prev = S
        # mark the minima
        cols = [3, 4, 5, 6]
        best = {c: int(np.nanargmin([row[c] for row in rows])) for c in cols}
        lr_best = 0
        for i in range(1, len(rows)):
            if stats.chi2.sf(rows[i][2], k * k) < 0.05:
                lr_best = i
        out = []
        for i, row in enumerate(rows):
            cells = [row[0], fmt_num(row[1]), (fmt_num(row[2]) + ("*" if i == lr_best else "")) if i else "NA"]
            for c in cols:
                cells.append(fmt_num(row[c]) + ("*" if best[c] == i else ""))
            out.append(cells)
        r = Report("VAR Lag Order Selection Criteria")
        r.header([f"Endogenous variables: {' '.join(e.upper() for e in self.endog_names)}",
                  f"Included observations: {T}"])
        r.table(["Lag", "LogL", "LR", "FPE", "AIC", "SC", "HQ"], out, ["int", "str", "str", "str", "str", "str", "str"])
        r.note("* indicates lag order selected by the criterion. LR: sequential modified LR test (5% level); FPE: Final "
               "prediction error; AIC/SC/HQ: Akaike, Schwarz, Hannan-Quinn.")
        return r

    def view_roots(self):
        roots = 1 / np.asarray(self.sm.roots)
        r = Report("Roots of Characteristic Polynomial")
        r.table(["Root", "Modulus"], [[_c(z), abs(z)] for z in sorted(roots, key=lambda z: -abs(z))])
        r.note("No root lies outside the unit circle. VAR satisfies the stability condition."
               if np.all(np.abs(roots) < 1) else "Warning: at least one root outside the unit circle. VAR not stable.")
        fig, ax = plots.new_figure(5, 5)
        t = np.linspace(0, 2 * np.pi, 300)
        ax.plot(np.cos(t), np.sin(t), color="#888", lw=0.8)
        ax.scatter(roots.real, roots.imag, color=plots.PALETTE[0], zorder=3)
        ax.set_aspect("equal")
        ax.set_title("Inverse Roots of AR Characteristic Polynomial")
        fig.tight_layout()
        r.figure(fig, "AR roots")
        return r

    def view_granger(self) -> Report:
        res = self.sm
        r = Report("VAR Granger Causality/Block Exogeneity Wald Tests")
        for dep in self.endog_names:
            rows = []
            others = [e for e in self.endog_names if e != dep]
            for ex in others:
                t = res.test_causality(self.endog_names.index(dep), [self.endog_names.index(ex)], kind="wald")
                rows.append([ex.upper(), float(t.test_statistic), int(t.df), float(t.pvalue)])
            if len(others) > 1:
                t = res.test_causality(self.endog_names.index(dep), [self.endog_names.index(e) for e in others],
                                       kind="wald")
                rows.append(["All", float(t.test_statistic), int(t.df), float(t.pvalue)])
            r.table(["Excluded", "Chi-sq", "df", "Prob."], rows, ["str", "num", "int", "prob"],
                    title=f"Dependent variable: {dep.upper()}")
        return r

    def view_impulse(self, h=10, type="cholesky", combine=False):  # noqa: A002
        h = int(h)
        irf = self.sm.irf(h)
        k = len(self.endog_names)
        if type == "generalized":
            resp = _girf(self.sm, h)
            se = None
        elif type == "residual":
            resp = irf.irfs
            se = irf.stderr(orth=False)
        else:
            resp = irf.orth_irfs
            se = irf.stderr(orth=True)
        panels = []
        names = [e.upper() for e in self.endog_names]
        rows = []
        for i in range(k):
            for j in range(k):
                y = resp[:, i, j]
                lo = hi = None
                if se is not None:
                    lo, hi = y - 2 * se[:, i, j], y + 2 * se[:, i, j]
                panels.append((f"Response of {names[i]} to {names[j]}", y, lo, hi))
        x = np.arange(0, h + 1)
        lab = {"cholesky": "Cholesky One S.D.", "generalized": "Generalized One S.D.", "residual": "Residual one unit"}[type]
        if combine:
            fig, axes = plots.new_figure(3.3 * k, 2.8, 1, k, squeeze=False)
            for i in range(k):
                ax = axes[0, i]
                for j in range(k):
                    ax.plot(x, resp[:, i, j], label=names[j], color=plots.PALETTE[j % 10])
                ax.axhline(0, color="#444", lw=0.6)
                ax.set_title(f"Response of {names[i]}", fontsize=9)
                ax.legend(frameon=False, fontsize=7)
            fig.suptitle(f"Response to {lab} Innovations", fontsize=10)
            fig.tight_layout()
        else:
            fig = plots.grid_plot(panels, k, f"Response to {lab} Innovations ± 2 S.E.", x=x)
        r = Report("Impulse Responses")
        r.header([f"Impulse: {lab} innovations" + (" (ordering: " + " ".join(names) + ")" if type == "cholesky" else "")])
        for i in range(k):
            r.table(["Period"] + names, [[p + 1] + list(resp[p, i]) for p in range(h + 1)],
                    ["int"] + ["num"] * k, title=f"Response of {names[i]}:")
        return [FigureOutput(fig, "Impulse Responses"), r]

    def view_decomp(self, h=10):
        h = int(h)
        fe = self.sm.fevd(h)
        names = [e.upper() for e in self.endog_names]
        k = len(names)
        r = Report("Variance Decomposition")
        mse = fe.decomp
        fc_se = np.sqrt(np.array([np.diag(m) for m in self.sm.mse(h)]))
        for i in range(k):
            rows = [[p + 1, fc_se[p, i]] + list(100 * mse[i, p]) for p in range(h)]
            r.table(["Period", "S.E."] + names, rows, ["int", "num"] + ["f4"] * k,
                    title=f"Variance Decomposition of {names[i]}:")
        r.note("Cholesky ordering: " + " ".join(names))
        fig, axes = plots.new_figure(3.2 * k, 2.8, 1, k, squeeze=False)
        for i in range(k):
            ax = axes[0, i]
            bottom = np.zeros(h)
            for j in range(k):
                ax.bar(np.arange(1, h + 1), 100 * mse[i, :, j], bottom=bottom, color=plots.PALETTE[j % 10],
                       label=names[j])
                bottom += 100 * mse[i, :, j]
            ax.set_title(f"Variance of {names[i]}", fontsize=9)
            ax.set_ylim(0, 100)
        axes[0, 0].legend(frameon=False, fontsize=7)
        fig.tight_layout()
        return [r, FigureOutput(fig, "Variance Decomposition")]

    def view_portmanteau(self, lags=12) -> Report:
        t = self.sm.test_whiteness(nlags=int(lags), adjusted=True)
        r = Report("VAR Residual Portmanteau Test for Autocorrelations")
        r.text("Null Hypothesis: No residual autocorrelations up to lag h")
        r.stats([("Adj. Q-Stat", float(t.test_statistic)), ("df", int(t.df))], [("Prob.", float(t.pvalue))])
        return r

    def view_normality(self) -> Report:
        resid = np.asarray(self.sm.resid)
        k = resid.shape[1]
        # Cholesky-orthogonalised residuals (Lutkepohl)
        S = np.cov(resid, rowvar=False)
        P = np.linalg.cholesky(S)
        u = (np.linalg.inv(P) @ (resid - resid.mean(0)).T).T
        n = len(u)
        rows = []
        tot = [0.0, 0.0, 0.0]
        for j in range(k):
            sk = stats.skew(u[:, j])
            ku = stats.kurtosis(u[:, j], fisher=False)
            c1 = n * sk ** 2 / 6
            c2 = n * (ku - 3) ** 2 / 24
            rows.append([j + 1, sk, c1, ku, c2, c1 + c2, stats.chi2.sf(c1 + c2, 2)])
            tot[0] += c1
            tot[1] += c2
        rows.append(["Joint", "", tot[0], "", tot[1], tot[0] + tot[1], stats.chi2.sf(tot[0] + tot[1], 2 * k)])
        r = Report("VAR Residual Normality Tests")
        r.header(["Orthogonalization: Cholesky (Lutkepohl)", "Null Hypothesis: Residuals are multivariate normal"])
        r.table(["Component", "Skewness", "Chi-sq (skew)", "Kurtosis", "Chi-sq (kurt)", "Jarque-Bera", "Prob."], rows,
                ["str", "num", "num", "num", "num", "num", "prob"])
        return r

    def view_residcor(self) -> Report:
        from .stats import corr_report
        return corr_report(pd.DataFrame(np.asarray(self.sm.resid), columns=self.endog_names))

    def view_resids(self):
        idx = self.df.index[self.p:]
        resid = np.asarray(self.sm.resid)
        return FigureOutput(plots.line_plot({e.upper(): pd.Series(resid[:, j], index=idx)
                                             for j, e in enumerate(self.endog_names)}, "VAR residuals",
                                            multiple=True, zero=True), "VAR residuals")

    def proc_forecast(self, h=12, suffix="_f"):
        h = int(h)
        fm = auto_expand(self.wf, h, self.mask)
        exf = None
        if self.exdf is not None:
            exall = pd.DataFrame({x: self.ev.eval(x, ignore_sample=True, as_series=True) for x in self.exog_names})
            exf = exall.values[fm]
        fc, lo, hi = self.sm.forecast_interval(np.asarray(self.df.values)[-self.p:], h, exog_future=exf)
        out = {}
        for j, e in enumerate(self.endog_names):
            s = pd.Series(np.nan, index=self.wf.data.index)
            s.iloc[np.flatnonzero(fm)] = fc[:, j]
            self.wf.set_series(sanitize_name(e) + suffix, s, respect_sample=False)
            out[e] = (s, lo[:, j], hi[:, j])
        k = len(self.endog_names)
        fig, axes = plots.new_figure(7.5, 2.4 * math.ceil(k / 2), math.ceil(k / 2), 2, squeeze=False)
        for j, e in enumerate(self.endog_names):
            ax = axes.ravel()[j]
            hist = self.df[e]
            ax.plot(hist.index[-4 * h:], hist.values[-4 * h:], color="#333", lw=1)
            fi = self.wf.data.index[fm]
            ax.plot(fi, fc[:, j], color=plots.PALETTE[0], lw=1.4)
            ax.fill_between(fi, lo[:, j], hi[:, j], color="#9fb8e3", alpha=0.4)
            ax.set_title(f"{e.upper()} forecast", fontsize=9)
        for j in range(k, axes.size):
            axes.ravel()[j].set_visible(False)
        fig.tight_layout()
        r = Report("VAR Forecast")
        labs = np.array(self.wf.labels())[fm]
        r.table(["Period"] + [e.upper() for e in self.endog_names], [[labs[i]] + list(fc[i]) for i in range(h)])
        r.note(f"Forecast series saved with suffix '{suffix}'. 95% intervals shown in graph.")
        return [r, FigureOutput(fig, "VAR forecast")]

    def proc_makeresids(self, prefix="resid_"):
        resid = np.asarray(self.sm.resid)
        idx = np.flatnonzero(self.mask)[self.p:]
        for j, e in enumerate(self.endog_names):
            s = pd.Series(np.nan, index=self.wf.data.index)
            s.iloc[idx] = resid[:, j]
            self.wf.set_series(prefix + sanitize_name(e), s, respect_sample=False)
        return f"Residuals saved as {prefix}<name>"


def _lagname(e: str, l: int) -> str:
    import re as _re
    if _re.match(r"^[a-z_][a-z0-9_]*$", e):
        return f"{e.upper()}(-{l})"
    m = _re.match(r"^(d|dlog|log)\(([a-z_][a-z0-9_]*)\)$", e)
    if m:
        return f"{m.group(1).upper()}({m.group(2).upper()}(-{l}))"
    return f"({e.upper()})(-{l})"


def _girf(res, h):
    """Generalized impulse responses (Pesaran & Shin 1998)."""
    Sigma = np.asarray(res.sigma_u)
    ma = res.ma_rep(h)
    k = Sigma.shape[0]
    out = np.zeros((h + 1, k, k))
    for j in range(k):
        e = np.zeros(k)
        e[j] = 1
        for p in range(h + 1):
            out[p, :, j] = ma[p] @ Sigma @ e / math.sqrt(Sigma[j, j])
    return out


# ============================================================================
# VECM
# ============================================================================

class VECMResult(ModelResult):
    object_type = "var"
    method_label = "Vector Error Correction"

    def __init__(self, wf, ev, endog: list[str], lags: int = 1, rank: int = 1, det: str = "co", name: str = ""):
        from statsmodels.tsa.vector_ar.vecm import VECM
        self.wf, self.ev, self.name = wf, ev, name
        self.endog_names = [e.lower() for e in endog]
        df = pd.DataFrame({e: ev.eval(e, ignore_sample=True, as_series=True) for e in self.endog_names},
                          index=wf.data.index)
        mask = wf.sample & df.notna().all(axis=1).values
        if not contiguous_check(mask):
            raise ModelError("VECM estimation requires a contiguous sample")
        self.mask, self.df = mask, df[mask]
        self.lags, self.rank, self.det = int(lags), int(rank), det
        self.sm = VECM(self.df.values, k_ar_diff=self.lags, coint_rank=self.rank, deterministic=det).fit()

    def report(self) -> Report:
        res = self.sm
        names = [e.upper() for e in self.endog_names]
        k = len(names)
        r = Report(f"VEC: {self.name.upper()}" if self.name else "Vector Error Correction Estimates")
        r.header(["Vector Error Correction Estimates", f"Included observations: {self.df.shape[0] - self.lags - 1}",
                  f"Cointegrating rank: {self.rank}; lagged differences: {self.lags}",
                  "Deterministic terms: " + {"n": "none (EViews case 1)", "ci": "restricted constant in CE (case 2)",
                                             "co": "unrestricted constant (case 3, EViews default)",
                                             "li": "restricted trend in CE, unrestricted constant (case 4)",
                                             "lo": "unrestricted constant and trend (case 5)"}.get(self.det, self.det),
                  "Standard errors in ( ) & t-statistics in [ ]"], stamp=True)
        beta = np.asarray(res.beta)
        rows = []
        for i, n in enumerate(names):
            rows.append([f"{n}(-1)"] + [beta[i, j] for j in range(self.rank)])
        try:
            sb = np.asarray(res.stderr_beta)
            rows2 = []
            for i, n in enumerate(names):
                rows2.append([f"{n}(-1)"] + [beta[i, j] for j in range(self.rank)])
                rows2.append([""] + [f"({fmt_num(sb[i, j])})" if i >= self.rank else "" for j in range(self.rank)])
            rows = rows2
        except Exception:
            pass
        if res.det_coef_coint.size:
            dc = np.asarray(res.det_coef_coint)
            rows.append(["C"] + [dc[0, j] for j in range(self.rank)])
        r.table(["Cointegrating Eq:"] + [f"CointEq{j + 1}" for j in range(self.rank)], rows)
        alpha, sa = np.asarray(res.alpha), np.asarray(res.stderr_alpha)
        G, sg = np.asarray(res.gamma), np.asarray(res.stderr_gamma)
        rows = []
        for j in range(self.rank):
            rows.append([f"CointEq{j + 1}"] + [alpha[i, j] for i in range(k)])
            rows.append([""] + [f"({fmt_num(sa[i, j])})" for i in range(k)])
            rows.append([""] + [f"[{alpha[i, j] / sa[i, j]:.5f}]" for i in range(k)])
        for l in range(self.lags):
            for m, n in enumerate(names):
                c = l * k + m
                rows.append([f"D({n}(-{l + 1}))"] + [G[i, c] for i in range(k)])
                rows.append([""] + [f"({fmt_num(sg[i, c])})" for i in range(k)])
                rows.append([""] + [f"[{G[i, c] / sg[i, c]:.5f}]" for i in range(k)])
        if res.det_coef.size:
            dc, sdc = np.asarray(res.det_coef), np.asarray(res.stderr_det_coef)
            for c in range(dc.shape[1]):
                rows.append(["C" if c == 0 else "@TREND"] + [dc[i, c] for i in range(k)])
                rows.append([""] + [f"({fmt_num(sdc[i, c])})" for i in range(k)])
        r.table(["Error Correction:"] + [f"D({n})" for n in names], rows)
        r.stats([("Log likelihood", float(res.llf)), ("Number of coefficients",
                                                       int(alpha.size + G.size + res.det_coef.size))], [])
        return r

    def views(self):
        return [View("output", "Estimation Output", self.report),
                View("impulse", "Impulse Responses", self.view_impulse, [Param("h", "Periods", "int", 10)]),
                View("cointgraph", "Cointegration Graph", self.view_coint)]

    def procs(self):
        return [View("forecast", "Forecast...", self.proc_forecast,
                     [Param("h", "Horizon", "int", 12), Param("suffix", "Suffix", "text", "_f")], kind="proc")]

    def view_impulse(self, h=10):
        h = int(h)
        irf = self.sm.irf(h)
        names = [e.upper() for e in self.endog_names]
        k = len(names)
        resp = irf.orth_irfs
        panels = [(f"Response of {names[i]} to {names[j]}", resp[:, i, j], None, None) for i in range(k)
                  for j in range(k)]
        return FigureOutput(plots.grid_plot(panels, k, "Response to Cholesky One S.D. Innovations",
                                            x=np.arange(h + 1)), "VECM impulse responses")

    def view_coint(self):
        beta = np.asarray(self.sm.beta)
        ce = self.df.values @ beta
        return FigureOutput(plots.line_plot({f"CointEq{j + 1}": pd.Series(ce[:, j], index=self.df.index)
                                             for j in range(self.rank)}, "Cointegrating relation(s)", zero=True),
                            "Cointegration graph")

    def proc_forecast(self, h=12, suffix="_f"):
        h = int(h)
        fm = auto_expand(self.wf, h, self.mask)
        fc = self.sm.predict(steps=h)
        for j, e in enumerate(self.endog_names):
            s = pd.Series(np.nan, index=self.wf.data.index)
            s.iloc[np.flatnonzero(fm)] = fc[:, j]
            self.wf.set_series(sanitize_name(e) + suffix, s, respect_sample=False)
        labs = np.array(self.wf.labels())[fm]
        r = Report("VECM Forecast")
        r.table(["Period"] + [e.upper() for e in self.endog_names], [[labs[i]] + list(fc[i]) for i in range(h)])
        return r


# ============================================================================
# ARDL with bounds test
# ============================================================================

class ARDLResult(EquationBase):
    method_label = "ARDL"

    def __init__(self, wf, ev, spec: str, maxlag: int = 4, maxdl: int = 4, ic: str = "aic", fixed: str | None = None,
                 trend: str = "c", case: int = 3, name=""):
        super().__init__(wf, ev, spec, name)
        from statsmodels.tsa.ardl import ARDL, UECM, ardl_select_order
        terms = [t for t in split_terms(spec) if t.lower() != "c"]
        y_name, x_names = terms[0], terms[1:]
        cols = {t: ev.eval(t, ignore_sample=True, as_series=True) for t in terms}
        df = pd.DataFrame(cols, index=wf.data.index)
        mask = wf.sample & df.notna().all(axis=1).values
        if not contiguous_check(mask):
            raise ModelError("ARDL requires a contiguous sample")
        df = df[mask]
        self.mask = mask
        y = df[y_name].reset_index(drop=True)
        X = df[x_names].reset_index(drop=True)
        X.columns = [c.upper() for c in X.columns]
        y.name = y_name.upper()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if fixed:
                lg = [int(v) for v in str(fixed).replace(",", " ").split()]
                p, qs = lg[0], lg[1:] + [lg[-1]] * (len(x_names) - len(lg) + 1)
                order = {c: qs[i] for i, c in enumerate(X.columns)}
                res = ARDL(y, p, X, order, trend=trend).fit()
                self.sel = None
            else:
                sel = ardl_select_order(y, int(maxlag), X, int(maxdl), ic={"sic": "bic", "sc": "bic"}.get(ic, ic),
                                        trend=trend)
                res = sel.model.fit()
                self.sel = sel
        self.sm = res
        self.p = res.model.ar_lags[-1] if res.model.ar_lags else 0
        self.q = {k: (max(v) if v else 0) for k, v in (res.model.dl_lags or {}).items()}
        self.yname = y_name.upper()
        self.x_names = list(X.columns)
        nm = []
        for n in res.params.index:
            s = str(n)
            if s == "const":
                nm.append("C")
            elif s == "trend":
                nm.append("@TREND")
            elif ".L" in s:
                base, l = s.split(".L")
                nm.append(f"{base.upper()}(-{l})" if l != "0" else base.upper())
            else:
                nm.append(s.upper())
        self.names = nm
        self.params = np.asarray(res.params)
        self.bse = np.asarray(res.bse)
        self.cov = np.asarray(res.cov_params())
        self.tvalues = self.params / self.bse
        n = int(res.nobs)
        self.df_resid = n - len(self.params)
        self.pvalues = pvalue_t(self.tvalues, self.df_resid)
        self.yv = np.asarray(res.fittedvalues + res.resid)
        self.resid_ = np.asarray(res.resid)
        self.fitted_ = np.asarray(res.fittedvalues)
        m2 = np.zeros_like(mask)
        m2[np.flatnonzero(mask)[-n:]] = True
        self.mask = m2
        self.case = int(case)
        self.trend = trend
        self.y, self.X = y, X
        ssr = float(self.resid_ @ self.resid_)
        tss = float(((self.yv - self.yv.mean()) ** 2).sum())
        k = len(self.params)
        aic, sc, hq = info_criteria(float(res.llf), k, n)
        r2 = 1 - ssr / tss
        self.stats_dict = {"r2": r2, "rbar2": 1 - (1 - r2) * (n - 1) / (n - k), "se": math.sqrt(ssr / (n - k)),
                           "ssr": ssr, "logl": float(res.llf), "aic": aic, "sc": sc, "hq": hq, "dw": _dw(self.resid_),
                           "meandep": float(self.yv.mean()), "sddep": float(self.yv.std(ddof=1))}
        self.logl = float(res.llf)

    def report(self) -> Report:
        s = self.stats_dict
        lab = f"ARDL({self.p}" + "".join(f", {self.q.get(c, 0)}" for c in self.x_names) + ")"
        r = Report(f"Equation: {self.name.upper()}" if self.name else "ARDL")
        hdr = [f"Dependent Variable: {self.yname}", "Method: ARDL", f"Selected Model: {lab}"]
        hdr += sample_header(self.wf, self.mask)
        if self.sel is not None:
            hdr.append("Model selection method: information criterion; dynamic regressors: " + " ".join(self.x_names))
        r.header(hdr, stamp=True)
        r.coef_table(self.names, self.params, self.bse, self.tvalues, self.pvalues)
        r.stats([("R-squared", s["r2"]), ("Adjusted R-squared", s["rbar2"]), ("S.E. of regression", s["se"]),
                 ("Sum squared resid", s["ssr"]), ("Log likelihood", s["logl"])],
                [("Mean dependent var", s["meandep"]), ("S.D. dependent var", s["sddep"]),
                 ("Akaike info criterion", s["aic"]), ("Schwarz criterion", s["sc"]), ("Durbin-Watson stat", s["dw"])])
        try:
            r.extend(self.view_bounds(), with_title=True)
        except Exception as e:
            r.note(f"Bounds test unavailable: {e}")
        return r

    def views(self):
        return self.base_views() + [View("bounds", "Long Run Form and Bounds Test", self.view_bounds,
                                         group="Coefficient Diagnostics")]

    def view_bounds(self) -> Report:
        from statsmodels.tsa.ardl import UECM
        order = {c: max(self.q.get(c, 0), 1) for c in self.x_names}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            u = UECM(self.y, max(self.p, 1), self.X, order, trend=self.trend).fit()
            bt = u.bounds_test(case=self.case)
        r = Report("ARDL Long Run Form and Bounds Test")
        r.header([f"Case {self.case}: " + {1: "No constant, no trend", 2: "Restricted constant, no trend",
                                           3: "Unrestricted constant, no trend", 4: "Unrestricted constant, restricted trend",
                                           5: "Unrestricted constant and trend"}[self.case],
                  "Null Hypothesis: No levels relationship"])
        lr_rows = []
        try:
            p = u.params
            ylev = [i for i in p.index if i.endswith(".L1") and i.split(".L1")[0].upper() == self.yname]
            phi = p[ylev[0]] if ylev else np.nan
            for c in self.x_names:
                key = [i for i in p.index if i.upper().startswith(c + ".L1")]
                if key:
                    lr_rows.append([c, -p[key[0]] / phi])
            if lr_rows:
                r.table(["Variable", "Long-run coefficient"], lr_rows, title="Levels Equation (point estimates)")
        except Exception:
            pass
        cv = bt.critical_values
        rows = [["F-statistic", float(bt.statistic), "", ""], ["k", len(self.x_names), "", ""]]
        r.table(["Test Statistic", "Value", "", ""], rows, ["str", "num", "str", "str"])
        crow = []
        for lvl in cv.index:
            crow.append([f"{100 - float(lvl):g}%", float(cv.loc[lvl].iloc[0]), float(cv.loc[lvl].iloc[1])])
        r.table(["Signif.", "I(0)", "I(1)"], crow, title="Critical values (asymptotic)")
        try:
            pv = np.atleast_1d(np.asarray(bt.pvalue, dtype=float))
            r.stats([("p-value (I(0) bound)", float(pv[0])), ("p-value (I(1) bound)", float(pv[-1]))], [])
        except Exception:
            pass
        st = float(bt.statistic)
        lo5, hi5 = float(cv.iloc[1, 0]), float(cv.iloc[1, 1])
        verdict = ("F above the 5% I(1) bound: a long-run levels relationship exists" if st > hi5 else
                   "F below the 5% I(0) bound: no levels relationship" if st < lo5 else
                   "F between the 5% bounds: inconclusive")
        r.note(verdict)
        return r


# ============================================================================
# Markov switching
# ============================================================================

class MarkovResult(ModelResult):
    object_type = "equation"
    method_label = "Markov Switching Regression"

    def __init__(self, wf, ev, spec: str, k: int = 2, ar: int = 0, switchvar: bool = True, name=""):
        import statsmodels.api as sm
        self.wf, self.ev, self.name = wf, ev, name
        d = build_design(wf, ev, spec, allow_arma=False)
        if not contiguous_check(d.mask):
            raise ModelError("Markov switching requires a contiguous sample")
        self.d, self.mask, self.yname, self.k = d, d.mask, d.yname, int(k)
        y = d.y.values
        cols = [c for c in d.xnames if c != "C"]
        exog = d.X[cols].values if cols else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if int(ar) > 0:
                mod = sm.tsa.MarkovAutoregression(y, k_regimes=self.k, order=int(ar), exog=exog,
                                                  trend="c" if "C" in d.xnames else "n",
                                                  switching_variance=bool(switchvar), switching_ar=False)
            else:
                mod = sm.tsa.MarkovRegression(y, k_regimes=self.k, exog=exog, trend="c" if "C" in d.xnames else "n",
                                              switching_variance=bool(switchvar))
            self.sm = mod.fit(search_reps=20, disp=False)
        self.ar = int(ar)
        self.exog_names = cols

    def report(self) -> Report:
        res = self.sm
        r = Report(f"Equation: {self.name.upper()}" if self.name else "Markov Switching")
        r.header([f"Dependent Variable: {self.yname}", f"Method: Markov Switching Regression ({self.k} regimes"
                  + (f", AR({self.ar})" if self.ar else "") + ")"] + sample_header(self.wf, self.mask)
                 + ["Number of states: " + str(self.k), "Ordinary standard errors & covariance using numeric Hessian"],
                 stamp=True)
        names = [self._pname(p) for p in res.model.param_names]
        pv = np.asarray(res.pvalues)
        r.coef_table(names, np.asarray(res.params), np.asarray(res.bse), np.asarray(res.tvalues), pv,
                     stat_label="z-Statistic")
        P = np.asarray(res.regime_transition)[:, :, 0]
        r.table([""] + [f"{j + 1}" for j in range(self.k)],
                [[f"{i + 1}"] + [P[j, i] for j in range(self.k)] for i in range(self.k)],
                title="Transition probabilities P(i, k) = P(s(t) = k | s(t-1) = i)")
        dur = np.asarray(res.expected_durations)
        r.table(["Regime", "Expected duration"], [[i + 1, dur[i]] for i in range(self.k)])
        aic, sc, hq = info_criteria(res.llf, len(res.params), int(res.nobs))
        r.stats([("Log likelihood", res.llf), ("Akaike info criterion", aic)], [("Schwarz criterion", sc),
                                                                              ("Hannan-Quinn criter.", hq)])
        return r

    def _pname(self, p):
        p = str(p)
        rx = __import__("re")
        m = rx.match(r"p\[(\d+)->(\d+)\]", p)
        if m:
            return f"P({int(m.group(1)) + 1}->{int(m.group(2)) + 1})"
        m = rx.match(r"(.+)\[(\d+)\]", p)
        if m:
            base, reg = m.group(1), int(m.group(2)) + 1
            if base == "const":
                base = "C"
            elif base.startswith("x"):
                try:
                    base = self.exog_names[int(base[1:]) - 1].upper()
                except Exception:
                    base = base.upper()
            elif base == "sigma2":
                base = "SIGMA^2"
            elif base.startswith("ar.L"):
                base = f"AR({base[4:]})"
            return f"{base} — Regime {reg}"
        return p.upper()

    def views(self):
        return [View("output", "Estimation Output", self.report),
                View("probs", "Regime Probabilities", self.view_probs,
                     [Param("type", "Type", "choice", "smoothed", ["smoothed", "filtered", "predicted"])])]

    def procs(self):
        return [View("makeprobs", "Make Regime Probability Series", self.proc_makeprobs,
                     [Param("prefix", "Prefix", "text", "prob_reg")], kind="proc")]

    def _probs(self, type_="smoothed"):
        res = self.sm
        P = {"smoothed": res.smoothed_marginal_probabilities, "filtered": res.filtered_marginal_probabilities,
             "predicted": res.predicted_marginal_probabilities}[type_]
        return np.asarray(P)

    def view_probs(self, type="smoothed"):  # noqa: A002
        P = self._probs(type)
        idx = np.flatnonzero(self.mask)[-len(P):]
        ix = self.wf.data.index[idx]
        fig, axes = plots.new_figure(7.5, 2.2 * self.k, self.k, 1, squeeze=False)
        for j in range(self.k):
            ax = axes[j, 0]
            ax.fill_between(ix, 0, P[:, j], color=plots.PALETTE[j], alpha=0.5)
            ax.set_ylim(0, 1)
            ax.set_title(f"{type.capitalize()} probability of regime {j + 1}", fontsize=9)
        fig.tight_layout()
        return FigureOutput(fig, "Regime probabilities")

    def proc_makeprobs(self, prefix="prob_reg"):
        P = self._probs("smoothed")
        idx = np.flatnonzero(self.mask)[-len(P):]
        for j in range(self.k):
            s = pd.Series(np.nan, index=self.wf.data.index)
            s.iloc[idx] = P[:, j]
            self.wf.set_series(f"{prefix}{j + 1}", s, respect_sample=False)
        return f"Saved {prefix}1..{prefix}{self.k}"


# ============================================================================
# filters
# ============================================================================

def hp_filter(wf, ev, expr, lam=None, trend_name=None, cycle_name=None):
    from statsmodels.tsa.filters.hp_filter import hpfilter
    s, m = _clean_series(wf, ev, expr)
    if not contiguous_check(m):
        raise ModelError("HP filter requires a contiguous sample")
    lam = float(lam) if lam else {"a": 100, "s": 400, "q": 1600, "m": 14400, "w": 270400, "d5": 6.25e6,
                                  "d7": 6.25e6}.get(wf.freq, 1600)
    cyc, tr = hpfilter(s[m].values, lamb=lam)
    trend_name = trend_name or f"{expr.lower()}_hpt"
    t = series_full(wf, tr, m)
    c = series_full(wf, cyc, m)
    wf.set_series(trend_name, t, respect_sample=False)
    if cycle_name:
        wf.set_series(cycle_name, c, respect_sample=False)
    fig, axes = plots.new_figure(7.5, 5, 2, 1, sharex=True)
    idx = wf.data.index[m] if not wf.is_panel else np.arange(m.sum())
    axes[0].plot(idx, s[m].values, color=plots.PALETTE[1], lw=1, label=expr.upper())
    axes[0].plot(idx, tr, color=plots.PALETTE[0], lw=1.5, label="Trend")
    axes[0].legend(frameon=False)
    axes[0].set_title(f"Hodrick-Prescott Filter (lambda={lam:g})")
    axes[1].plot(idx, cyc, color=plots.PALETTE[2], lw=1)
    axes[1].axhline(0, color="#444", lw=0.6)
    axes[1].set_title("Cycle")
    fig.tight_layout()
    return [FigureOutput(fig, "HP filter"), f"Trend saved to {trend_name.upper()}" +
            (f", cycle to {cycle_name.upper()}" if cycle_name else "")]


def bp_filter(wf, ev, expr, kind="bk", low=6, high=32, K=12, name=None):
    from statsmodels.tsa.filters import bk_filter, cf_filter
    s, m = _clean_series(wf, ev, expr)
    v = s[m].values
    if kind == "cf":
        cyc, tr = cf_filter.cffilter(v, low=float(low), high=float(high))
        full = series_full(wf, cyc, m)
    else:
        cyc = bk_filter.bkfilter(v, low=float(low), high=float(high), K=int(K))
        mm = m.copy()
        idx = np.flatnonzero(m)
        mm[:] = False
        mm[idx[int(K):len(idx) - int(K)]] = True
        full = series_full(wf, cyc, mm)
    name = name or f"{expr.lower()}_{kind}"
    wf.set_series(name, full, respect_sample=False)
    return [FigureOutput(plots.line_plot({name: full[m]}, f"{'Baxter-King' if kind == 'bk' else 'Christiano-Fitzgerald'}"
                                         f" band-pass cycle ({low}-{high} periods)", zero=True), "Band-pass filter"),
            f"Cycle saved to {name.upper()}"]
