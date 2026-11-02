"""
Panel data econometrics.

* Static panels (linearmodels): pooled OLS, fixed effects (cross-section,
  period, two-way), random effects (Swamy-Arora), between, first differences,
  Fama-MacBeth; ordinary / White / clustered / Driscoll-Kraay covariances.
* Specification tests: redundant fixed effects (F, LR), Hausman, Breusch-Pagan
  and Honda LM, Pesaran CD, Wooldridge serial correlation, modified Wald
  groupwise heteroskedasticity.
* Dynamic panel GMM: Arellano-Bond difference GMM and Blundell-Bond system GMM,
  one- and two-step with Windmeijer-corrected standard errors, Hansen/Sargan
  over-identification tests and Arellano-Bond AR(m) tests.
* Panel unit roots (LLC, IPS, Fisher-ADF/PP, Hadri) and Dumitrescu-Hurlin
  panel Granger causality.
"""
from __future__ import annotations

import math
import warnings
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import stats

from ..report import FigureOutput, Report, fmt_num
from . import plots
from .base import (ModelError, ModelResult, Param, View, build_design, info_criteria, pvalue_t, pvalue_z, safe_inv,
                   series_full, split_terms, wald_from_restrictions)
from .regression import EquationBase, _dw


def _require_panel(wf):
    if not wf.is_panel:
        raise ModelError("This procedure requires a panel workfile (import with panel structure: id and time)")


def panel_info(idx: pd.MultiIndex) -> tuple[int, int, int, bool]:
    ids = idx.get_level_values(0)
    t = idx.get_level_values(1)
    N = len(pd.unique(ids))
    T = len(pd.unique(t))
    n = len(idx)
    return N, T, n, n == N * T


# ============================================================================
# static panel models
# ============================================================================

METHOD_LABEL = {
    "pooled": "Panel Least Squares", "fe": "Panel Least Squares", "te": "Panel Least Squares",
    "twoway": "Panel Least Squares", "re": "Panel EGLS (Cross-section random effects)",
    "be": "Panel Between Estimator", "fd": "Panel First-Difference Least Squares",
    "fm": "Fama-MacBeth (cross-sectional regressions)",
}


class PanelResult(EquationBase):
    object_type = "panel"

    def __init__(self, wf, ev, spec: str, method: str = "fe", cov: str = "ordinary", hac_lags=None, name=""):
        _require_panel(wf)
        super().__init__(wf, ev, spec, name)
        from linearmodels.panel import (BetweenOLS, FamaMacBeth, FirstDifferenceOLS, PanelOLS, PooledOLS,
                                        RandomEffects)
        method = method.lower()
        self.method = method
        d = build_design(wf, ev, spec, allow_arma=False)
        self.d, self.mask, self.yname = d, d.mask, d.yname
        y, X = d.y, d.X.copy()
        if method in ("fd",) and "C" in X.columns:
            X = X.drop(columns=["C"])
        self.xnames = list(X.columns)
        cv = (cov or "ordinary").lower()
        fit_kw = {}
        if cv in ("white", "robust", "hc1"):
            fit_kw = {"cov_type": "robust"}
            self.cov_label = "White cross-section standard errors & covariance (d.f. corrected)"
        elif cv in ("cluster", "clustered", "cluster_entity", "entity"):
            fit_kw = {"cov_type": "clustered", "cluster_entity": True}
            self.cov_label = "Cross-section clustered standard errors & covariance"
        elif cv in ("cluster_time", "time", "period"):
            fit_kw = {"cov_type": "clustered", "cluster_time": True}
            self.cov_label = "Period clustered standard errors & covariance"
        elif cv in ("cluster2", "twoway", "both"):
            fit_kw = {"cov_type": "clustered", "cluster_entity": True, "cluster_time": True}
            self.cov_label = "Two-way (cross-section & period) clustered standard errors"
        elif cv in ("dk", "driscoll", "kernel", "hac", "scc"):
            fit_kw = {"cov_type": "kernel"}
            if hac_lags is not None:
                fit_kw["bandwidth"] = hac_lags
            self.cov_label = "Driscoll-Kraay (spatial correlation consistent) standard errors"
        else:
            fit_kw = {"cov_type": "unadjusted"}
            self.cov_label = ""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if method == "pooled":
                mod = PooledOLS(y, X)
            elif method == "fe":
                mod = PanelOLS(y, X, entity_effects=True, drop_absorbed=True)
            elif method == "te":
                mod = PanelOLS(y, X, time_effects=True, drop_absorbed=True)
            elif method == "twoway":
                mod = PanelOLS(y, X, entity_effects=True, time_effects=True, drop_absorbed=True)
            elif method == "re":
                mod = RandomEffects(y, X)
            elif method == "be":
                mod = BetweenOLS(y, X)
                if fit_kw.get("cov_type") == "clustered" or fit_kw.get("cov_type") == "kernel":
                    fit_kw = {"cov_type": "robust"}
            elif method == "fd":
                mod = FirstDifferenceOLS(y, X)
            elif method == "fm":
                mod = FamaMacBeth(y, X)
                fit_kw = {"cov_type": "kernel"} if cv in ("hac", "kernel", "nw") else {"cov_type": "unadjusted"}
                self.cov_label = "Fama-MacBeth standard errors" + (" (Newey-West)" if fit_kw["cov_type"] == "kernel" else "")
            else:
                raise ModelError(f"Unknown panel method '{method}' (pooled, fe, te, twoway, re, be, fd, fm)")
            res = mod.fit(**fit_kw)
        self.lm = res
        names = [n for n in self.xnames if n in res.params.index]
        self.names = names
        self.params = np.asarray(res.params[names])
        self.bse = np.asarray(res.std_errors[names])
        self.cov = np.asarray(res.cov.loc[names, names])
        self.tvalues = self.params / self.bse
        self.df_resid = int(res.df_resid)
        self.pvalues = pvalue_t(self.tvalues, self.df_resid)
        N, T, n, bal = panel_info(y.index)
        self.N, self.T, self.n, self.balanced = N, T, n, bal
        # residuals on the original rows (where available)
        resid = res.resids
        if method in ("be", "fd", "fm"):
            fitted = pd.Series(X[names].values @ self.params, index=X.index)
            if method == "fd":
                fitted = fitted
            e_full = y - fitted
            if method == "fd":
                e_use = resid.reindex(y.index)
            else:
                e_use = e_full
        else:
            e_use = resid.reindex(y.index)
        self.yv = y.values
        self.resid_series = e_use
        self.resid_ = np.nan_to_num(e_use.values)
        self.fitted_ = self.yv - self.resid_
        self._stats(y, res)

    def _stats(self, y, res):
        n = self.n
        e = self.resid_series.dropna().values
        ssr = float(e @ e)
        k = len(self.params)
        neff = len(e)
        if self.method in ("fe", "twoway", "te"):
            k_eff = k + (self.N - (1 if "C" in self.names else 0) if self.method in ("fe", "twoway") else 0) + \
                    (self.T - 1 if self.method in ("te", "twoway") else 0)
        else:
            k_eff = k
        try:
            r2 = float(res.rsquared_overall) if self.method in ("re", "be") else float(res.rsquared)
        except Exception:
            r2 = np.nan
        if self.method in ("fe", "twoway", "te"):
            tss = float(((self.yv - self.yv.mean()) ** 2).sum())
            r2 = 1 - ssr / tss  # LSDV R-squared (EViews convention)
        logl = -neff / 2 * (1 + math.log(2 * math.pi) + math.log(ssr / neff)) if neff else np.nan
        aic, sc, hq = info_criteria(logl, k_eff, neff) if neff else (np.nan,) * 3
        F = Fp = np.nan
        try:
            fs = res.f_statistic
            F, Fp = float(fs.stat), float(fs.pval)
        except Exception:
            pass
        self.logl = logl
        dw = np.nan
        try:
            ee = self.resid_series.dropna()
            num = ee.groupby(level=0).diff().dropna()
            dw = float((num ** 2).sum() / (ee ** 2).sum())
        except Exception:
            pass
        self.stats_dict = {"r2": r2, "rbar2": 1 - (1 - r2) * (neff - 1) / max(neff - k_eff, 1), "ssr": ssr,
                           "se": math.sqrt(ssr / max(neff - k_eff, 1)), "logl": logl, "aic": aic, "sc": sc, "hq": hq,
                           "f": F, "fprob": Fp, "dw": dw, "meandep": float(self.yv.mean()),
                           "sddep": float(self.yv.std(ddof=1)), "regobs": n, "ncoef": k}
        try:
            self.stats_dict["r2within"] = float(res.rsquared_within)
            self.stats_dict["r2between"] = float(res.rsquared_between)
            self.stats_dict["r2overall"] = float(res.rsquared_overall)
        except Exception:
            pass

    def report(self) -> Report:
        s = self.stats_dict
        r = Report(f"Equation: {self.name.upper()}" if self.name else "Panel Equation")
        hdr = [f"Dependent Variable: {self.yname}", f"Method: {METHOD_LABEL[self.method]}",
               f"Sample: {self.wf.sample_label(self.mask)}", f"Periods included: {self.T}",
               f"Cross-sections included: {self.N}",
               f"Total panel ({'balanced' if self.balanced else 'unbalanced'}) observations: {self.n}"]
        if self.method == "re":
            hdr.append("Swamy and Arora estimator of component variances")
        if self.cov_label:
            hdr.append(self.cov_label)
        r.header(hdr, stamp=True)
        r.coef_table(self.names, self.params, self.bse, self.tvalues, self.pvalues)
        eff = {"fe": "Cross-section fixed (dummy variables)", "te": "Period fixed (dummy variables)",
               "twoway": "Cross-section fixed (dummy variables)\nPeriod fixed (dummy variables)"}.get(self.method)
        if eff:
            r.section("Effects Specification")
            r.text(eff)
        if self.method == "re":
            try:
                vd = self.lm.variance_decomposition
                se_e, se_r = math.sqrt(float(vd["Effects"])), math.sqrt(float(vd["Residual"]))
                rho = float(vd["Percent due to Effects"])
                r.table(["Effects Specification", "S.D.", "Rho"],
                        [["Cross-section random", se_e, rho], ["Idiosyncratic random", se_r, 1 - rho]],
                        ["str", "num", "f4"])
                th = self.lm.theta
                r.note(f"Theta (quasi-demeaning) = {fmt_num(float(np.asarray(th).mean()))}")
            except Exception:
                pass
        left = [("R-squared", s["r2"]), ("Adjusted R-squared", s["rbar2"]), ("S.E. of regression", s["se"]),
                ("Sum squared resid", s["ssr"]), ("Log likelihood", s["logl"])]
        if not math.isnan(s["f"]):
            left += [("F-statistic", s["f"]), ("Prob(F-statistic)", s["fprob"])]
        right = [("Mean dependent var", s["meandep"]), ("S.D. dependent var", s["sddep"]),
                 ("Akaike info criterion", s["aic"]), ("Schwarz criterion", s["sc"]),
                 ("Hannan-Quinn criter.", s["hq"]), ("Durbin-Watson stat", s["dw"])]
        if "r2within" in s:
            right += [("R-squared (within)", s["r2within"]), ("R-squared (between)", s["r2between"])]
        r.stats(left, right)
        return r

    def views(self):
        v = [x for x in self.base_views() if x.key in ("output", "wald", "coefcov", "cinterval", "hist")]
        v += [View("resids", "Residual Graph (stacked)", self.view_resid_graph),
              View("effects", "Estimated Effects", self.view_effects, group="Panel Tests"),
              View("fixedtest", "Redundant Fixed Effects - Likelihood Ratio", self.view_fixed_test, group="Panel Tests"),
              View("hausman", "Correlated Random Effects - Hausman Test", self.view_hausman, group="Panel Tests"),
              View("lmtest", "Breusch-Pagan / Honda LM Test for Random Effects", self.view_lm, group="Panel Tests"),
              View("cdtest", "Cross-section Dependence Tests (Pesaran CD, BP-LM)", self.view_cd, group="Panel Tests"),
              View("serialtest", "Wooldridge Serial Correlation Test", self.view_wooldridge, group="Panel Tests"),
              View("hettest", "Modified Wald Test (groupwise heteroskedasticity)", self.view_groupwise,
                   group="Panel Tests")]
        return v

    # ---- helpers
    def _refit(self, method, cov="ordinary"):
        old = self.wf._sample
        try:
            self.wf._sample = self.mask.copy()
            return PanelResult(self.wf, self.ev, self.spec, method=method, cov=cov)
        finally:
            self.wf._sample = old

    def view_resid_graph(self):
        e = self.resid_series.dropna()
        fig, ax = plots.new_figure(7.5, 4)
        ax.plot(np.arange(len(e)), e.values, color=plots.PALETTE[0], lw=0.7)
        ax.axhline(0, color="#444", lw=0.6)
        ids = e.index.get_level_values(0)
        b = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
        for x in b:
            ax.axvline(x, color="#ddd", lw=0.5)
        ax.set_title("Stacked residuals (cross-section boundaries marked)")
        fig.tight_layout()
        return FigureOutput(fig, "Panel residuals")

    def view_effects(self):
        try:
            eff = self.lm.estimated_effects
        except Exception:
            raise ModelError("Estimated effects are not available for this estimator")
        e = eff.groupby(level=0).first().iloc[:, 0] if self.method in ("fe", "re", "twoway") else \
            eff.groupby(level=1).first().iloc[:, 0]
        r = Report("Estimated Effects")
        r.table(["Cross-section" if self.method != "te" else "Period", "Effect"], [[str(i), float(v)] for i, v in
                                                                                    e.items()])
        return r

    def view_fixed_test(self):
        if self.method not in ("fe", "te", "twoway"):
            fe = self._refit("fe")
        else:
            fe = self
        pooled = self._refit("pooled")
        r = Report("Redundant Fixed Effects Tests")
        r.text(f"Equation: {self.name.upper() or 'untitled'}\nTest cross-section fixed effects")
        try:
            fp = fe.lm.f_pooled
            F, Fp, df = float(fp.stat), float(fp.pval), fp.df
            df_str = f"({fp.df}, {fp.df_denom})"
        except Exception:
            F = Fp = np.nan
            df_str = ""
        LR = 2 * (fe.logl - pooled.logl)
        dfl = fe.N - 1 if fe.method == "fe" else fe.T - 1 if fe.method == "te" else fe.N + fe.T - 2
        r.table(["Effects Test", "Statistic", "d.f.", "Prob."],
                [["Cross-section F", F, df_str, Fp], ["Cross-section Chi-square", LR, str(dfl), stats.chi2.sf(LR, dfl)]],
                ["str", "num", "str", "prob"])
        r.note("H0: the fixed effects are jointly redundant (pooled OLS is adequate).")
        return r

    def view_hausman(self):
        # classical Hausman: both estimators with conventional covariance matrices
        fe = self._refit("fe", "ordinary")
        re = self._refit("re", "ordinary")
        common = [n for n in fe.names if n in re.names and n != "C"]
        if not common:
            raise ModelError("No common time-varying coefficients for the Hausman test")
        bi = [fe.names.index(n) for n in common]
        ri = [re.names.index(n) for n in common]
        d = fe.params[bi] - re.params[ri]
        V = fe.cov[np.ix_(bi, bi)] - re.cov[np.ix_(ri, ri)]
        H = float(d @ np.linalg.pinv(V) @ d)
        df = len(common)
        r = Report("Correlated Random Effects - Hausman Test")
        r.text("Test cross-section random effects\nH0: random effects are uncorrelated with regressors (RE consistent)")
        invalid = H < 0
        if invalid:
            H = 0.0
        r.table(["Test Summary", "Chi-Sq. Statistic", "Chi-Sq. d.f.", "Prob."],
                [["Cross-section random", H, df, stats.chi2.sf(H, df)]], ["str", "num", "int", "prob"])
        if invalid:
            r.warn("Cross-section test variance is invalid. Hausman statistic set to zero (see robust test below).")
        rows = []
        for j, n in enumerate(common):
            vd = V[j, j]
            rows.append([n, fe.params[bi[j]], re.params[ri[j]], vd,
                         stats.chi2.sf(d[j] ** 2 / vd, 1) if vd > 0 else np.nan])
        r.table(["Variable", "Fixed", "Random", "Var(Diff.)", "Prob."], rows, ["str", "num", "num", "num", "prob"],
                title="Cross-section random effects test comparisons")
        # regression-based (Mundlak / Wooldridge) robust version
        try:
            from linearmodels.panel import RandomEffects
            d0 = build_design(self.wf, self.ev, self.spec, mask=self.mask, allow_arma=False)
            X = d0.X.copy()
            mcols = []
            for n in common:
                if n in X.columns:
                    X[f"MEAN({n})"] = X[n].groupby(level=0).transform("mean")
                    mcols.append(f"MEAN({n})")
            res = RandomEffects(d0.y, X).fit(cov_type="clustered", cluster_entity=True)
            b = np.asarray(res.params[mcols])
            Vm = np.asarray(res.cov.loc[mcols, mcols])
            W = float(b @ safe_inv(Vm) @ b)
            r.table(["Robust test (Mundlak, cluster-robust)", "Chi-Sq. Statistic", "Chi-Sq. d.f.", "Prob."],
                    [["Group means of regressors = 0", W, len(mcols), stats.chi2.sf(W, len(mcols))]],
                    ["str", "num", "int", "prob"])
        except Exception as e:  # pragma: no cover
            r.note(f"Robust Mundlak test unavailable: {e}")
        return r

    def view_lm(self):
        pooled = self if self.method == "pooled" else self._refit("pooled")
        e = pooled.resid_series.dropna()
        g = e.groupby(level=0)
        Ti = g.size().values
        s_i = g.sum().values
        n = len(e)
        ssq = float((e ** 2).sum())
        LM = n ** 2 / (2 * float(np.sum(Ti * (Ti - 1)))) * (float(np.sum(s_i ** 2)) / ssq - 1) ** 2
        honda = math.copysign(math.sqrt(LM), float(np.sum(s_i ** 2)) / ssq - 1)
        # time effects
        gt = e.groupby(level=1)
        Nt = gt.size().values
        s_t = gt.sum().values
        LMt = n ** 2 / (2 * float(np.sum(Nt * (Nt - 1)))) * (float(np.sum(s_t ** 2)) / ssq - 1) ** 2
        r = Report("Lagrange Multiplier Tests for Random Effects")
        r.text("Null hypotheses: No effects. Alternative one-sided for Honda.")
        r.table(["Test", "Cross-section", "Period", "Both"],
                [["Breusch-Pagan", LM, LMt, LM + LMt],
                 ["  (Prob.)", stats.chi2.sf(LM, 1), stats.chi2.sf(LMt, 1), stats.chi2.sf(LM + LMt, 2)],
                 ["Honda", honda, math.copysign(math.sqrt(LMt), float(np.sum(s_t ** 2)) / ssq - 1),
                  (honda + math.copysign(math.sqrt(LMt), float(np.sum(s_t ** 2)) / ssq - 1)) / math.sqrt(2)],
                 ["  (Prob.)", stats.norm.sf(honda), stats.norm.sf(math.copysign(math.sqrt(LMt), float(np.sum(s_t ** 2)) / ssq - 1)),
                  np.nan]], ["str", "num", "num", "num"])
        return r

    def view_cd(self):
        e = self.resid_series.dropna().unstack(level=0)
        N = e.shape[1]
        rows_rho, Tij = [], []
        lm = 0.0
        cd = 0.0
        cnt = 0
        for i in range(N):
            for j in range(i + 1, N):
                ab = e.iloc[:, [i, j]].dropna()
                T = len(ab)
                if T < 3:
                    continue
                rho = float(np.corrcoef(ab.values.T)[0, 1])
                cd += math.sqrt(T) * rho
                lm += T * rho ** 2
                rows_rho.append(abs(rho))
                Tij.append(T)
                cnt += 1
        CD = math.sqrt(2 / (N * (N - 1))) * cd
        Tbar = np.mean(Tij) if Tij else np.nan
        sLM = math.sqrt(1 / (N * (N - 1))) * (lm - cnt)
        r = Report("Residual Cross-Section Dependence Test")
        r.text("Null hypothesis: No cross-section dependence (correlation) in residuals")
        r.table(["Test", "Statistic", "d.f.", "Prob."],
                [["Breusch-Pagan LM", lm, cnt, stats.chi2.sf(lm, cnt)],
                 ["Pesaran scaled LM", sLM, "", 2 * stats.norm.sf(abs(sLM))],
                 ["Pesaran CD", CD, "", 2 * stats.norm.sf(abs(CD))]], ["str", "num", "str", "prob"])
        r.note(f"Average |rho| = {fmt_num(np.mean(rows_rho) if rows_rho else np.nan)};  average T(ij) = {fmt_num(Tbar)}")
        return r

    def view_wooldridge(self):
        fd = self._refit("fd") if self.method != "fd" else self
        e = fd.lm.resids.dropna()
        el = e.groupby(level=0).shift(1)
        df = pd.DataFrame({"e": e, "el": el}).dropna()
        x = df["el"].values
        y = df["e"].values
        b = float(x @ y / (x @ x))
        u = y - b * x
        # cluster-robust variance by cross-section
        g = df.index.get_level_values(0)
        meat = 0.0
        for gid in pd.unique(g):
            m = g == gid
            meat += float((x[m] * u[m]).sum()) ** 2
        V = meat / (x @ x) ** 2
        F = (b + 0.5) ** 2 / V
        N = len(pd.unique(g))
        r = Report("Wooldridge Test for Serial Correlation in Panel Data")
        r.text("H0: no first-order serial correlation in the idiosyncratic errors")
        r.stats([("F-statistic", F), ("Coefficient on lagged FD residual", b)],
                [(f"Prob. F(1,{N - 1})", stats.f.sf(F, 1, N - 1)), ("(equals -0.5 under H0)", "")])
        return r

    def view_groupwise(self):
        e = self.resid_series.dropna()
        g = e.groupby(level=0)
        s2 = float((e ** 2).sum() / len(e))
        W = 0.0
        N = 0
        for _, ei in g:
            v = ei.values
            Ti = len(v)
            if Ti < 2:
                continue
            si = float(np.mean(v ** 2))
            Vi = float(np.sum((v ** 2 - si) ** 2)) / (Ti * (Ti - 1))
            if Vi > 0:
                W += (si - s2) ** 2 / Vi
                N += 1
        r = Report("Modified Wald Test for Groupwise Heteroskedasticity")
        r.text("H0: sigma(i)^2 = sigma^2 for all cross-sections")
        r.stats([("Chi-square statistic", W)], [(f"Prob. Chi-Square({N})", stats.chi2.sf(W, N))])
        return r


# ============================================================================
# Dynamic panel GMM
# ============================================================================

class DPDResult(ModelResult):
    object_type = "panel"
    method_label = "Panel Generalized Method of Moments"

    def __init__(self, wf, ev, spec: str, lags: int = 1, system: bool = False, twostep: bool = True,
                 endog: list[str] | None = None, predet: list[str] | None = None, maxlag: int = 99,
                 collapse: bool = False, timedum: bool = True, name: str = ""):
        _require_panel(wf)
        self.wf, self.ev, self.name, self.spec = wf, ev, name, spec
        terms = [t for t in split_terms(spec) if t.lower() != "c"]
        if not terms:
            raise ModelError("Specify the dependent variable and regressors")
        self.yname = terms[0].upper()
        xs = terms[1:]
        endog = [e.lower() for e in (endog or [])]
        predet = [p.lower() for p in (predet or [])]
        self.L, self.system, self.twostep, self.collapse = int(lags), bool(system), bool(twostep), bool(collapse)
        self.maxlag = int(maxlag)
        idx = wf.data.index
        ids = idx.get_level_values(0)
        times = idx.get_level_values(1)
        tgrid = sorted(pd.unique(times))
        tpos = {t: i for i, t in enumerate(tgrid)}
        Tg = len(tgrid)
        yS = ev.eval(terms[0], ignore_sample=True, as_series=True)
        Xs = {x: ev.eval(x, ignore_sample=True, as_series=True) for x in xs}
        smp = wf.sample
        # assemble per-individual arrays on the time grid
        uid = list(dict.fromkeys(ids))
        Y = np.full((len(uid), Tg), np.nan)
        XX = np.full((len(uid), Tg, len(xs)), np.nan)
        pos_i = {u: i for i, u in enumerate(uid)}
        ii = np.array([pos_i[u] for u in ids])
        tt = np.array([tpos[t] for t in times])
        ok = smp
        Y[ii[ok], tt[ok]] = yS.values[ok]
        for j, x in enumerate(xs):
            XX[ii[ok], tt[ok], j] = Xs[x].values[ok]
        self.tgrid = tgrid
        # regressor list: L lags of y, xs, time dummies
        names = [f"{self.yname}(-{l})" for l in range(1, self.L + 1)] + [x.upper() for x in xs]
        kinds = ["ylag"] * self.L + ["x"] * len(xs)
        N = len(uid)

        def ylag(i, t, l):
            return Y[i, t - l] if t - l >= 0 else np.nan

        # valid periods for the difference equation: need y_t, y_{t-1}, ..., y_{t-L-1} and x_t, x_{t-1}
        first_t = self.L + 1
        td_cols = []
        if timedum:
            td_cols = list(range(first_t + 1 if not self.system else first_t + 1, Tg))  # drop base period(s)
        self.td_cols = td_cols
        names += [f"@YEAR={_tlab(tgrid[c])}" for c in td_cols]
        kinds += ["td"] * len(td_cols)
        if self.system:
            names.append("C")
            kinds.append("const")
        K = len(names)
        self.names, self.kinds = names, kinds

        def wrow(i, t):
            r = [ylag(i, t, l) for l in range(1, self.L + 1)] + list(XX[i, t, :])
            r += [1.0 if t == c else 0.0 for c in td_cols]
            if self.system:
                r.append(1.0)
            return np.array(r, dtype=float)

        gmm_vars = [("y", None, 2)]  # (kind, index, min lag for diff eq)
        for j, x in enumerate(xs):
            if x in endog:
                gmm_vars.append(("x", j, 2))
            elif x in predet:
                gmm_vars.append(("x", j, 1))

        def val(kind, j, i, t):
            if t < 0:
                return np.nan
            return Y[i, t] if kind == "y" else XX[i, t, j]

        exog_idx = [j for j, x in enumerate(xs) if x not in endog and x not in predet]
        # build instrument keys and per-individual blocks
        rows_y, rows_X, rows_Z, rows_meta = [], [], [], []
        keys: dict = {}

        def key(k):
            if k not in keys:
                keys[k] = len(keys)
            return keys[k]

        blocks = []
        for i in range(N):
            yi, Xi, Zi, meta = [], [], [], []
            for t in range(first_t, Tg):
                w_t, w_tm = wrow(i, t), wrow(i, t - 1)
                if np.isnan(Y[i, t]) or np.isnan(Y[i, t - 1]) or np.isnan(w_t).any() or np.isnan(w_tm).any():
                    continue
                dy = Y[i, t] - Y[i, t - 1]
                dw = w_t - w_tm
                if self.system:
                    dw[-1] = 0.0
                z = {}
                for (kd, j, lmin) in gmm_vars:
                    for l in range(lmin, min(self.maxlag, t) + 1):
                        v = val(kd, j, i, t - l)
                        if np.isnan(v):
                            continue
                        k_ = ("gd", kd, j, l) if self.collapse else ("gd", kd, j, t, l)
                        z[key(k_)] = v
                for j in exog_idx:
                    z[key(("ivd", j))] = XX[i, t, j] - XX[i, t - 1, j]
                for c in td_cols:
                    z[key(("tdd", c))] = (1.0 if t == c else 0.0) - (1.0 if t - 1 == c else 0.0)
                yi.append(dy)
                Xi.append(dw)
                Zi.append(z)
                meta.append(("d", t))
            if self.system:
                for t in range(first_t, Tg):
                    w_t = wrow(i, t)
                    if np.isnan(Y[i, t]) or np.isnan(w_t).any():
                        continue
                    z = {}
                    for (kd, j, lmin) in gmm_vars:
                        l = lmin - 1
                        a, b = val(kd, j, i, t - l), val(kd, j, i, t - l - 1)
                        if np.isnan(a) or np.isnan(b):
                            continue
                        k_ = ("gl", kd, j) if self.collapse else ("gl", kd, j, t)
                        z[key(k_)] = a - b
                    for j in exog_idx:
                        z[key(("ivl", j))] = XX[i, t, j]
                    for c in td_cols:
                        z[key(("tdl", c))] = 1.0 if t == c else 0.0
                    z[key(("const",))] = 1.0
                    yi.append(Y[i, t])
                    Xi.append(w_t)
                    Zi.append(z)
                    meta.append(("l", t))
            if yi:
                blocks.append((i, np.array(yi), np.array(Xi), Zi, meta))
        if not blocks:
            raise ModelError("No usable observations for dynamic panel GMM")
        Lz = len(keys)
        self.blocks = []
        for (i, yi, Xi, Zi, meta) in blocks:
            Z = np.zeros((len(yi), Lz))
            for r_, z in enumerate(Zi):
                for c, v in z.items():
                    Z[r_, c] = v
            self.blocks.append((i, yi, Xi, Z, meta))
        # drop all-zero instrument columns
        colsum = sum(np.abs(b[3]).sum(axis=0) for b in self.blocks)
        keep = colsum > 0
        self.blocks = [(i, yi, Xi, Z[:, keep], m) for (i, yi, Xi, Z, m) in self.blocks]
        # drop time-dummy regressors that are identically zero
        Xall = np.vstack([b[2] for b in self.blocks])
        kx = np.abs(Xall).sum(axis=0) > 0
        if not kx.all():
            self.names = [n for n, k in zip(self.names, kx) if k]
            self.kinds = [n for n, k in zip(self.kinds, kx) if k]
            self.blocks = [(i, yi, Xi[:, kx], Z, m) for (i, yi, Xi, Z, m) in self.blocks]
        self.N = len(self.blocks)
        self.nobs_d = sum(sum(1 for m in b[4] if m[0] == "d") for b in self.blocks)
        self.nobs_l = sum(sum(1 for m in b[4] if m[0] == "l") for b in self.blocks)
        self.nz = int(keep.sum())
        self._estimate()

    def _H(self, meta):
        n = len(meta)
        H = np.zeros((n, n))
        for a in range(n):
            ka, ta = meta[a]
            if ka == "d":
                H[a, a] = 2.0
                for b in range(n):
                    kb, tb = meta[b]
                    if kb == "d" and abs(ta - tb) == 1:
                        H[a, b] = -1.0
            else:
                H[a, a] = 1.0
        return H

    def _estimate(self):
        B = self.blocks
        K = len(self.names)
        ZX = sum(Z.T @ X for (_, y, X, Z, m) in B)
        Zy = sum(Z.T @ y for (_, y, X, Z, m) in B)
        A1 = sum(Z.T @ self._H(m) @ Z for (_, y, X, Z, m) in B)
        W1 = np.linalg.pinv(A1)
        M1 = ZX.T @ W1 @ ZX
        M1i = safe_inv(M1)
        th1 = M1i @ ZX.T @ W1 @ Zy
        u1 = [y - X @ th1 for (_, y, X, Z, m) in B]
        S1 = sum(Z.T @ np.outer(u, u) @ Z for (_, y, X, Z, m), u in zip(B, u1))
        V1r = M1i @ ZX.T @ W1 @ S1 @ W1 @ ZX @ M1i
        self.th1, self.V1r = th1, V1r
        # Sargan (non-robust, one-step)
        Zu1 = sum(Z.T @ u for (_, y, X, Z, m), u in zip(B, u1))
        ud = np.concatenate([u[[k for k, mm in enumerate(m) if mm[0] == "d"]] for (_, y, X, Z, m), u in zip(B, u1)])
        s2 = float(ud @ ud) / (2 * max(len(ud) - K, 1)) if not self.system else \
            float(np.concatenate(u1) @ np.concatenate(u1)) / max(sum(len(u) for u in u1) - K, 1)
        self.sargan = float(Zu1 @ W1 @ Zu1) / s2 if not self.system else float(Zu1 @ W1 @ Zu1) / s2
        self.df_over = self.nz - K
        if self.twostep:
            W2 = np.linalg.pinv(S1)
            M2 = ZX.T @ W2 @ ZX
            V2 = safe_inv(M2)
            th2 = V2 @ ZX.T @ W2 @ Zy
            u2 = [y - X @ th2 for (_, y, X, Z, m) in B]
            Zu2 = sum(Z.T @ u for (_, y, X, Z, m), u in zip(B, u2))
            D = np.zeros((K, K))
            for k in range(K):
                dA = -sum(Z.T @ (np.outer(X[:, k], u) + np.outer(u, X[:, k])) @ Z
                          for (_, y, X, Z, m), u in zip(B, u1))
                D[:, k] = -V2 @ ZX.T @ W2 @ dA @ W2 @ Zu2
            Vw = V2 + D @ V2 + V2 @ D.T + D @ V1r @ D.T
            self.theta, self.cov, self.u, self.W, self.Minv = th2, (Vw + Vw.T) / 2, u2, W2, V2
            self.hansen = float(Zu2 @ W2 @ Zu2)
            self.V2_uncorrected = V2
        else:
            self.theta, self.cov, self.u, self.W, self.Minv = th1, V1r, u1, W1, M1i
            S1i = np.linalg.pinv(S1)
            self.hansen = float(Zu1 @ S1i @ Zu1)
        self.bse = np.sqrt(np.clip(np.diag(self.cov), 0, None))
        self.zstat = self.theta / self.bse
        self.pvalues = pvalue_z(self.zstat)
        self.ar_tests = [self._ar_test(m) for m in (1, 2, 3)]
        sel = [i for i, k in enumerate(self.kinds) if k in ("ylag", "x")]
        b = self.theta[sel]
        self.wald = float(b @ safe_inv(self.cov[np.ix_(sel, sel)]) @ b)
        self.wald_df = len(sel)

    def _ar_test(self, order):
        B = self.blocks
        ZX = sum(Z.T @ X for (_, y, X, Z, m) in B)
        num = 0.0
        t1 = 0.0
        wX = np.zeros(len(self.names))
        Zuw = np.zeros(B[0][3].shape[1])
        for (_, y, X, Z, m), u in zip(B, self.u):
            dpos = [k for k, mm in enumerate(m) if mm[0] == "d"]
            tmap = {m[k][1]: k for k in dpos}
            w = np.zeros(len(u))
            for k in dpos:
                tl = m[k][1] - order
                if tl in tmap:
                    w[k] = u[tmap[tl]]
            wd = np.zeros(len(u))
            wd[dpos] = w[dpos]
            ud = np.zeros(len(u))
            ud[dpos] = u[dpos]
            num += float(wd @ ud)
            t1 += float(wd @ ud) ** 2
            wX += wd @ X
            Zuw += (Z.T @ u) * float(wd @ ud)
        A = self.Minv @ ZX.T @ self.W
        var = t1 - 2 * float(wX @ A @ Zuw) + float(wX @ self.cov @ wX)
        if var <= 0:
            return (np.nan, np.nan)
        z = num / math.sqrt(var)
        return (z, float(2 * stats.norm.sf(abs(z))))

    def report(self) -> Report:
        r = Report(f"Equation: {self.name.upper()}" if self.name else "Dynamic Panel GMM")
        kind = "System GMM (Blundell-Bond)" if self.system else "Difference GMM (Arellano-Bond)"
        hdr = [f"Dependent Variable: {self.yname}", f"Method: Panel Generalized Method of Moments — {kind}",
               f"Transformation: First Differences{' + levels' if self.system else ''}",
               f"Cross-sections included: {self.N}",
               f"Observations: {self.nobs_d} (differences)" + (f" + {self.nobs_l} (levels)" if self.system else ""),
               f"{'Two-step' if self.twostep else 'One-step'} estimates; "
               + ("Windmeijer (2005) finite-sample corrected standard errors" if self.twostep else
                  "robust (cluster) standard errors"),
               f"Number of instruments: {self.nz}" + (" (collapsed)" if self.collapse else "")
               + (f"; GMM lags limited to {self.maxlag}" if self.maxlag < 99 else "")]
        r.header(hdr, stamp=True)
        r.coef_table(self.names, self.theta, self.bse, self.zstat, self.pvalues, stat_label="z-Statistic")
        ar = self.ar_tests
        r.stats([("Hansen J-statistic", self.hansen), ("Prob(J-statistic)", stats.chi2.sf(self.hansen, self.df_over)),
                 ("Sargan statistic (one-step)", self.sargan),
                 ("Prob(Sargan)", stats.chi2.sf(self.sargan, self.df_over)),
                 ("Wald chi2 (regressors)", self.wald), ("Prob(Wald)", stats.chi2.sf(self.wald, self.wald_df))],
                [("Instrument rank", self.nz), ("Overid. d.f.", self.df_over),
                 ("AR(1) z (Arellano-Bond)", ar[0][0]), ("Prob AR(1)", ar[0][1]),
                 ("AR(2) z (Arellano-Bond)", ar[1][0]), ("Prob AR(2)", ar[1][1])])
        r.note("AR(m): H0 no m-th order serial correlation in differenced residuals (AR(1) rejection expected; "
               "AR(2) should not reject).  Hansen/Sargan: H0 instruments are valid.")
        if self.nz > self.N:
            r.warn(f"Instrument count ({self.nz}) exceeds the number of groups ({self.N}); consider collapse=1 or "
                   "maxlag=.")
        return r

    def stat(self, key):
        d = {"hansen": self.hansen, "sargan": self.sargan, "ninst": self.nz, "ngroups": self.N}
        if key in d:
            return d[key]
        if key == "coefs":
            return lambda i=None: self.theta if i is None else float(self.theta[int(i) - 1])
        raise ModelError(f"@{key} not available")

    def views(self):
        return [View("output", "Estimation Output", self.report),
                View("artest", "Arellano-Bond Serial Correlation Tests", self.view_ar),
                View("onestep", "One-step estimates", self.view_onestep),
                View("wald", "Wald Test - Coefficient Restrictions", lambda restrictions="c(1)=0":
                     wald_from_restrictions(restrictions, self.theta, self.cov),
                     [Param("restrictions", "Restrictions", "text", "c(1)=0")])]

    def view_ar(self):
        r = Report("Arellano-Bond Serial Correlation Test")
        r.table(["Test order", "z-statistic", "Prob."],
                [[f"AR({m + 1})", z, p] for m, (z, p) in enumerate(self.ar_tests)], ["str", "num", "prob"])
        return r

    def view_onestep(self):
        se = np.sqrt(np.clip(np.diag(self.V1r), 0, None))
        r = Report("One-step GMM estimates (robust)")
        r.coef_table(self.names, self.th1, se, self.th1 / se, pvalue_z(self.th1 / se), stat_label="z-Statistic")
        return r


def _tlab(t):
    try:
        return str(pd.Timestamp(t).year)
    except Exception:
        return str(t)


# ============================================================================
# panel unit-root tests
# ============================================================================

def _panel_matrix(wf, ev, expr) -> list[np.ndarray]:
    s = ev.eval(expr, ignore_sample=True, as_series=True)
    s = s[wf.sample]
    out = []
    for _, g in s.groupby(level=0, sort=False):
        v = g.dropna().values.astype(float)
        if len(v) >= 6:
            out.append(v)
    return out


def _adf_reg(v, p, trend):
    dv = np.diff(v)
    T = len(dv) - p
    if T <= p + 3:
        return None
    Y = dv[p:]
    cols = [v[p:-1]]
    for j in range(1, p + 1):
        cols.append(dv[p - j:-j])
    if trend in ("c", "ct"):
        cols.append(np.ones(T))
    if trend == "ct":
        cols.append(np.arange(T, dtype=float))
    X = np.column_stack(cols)
    b, *_ = np.linalg.lstsq(X, Y, rcond=None)
    e = Y - X @ b
    s2 = e @ e / (T - X.shape[1])
    se = math.sqrt(s2 * safe_inv(X.T @ X)[0, 0])
    return b[0] / se, b[0], se, e, X, Y, s2


def _select_lag(v, maxp, trend):
    best, bp = None, 0
    for p in range(0, maxp + 1):
        r = _adf_reg(v[maxp - p:] if p < maxp else v, p, trend)
        if r is None:
            continue
        e = r[3]
        n = len(e)
        k = r[4].shape[1]
        sic = math.log(e @ e / n) + k * math.log(n) / n
        if best is None or sic < best:
            best, bp = sic, p
    return bp


@lru_cache(maxsize=256)
def _ips_moments(T: int, p: int, trend: str, reps: int = 2000):
    rng = np.random.default_rng(12345)
    ts = []
    for _ in range(reps):
        v = np.cumsum(rng.standard_normal(T))
        r = _adf_reg(v, p, trend)
        if r is not None:
            ts.append(r[0])
    ts = np.array(ts)
    return float(ts.mean()), float(ts.var())


def panel_unit_root(wf, ev, expr: str, trend: str = "c", maxlag: int | None = None, lags: int | None = None,
                    tests: str = "llc ips adf pp hadri") -> Report:
    from arch.unitroot import ADF, PhillipsPerron
    _require_panel(wf)
    trend = {"none": "n", "n": "n", "c": "c", "const": "c", "ct": "ct", "trend": "ct"}.get(trend, "c")
    units = _panel_matrix(wf, ev, expr)
    N = len(units)
    if N < 2:
        raise ModelError("Need at least two cross-sections with 6+ observations")
    Tmin = min(len(v) for v in units)
    maxp = maxlag if maxlag is not None else max(0, min(int(12 * (Tmin / 100) ** 0.25), Tmin // 4 - 1))
    pl = [lags if lags is not None else _select_lag(v, maxp, trend) for v in units]
    tests = tests.split()
    rows = []
    r = Report(f"Panel unit root test: Summary — {expr.upper()}")
    r.header([f"Series: {expr.upper()}", f"Exogenous variables: {'Individual effects' if trend == 'c' else 'Individual effects, individual linear trends' if trend == 'ct' else 'None'}",
              f"Automatic selection of maximum lags ({maxp}) by SIC" if lags is None else f"User-specified lags: {lags}",
              f"Cross-sections: {N};  total observations: {sum(len(v) for v in units)}"], stamp=True)
    # IPS
    tstats = []
    for v, p in zip(units, pl):
        rr = _adf_reg(v, p, trend if trend != "n" else "c")
        tstats.append(rr[0] if rr else np.nan)
    tstats = np.array(tstats)
    if "ips" in tests and trend != "n":
        Es, Vs = zip(*[_ips_moments(len(v), p, trend) for v, p in zip(units, pl)])
        W = math.sqrt(N) * (np.nanmean(tstats) - np.mean(Es)) / math.sqrt(np.mean(Vs))
        rows.append(["Im, Pesaran and Shin W-stat", W, stats.norm.cdf(W), N, "Unit root (individual)"])
    # LLC
    if "llc" in tests:
        z, p_ = _llc(units, pl, trend)
        rows.append(["Levin, Lin & Chu t*", z, p_, N, "Unit root (common)"])
    # Fisher
    if "adf" in tests:
        pv = []
        for v, p in zip(units, pl):
            try:
                pv.append(float(ADF(v, lags=p, trend=trend).pvalue))
            except Exception:
                pass
        pv = np.clip(np.array(pv), 1e-12, 1 - 1e-12)
        P = -2 * np.log(pv).sum()
        Z = stats.norm.ppf(pv).sum() / math.sqrt(len(pv))
        rows.append(["ADF - Fisher Chi-square", P, stats.chi2.sf(P, 2 * len(pv)), len(pv), "Unit root (individual)"])
        rows.append(["ADF - Choi Z-stat", Z, stats.norm.cdf(Z), len(pv), "Unit root (individual)"])
    if "pp" in tests:
        pv = []
        for v in units:
            try:
                pv.append(float(PhillipsPerron(v, trend=trend if trend != "n" else "n").pvalue))
            except Exception:
                pass
        pv = np.clip(np.array(pv), 1e-12, 1 - 1e-12)
        P = -2 * np.log(pv).sum()
        Z = stats.norm.ppf(pv).sum() / math.sqrt(len(pv))
        rows.append(["PP - Fisher Chi-square", P, stats.chi2.sf(P, 2 * len(pv)), len(pv), "Unit root (individual)"])
        rows.append(["PP - Choi Z-stat", Z, stats.norm.cdf(Z), len(pv), "Unit root (individual)"])
    if "hadri" in tests and trend != "n":
        Z = _hadri(units, trend)
        rows.append(["Hadri Z-stat (heteroskedastic)", Z, stats.norm.sf(Z), N, "Stationarity (common)"])
    r.table(["Method", "Statistic", "Prob.**", "Cross-sections", "Null"], rows, ["str", "num", "prob", "int", "str"])
    r.note("** Probabilities for Fisher tests use an asymptotic Chi-square distribution; all other tests assume "
           "asymptotic normality.  IPS moments E[t], Var[t] are obtained by Monte Carlo for each (T, p);  the LLC "
           "statistic is standardised with its Monte Carlo null distribution.")
    return r


def _llc(units, pl, trend, reps: int = 300):
    def stat(units_):
        es, vs = [], []
        sig_ratio = []
        for v, p in zip(units_, pl):
            dv = np.diff(v)
            T = len(dv) - p
            if T <= p + 3:
                continue
            Y = dv[p:]
            lag = v[p:-1]
            cols = [dv[p - j:-j] for j in range(1, p + 1)]
            if trend in ("c", "ct"):
                cols.append(np.ones(T))
            if trend == "ct":
                cols.append(np.arange(T, dtype=float))
            Xd = np.column_stack(cols) if cols else np.zeros((T, 0))
            if Xd.shape[1]:
                e = Y - Xd @ np.linalg.lstsq(Xd, Y, rcond=None)[0]
                vv = lag - Xd @ np.linalg.lstsq(Xd, lag, rcond=None)[0]
            else:
                e, vv = Y, lag
            b = float(vv @ e / (vv @ vv))
            s = math.sqrt(float(((e - b * vv) ** 2).sum()) / max(T - p - 1, 1))
            es.append(e / s)
            vs.append(vv / s)
            lr = np.var(dv)
            sig_ratio.append(math.sqrt(lr) / s)
        E, V = np.concatenate(es), np.concatenate(vs)
        delta = float(V @ E / (V @ V))
        e2 = E - delta * V
        s2 = float(e2 @ e2) / (len(E) - len(es))
        return delta / math.sqrt(s2 / float(V @ V))

    t_obs = stat(units)
    rng = np.random.default_rng(2024)
    sims = []
    for _ in range(reps):
        sims.append(stat([np.cumsum(rng.standard_normal(len(v))) for v in units]))
    sims = np.array(sims)
    z = (t_obs - sims.mean()) / sims.std()
    return z, float(stats.norm.cdf(z))


def _hadri(units, trend):
    lms = []
    for v in units:
        T = len(v)
        X = np.ones((T, 1)) if trend == "c" else np.column_stack([np.ones(T), np.arange(T)])
        e = v - X @ np.linalg.lstsq(X, v, rcond=None)[0]
        S = np.cumsum(e)
        s2 = e @ e / T
        lms.append(float((S ** 2).sum() / (T ** 2 * s2)))
    N = len(lms)
    xi, zeta2 = (1 / 6, 1 / 45) if trend == "c" else (1 / 15, 11 / 6300)
    return math.sqrt(N) * (np.mean(lms) - xi) / math.sqrt(zeta2)


def dumitrescu_hurlin(wf, ev, names: list[str], lags: int = 2) -> Report:
    _require_panel(wf)
    K = int(lags)
    df = pd.DataFrame({n: ev.eval(n, ignore_sample=True, as_series=True) for n in names}, index=wf.data.index)
    df = df[wf.sample]
    rows = []
    for a in names:
        for b in names:
            if a == b:
                continue
            Ws, Es, Vs = [], [], []
            for _, g in df[[a, b]].groupby(level=0, sort=False):
                g = g.dropna()
                y, x = g[b].values, g[a].values
                T = len(y) - K
                if T <= 2 * K + 2:
                    continue
                Y = y[K:]
                ylags = np.column_stack([y[K - l:-l] for l in range(1, K + 1)])
                xlags = np.column_stack([x[K - l:-l] for l in range(1, K + 1)])
                Xr = np.column_stack([np.ones(T), ylags])
                Xu = np.column_stack([Xr, xlags])
                er = Y - Xr @ np.linalg.lstsq(Xr, Y, rcond=None)[0]
                eu = Y - Xu @ np.linalg.lstsq(Xu, Y, rcond=None)[0]
                W = (er @ er - eu @ eu) / (eu @ eu / (T - 2 * K - 1))
                Ws.append(W)
                if T - 2 * K - 5 > 0:
                    Es.append(K * (T - 2 * K - 1) / (T - 2 * K - 3))
                    Vs.append(2 * K * (T - 2 * K - 1) ** 2 * (T - K - 3) / ((T - 2 * K - 3) ** 2 * (T - 2 * K - 5)))
            if not Ws:
                continue
            N = len(Ws)
            Wbar = float(np.mean(Ws))
            Zbar = math.sqrt(N / (2 * K)) * (Wbar - K)
            if len(Es) == N:
                Zt = math.sqrt(N) * (Wbar - np.mean(Es)) / math.sqrt(np.mean(Vs))
                pt = 2 * stats.norm.sf(abs(Zt))
            else:
                Zt = pt = np.nan
            rows.append([f"{a.upper()} does not homogeneously cause {b.upper()}", Wbar, Zbar, 2 * stats.norm.sf(abs(Zbar)),
                         Zt, pt])
    r = Report("Pairwise Dumitrescu Hurlin Panel Causality Tests")
    r.header([f"Sample: {wf.sample_label()}", f"Lags: {K}"])
    r.table(["Null Hypothesis:", "W-Stat.", "Zbar-Stat.", "Prob.", "Ztilde", "Prob."], rows,
            ["str", "num", "num", "prob", "num", "prob"])
    r.note("Ztilde (finite-T correction) requires T > 2K + 5 time periods per cross-section.")
    return r


def panel_describe(wf, ev, names: list[str]) -> Report:
    _require_panel(wf)
    r = Report("Panel Summary Statistics (overall / between / within)")
    rows = []
    for n in names:
        s = ev.eval(n, ignore_sample=True, as_series=True)[wf.sample].dropna()
        m = s.groupby(level=0).transform("mean")
        within = s - m + s.mean()
        between = s.groupby(level=0).mean()
        rows.append([n.upper(), "overall", s.mean(), s.std(), s.min(), s.max(), len(s)])
        rows.append(["", "between", "", between.std(), between.min(), between.max(), len(between)])
        rows.append(["", "within", "", within.std(), within.min(), within.max(),
                     f"{len(s) / max(len(between), 1):.1f}"])
    r.table(["Variable", "", "Mean", "Std. Dev.", "Min", "Max", "Obs (N / T-bar)"], rows,
            ["str", "str", "num", "num", "num", "num", "str"])
    return r
