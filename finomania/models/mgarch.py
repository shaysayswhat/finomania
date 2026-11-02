"""
Multivariate GARCH: CCC (Bollerslev 1990), DCC (Engle 2002), asymmetric DCC
(Cappiello, Engle & Sheppard 2006) with Gaussian or multivariate-t second step,
and diagonal / scalar BEKK(1,1) with variance targeting.

DCC/CCC are estimated by two-step QML: univariate ARCH models (any
specification supported by Finomania) followed by the correlation dynamics.
Second-step standard errors are conditional on the first step.
"""
from __future__ import annotations

import itertools
import math
import warnings

import numpy as np
import pandas as pd
from scipy import optimize, special, stats
from scipy.signal import lfilter

from ..core.workfile import sanitize_name
from ..report import FigureOutput, Report, fmt_num
from . import plots
from .base import (ModelError, ModelResult, Param, View, contiguous_check, info_criteria, numeric_hessian,
                   pvalue_z, safe_inv)
from .garch import GarchResult, norm_dist, norm_model

LOG2PI = math.log(2 * math.pi)


def _corr_from_q(Q: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.einsum("tii->ti", Q))
    return Q / (d[:, :, None] * d[:, None, :])


def _filter_q(Z: np.ndarray, a: float, b: float, g: float = 0.0, Qbar=None, Nbar=None) -> np.ndarray:
    T, k = Z.shape
    Qbar = Z.T @ Z / T if Qbar is None else Qbar
    S = np.einsum("ti,tj->tij", Z, Z).reshape(T, k * k)
    x = np.empty((T, k * k))
    const = (1 - a - b) * Qbar
    if g:
        Nz = Z * (Z < 0)
        Nbar = Nz.T @ Nz / T if Nbar is None else Nbar
        const = const - g * Nbar
        SN = np.einsum("ti,tj->tij", Nz, Nz).reshape(T, k * k)
    x[0] = Qbar.ravel()
    x[1:] = const.ravel() + a * S[:-1]
    if g:
        x[1:] += g * SN[:-1]
    Q = np.empty_like(x)
    Q[0] = Qbar.ravel()
    if T > 1:
        Q[1:] = lfilter([1.0], [1.0, -b], x[1:], axis=0, zi=(b * Qbar.ravel())[None, :])[0]
    return Q.reshape(T, k, k)


def _corr_loglik_terms(Z, R, dist="normal", nu=None):
    sign, logdet = np.linalg.slogdet(R)
    if np.any(sign <= 0):
        return None
    sol = np.linalg.solve(R, Z[:, :, None])[:, :, 0]
    quad = np.einsum("ti,ti->t", Z, sol)
    k = Z.shape[1]
    if dist == "t":
        c = special.gammaln((nu + k) / 2) - special.gammaln(nu / 2) - k / 2 * math.log(math.pi * (nu - 2))
        return c - 0.5 * logdet - (nu + k) / 2 * np.log1p(quad / (nu - 2))
    return -0.5 * (logdet + quad - np.einsum("ti,ti->t", Z, Z))


class MGarchBase(ModelResult):
    object_type = "mgarch"

    def _setup(self, wf, ev, names: list[str]):
        if len(names) < 2:
            raise ModelError("Multivariate GARCH needs at least two series")
        self.wf, self.ev = wf, ev
        self.series_names = [n.lower() for n in names]
        df = pd.DataFrame({n: ev.eval(n, ignore_sample=True, as_series=True) for n in self.series_names},
                          index=wf.data.index)
        mask = wf.sample & df.notna().all(axis=1).values
        if not contiguous_check(mask):
            raise ModelError("Multivariate GARCH requires a contiguous common sample")
        self.base_mask = mask
        self.df = df
        self.k = len(names)

    # ---- shared helpers
    def _pairs(self):
        return list(itertools.combinations(range(self.k), 2))

    def _idx(self):
        return self.wf.data.index[self.mask]

    def stat(self, key):
        d = getattr(self, "stats_dict", {})
        if key in d:
            return d[key]
        raise ModelError(f"@{key} not available")

    # conditional covariance H_t (T,k,k)
    def H(self) -> np.ndarray:
        raise NotImplementedError

    def corr(self) -> np.ndarray:
        H = self.H()
        d = np.sqrt(np.einsum("tii->ti", H))
        return H / (d[:, :, None] * d[:, None, :])

    def views_common(self):
        return [
            View("output", "Estimation Output", self.report),
            View("corrgraph", "Conditional Correlations Graph", self.view_corr),
            View("covgraph", "Conditional Covariance / Variance Graph", self.view_cov),
            View("corrtable", "Conditional Correlation Summary", self.view_corr_summary),
            View("hedge", "Hedge Ratios & Min-Variance Weights", self.view_hedge),
        ]

    def procs_common(self):
        return [
            View("makecorr", "Make Conditional Correlation Series", self.proc_makecorr,
                 [Param("prefix", "Prefix", "text", "corr_")], kind="proc"),
            View("makecov", "Make Conditional (Co)variance Series", self.proc_makecov,
                 [Param("prefix", "Prefix", "text", "cov_")], kind="proc"),
            View("makeweights", "Make Min-Variance Portfolio Weights", self.proc_makeweights,
                 [Param("prefix", "Prefix", "text", "w_")], kind="proc"),
        ]

    def view_corr(self):
        R = self.corr()
        idx = self._idx()
        data = {f"{self.series_names[i].upper()}–{self.series_names[j].upper()}": pd.Series(R[:, i, j], index=idx)
                for i, j in self._pairs()}
        fig = plots.line_plot(data, "Conditional correlations", multiple=len(data) > 3)
        return FigureOutput(fig, "Conditional correlations")

    def view_cov(self):
        H = self.H()
        idx = self._idx()
        data = {}
        for i in range(self.k):
            data[f"Var({self.series_names[i].upper()})"] = pd.Series(H[:, i, i], index=idx)
        for i, j in self._pairs():
            data[f"Cov({self.series_names[i].upper()},{self.series_names[j].upper()})"] = pd.Series(H[:, i, j],
                                                                                              index=idx)
        return FigureOutput(plots.line_plot(data, "Conditional covariances", multiple=True), "Conditional covariance")

    def view_corr_summary(self):
        R = self.corr()
        rows = []
        for i, j in self._pairs():
            v = R[:, i, j]
            rows.append([f"{self.series_names[i].upper()}–{self.series_names[j].upper()}", v.mean(), np.median(v),
                         v.min(), v.max(), v.std(), v[-1]])
        r = Report("Conditional Correlation Summary")
        r.table(["Pair", "Mean", "Median", "Min", "Max", "Std. Dev.", "Last"], rows)
        return r

    def view_hedge(self):
        H = self.H()
        idx = self._idx()
        rows, data = [], {}
        for i, j in self._pairs():
            hr = H[:, i, j] / H[:, j, j]
            data[f"{self.series_names[i].upper()} hedged with {self.series_names[j].upper()}"] = pd.Series(hr, index=idx)
            rows.append([f"long {self.series_names[i].upper()} / short {self.series_names[j].upper()}", hr.mean(),
                         hr.std(), hr.min(), hr.max(), hr[-1]])
        ones = np.ones(self.k)
        Hinv = np.linalg.inv(H)
        w = Hinv @ ones
        w = w / w.sum(axis=1, keepdims=True)
        r = Report("Dynamic Hedge Ratios (beta = H_ij / H_jj)")
        r.table(["Hedge", "Mean", "Std. Dev.", "Min", "Max", "Last"], rows)
        r.table(["Asset", "Mean weight", "Std. Dev.", "Last"],
                [[self.series_names[i].upper(), w[:, i].mean(), w[:, i].std(), w[-1, i]] for i in range(self.k)],
                title="Global minimum-variance portfolio weights")
        fig = plots.line_plot(data, "Dynamic hedge ratios", multiple=len(data) > 3)
        return [r, FigureOutput(fig, "Hedge ratios")]

    def _save(self, name, values):
        s = pd.Series(np.nan, index=self.wf.data.index)
        m = self.mask
        s.iloc[np.flatnonzero(m)] = values
        self.wf.set_series(name, s, respect_sample=False)

    def proc_makecorr(self, prefix="corr_"):
        R = self.corr()
        made = []
        for i, j in self._pairs():
            nm = f"{prefix}{sanitize_name(self.series_names[i])}_{sanitize_name(self.series_names[j])}"
            self._save(nm, R[:, i, j])
            made.append(nm.upper())
        return "Created " + ", ".join(made)

    def proc_makecov(self, prefix="cov_"):
        H = self.H()
        made = []
        for i in range(self.k):
            nm = f"var_{sanitize_name(self.series_names[i])}" if prefix == "cov_" else f"{prefix}{sanitize_name(self.series_names[i])}"
            self._save(nm, H[:, i, i])
            made.append(nm.upper())
        for i, j in self._pairs():
            nm = f"{prefix}{sanitize_name(self.series_names[i])}_{sanitize_name(self.series_names[j])}"
            self._save(nm, H[:, i, j])
            made.append(nm.upper())
        return "Created " + ", ".join(made)

    def proc_makeweights(self, prefix="w_"):
        H = self.H()
        w = np.linalg.solve(H, np.ones((len(H), self.k, 1)))[:, :, 0]
        w = w / w.sum(axis=1, keepdims=True)
        for i in range(self.k):
            self._save(f"{prefix}{sanitize_name(self.series_names[i])}", w[:, i])
        return f"Minimum-variance weights saved with prefix {prefix.upper()}"


# ============================================================================
# DCC / ADCC / CCC
# ============================================================================

class DCCResult(MGarchBase):
    def __init__(self, wf, ev, names: list[str], kind: str = "dcc", model: str = "garch", p: int = 1, q: int = 1,
                 o: int = 0, dist: str = "normal", mean: str = "c", mvdist: str = "normal", cov: str = "bw", name=""):
        self._setup(wf, ev, names)
        self.name = name
        self.kind = kind.lower()               # dcc | adcc | ccc
        self.model, self.dist, self.mvdist = norm_model(model), norm_dist(dist), norm_dist(mvdist)
        self.method_label = {"dcc": "DCC-GARCH", "adcc": "Asymmetric DCC-GARCH", "ccc": "CCC-GARCH"}[self.kind]
        mean = (mean or "c").lower()
        mspec = {"c": "c", "const": "c", "constant": "c", "zero": "", "none": "", "ar1": "c ar(1)",
                 "ar(1)": "c ar(1)"}.get(mean, "c")
        # step 1 -- univariate models on the common sample
        saved = wf._sample
        wf._sample = self.base_mask.copy()
        try:
            self.uni = [GarchResult(wf, ev, f"{n} {mspec}".strip(), model=self.model, p=p, q=q, o=o, dist=self.dist,
                                    cov=cov) for n in self.series_names]
        finally:
            wf._sample = saved
        mask = self.uni[0].mask.copy()
        for u in self.uni[1:]:
            mask &= u.mask
        self.mask = mask
        Zs, Hs = [], []
        for u in self.uni:
            sel = np.isin(np.flatnonzero(u.mask), np.flatnonzero(mask))
            Zs.append(u.std_resid()[sel])
            Hs.append(u.h_[sel])
        self.Z = np.column_stack(Zs)
        self.Hd = np.column_stack(Hs)
        T, k = self.Z.shape
        self.T = T
        self.Qbar = self.Z.T @ self.Z / T
        Nz = self.Z * (self.Z < 0)
        self.Nbar = Nz.T @ Nz / T
        self.uni_ll = sum(float(u.llf) for u in self.uni)
        self._fit_step2(cov)

    # ---- step 2
    def _unpack(self, th):
        a = b = g = 0.0
        nu = None
        i = 0
        if self.kind in ("dcc", "adcc"):
            a, b = th[0], th[1]
            i = 2
            if self.kind == "adcc":
                g = th[2]
                i = 3
        if self.mvdist == "t":
            nu = th[i]
        return a, b, g, nu

    def _R(self, a, b, g):
        if self.kind == "ccc":
            D = np.sqrt(np.diag(self.Qbar))
            R0 = self.Qbar / np.outer(D, D)
            return np.broadcast_to(R0, (self.T, self.k, self.k))
        return _corr_from_q(_filter_q(self.Z, a, b, g, self.Qbar, self.Nbar))

    def _terms(self, th):
        a, b, g, nu = self._unpack(th)
        R = self._R(a, b, g)
        return _corr_loglik_terms(self.Z, R, "t" if self.mvdist == "t" else "normal", nu)

    def _negll(self, th):
        lt = self._terms(th)
        if lt is None or not np.all(np.isfinite(lt)):
            return 1e10
        return -float(lt.sum()) / self.T

    def _fit_step2(self, cov):
        names, x0s, bnds, cons = [], [], [], []
        if self.kind in ("dcc", "adcc"):
            names += ["DCC alpha (a)", "DCC beta (b)"]
            bnds += [(0.0, 0.5), (0.0, 0.9999)]
            if self.kind == "adcc":
                names.append("ADCC gamma (g)")
                bnds.append((0.0, 0.5))
                Qm = np.linalg.inv(np.linalg.cholesky(self.Qbar))
                delta = float(np.max(np.linalg.eigvalsh(Qm @ self.Nbar @ Qm.T)))
                cons.append({"type": "ineq", "fun": lambda th: 0.9999 - th[0] - th[1] - delta * th[2]})
                self.delta = delta
            else:
                cons.append({"type": "ineq", "fun": lambda th: 0.9999 - th[0] - th[1]})
        if self.mvdist == "t":
            names.append("T-DIST. DOF (multivariate)")
            bnds.append((2.1, 300.0))
        self.p2names = names
        if not names:
            self.p2 = np.zeros(0)
            self.p2se = np.zeros(0)
            self.p2cov = np.zeros((0, 0))
            lt = self._terms(np.zeros(0))
            self.ll2 = float(lt.sum())
            return
        starts = []
        for a, b in ((0.02, 0.95), (0.05, 0.90), (0.01, 0.98), (0.1, 0.8)):
            th = []
            if self.kind in ("dcc", "adcc"):
                th += [a, b]
                if self.kind == "adcc":
                    th.append(0.01)
            if self.mvdist == "t":
                th.append(8.0)
            starts.append(np.array(th))
        best = min(starts, key=self._negll)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = optimize.minimize(self._negll, best, method="SLSQP", bounds=bnds, constraints=cons,
                                    options={"maxiter": 500, "ftol": 1e-12})
        th = res.x
        self.p2 = th
        self.converged = bool(res.success)
        lt = self._terms(th)
        self.ll2 = float(lt.sum())
        H = numeric_hessian(lambda x: -1e10 if self._terms(x) is None else float(self._terms(x).sum()), th, rel=1e-4)
        G = np.zeros((self.T, len(th)))
        for i in range(len(th)):
            h = 1e-5 * max(abs(th[i]), 1e-3)
            e = np.zeros(len(th))
            e[i] = h
            lp, lm = self._terms(th + e), self._terms(th - e)
            if lp is not None and lm is not None:
                G[:, i] = (lp - lm) / (2 * h)
        Hinv = safe_inv(-H)
        V = Hinv @ (G.T @ G) @ Hinv if cov == "bw" else Hinv
        self.p2cov = (V + V.T) / 2
        self.p2se = np.sqrt(np.clip(np.diag(self.p2cov), 0, None))

    # ---- likelihood / covariance
    def total_loglik(self) -> float:
        if self.mvdist == "t":
            # joint density of returns: mvt(z; R, nu) with Jacobian of z = D^-1 e
            return self.ll2 - 0.5 * float(np.log(self.Hd).sum())
        return self.uni_ll + self.ll2

    def H(self):
        a, b, g, _ = self._unpack(self.p2) if len(self.p2) else (0, 0, 0, None)
        R = np.array(self._R(a, b, g))
        D = np.sqrt(self.Hd)
        return R * D[:, :, None] * D[:, None, :]

    def corr(self):
        a, b, g, _ = self._unpack(self.p2) if len(self.p2) else (0, 0, 0, None)
        return np.array(self._R(a, b, g))

    def report(self) -> Report:
        r = Report(f"System: {self.name.upper()}" if self.name else self.method_label)
        u0 = self.uni[0]
        uni_lab = u0.report().blocks[0].lines
        modelline = next((l for l in uni_lab if l.startswith("Model:")), "")
        r.header([f"Method: {self.method_label} — two-step quasi-maximum likelihood",
                  f"Series: {' '.join(n.upper() for n in self.series_names)}",
                  f"Sample: {self.wf.sample_label(self.mask)}", f"Included observations: {self.T}",
                  f"Univariate step: {modelline.replace('Model: ', '')}, {u0.method_label.replace('ML ARCH - ', '')}",
                  f"Correlation step: {'multivariate Student t' if self.mvdist == 't' else 'Gaussian'} "
                  f"({'Engle (2002)' if self.kind == 'dcc' else 'Cappiello-Engle-Sheppard (2006)' if self.kind == 'adcc' else 'Bollerslev (1990)'})",
                  "Q(t) = (1-a-b)Qbar + a z(t-1)z(t-1)' + b Q(t-1)" + (" + g[n n' - Nbar]" if self.kind == "adcc" else "")
                  if self.kind != "ccc" else "R(t) = R (constant)"], stamp=True)
        names, coef, se = [], [], []
        breaks = {}
        for u in self.uni:
            breaks[len(names)] = f"Univariate: {u.yname}"
            nm = getattr(u.fit, "n_mean", 0)
            names += [("C (variance)" if (i >= nm and n == "C") else n) for i, n in enumerate(u.names)]
            coef += list(u.params)
            se += list(u.bse)
        if len(self.p2):
            breaks[len(names)] = "Correlation dynamics"
            names += self.p2names
            coef += list(self.p2)
            se += list(self.p2se)
        coef, se = np.array(coef), np.array(se)
        z = coef / np.where(se > 0, se, np.nan)
        r.coef_table(names, coef, se, z, pvalue_z(z), stat_label="z-Statistic", section_breaks=breaks)
        ll = self.total_loglik()
        npar = len(coef)
        aic, sc, hq = info_criteria(ll, npar, self.T)
        left = [("Log likelihood (total)", ll), ("Log likelihood (correlation step)", self.ll2),
                ("Akaike info criterion", aic), ("Schwarz criterion", sc)]
        right = [("Hannan-Quinn criter.", hq), ("Number of parameters", npar)]
        if self.kind in ("dcc", "adcc"):
            pers = self.p2[0] + self.p2[1] + (self.delta * self.p2[2] if self.kind == "adcc" else 0)
            right.append(("Correlation persistence", pers))
        r.stats(left, right)
        Dq = np.sqrt(np.diag(self.Qbar))
        Rb = self.Qbar / np.outer(Dq, Dq)
        labs = [n.upper() for n in self.series_names]
        r.table([""] + labs, [[labs[i]] + list(Rb[i]) for i in range(self.k)],
                title="Unconditional correlation of standardized residuals (Qbar)")
        r.note("Correlation-step standard errors are conditional on the first-step estimates "
               "(Bollerslev-Wooldridge sandwich).")
        self.stats_dict = {"logl": ll, "aic": aic, "sc": sc, "hq": hq}
        return r

    def views(self):
        v = self.views_common()
        v.append(View("cctest", "Test of Constant Correlation (Engle-Sheppard)", self.view_cctest,
                      [Param("lags", "Lags", "int", 5)]))
        if self.kind != "ccc":
            v.append(View("lrccc", "LR Test: DCC vs CCC", self.view_lr_ccc))
        v.append(View("univariate", "Univariate Models Output", self.view_univariate))
        return v

    def procs(self):
        p = self.procs_common()
        if self.kind != "ccc":
            p.append(View("forecast", "Forecast Correlations...", self.proc_forecast,
                          [Param("h", "Horizon", "int", 20)], kind="proc"))
        return p

    def view_univariate(self):
        r = Report("Univariate first-step models")
        for u in self.uni:
            r.extend(u.report())
        return r

    def view_cctest(self, lags=5):
        s = int(lags)
        Dq = np.sqrt(np.diag(self.Qbar))
        Rc = self.Qbar / np.outer(Dq, Dq)
        w, V = np.linalg.eigh(Rc)
        Rm12 = V @ np.diag(w ** -0.5) @ V.T
        zh = self.Z @ Rm12.T
        iu = np.triu_indices(self.k, 1)
        Y = np.einsum("ti,tj->tij", zh, zh)[:, iu[0], iu[1]]  # T x m
        T, m = Y.shape
        yv = Y[s:].ravel()
        Xs = [np.ones(len(yv))]
        for l in range(1, s + 1):
            Xs.append(Y[s - l:T - l].ravel())
        X = np.column_stack(Xs)
        b = np.linalg.lstsq(X, yv, rcond=None)[0]
        e = yv - X @ b
        s2 = e @ e / (len(yv) - X.shape[1])
        stat = float(b @ X.T @ X @ b / s2)
        df = s + 1
        r = Report("Engle-Sheppard Test of Constant Conditional Correlation")
        r.text("Null hypothesis: correlations are constant (CCC)")
        r.stats([("Chi-square statistic", stat), ("Lags", s)], [("Degrees of freedom", df),
                                                             ("Probability", float(stats.chi2.sf(stat, df)))])
        return r

    def view_lr_ccc(self):
        R0 = np.broadcast_to(self.Qbar / np.outer(np.sqrt(np.diag(self.Qbar)), np.sqrt(np.diag(self.Qbar))),
                             (self.T, self.k, self.k))
        if self.mvdist == "t":
            nu = self.p2[-1]
            l0 = float(_corr_loglik_terms(self.Z, R0, "t", nu).sum())
        else:
            l0 = float(_corr_loglik_terms(self.Z, R0).sum())
        LR = 2 * (self.ll2 - l0)
        df = 2 if self.kind == "dcc" else 3
        r = Report("Likelihood Ratio Test: dynamic vs constant correlation")
        r.stats([("LR statistic", LR), ("Restricted LogL (CCC)", l0)],
                [("df", df), ("Probability (chi-square)", float(stats.chi2.sf(LR, df)))])
        r.note("Under H0 the DCC parameters lie on the boundary; the chi-square p-value is conservative.")
        return r

    def proc_forecast(self, h=20):
        h = int(h)
        a, b, g, _ = self._unpack(self.p2)
        Q = _filter_q(self.Z, a, b, g, self.Qbar, self.Nbar)
        zT = self.Z[-1]
        Q1 = (1 - a - b) * self.Qbar - g * self.Nbar + a * np.outer(zT, zT) + b * Q[-1]
        if g:
            nT = zT * (zT < 0)
            Q1 += g * np.outer(nT, nT)
        d1 = np.sqrt(np.diag(Q1))
        R1 = Q1 / np.outer(d1, d1)
        Dq = np.sqrt(np.diag(self.Qbar))
        Rbar = self.Qbar / np.outer(Dq, Dq)
        pers = a + b + (self.delta * g if self.kind == "adcc" else 0)
        rows = []
        Rf = []
        for s in range(1, h + 1):
            Rs = (1 - pers ** (s - 1)) * Rbar + pers ** (s - 1) * R1
            Rf.append(Rs)
            rows.append([s] + [Rs[i, j] for i, j in self._pairs()])
        r = Report("DCC correlation forecasts (Engle-Sheppard approximation)")
        r.table(["Step"] + [f"{self.series_names[i].upper()}–{self.series_names[j].upper()}" for i, j in self._pairs()],
                rows, ["int"] + ["num"] * len(self._pairs()))
        Rf = np.array(Rf)
        fig, ax = plots.new_figure()
        for n, (i, j) in enumerate(self._pairs()):
            ax.plot(np.arange(1, h + 1), Rf[:, i, j], color=plots.PALETTE[n % 10],
                    label=f"{self.series_names[i].upper()}–{self.series_names[j].upper()}")
            ax.axhline(Rbar[i, j], color=plots.PALETTE[n % 10], ls=":", lw=0.8)
        ax.legend(frameon=False)
        ax.set_title("Correlation forecasts (dotted: unconditional)")
        ax.set_xlabel("Steps ahead")
        fig.tight_layout()
        return [r, FigureOutput(fig, "DCC forecast")]


# ============================================================================
# BEKK
# ============================================================================

class BEKKResult(MGarchBase):
    method_label = "BEKK-GARCH"

    def __init__(self, wf, ev, names: list[str], kind: str = "diagonal", cov: str = "bw", name=""):
        self._setup(wf, ev, names)
        self.name = name
        self.kind = kind
        self.mask = self.base_mask
        Y = self.df[self.mask].values
        self.mu = Y.mean(0)
        self.E = Y - self.mu
        T, k = self.E.shape
        self.T = T
        self.S = self.E.T @ self.E / T
        if k > 6 and kind == "diagonal":
            warnings.warn("Diagonal BEKK with many series may be slow")
        self._fit(cov)

    def _ab(self, th):
        k = self.k
        if self.kind == "scalar":
            return np.full(k, th[0]), np.full(k, th[1])
        return th[:k], th[k:2 * k]

    def _Hpath(self, th):
        a, b = self._ab(th)
        T, k = self.E.shape
        S = self.S
        C = S - np.outer(a, a) * S - np.outer(b, b) * S
        if np.min(np.linalg.eigvalsh(C)) <= 0:
            return None
        H = np.empty((T, k, k))
        EE = np.einsum("ti,tj->tij", self.E, self.E)
        for i in range(k):
            for j in range(i, k):
                aa, bb = a[i] * a[j], b[i] * b[j]
                x = np.empty(T)
                x[0] = S[i, j]
                x[1:] = C[i, j] + aa * EE[:-1, i, j]
                y = np.empty(T)
                y[0] = S[i, j]
                y[1:] = lfilter([1.0], [1.0, -bb], x[1:], zi=[bb * S[i, j]])[0]
                H[:, i, j] = y
                H[:, j, i] = y
        return H

    def _terms(self, th):
        H = self._Hpath(th)
        if H is None:
            return None
        sign, logdet = np.linalg.slogdet(H)
        if np.any(sign <= 0):
            return None
        sol = np.linalg.solve(H, self.E[:, :, None])[:, :, 0]
        quad = np.einsum("ti,ti->t", self.E, sol)
        return -0.5 * (self.k * LOG2PI + logdet + quad)

    def _negll(self, th):
        lt = self._terms(th)
        if lt is None or not np.all(np.isfinite(lt)):
            return 1e10
        return -float(lt.sum()) / self.T

    def _fit(self, cov):
        k = self.k
        if self.kind == "scalar":
            x0 = np.array([0.25, 0.95])
            bnds = [(0.0, 0.99), (0.0, 0.9999)]
            cons = [{"type": "ineq", "fun": lambda th: 0.9999 - th[0] ** 2 - th[1] ** 2}]
            self.pnames = ["a (scalar)", "b (scalar)"]
        else:
            x0 = np.concatenate([np.full(k, 0.25), np.full(k, 0.95)])
            bnds = [(0.0, 0.99)] * k + [(0.0, 0.9999)] * k
            cons = [{"type": "ineq", "fun": (lambda th, i=i: 0.9999 - th[i] ** 2 - th[k + i] ** 2)} for i in range(k)]
            self.pnames = [f"A({i + 1},{i + 1}) [{n.upper()}]" for i, n in enumerate(self.series_names)] + \
                          [f"B({i + 1},{i + 1}) [{n.upper()}]" for i, n in enumerate(self.series_names)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = optimize.minimize(self._negll, x0, method="SLSQP", bounds=bnds, constraints=cons,
                                    options={"maxiter": 500, "ftol": 1e-12})
        th = res.x
        self.converged = bool(res.success)
        self.theta = th
        lt = self._terms(th)
        self.llf = float(lt.sum())
        Hs = numeric_hessian(lambda x: -1e10 if self._terms(x) is None else float(self._terms(x).sum()), th, rel=1e-4)
        G = np.zeros((self.T, len(th)))
        for i in range(len(th)):
            h = 1e-5 * max(abs(th[i]), 1e-3)
            e = np.zeros(len(th))
            e[i] = h
            lp, lm = self._terms(th + e), self._terms(th - e)
            if lp is not None and lm is not None:
                G[:, i] = (lp - lm) / (2 * h)
        Hinv = safe_inv(-Hs)
        V = Hinv @ (G.T @ G) @ Hinv if cov == "bw" else Hinv
        self.covm = (V + V.T) / 2
        self.bse = np.sqrt(np.clip(np.diag(self.covm), 0, None))
        self._H = self._Hpath(th)

    def H(self):
        return self._H

    def report(self) -> Report:
        r = Report(f"System: {self.name.upper()}" if self.name else "BEKK-GARCH")
        r.header([f"Method: {self.kind.capitalize()} BEKK(1,1) — Gaussian maximum likelihood with variance targeting",
                  f"Series: {' '.join(n.upper() for n in self.series_names)}",
                  f"Sample: {self.wf.sample_label(self.mask)}", f"Included observations: {self.T}",
                  "H(t) = C'C + A e(t-1)e(t-1)' A + B H(t-1) B,   C'C = S - A S A - B S B",
                  "Mean equations: constants (sample means)"], stamp=True)
        z = self.theta / np.where(self.bse > 0, self.bse, np.nan)
        r.coef_table(self.pnames, self.theta, self.bse, z, pvalue_z(z), stat_label="z-Statistic")
        npar = len(self.theta) + self.k + self.k * (self.k + 1) // 2
        aic, sc, hq = info_criteria(self.llf, npar, self.T)
        a, b = self._ab(self.theta)
        r.stats([("Log likelihood", self.llf), ("Akaike info criterion", aic), ("Schwarz criterion", sc)],
                [("Hannan-Quinn criter.", hq), ("Parameters (incl. targeted)", npar),
                 ("Max persistence a²+b²", float(np.max(a ** 2 + b ** 2)))])
        labs = [n.upper() for n in self.series_names]
        r.table([""] + labs, [[labs[i]] + list(self.S[i]) for i in range(self.k)],
                title="Targeted unconditional covariance (S)")
        self.stats_dict = {"logl": self.llf, "aic": aic, "sc": sc, "hq": hq}
        return r

    def views(self):
        return self.views_common()

    def procs(self):
        return self.procs_common()
