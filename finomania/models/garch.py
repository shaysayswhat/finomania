"""
Univariate ARCH family.

Two engines produce a common result object:

* a native maximum-likelihood engine implementing the EViews specifications
  GARCH(p,q), TARCH/GJR, EGARCH, PARCH (APARCH), Component GARCH and IGARCH,
  with ARMA mean terms, ARCH-in-mean (sd / var / log var), exogenous
  variance regressors and Normal / Student-t / GED / skewed-t errors;
* the ``arch`` package for FIGARCH, HARCH and EWMA (RiskMetrics).

Coefficient covariance: Bollerslev-Wooldridge robust (default), outer
product of gradients, or inverse Hessian.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize, special, stats
from scipy.signal import lfilter, lfiltic

from ..report import FigureOutput, Report, fmt_num
from . import plots
from .base import (ModelError, ModelResult, Param, View, build_design, contiguous_check, info_criteria,
                   jarque_bera, numeric_hessian, pvalue_z, safe_inv, sample_header, series_full, split_at,
                   split_terms, wald_from_restrictions)
from .regression import _dw, arch_lm_test

SQ2PI = math.sqrt(2 / math.pi)
LOG2PI = math.log(2 * math.pi)

NATIVE_MODELS = {"garch", "arch", "tarch", "gjr", "egarch", "parch", "aparch", "cgarch", "igarch"}
ARCHPKG_MODELS = {"figarch", "harch", "ewma", "riskmetrics", "zarch", "tgarch"}
DIST_LABEL = {"normal": "Normal distribution", "t": "Student's t distribution",
              "ged": "Generalized error distribution (GED)", "skewt": "Skewed Student's t distribution"}


def norm_model(m: str) -> str:
    m = (m or "garch").lower()
    return {"gjr": "tarch", "gjrgarch": "tarch", "gjr-garch": "tarch", "aparch": "parch", "riskmetrics": "ewma",
            "tgarch": "zarch", "component": "cgarch", "integrated": "igarch"}.get(m, m)


def norm_dist(d: str) -> str:
    d = (d or "normal").lower()
    return {"n": "normal", "norm": "normal", "gaussian": "normal", "student": "t", "tdist": "t", "studentt": "t",
            "skewstudent": "skewt", "skew-t": "skewt", "sst": "skewt", "skt": "skewt", "gev": "ged",
            "generror": "ged"}.get(d, d)


# ============================================================================
# densities (log-likelihood contributions)
# ============================================================================

def loglik_terms(e: np.ndarray, h: np.ndarray, dist: str, dp: np.ndarray) -> np.ndarray:
    if dist == "normal":
        return -0.5 * (LOG2PI + np.log(h) + e * e / h)
    if dist == "t":
        nu = dp[0]
        c = special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2) - 0.5 * math.log(math.pi * (nu - 2))
        return c - 0.5 * np.log(h) - (nu + 1) / 2 * np.log1p(e * e / (h * (nu - 2)))
    if dist == "ged":
        r = dp[0]
        lam = math.sqrt(2 ** (-2 / r) * math.exp(special.gammaln(1 / r) - special.gammaln(3 / r)))
        c = math.log(r) - math.log(lam) - (1 + 1 / r) * math.log(2) - special.gammaln(1 / r)
        z = e / np.sqrt(h)
        return c - 0.5 * np.abs(z / lam) ** r - 0.5 * np.log(h)
    if dist == "skewt":
        nu, lam = dp[0], dp[1]
        c = math.exp(special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2)) / math.sqrt(math.pi * (nu - 2))
        a = 4 * lam * c * (nu - 2) / (nu - 1)
        b = math.sqrt(max(1 + 3 * lam * lam - a * a, 1e-12))
        z = e / np.sqrt(h)
        s = np.where(z < -a / b, 1 - lam, 1 + lam)
        return (math.log(b) + math.log(c) - (nu + 1) / 2 * np.log1p(((b * z + a) / s) ** 2 / (nu - 2))
                - 0.5 * np.log(h))
    raise ModelError(f"Unknown distribution {dist}")


def dist_draw(rng, dist, dp, size):
    if dist == "normal":
        return rng.standard_normal(size)
    if dist == "t":
        nu = dp[0]
        return rng.standard_t(nu, size) * math.sqrt((nu - 2) / nu)
    if dist == "ged":
        r = dp[0]
        g = rng.gamma(1 / r, 1, size)
        x = np.sign(rng.random(size) - 0.5) * g ** (1 / r)
        return x / x.std() if np.size(x) > 50 else x / math.sqrt(math.gamma(3 / r) / math.gamma(1 / r))
    return rng.standard_normal(size)


# ============================================================================
# native engine
# ============================================================================

@dataclass
class GarchSpec:
    model: str = "garch"
    p: int = 1                 # ARCH order
    q: int = 1                 # GARCH order
    o: int = 0                 # asymmetric (threshold) order
    dist: str = "normal"
    archm: str | None = None   # None | sd | var | log
    power: float | None = None  # fixed PARCH power (None = estimated)


class NativeGarch:
    def __init__(self, y: np.ndarray, X: np.ndarray, xnames: list[str], ar: list[int], ma: list[int],
                 Z: np.ndarray | None, znames: list[str], spec: GarchSpec):
        self.y, self.X, self.xnames, self.ar, self.ma = y, X, xnames, list(ar), list(ma)
        self.Z = Z if Z is not None and Z.size else None
        self.znames = znames
        self.spec = spec
        self.T = len(y)
        self.t0 = max(self.ar) if self.ar else 0
        if self.t0 >= self.T - 10:
            raise ModelError("Too few observations for the ARCH model")
        sp = spec
        if sp.model == "cgarch" and (sp.p != 1 or sp.q != 1):
            sp.p, sp.q = 1, 1
        if sp.model == "igarch" and sp.q < 1:
            raise ModelError("IGARCH requires at least one GARCH term")
        if sp.model in ("tarch", "egarch", "parch") and sp.o == 0:
            sp.o = 1
        if sp.model == "parch":
            sp.o = min(sp.o, sp.p)
        self._layout()

    # ------------------------------------------------------------ layout
    def _layout(self):
        sp = self.spec
        names, kinds = [], []
        for n in self.xnames:
            names.append(n)
            kinds.append("mean")
        for i in self.ar:
            names.append(f"AR({i})")
            kinds.append("ar")
        for j in self.ma:
            names.append(f"MA({j})")
            kinds.append("ma")
        if sp.archm:
            names.append({"sd": "@SQRT(GARCH)", "var": "GARCH", "log": "LOG(GARCH)"}[sp.archm])
            kinds.append("archm")
        self.n_mean = len(names)
        m = sp.model
        if m in ("garch", "arch", "tarch"):
            names.append("C")
            kinds.append("omega")
            names += [f"RESID(-{i})^2" for i in range(1, sp.p + 1)]
            kinds += ["alpha"] * sp.p
            if m == "tarch":
                names += [f"RESID(-{k})^2*(RESID(-{k})<0)" for k in range(1, sp.o + 1)]
                kinds += ["gamma"] * sp.o
            names += [f"GARCH(-{j})" for j in range(1, sp.q + 1)]
            kinds += ["beta"] * sp.q
        elif m == "igarch":
            names += [f"RESID(-{i})^2" for i in range(1, sp.p + 1)]
            kinds += ["alpha"] * sp.p
            names += [f"GARCH(-{j})" for j in range(1, sp.q)]
            kinds += ["beta"] * (sp.q - 1)
        elif m == "egarch":
            names.append("C")
            kinds.append("omega")
            names += [f"ABS(RESID(-{i})/@SQRT(GARCH(-{i})))" for i in range(1, sp.p + 1)]
            kinds += ["alpha"] * sp.p
            names += [f"RESID(-{k})/@SQRT(GARCH(-{k}))" for k in range(1, sp.o + 1)]
            kinds += ["gamma"] * sp.o
            names += [f"LOG(GARCH(-{j}))" for j in range(1, sp.q + 1)]
            kinds += ["beta"] * sp.q
        elif m == "parch":
            names.append("C")
            kinds.append("omega")
            names += [f"ALPHA({i}) [(|RESID(-{i})|-GAMMA*RESID(-{i}))^DELTA]" for i in range(1, sp.p + 1)]
            kinds += ["alpha"] * sp.p
            names += [f"GAMMA({k})" for k in range(1, sp.o + 1)]
            kinds += ["gamma"] * sp.o
            names += [f"BETA({j}) [@SQRT(GARCH(-{j}))^DELTA]" for j in range(1, sp.q + 1)]
            kinds += ["beta"] * sp.q
            if sp.power is None:
                names.append("DELTA (power)")
                kinds.append("delta")
        elif m == "cgarch":
            names += ["Perm: C", "Perm: [Q-C]", "Perm: [ARCH-GARCH]", "Tran: [ARCH-Q]", "Tran: [GARCH-Q]"]
            kinds += ["omega", "rho", "phi", "alpha", "beta"]
        else:
            raise ModelError(f"Model {m} not handled by native engine")
        for zn in self.znames:
            names.append(zn)
            kinds.append("delta_z")
        if sp.dist == "t":
            names.append("T-DIST. DOF")
            kinds.append("nu")
        elif sp.dist == "ged":
            names.append("GED PARAMETER")
            kinds.append("ged")
        elif sp.dist == "skewt":
            names += ["T-DIST. DOF", "SKEWNESS (lambda)"]
            kinds += ["nu", "skew"]
        self.names, self.kinds = names, kinds
        self.k = len(names)

    def idx(self, kind):
        return [i for i, k in enumerate(self.kinds) if k == kind]

    # ------------------------------------------------------------ filter
    def filter(self, th: np.ndarray):
        """Return (eps, h) for parameter vector th; raises FloatingPointError on invalid variances."""
        sp = self.spec
        kinds = self.kinds
        T, t0 = self.T, self.t0
        nx = len(self.xnames)
        b = th[:nx]
        pos = nx
        phi = th[pos:pos + len(self.ar)]
        pos += len(self.ar)
        tht = th[pos:pos + len(self.ma)]
        pos += len(self.ma)
        lam = 0.0
        if sp.archm:
            lam = th[pos]
            pos += 1
        xb = self.X @ b if nx else np.zeros(T)
        zd = None
        iz = self.idx("delta_z")
        if iz:
            zd = self.Z @ th[iz]
        bc = self.bc
        m = sp.model
        alpha = th[self.idx("alpha")]
        beta = th[self.idx("beta")]
        gamma = th[self.idx("gamma")]
        omega = th[self.idx("omega")[0]] if self.idx("omega") else 0.0
        if m == "igarch":
            beta = np.concatenate([beta, [1.0 - alpha.sum() - beta.sum()]])

        simple = (not sp.archm) and not self.ma and m in ("garch", "arch", "tarch", "igarch")
        if simple:
            # residuals are independent of the variance -> vectorised path
            u = self.y - xb
            eps = u.copy()
            for i, ph in zip(self.ar, phi):
                eps[i:] -= ph * u[:-i]
            eps[:t0] = 0.0
            e2 = eps * eps
            e2[:t0] = bc
            T_ = T - t0
            P = max(sp.p, sp.o, sp.q, 1)
            # build driving term for t >= t0 using presample backcast
            ext_e2 = np.concatenate([np.full(P, bc), e2[t0:]])
            neg = np.concatenate([np.full(P, 0.5), (eps[t0:] < 0).astype(float)])
            drive = np.full(T_, omega)
            for i, a in enumerate(alpha, start=1):
                drive += a * ext_e2[P - i:P - i + T_]
            for k, g in enumerate(gamma, start=1):
                drive += g * (ext_e2 * neg)[P - k:P - k + T_]
            if zd is not None:
                drive += zd[t0:]
            q = len(beta)
            if q:
                a_ = np.concatenate([[1.0], -beta])
                zi = lfiltic([1.0], a_, y=np.full(q, bc), x=np.zeros(1))
                h_ = lfilter([1.0], a_, drive, zi=zi)[0]
            else:
                h_ = drive
            h = np.full(T, bc)
            h[t0:] = h_
            if not np.all(np.isfinite(h_)) or h_.min() <= 0:
                raise FloatingPointError
            return eps, h

        # general recursive path (ARCH-M, MA terms, EGARCH, PARCH, CGARCH)
        y = self.y.tolist()
        xbl = xb.tolist()
        zdl = zd.tolist() if zd is not None else None
        h = [bc] * T
        eps = [0.0] * T
        u = [0.0] * T
        lam_f = {"sd": math.sqrt, "var": (lambda v: v), "log": math.log, None: None}[sp.archm]
        g_bc = lam * lam_f(bc) if sp.archm else 0.0
        for t in range(t0):
            u[t] = y[t] - xbl[t] - g_bc
        al, be, ga = alpha.tolist(), beta.tolist(), gamma.tolist()
        phl, thl = phi.tolist(), tht.tolist()
        p, q, o = len(al), len(be), len(ga)
        logbc = math.log(bc)
        if m == "parch":
            delta = sp.power if sp.power is not None else th[self.idx("delta")[0]]
            sd_pre = bc ** (delta / 2)
            sdel = [sd_pre] * T
        if m == "cgarch":
            rho = th[self.idx("rho")[0]]
            phc = th[self.idx("phi")[0]]
            qq = [bc] * T
        if m == "egarch":
            lh = [logbc] * T
        for t in range(t0, T):
            # ---- variance
            if m in ("garch", "arch", "tarch", "igarch"):
                v = omega
                for i in range(p):
                    s = t - 1 - i
                    v += al[i] * (eps[s] * eps[s] if s >= t0 else bc)
                for k in range(o):
                    s = t - 1 - k
                    if s >= t0:
                        if eps[s] < 0:
                            v += ga[k] * eps[s] * eps[s]
                    else:
                        v += ga[k] * 0.5 * bc
                for j in range(q):
                    s = t - 1 - j
                    v += be[j] * (h[s] if s >= t0 else bc)
                if zdl is not None:
                    v += zdl[t]
                ht = v
            elif m == "egarch":
                v = omega
                for i in range(p):
                    s = t - 1 - i
                    v += al[i] * (abs(eps[s]) / math.sqrt(h[s]) if s >= t0 else SQ2PI)
                for k in range(o):
                    s = t - 1 - k
                    if s >= t0:
                        v += ga[k] * eps[s] / math.sqrt(h[s])
                for j in range(q):
                    s = t - 1 - j
                    v += be[j] * (lh[s] if s >= t0 else logbc)
                if zdl is not None:
                    v += zdl[t]
                if v > 700:
                    raise FloatingPointError
                lh[t] = v
                ht = math.exp(v)
            elif m == "parch":
                v = omega
                for i in range(p):
                    s = t - 1 - i
                    gk = ga[i] if i < o else 0.0
                    if s >= t0:
                        a_ = abs(eps[s]) - gk * eps[s]
                        v += al[i] * (a_ ** delta if a_ > 0 else 0.0)
                    else:
                        v += al[i] * sd_pre
                for j in range(q):
                    s = t - 1 - j
                    v += be[j] * (sdel[s] if s >= t0 else sd_pre)
                if zdl is not None:
                    v += zdl[t]
                if v <= 0:
                    raise FloatingPointError
                sdel[t] = v
                ht = v ** (2.0 / delta)
            else:  # cgarch
                s = t - 1
                e2p = eps[s] * eps[s] if s >= t0 else bc
                hp = h[s] if s >= t0 else bc
                qp = qq[s] if s >= t0 else bc
                qt = omega + rho * (qp - omega) + phc * (e2p - hp)
                if zdl is not None:
                    qt += zdl[t]
                qq[t] = qt
                ht = qt + al[0] * (e2p - qp) + be[0] * (hp - qp)
            if not (ht > 0) or ht != ht or ht == float("inf"):
                raise FloatingPointError
            h[t] = ht
            # ---- mean
            mu = xbl[t]
            if sp.archm:
                mu += lam * lam_f(ht)
            ut = y[t] - mu
            e = ut
            for i, ph in enumerate(phl):
                e -= ph * u[t - self.ar[i]]
            for j, tj in enumerate(thl):
                s = t - self.ma[j]
                if s >= t0:
                    e -= tj * eps[s]
            u[t] = ut
            eps[t] = e
        return np.array(eps), np.array(h)

    def ll_terms(self, th):
        try:
            with np.errstate(all="raise"):
                eps, h = self.filter(th)
        except (FloatingPointError, OverflowError, ValueError, ZeroDivisionError):
            return None
        dp = th[self.idx("nu") + self.idx("ged") + self.idx("skew")]
        lt = loglik_terms(eps[self.t0:], h[self.t0:], self.spec.dist, dp)
        if not np.all(np.isfinite(lt)):
            return None
        return lt

    def negll(self, th):
        lt = self.ll_terms(th)
        if lt is None:
            return 1e10
        return -float(lt.sum()) / self.T

    # ------------------------------------------------------------ starting values / bounds
    def start_values(self, b0: np.ndarray, s2: float) -> list[np.ndarray]:
        sp = self.spec
        base = np.zeros(self.k)
        base[: len(self.xnames)] = b0
        cands = []
        m = sp.model
        grid = [(0.05, 0.90), (0.10, 0.80), (0.15, 0.80), (0.05, 0.94), (0.20, 0.60)]
        for a, bb in grid:
            th = base.copy()
            ia, ib, ig = self.idx("alpha"), self.idx("beta"), self.idx("gamma")
            if m in ("garch", "arch", "tarch"):
                if not ib:
                    a, bb = 0.3, 0.0
                th[ia] = a / max(len(ia), 1)
                th[ib] = bb / max(len(ib), 1)
                if ig:
                    th[ig] = 0.05
                th[self.idx("omega")] = s2 * max(1 - a - bb - (0.025 if ig else 0), 0.02)
            elif m == "igarch":
                th[ia] = a / max(len(ia), 1)
                if ib:
                    th[ib] = (1 - a) / (len(ib) + 1)
            elif m == "egarch":
                th[ia] = a + 0.05
                th[ig] = 0.0
                th[ib] = min(bb + 0.05, 0.98) / max(len(ib), 1)
                th[self.idx("omega")] = (1 - th[ib].sum()) * math.log(s2) - th[ia].sum() * SQ2PI
            elif m == "parch":
                d = sp.power if sp.power is not None else 2.0
                th[ia] = a
                th[ig] = 0.0
                th[ib] = bb
                if self.idx("delta"):
                    th[self.idx("delta")] = d
                th[self.idx("omega")] = s2 ** (d / 2) * max(1 - a - bb, 0.02)
            elif m == "cgarch":
                th[self.idx("omega")] = s2
                th[self.idx("rho")] = 0.98
                th[self.idx("phi")] = 0.03
                th[ia] = a
                th[ib] = min(bb, 0.97 - a - 0.02)
            if self.idx("nu"):
                th[self.idx("nu")] = 8.0
            if self.idx("ged"):
                th[self.idx("ged")] = 1.5
            if self.idx("skew"):
                th[self.idx("skew")] = 0.0
            cands.append(th)
        return cands

    def bounds(self, s2):
        bnd = []
        m = self.spec.model
        for k in self.kinds:
            if k in ("mean", "archm", "delta_z"):
                bnd.append((None, None))
            elif k in ("ar", "ma"):
                bnd.append((-0.9999, 0.9999))
            elif k == "omega":
                bnd.append((None, None) if m == "egarch" else (1e-12 * s2 ** (1 if m != "parch" else 0.5), None))
            elif k == "alpha":
                bnd.append((None, None) if m == "egarch" else (0.0, 1.0))
            elif k == "gamma":
                bnd.append((None, None) if m == "egarch" else (-1.0, 1.0))
            elif k == "beta":
                bnd.append((-0.9999, 0.9999) if m == "egarch" else (0.0, 1.0))
            elif k == "delta":
                bnd.append((0.2, 4.0))
            elif k == "rho":
                bnd.append((0.0, 0.99999))
            elif k == "phi":
                bnd.append((0.0, 1.0))
            elif k == "nu":
                bnd.append((2.05, 500.0))
            elif k == "ged":
                bnd.append((0.05, 50.0))
            elif k == "skew":
                bnd.append((-0.99, 0.99))
            else:
                bnd.append((None, None))
        return bnd

    def constraints(self):
        m = self.spec.model
        ia, ib, ig = self.idx("alpha"), self.idx("beta"), self.idx("gamma")
        cons = []
        if m in ("garch", "arch", "tarch"):
            cons.append(lambda th: 0.99999 - th[ia].sum() - th[ib].sum() - 0.5 * th[ig].sum())
            for i, g in zip(ia, ig):
                cons.append(lambda th, i=i, g=g: th[i] + th[g])
        elif m == "igarch":
            cons.append(lambda th: 1.0 - th[ia].sum() - th[ib].sum())
        elif m == "egarch":
            cons.append(lambda th: 0.99999 - th[ib].sum())
        elif m == "parch":
            cons.append(lambda th: 0.99999 - th[ia].sum() - th[ib].sum())
        elif m == "cgarch":
            ir, ip = self.idx("rho")[0], self.idx("phi")[0]
            cons.append(lambda th: th[ir] - th[ia].sum() - th[ib].sum() - 1e-6)
            cons.append(lambda th: th[ib].sum() - th[ip])
        return cons

    # ------------------------------------------------------------ estimation
    def fit(self, b0: np.ndarray, cov: str = "bw", maxiter: int = 1000):
        # presample backcast from OLS residuals (exponential smoothing, arch-style)
        u0 = self.y - (self.X @ b0 if len(b0) else 0)
        u0 = u0[self.t0:]
        s2 = float(np.mean(u0 ** 2))
        tau = min(75, len(u0))
        w = 0.94 ** np.arange(tau)
        w /= w.sum()
        self.bc = float((u0[:tau] ** 2 * w).sum())
        cands = self.start_values(b0, s2)
        vals = [self.negll(c) for c in cands]
        x0 = cands[int(np.argmin(vals))]
        scale = np.maximum(np.abs(x0), 1e-3)
        for i, k in enumerate(self.kinds):
            if k in ("alpha", "beta", "gamma", "rho", "phi", "ar", "ma", "skew"):
                scale[i] = 1.0
            elif k == "omega" and self.spec.model != "egarch":
                scale[i] = max(abs(x0[i]), 1e-12)
            elif k in ("mean", "archm", "delta_z"):
                scale[i] = max(abs(x0[i]), math.sqrt(s2) * 0.1, 1e-8)
        bnds = self.bounds(s2)
        sb = []
        for (lo, hi), sc in zip(bnds, scale):
            sb.append((None if lo is None else lo / sc, None if hi is None else hi / sc))
        cons = [{"type": "ineq", "fun": (lambda z, f=f: f(z * scale))} for f in self.constraints()]
        f = lambda z: self.negll(z * scale)  # noqa: E731
        z0 = x0 / scale
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = optimize.minimize(f, z0, method="SLSQP", bounds=sb, constraints=cons,
                                    options={"maxiter": maxiter, "ftol": 1e-10})
            # polish from the solution once more (helps SLSQP stop at a true optimum)
            res2 = optimize.minimize(f, res.x, method="SLSQP", bounds=sb, constraints=cons,
                                     options={"maxiter": maxiter, "ftol": 1e-12})
        nit = int(res.nit) + int(res2.nit)
        if res2.fun <= res.fun:
            res = res2
        self.opt = res
        th = res.x * scale
        self.theta = th
        self.converged = bool(res.success) and res.fun < 1e9
        self.niter = nit
        lt = self.ll_terms(th)
        if lt is None:
            raise ModelError("ARCH estimation failed: invalid variance at the optimum (try other starting values)")
        self.llf = float(lt.sum())
        self.eps, self.h = self.filter(th)
        self._covariance(cov)
        return self

    def _covariance(self, cov):
        th = self.theta
        k = self.k
        T = self.T

        def total(x):
            lt = self.ll_terms(x)
            return -1e10 if lt is None else float(lt.sum())

        H = numeric_hessian(total, th, rel=1e-4)
        G = np.zeros((T - self.t0, k))
        l0 = self.ll_terms(th)
        for i in range(k):
            hstep = 1e-5 * max(abs(th[i]), 1e-3)
            e = np.zeros(k)
            e[i] = hstep
            lp, lm = self.ll_terms(th + e), self.ll_terms(th - e)
            if lp is not None and lm is not None:
                G[:, i] = (lp - lm) / (2 * hstep)
            elif lp is not None:
                G[:, i] = (lp - l0) / hstep
            elif lm is not None:
                G[:, i] = (l0 - lm) / hstep
        Hinv = safe_inv(-H)
        OPG = G.T @ G
        cov = (cov or "bw").lower()
        if cov in ("opg", "bhhh"):
            V = safe_inv(OPG)
            self.cov_label = "Coefficient covariance computed using outer product of gradients"
        elif cov in ("hessian", "ordinary", "hess"):
            V = Hinv
            self.cov_label = "Coefficient covariance computed using observed Hessian"
        else:
            V = Hinv @ OPG @ Hinv
            self.cov_label = "Bollerslev-Wooldridge robust standard errors & covariance"
        V = (V + V.T) / 2
        self.cov = V
        self.bse = np.sqrt(np.clip(np.diag(V), 0, None))

    # ------------------------------------------------------------ simulation forecast
    def simulate_forecast(self, horizon: int, Xf: np.ndarray, Zf: np.ndarray | None, nsim: int = 2000, seed=0):
        """Bootstrap-simulate mean and variance forecasts."""
        rng = np.random.default_rng(seed)
        sp = self.spec
        th = self.theta
        z_hist = (self.eps / np.sqrt(self.h))[self.t0:]
        z_hist = (z_hist - z_hist.mean()) / z_hist.std()
        nx = len(self.xnames)
        b = th[:nx]
        phi = th[nx:nx + len(self.ar)]
        tht = th[nx + len(self.ar): nx + len(self.ar) + len(self.ma)]
        lam = th[self.n_mean - 1] if sp.archm else 0.0
        T = self.T
        L = 60
        eps = np.tile(self.eps[-L:], (nsim, 1))
        h = np.tile(self.h[-L:], (nsim, 1))
        y_hist = self.y[-L:]
        xb_hist = (self.X[-L:] @ b) if nx else np.zeros(L)
        lamf = {"sd": np.sqrt, "var": lambda v: v, "log": np.log}.get(sp.archm)
        u = np.tile(y_hist - xb_hist - (lam * lamf(self.h[-L:]) if sp.archm else 0), (nsim, 1))
        extra = {}
        m = sp.model
        alpha = th[self.idx("alpha")]
        beta = th[self.idx("beta")]
        gamma = th[self.idx("gamma")]
        omega = th[self.idx("omega")[0]] if self.idx("omega") else 0.0
        if m == "igarch":
            beta = np.concatenate([beta, [1 - alpha.sum() - beta.sum()]])
        if m == "parch":
            delta = sp.power if sp.power is not None else th[self.idx("delta")[0]]
            sdel = h ** (delta / 2)
        if m == "cgarch":
            rho = th[self.idx("rho")[0]]
            phc = th[self.idx("phi")[0]]
            # reconstruct permanent component by re-running the recursion
            qhist = self._cgarch_q()
            qq = np.tile(qhist[-L:], (nsim, 1))
        zd = (Zf @ th[self.idx("delta_z")]) if (Zf is not None and self.idx("delta_z")) else np.zeros(horizon)
        mean_f = np.zeros((nsim, horizon))
        var_f = np.zeros((nsim, horizon))
        for s in range(horizon):
            if m in ("garch", "arch", "tarch", "igarch"):
                v = np.full(nsim, omega)
                for i, a in enumerate(alpha, start=1):
                    v += a * eps[:, -i] ** 2
                for k, g in enumerate(gamma, start=1):
                    v += g * eps[:, -k] ** 2 * (eps[:, -k] < 0)
                for j, bb in enumerate(beta, start=1):
                    v += bb * h[:, -j]
                ht = v + zd[s]
            elif m == "egarch":
                v = np.full(nsim, omega)
                for i, a in enumerate(alpha, start=1):
                    v += a * np.abs(eps[:, -i]) / np.sqrt(h[:, -i])
                for k, g in enumerate(gamma, start=1):
                    v += g * eps[:, -k] / np.sqrt(h[:, -k])
                for j, bb in enumerate(beta, start=1):
                    v += bb * np.log(h[:, -j])
                ht = np.exp(v + zd[s])
            elif m == "parch":
                v = np.full(nsim, omega)
                for i, a in enumerate(alpha, start=1):
                    gk = gamma[i - 1] if i - 1 < len(gamma) else 0.0
                    v += a * np.clip(np.abs(eps[:, -i]) - gk * eps[:, -i], 0, None) ** delta
                for j, bb in enumerate(beta, start=1):
                    v += bb * sdel[:, -j]
                v = np.maximum(v + zd[s], 1e-300)
                sdel = np.column_stack([sdel, v])
                ht = v ** (2 / delta)
            else:
                qt = omega + rho * (qq[:, -1] - omega) + phc * (eps[:, -1] ** 2 - h[:, -1]) + zd[s]
                ht = qt + alpha[0] * (eps[:, -1] ** 2 - qq[:, -1]) + beta[0] * (h[:, -1] - qq[:, -1])
                qq = np.column_stack([qq, qt])
            ht = np.maximum(ht, 1e-300)
            z = rng.choice(z_hist, nsim)
            e = z * np.sqrt(ht)
            mu = (Xf[s] @ b if nx else 0.0) + (lam * lamf(ht) if sp.archm else 0.0)
            ut = e.copy()
            for i, ph in zip(self.ar, phi):
                ut += ph * u[:, -i]
            for j, tj in zip(self.ma, tht):
                ut += tj * eps[:, -j]
            eps = np.column_stack([eps, e])
            h = np.column_stack([h, ht])
            u = np.column_stack([u, ut])
            mean_f[:, s] = mu + ut
            var_f[:, s] = ht
        _ = T, extra
        return mean_f, var_f

    def _cgarch_q(self):
        th = self.theta
        omega = th[self.idx("omega")[0]]
        rho = th[self.idx("rho")[0]]
        phc = th[self.idx("phi")[0]]
        q = np.full(self.T, self.bc)
        for t in range(self.t0, self.T):
            s = t - 1
            e2p = self.eps[s] ** 2 if s >= self.t0 else self.bc
            hp = self.h[s] if s >= self.t0 else self.bc
            qp = q[s] if s >= self.t0 else self.bc
            q[t] = omega + rho * (qp - omega) + phc * (e2p - hp)
        return q

    def uncond_var(self) -> float:
        th = self.theta
        m = self.spec.model
        a = th[self.idx("alpha")].sum()
        b = th[self.idx("beta")].sum()
        g = th[self.idx("gamma")].sum()
        if m in ("garch", "arch", "tarch"):
            pers = a + b + 0.5 * g
            return th[self.idx("omega")[0]] / (1 - pers) if pers < 1 else np.nan
        if m == "cgarch":
            return th[self.idx("omega")[0]]
        return float(np.mean(self.h[self.t0:]))

    def persistence(self) -> float:
        th = self.theta
        m = self.spec.model
        a = th[self.idx("alpha")].sum()
        b = th[self.idx("beta")].sum()
        g = th[self.idx("gamma")].sum()
        if m in ("garch", "arch", "tarch"):
            return a + b + 0.5 * g
        if m == "igarch":
            return 1.0
        if m == "egarch":
            return b
        if m == "cgarch":
            return th[self.idx("rho")[0]]
        if m == "parch":
            return a + b
        return np.nan


# ============================================================================
# arch-package engine (FIGARCH, HARCH, EWMA, Zakoian TGARCH)
# ============================================================================

class ArchPkgGarch:
    def __init__(self, y, X, xnames, ar, spec: GarchSpec, harch_lags=(1, 5, 22)):
        from arch.univariate import (ARX, FIGARCH, GARCH, HARCH, LS, ConstantMean, EWMAVariance, GeneralizedError,
                                     Normal, SkewStudent, StudentsT, ZeroMean)
        self.spec = spec
        self.y = y
        has_c = "C" in xnames
        others = [i for i, n in enumerate(xnames) if n != "C"]
        Xo = X[:, others] if others else None
        self.scale = 1.0
        sd = float(np.std(y))
        if sd < 0.5 or sd > 1000:
            self.scale = 10 ** round(math.log10(1.0 / sd))
        ys = y * self.scale
        if ar:
            mean = ARX(ys, x=Xo, lags=list(ar), constant=has_c)
        elif Xo is not None:
            mean = LS(ys, x=np.column_stack([np.ones(len(y)), Xo]) if has_c else Xo)
        elif has_c:
            mean = ConstantMean(ys)
        else:
            mean = ZeroMean(ys)
        m = spec.model
        if m == "figarch":
            vol = FIGARCH(p=min(spec.p, 1), q=min(spec.q, 1))
        elif m == "harch":
            vol = HARCH(lags=list(harch_lags))
        elif m == "ewma":
            vol = EWMAVariance(lam=None)
        elif m == "zarch":
            vol = GARCH(p=spec.p, o=max(spec.o, 1), q=spec.q, power=1.0)
        else:
            raise ModelError(f"{m} not supported by arch engine")
        mean.volatility = vol
        mean.distribution = {"normal": Normal(), "t": StudentsT(), "skewt": SkewStudent(),
                             "ged": GeneralizedError()}[spec.dist]
        self.mod = mean
        self.ar, self.xnames = list(ar), xnames
        self.T = len(y)
        self.t0 = max(ar) if ar else 0

    def fit(self, cov="bw"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = self.mod.fit(disp="off", cov_type="classic" if cov in ("hessian", "ordinary") else "robust",
                               options={"maxiter": 2000})
        self.res = res
        s = self.scale
        names = []
        theta, bse = [], []
        for pn, v, se in zip(res.params.index, res.params.values, res.std_err.values):
            f = 1.0
            nm = pn
            if pn in ("Const", "mu"):
                nm, f = "C", 1 / s
            elif pn.startswith("x"):
                try:
                    idx = int(pn[1:])
                    other = [n for n in self.xnames if n != "C"]
                    nm = other[idx] if idx < len(other) else pn
                except ValueError:
                    nm = pn.upper()
                f = 1 / s
            elif pn.startswith("y[") or pn.startswith("y.L"):
                nm = f"AR({pn.split('[')[-1].rstrip(']') if '[' in pn else pn[3:]})"
            elif pn == "omega":
                nm = "C (variance)"
                f = 1 / s ** (2 if self.spec.model != "zarch" else 1)
            elif pn == "nu":
                nm = "T-DIST. DOF"
            elif pn == "lambda" and self.spec.model != "ewma":
                nm = "SKEWNESS (lambda)"
            elif pn == "eta":
                nm = "T-DIST. DOF"
            else:
                nm = pn.upper()
            names.append(nm)
            theta.append(v * f)
            bse.append(se * f)
        self.names = names
        self.theta = np.array(theta)
        self.bse = np.array(bse)
        self.cov = np.diag(self.bse ** 2)
        try:
            cv = np.asarray(res.param_cov)
            fvec = np.where(np.abs(np.asarray(res.params.values)) > 0, self.theta / np.where(res.params.values == 0, 1,
                                                                                              res.params.values), 1)
            self.cov = cv * np.outer(fvec, fvec)
        except Exception:
            pass
        self.llf = float(res.loglikelihood) - self.T * math.log(s) + (0 if True else 0)
        cvar = np.asarray(res.conditional_volatility) ** 2 / s ** 2
        resid = np.asarray(res.resid) / s
        cvar = np.where(np.isnan(cvar), np.nanmean(cvar), cvar)
        resid = np.where(np.isnan(resid), 0.0, resid)
        self.h, self.eps = cvar, resid
        self.converged = res.convergence_flag == 0
        self.niter = int(getattr(res, "iterations", 0) or 0)
        self.cov_label = "Bollerslev-Wooldridge robust standard errors & covariance" if cov not in ("hessian",
                                                                                                    "ordinary") else \
            "Coefficient covariance computed using observed Hessian"
        k_mean = sum(1 for n in names if n in self.xnames or n.startswith("AR("))
        self.n_mean = k_mean
        self.k = len(names)
        self.kinds = ["mean"] * k_mean + ["var"] * (len(names) - k_mean)
        return self

    def simulate_forecast(self, horizon, Xf, Zf=None, nsim=2000, seed=0):
        method = "analytic" if self.spec.model in ("figarch", "harch", "ewma") else "simulation"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                fc = self.res.forecast(horizon=horizon, method=method, reindex=False,
                                       x=None)
            except Exception:
                fc = self.res.forecast(horizon=horizon, method="simulation", simulations=nsim, reindex=False)
        s = self.scale
        mean = np.asarray(fc.mean)[-1] / s
        var = np.asarray(fc.variance)[-1] / s ** 2
        return np.tile(mean, (1, 1)), np.tile(var, (1, 1))

    def uncond_var(self):
        return float(np.mean(self.h))

    def persistence(self):
        return np.nan


# ============================================================================
# result wrapper
# ============================================================================

class GarchResult(ModelResult):
    object_type = "equation"

    def __init__(self, wf, ev, spec_text: str, model="garch", p=1, q=1, o=0, dist="normal", archm=None,
                 power=None, cov="bw", engine="auto", name=""):
        self.wf, self.ev, self.name = wf, ev, name
        self.spec_text = spec_text
        model, dist = norm_model(model), norm_dist(dist)
        if model == "arch":
            q = 0
        if dist not in DIST_LABEL:
            raise ModelError(f"Unknown error distribution '{dist}' (normal, t, ged, skewt)")
        archm = None if archm in (None, "", "none", "0") else str(archm).lower()
        if archm not in (None, "sd", "var", "log"):
            raise ModelError("archm must be sd, var or log")
        parts = split_at(spec_text)
        mean_spec = parts[0]
        var_spec = parts[1] if len(parts) > 1 else ""
        d = build_design(wf, ev, mean_spec)
        if d.sar or d.sma:
            raise ModelError("Seasonal ARMA terms are not supported in ARCH mean equations")
        zcols, znames = None, []
        if var_spec.strip():
            Zd = build_design(wf, ev, var_spec, require_y=False, allow_arma=False)
            mask = d.mask & Zd.mask
            d = build_design(wf, ev, mean_spec, mask=mask)
            Zd = build_design(wf, ev, var_spec, mask=d.mask, require_y=False, allow_arma=False)
            zcols = Zd.X[[c for c in Zd.xnames if c != "C"]].values
            znames = [c for c in Zd.xnames if c != "C"]
        if not contiguous_check(d.mask):
            raise ModelError("ARCH estimation requires a contiguous sample")
        self.d, self.yname = d, d.yname
        y = d.y.values
        X = d.X.values
        spec = GarchSpec(model=model, p=int(p), q=int(q), o=int(o), dist=dist, archm=archm,
                         power=float(power) if power not in (None, "") else None)
        use_pkg = model in ARCHPKG_MODELS or engine == "arch"
        if use_pkg and (archm or d.ma or znames):
            raise ModelError(f"{model.upper()} does not support ARCH-M, MA terms or variance regressors")
        # OLS start for mean parameters
        b0 = np.linalg.lstsq(X, y, rcond=None)[0] if X.shape[1] else np.zeros(0)
        if use_pkg:
            self.fit = ArchPkgGarch(y, X, d.xnames, d.ar, spec).fit(cov)
            self.engine = "arch"
        else:
            self.fit = NativeGarch(y, X, d.xnames, d.ar, d.ma, zcols, znames, spec).fit(b0, cov)
            self.engine = "native"
        self.spec = spec
        f = self.fit
        t0 = f.t0
        mask = d.mask.copy()
        if t0:
            idx = np.flatnonzero(mask)
            mask[idx[:t0]] = False
        self.mask = mask
        self.names = f.names
        self.params = f.theta
        self.bse = f.bse
        self.cov = f.cov
        self.zstat = self.params / np.where(self.bse > 0, self.bse, np.nan)
        self.pvalues = pvalue_z(self.zstat)
        self.eps_ = f.eps[t0:]
        self.h_ = f.h[t0:]
        self.yv = y[t0:]
        self.fitted_ = self.yv - self.eps_
        self.llf = f.llf
        self.method_label = f"ML ARCH - {DIST_LABEL[dist]}"
        n = len(self.yv)
        k = len(self.params)
        e = self.eps_
        ssr = float(e @ e)
        tss = float(((self.yv - self.yv.mean()) ** 2).sum())
        r2 = 1 - ssr / tss if tss > 0 else np.nan
        kmean = f.n_mean if hasattr(f, "n_mean") else len(d.xnames)
        aic, sc, hq = info_criteria(self.llf, k, n)
        self.stats_dict = {"r2": r2, "rbar2": 1 - (1 - r2) * (n - 1) / max(n - kmean, 1),
                           "se": math.sqrt(ssr / max(n - kmean, 1)), "ssr": ssr, "logl": self.llf, "aic": aic,
                           "sc": sc, "hq": hq, "dw": _dw(e), "meandep": float(self.yv.mean()),
                           "sddep": float(self.yv.std(ddof=1)), "regobs": n, "ncoef": k,
                           "persistence": f.persistence(), "uncondvar": f.uncond_var()}

    # ------------------------------------------------------------ access
    def garch(self) -> pd.Series:
        return series_full(self.wf, self.h_, self.mask)

    def resid(self, standardized=False) -> pd.Series:
        v = self.eps_ / np.sqrt(self.h_) if standardized else self.eps_
        return series_full(self.wf, v, self.mask)

    def std_resid(self) -> np.ndarray:
        return self.eps_ / np.sqrt(self.h_)

    def stat(self, key):
        key = key.lower()
        if key == "coefs":
            return lambda i=None: self.params if i is None else float(self.params[int(i) - 1])
        if key == "stderrs":
            return lambda i=None: self.bse if i is None else float(self.bse[int(i) - 1])
        if key == "tstats":
            return lambda i=None: self.zstat if i is None else float(self.zstat[int(i) - 1])
        if key in self.stats_dict:
            return self.stats_dict[key]
        raise ModelError(f"@{key} not available")

    # ------------------------------------------------------------ output
    def _variance_equation_text(self) -> str:
        sp = self.spec
        m = sp.model
        if self.engine == "arch":
            return {"figarch": "FIGARCH(p,d,q): (1-beta L) h = omega + [1 - beta L - phi L (1-L)^d] e^2",
                    "harch": "HARCH: h = omega + sum_i alpha_i (avg of e^2 over lag horizon i)",
                    "ewma": "EWMA: h = (1-lambda) e(-1)^2 + lambda h(-1)",
                    "zarch": "TGARCH (Zakoian): sigma = omega + alpha|e(-1)| + gamma|e(-1)|(e(-1)<0) + beta sigma(-1)"}[m]
        if m in ("garch", "arch"):
            s = "GARCH = C + " + " + ".join([f"a{i}*RESID(-{i})^2" for i in range(1, sp.p + 1)] +
                                            [f"b{j}*GARCH(-{j})" for j in range(1, sp.q + 1)])
        elif m == "tarch":
            s = ("GARCH = C + " + " + ".join([f"a{i}*RESID(-{i})^2" for i in range(1, sp.p + 1)] +
                                             [f"g{k}*RESID(-{k})^2*(RESID(-{k})<0)" for k in range(1, sp.o + 1)] +
                                             [f"b{j}*GARCH(-{j})" for j in range(1, sp.q + 1)]))
        elif m == "igarch":
            s = "GARCH = " + " + ".join([f"a{i}*RESID(-{i})^2" for i in range(1, sp.p + 1)] +
                                        [f"b{j}*GARCH(-{j})" for j in range(1, sp.q + 1)]) + "   (sum of coefs = 1)"
        elif m == "egarch":
            s = ("LOG(GARCH) = C + " + " + ".join([f"a{i}*ABS(RESID(-{i})/@SQRT(GARCH(-{i})))" for i in range(1, sp.p + 1)]
                                                  + [f"g{k}*RESID(-{k})/@SQRT(GARCH(-{k}))" for k in range(1, sp.o + 1)]
                                                  + [f"b{j}*LOG(GARCH(-{j}))" for j in range(1, sp.q + 1)]))
        elif m == "parch":
            s = ("@SQRT(GARCH)^DELTA = C + " + " + ".join(
                [f"a{i}*(ABS(RESID(-{i})) - g{i}*RESID(-{i}))^DELTA" for i in range(1, sp.p + 1)]
                + [f"b{j}*@SQRT(GARCH(-{j}))^DELTA" for j in range(1, sp.q + 1)]))
        else:
            s = ("Q = C + rho*(Q(-1) - C) + phi*(RESID(-1)^2 - GARCH(-1));  "
                 "GARCH = Q + alpha*(RESID(-1)^2 - Q(-1)) + beta*(GARCH(-1) - Q(-1))")
        if getattr(self.fit, "znames", None):
            s += " + " + " + ".join(self.fit.znames)
        return s

    def report(self) -> Report:
        s = self.stats_dict
        sp = self.spec
        r = Report(f"Equation: {self.name.upper()}" if self.name else "Equation")
        lab = {"garch": f"GARCH({sp.p},{sp.q})", "arch": f"ARCH({sp.p})", "tarch": f"TARCH({sp.p},{sp.o},{sp.q})",
               "egarch": f"EGARCH({sp.p},{sp.o},{sp.q})", "parch": f"PARCH({sp.p},{sp.o},{sp.q})",
               "cgarch": "Component GARCH(1,1)", "igarch": f"IGARCH({sp.p},{sp.q})",
               "figarch": "FIGARCH(1,d,1)", "harch": "HARCH", "ewma": "EWMA (RiskMetrics)",
               "zarch": "TGARCH (Zakoian)"}[sp.model]
        if sp.archm:
            lab += f"-M ({sp.archm})"
        hdr = [f"Dependent Variable: {self.yname}", f"Method: {self.method_label} ({'SLSQP' if self.engine == 'native' else 'arch package'})"]
        hdr += sample_header(self.wf, self.mask)
        hdr.append(("Convergence achieved" if self.fit.converged else "WARNING: convergence not achieved")
                   + (f" after {self.fit.niter} iterations" if self.fit.niter else ""))
        hdr.append(self.fit.cov_label)
        hdr.append("Presample variance: backcast (parameter = 0.94)" if self.engine == "native" else
                   "Presample variance: backcast (arch package)")
        hdr.append(f"Model: {lab}")
        hdr.append(self._variance_equation_text())
        r.header(hdr, stamp=True)
        nm = self.fit.n_mean if hasattr(self.fit, "n_mean") else 0
        breaks = {nm: "Variance Equation"} if nm < len(self.names) else {}
        r.coef_table(self.names, self.params, self.bse, self.zstat, self.pvalues, stat_label="z-Statistic",
                     section_breaks=breaks)
        r.stats([("R-squared", s["r2"]), ("Adjusted R-squared", s["rbar2"]), ("S.E. of regression", s["se"]),
                 ("Sum squared resid", s["ssr"]), ("Log likelihood", s["logl"]), ("Durbin-Watson stat", s["dw"])],
                [("Mean dependent var", s["meandep"]), ("S.D. dependent var", s["sddep"]),
                 ("Akaike info criterion", s["aic"]), ("Schwarz criterion", s["sc"]),
                 ("Hannan-Quinn criter.", s["hq"]), ("Persistence", s["persistence"])])
        if sp.model == "igarch" and self.engine == "native":
            al = self.params[self.fit.idx("alpha")].sum()
            be = self.params[self.fit.idx("beta")].sum()
            r.note(f"Implied GARCH(-{sp.q}) coefficient = 1 - sum of other ARCH/GARCH terms = {fmt_num(1 - al - be)}")
        if sp.model in ("garch", "tarch") and s["persistence"] < 1:
            hl = math.log(0.5) / math.log(s["persistence"]) if 0 < s["persistence"] < 1 else np.nan
            r.note(f"Unconditional variance = {fmt_num(s['uncondvar'])};  volatility half-life = {fmt_num(hl)} periods")
        return r

    # ------------------------------------------------------------ views
    def views(self):
        return [
            View("output", "Estimation Output", self.report),
            View("garchgraph", "GARCH Graph: Conditional Standard Deviation", lambda: self.view_garch_graph(False),
                 group="GARCH Graph"),
            View("garchvar", "GARCH Graph: Conditional Variance", lambda: self.view_garch_graph(True),
                 group="GARCH Graph"),
            View("resids", "Actual, Fitted, Residual Graph", self.view_afr),
            View("stdresid", "Standardized Residual Graph", self.view_stdresid),
            View("correl", "Correlogram - Q-statistics (std. resid)", self.view_correl,
                 [Param("lags", "Lags", "int", 24)], group="Residual Diagnostics"),
            View("correlsq", "Correlogram Squared Std. Residuals", self.view_correl_sq,
                 [Param("lags", "Lags", "int", 24)], group="Residual Diagnostics"),
            View("hist", "Histogram - Normality Test", self.view_hist, group="Residual Diagnostics"),
            View("archtest", "ARCH LM Test (std. resid)", self.view_archtest, [Param("lags", "Lags", "int", 5)],
                 group="Residual Diagnostics"),
            View("signbias", "Engle-Ng Sign Bias Test", self.view_signbias, group="Residual Diagnostics"),
            View("nic", "News Impact Curve", self.view_nic),
            View("wald", "Wald Test - Coefficient Restrictions", self.view_wald,
                 [Param("restrictions", "Restrictions", "text", "c(2)=0")], group="Coefficient Diagnostics"),
            View("coefcov", "Covariance Matrix", self.view_coefcov, group="Coefficient Diagnostics"),
        ]

    def procs(self):
        return [
            View("forecast", "Forecast...", self.proc_forecast,
                 [Param("h", "Horizon", "int", 20), Param("name", "Mean forecast name", "text",
                                                          f"{self.yname.lower()}_f"),
                  Param("varname", "Variance forecast name", "text", f"{self.yname.lower()}_vf")], kind="proc"),
            View("makegarch", "Make GARCH Variance Series", self.proc_makegarch,
                 [Param("name", "Name", "text", "garch01")], kind="proc"),
            View("makeresids", "Make Residual Series", self.proc_makeresids,
                 [Param("name", "Name", "text", "resid01"), Param("std", "Standardized", "bool", False)], kind="proc"),
        ]

    def view_garch_graph(self, variance=False):
        s = self.garch()[self.mask]
        if not variance:
            s = np.sqrt(s)
        t = "Conditional variance" if variance else "Conditional standard deviation"
        return FigureOutput(plots.line_plot({t: s}, t), t)

    def view_afr(self):
        idx = self.wf.data.index[self.mask]
        return FigureOutput(plots.afr_plot(pd.Series(self.yv, idx), pd.Series(self.fitted_, idx),
                                           pd.Series(self.eps_, idx)), "Actual, Fitted, Residual")

    def view_stdresid(self):
        z = self.resid(True)[self.mask]
        fig = plots.line_plot({"Standardized residuals": z}, "Standardized residuals", zero=True)
        return FigureOutput(fig, "Standardized residuals")

    def view_correl(self, lags=24):
        from .stats import correlogram_report
        return correlogram_report(self.std_resid(), "STD. RESID", int(lags), "Correlogram of Standardized Residuals")

    def view_correl_sq(self, lags=24):
        from .stats import correlogram_report
        return correlogram_report(self.std_resid() ** 2, "STD. RESID^2", int(lags),
                                  "Correlogram of Standardized Residuals Squared")

    def view_hist(self):
        from .stats import histogram_output
        return histogram_output(self.std_resid(), "Standardized Residuals", self.wf.sample_label(self.mask))

    def view_archtest(self, lags=5):
        return arch_lm_test(self.std_resid(), int(lags), title="Heteroskedasticity Test: ARCH (standardized residuals)")

    def view_signbias(self):
        z = self.std_resid()
        e = self.eps_
        z2 = z[1:] ** 2
        neg = (e[:-1] < 0).astype(float)
        pos = 1 - neg
        Z = np.column_stack([np.ones(len(z2)), neg, neg * z[:-1], pos * z[:-1]])
        from .regression import aux_ols
        b, u, ssr, r2 = aux_ols(z2, Z)
        n = len(z2)
        se = np.sqrt(np.diag(ssr / (n - 4) * safe_inv(Z.T @ Z)))
        t = b / se
        F = (r2 / 3) / ((1 - r2) / (n - 4))
        r = Report("Engle-Ng Sign and Size Bias Tests")
        r.text("Regression of squared standardized residuals on sign/size indicators of lagged residuals")
        rows = [["Sign Bias", t[1], 2 * stats.t.sf(abs(t[1]), n - 4)],
                ["Negative Size Bias", t[2], 2 * stats.t.sf(abs(t[2]), n - 4)],
                ["Positive Size Bias", t[3], 2 * stats.t.sf(abs(t[3]), n - 4)],
                ["Joint Effect (F)", F, stats.f.sf(F, 3, n - 4)]]
        r.table(["Test", "Statistic", "Prob."], rows, ["str", "num", "prob"])
        r.note("Significant statistics indicate asymmetry not captured by the variance specification.")
        return r

    def view_nic(self):
        sp = self.spec
        th = self.params
        f = self.fit
        hbar = float(np.mean(self.h_))
        sd = math.sqrt(hbar)
        e = np.linspace(-4 * sd, 4 * sd, 201)
        if self.engine != "native":
            raise ModelError("News impact curve available for native GARCH/TARCH/EGARCH/PARCH/CGARCH models")
        al = th[f.idx("alpha")]
        be = th[f.idx("beta")]
        ga = th[f.idx("gamma")]
        om = th[f.idx("omega")[0]] if f.idx("omega") else 0.0
        m = sp.model
        if m in ("garch", "arch", "tarch", "igarch"):
            if m == "igarch":
                be = np.concatenate([be, [1 - al.sum() - be.sum()]])
            h = om + al[0] * e ** 2 + (ga[0] * e ** 2 * (e < 0) if len(ga) else 0) + be.sum() * hbar
        elif m == "egarch":
            h = np.exp(om + al[0] * np.abs(e) / sd + (ga[0] * e / sd if len(ga) else 0) + be.sum() * math.log(hbar))
        elif m == "parch":
            dl = sp.power if sp.power is not None else th[f.idx("delta")[0]]
            g = ga[0] if len(ga) else 0
            h = (om + al[0] * np.clip(np.abs(e) - g * e, 0, None) ** dl + be.sum() * hbar ** (dl / 2)) ** (2 / dl)
        else:
            rho, phc = th[f.idx("rho")[0]], th[f.idx("phi")[0]]
            qbar = om
            q = om + rho * (qbar - om) + phc * (e ** 2 - hbar)
            h = q + al[0] * (e ** 2 - qbar) + be[0] * (hbar - qbar)
        fig, ax = plots.new_figure(6.5, 4.2)
        ax.plot(e, h, color=plots.PALETTE[0], lw=1.6)
        ax.axvline(0, color="#888", lw=0.6)
        ax.set_xlabel("RESID(-1)")
        ax.set_ylabel("Conditional variance")
        ax.set_title(f"News Impact Curve ({m.upper()})")
        fig.tight_layout()
        return FigureOutput(fig, "News impact curve")

    def view_wald(self, restrictions="c(2)=0"):
        return wald_from_restrictions(restrictions, self.params, self.cov)

    def view_coefcov(self):
        r = Report("Coefficient Covariance Matrix")
        r.table([""] + self.names, [[n] + list(self.cov[i]) for i, n in enumerate(self.names)])
        return r

    def proc_makegarch(self, name="garch01"):
        self.wf.set_series(name, self.garch(), respect_sample=False)
        return f"Conditional variance saved to {name.upper()}"

    def proc_makeresids(self, name="resid01", std=False):
        self.wf.set_series(name, self.resid(bool(std)), respect_sample=False)
        return f"{'Standardized r' if std else 'R'}esiduals saved to {name.upper()}"

    def proc_forecast(self, h=20, name="", varname=""):
        from .timeseries import auto_expand
        h = int(h)
        name = name or f"{self.yname.lower()}_f"
        varname = varname or f"{self.yname.lower()}_vf"
        others = [c for c in self.d.xnames if c != "C"] + list(getattr(self.fit, "znames", []) or [])
        last = np.flatnonzero(self.mask)[-1]
        if others and last + h > self.wf.nobs - 1:
            raise ModelError("Out-of-sample forecasts need future values of the regressors "
                             f"({', '.join(others)}): extend the workfile (expand) and fill them first")
        fm = auto_expand(self.wf, h, self.mask)
        Xf = np.zeros((h, len(self.d.xnames)))
        for j, c in enumerate(self.d.xnames):
            if c == "C":
                Xf[:, j] = 1.0
            else:
                v = self.ev.eval(c.lower(), ignore_sample=True, as_series=True).values[fm]
                if np.isnan(v).any():
                    raise ModelError(f"Regressor {c} has missing values in the forecast period")
                Xf[:, j] = v
        Zf = None
        if self.engine == "native" and self.fit.Z is not None:
            Zf = np.column_stack([self.ev.eval(z.lower(), ignore_sample=True, as_series=True).values[fm]
                                  for z in self.fit.znames])
        mf, vf = self.fit.simulate_forecast(h, Xf, Zf)
        mean = mf.mean(axis=0)
        var = vf.mean(axis=0)
        lo = np.quantile(mf, 0.025, axis=0) if mf.shape[0] > 1 else mean - 1.96 * np.sqrt(var)
        hi = np.quantile(mf, 0.975, axis=0) if mf.shape[0] > 1 else mean + 1.96 * np.sqrt(var)
        s1 = pd.Series(np.nan, index=self.wf.data.index)
        s2 = pd.Series(np.nan, index=self.wf.data.index)
        s1.iloc[np.flatnonzero(fm)] = mean
        s2.iloc[np.flatnonzero(fm)] = var
        self.wf.set_series(name, s1, respect_sample=False)
        self.wf.set_series(varname, s2, respect_sample=False)
        r = Report(f"ARCH Forecast: {name.upper()}")
        r.header([f"Horizon: {h}", f"Mean forecast saved to {name.upper()}, variance forecast to {varname.upper()}",
                  "Multi-step forecasts by bootstrap simulation of standardized residuals" if self.engine == "native"
                  else "Forecasts from the arch package"])
        labs = np.array(self.wf.labels())[fm]
        r.table(["Period", "Mean", "Variance", "Std. Dev.", "Lower 95%", "Upper 95%"],
                [[labs[i], mean[i], var[i], math.sqrt(var[i]), lo[i], hi[i]] for i in range(h)])
        fig, axes = plots.new_figure(7.5, 5, 2, 1)
        hist_n = min(len(self.h_), 6 * h)
        hi_idx = self.wf.data.index[self.mask][-hist_n:]
        fi = self.wf.data.index[fm]
        axes[0].plot(hi_idx, self.yv[-hist_n:], color="#555", lw=0.9, label=self.yname)
        axes[0].plot(fi, mean, color=plots.PALETTE[0], lw=1.5, label="Forecast")
        axes[0].fill_between(fi, lo, hi, color="#9fb8e3", alpha=0.4, label="95% band")
        axes[0].legend(frameon=False)
        axes[0].set_title("Mean forecast")
        axes[1].plot(hi_idx, np.sqrt(self.h_[-hist_n:]), color="#555", lw=0.9, label="Conditional S.D.")
        axes[1].plot(fi, np.sqrt(var), color=plots.PALETTE[1], lw=1.5, label="Forecast S.D.")
        axes[1].legend(frameon=False)
        axes[1].set_title("Volatility forecast")
        fig.tight_layout()
        return [r, FigureOutput(fig, "ARCH forecast")]
