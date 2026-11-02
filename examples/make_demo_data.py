"""Generate the SIMULATED demo datasets shipped with Finomania.

All numbers are synthetic (random draws from known models) so that the
examples run offline and estimates can be compared with the true parameters.

    python examples/make_demo_data.py
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data")


def stocks(rng):
    """Daily prices for 4 'stocks' and an index: DCC-GARCH returns with fat tails."""
    dates = pd.bdate_range("2015-01-01", "2024-12-31")
    T, k = len(dates), 4
    a, b = 0.03, 0.95
    Rbar = np.array([[1, .6, .45, .35], [.6, 1, .5, .4], [.45, .5, 1, .3], [.35, .4, .3, 1]])
    om = np.array([0.02, 0.03, 0.025, 0.04])
    al = np.array([0.06, 0.08, 0.07, 0.09])
    ga = np.array([0.08, 0.06, 0.07, 0.05])
    be = np.array([0.88, 0.86, 0.87, 0.85])
    mu = np.array([0.04, 0.03, 0.05, 0.02])
    h = om / (1 - al - be - ga / 2)
    Q = Rbar.copy()
    z_prev = np.zeros(k)
    e_prev = np.zeros(k)
    R = np.zeros((T, k))
    for t in range(T):
        if t:
            Q = (1 - a - b) * Rbar + a * np.outer(z_prev, z_prev) + b * Q
            h = om + al * e_prev ** 2 + ga * e_prev ** 2 * (e_prev < 0) + be * h
        d = np.sqrt(np.diag(Q))
        Rt = Q / np.outer(d, d)
        nu = 6
        z = np.linalg.cholesky(Rt) @ (rng.standard_t(nu, k) / np.sqrt(nu / (nu - 2)))
        e = np.sqrt(h) * z
        R[t] = mu + e
        z_prev, e_prev = z, e
    prices = 100 * np.exp(np.cumsum(R / 100, axis=0))
    idx = 1000 * np.exp(np.cumsum(R.mean(1) / 100 * 0.9 + 0.002 * rng.normal(size=T), axis=0))
    df = pd.DataFrame(prices, index=dates, columns=["alpha", "beta", "gamma", "delta"]).round(4)
    df["index"] = idx.round(3)
    df["volume"] = (1e6 * np.exp(rng.normal(0, 0.3, T))).round(0)
    df.index.name = "date"
    df.to_csv(os.path.join(OUT, "stocks_daily.csv"))


def macro(rng):
    """Quarterly macro data: cointegrated consumption/income, inflation, interest rate."""
    dates = pd.date_range("1980-01-01", "2024-10-01", freq="QS")
    T = len(dates)
    inc = np.cumsum(0.5 + rng.normal(0, 1, T)) + 100
    cons = 5 + 0.9 * inc + np.zeros(T)
    u = 0.0
    for t in range(T):
        u = 0.7 * u + rng.normal(0, 1)
        cons[t] += u
    infl = np.zeros(T)
    rate = np.zeros(T)
    for t in range(1, T):
        infl[t] = 0.5 + 0.75 * infl[t - 1] + rng.normal(0, 0.5)
        rate[t] = 0.3 + 0.6 * rate[t - 1] + 0.35 * infl[t - 1] + rng.normal(0, 0.4)
    q = [f"{d.year}Q{(d.month - 1) // 3 + 1}" for d in dates]
    df = pd.DataFrame({"date": q, "cons": cons.round(3), "income": inc.round(3), "infl": infl.round(3),
                       "rate": rate.round(3)})
    df.to_csv(os.path.join(OUT, "macro_quarterly.csv"), index=False)


def portfolios(rng):
    """Monthly returns (percent) of 10 portfolios generated from a 5-factor model, plus a factor file
    in Kenneth French's CSV layout (for offline use)."""
    dates = pd.date_range("1990-01-01", "2024-12-01", freq="MS")
    T = len(dates)
    means = np.array([0.6, 0.2, 0.3, 0.25, 0.3])
    sds = np.array([4.4, 3.0, 3.0, 2.2, 2.0])
    C = np.array([[1, .25, -.2, -.25, -.35], [.25, 1, .0, -.3, -.05], [-.2, .0, 1, .1, .65],
                  [-.25, -.3, .1, 1, .1], [-.35, -.05, .65, .1, 1]])
    F = means + (np.linalg.cholesky(C) @ rng.normal(size=(5, T))).T * sds
    mom = 0.6 + 4.0 * rng.standard_t(5, T) / np.sqrt(5 / 3)
    rf = np.clip(0.4 + np.cumsum(rng.normal(0, 0.02, T)), 0, 0.6)
    lines = ["This file is SIMULATED demo data in the layout of the Kenneth French data library.", "",
             ",Mkt-RF,SMB,HML,RMW,CMA,RF"]
    for d, f, r in zip(dates, F, rf):
        lines.append(f"{d.year}{d.month:02d}," + ",".join(f"{v:.2f}" for v in f) + f",{r:.2f}")
    lines += ["", " Annual Factors: January-December ", ",Mkt-RF,SMB,HML,RMW,CMA,RF", "", "Simulated data"]
    with open(os.path.join(OUT, "ff5_factors_simulated.csv"), "w") as fh:
        fh.write("\n".join(lines))
    lines = ["SIMULATED momentum factor", "", ",Mom   "]
    for d, m in zip(dates, mom):
        lines.append(f"{d.year}{d.month:02d},{m:.2f}")
    with open(os.path.join(OUT, "mom_factor_simulated.csv"), "w") as fh:
        fh.write("\n".join(lines))
    B = np.column_stack([rng.normal(1.0, 0.15, 10), np.linspace(-0.3, 1.2, 10), np.linspace(1.0, -0.4, 10),
                         rng.normal(0.2, 0.2, 10), rng.normal(0.1, 0.2, 10)])
    alpha = np.zeros(10)
    alpha[[0, 9]] = [0.25, -0.2]
    P = rf[:, None] + alpha + F @ B.T + rng.normal(0, 1.4, (T, 10))
    df = pd.DataFrame(P.round(3), columns=[f"p{i + 1}" for i in range(10)])
    df.insert(0, "date", [f"{d.year}-{d.month:02d}" for d in dates])
    df.to_csv(os.path.join(OUT, "portfolios_monthly.csv"), index=False)


def firm_panel(rng):
    """Annual firm panel: investment dynamics with firm effects."""
    rows = []
    for i in range(150):
        eta = rng.normal(0, 0.05)
        size = rng.normal(6, 1.2)
        inv = 0.1 + eta
        q = 1 + abs(rng.normal(0, 0.5))
        for t in range(2000, 2024):
            q = max(0.2, 0.7 * q + 0.3 * (1.5 + eta * 5) + rng.normal(0, 0.25))
            cf = 0.08 + 0.5 * eta + rng.normal(0, 0.04)
            lev = np.clip(0.3 + rng.normal(0, 0.1) - 0.5 * eta, 0, 0.95)
            size += 0.03 + rng.normal(0, 0.1)
            inv = 0.02 + 0.45 * inv + 0.03 * q + 0.25 * cf - 0.05 * lev + eta + rng.normal(0, 0.03)
            ret = 0.08 + 0.5 * (inv - 0.1) + rng.normal(0, 0.25)
            rows.append((f"firm{i + 1:03d}", t, round(inv, 5), round(q, 4), round(cf, 5), round(lev, 4),
                         round(size, 4), round(ret, 5)))
    df = pd.DataFrame(rows, columns=["firm", "year", "inv", "q", "cf", "lev", "size", "ret"])
    df.to_csv(os.path.join(OUT, "firm_panel.csv"), index=False)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(20261103)
    stocks(rng)
    macro(rng)
    portfolios(rng)
    firm_panel(rng)
    print("Demo data written to", OUT)
