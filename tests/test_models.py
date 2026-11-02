"""Model validation against reference implementations and simulated data with known parameters."""
import warnings

import numpy as np
import pandas as pd
import pytest

from finomania.core.expr import Evaluator
from finomania.core.workfile import Workfile

warnings.simplefilter("ignore")


def _wf_u(n, **series):
    wf = Workfile.create("u", n)
    for k, v in series.items():
        wf.set_series(k, v)
    return wf, Evaluator(wf)


def test_ols_matches_statsmodels():
    import statsmodels.api as sm
    from finomania.models.regression import LSResult
    rng = np.random.default_rng(0)
    x, z = rng.normal(size=200), rng.normal(size=200)
    y = 1 + 2 * x - 0.5 * z + rng.normal(size=200)
    wf, ev = _wf_u(200, x=x, z=z, y=y)
    eq = LSResult(wf, ev, "y c x z")
    ref = sm.OLS(y, sm.add_constant(np.column_stack([x, z]))).fit()
    np.testing.assert_allclose(eq.params, ref.params, rtol=1e-10)
    np.testing.assert_allclose(eq.bse, ref.bse, rtol=1e-10)
    assert eq.stats_dict["r2"] == pytest.approx(ref.rsquared)
    assert eq.stats_dict["logl"] == pytest.approx(ref.llf)
    assert eq.stats_dict["aic"] == pytest.approx(ref.aic / 200)   # EViews per-observation AIC


@pytest.fixture(scope="module")
def gjr_data():
    rng = np.random.default_rng(42)
    T = 2000
    om, a, g, b = 0.02, 0.05, 0.10, 0.88
    e, h = np.zeros(T), np.zeros(T)
    h[0] = om / (1 - a - b - g / 2)
    z = rng.standard_t(6, T) / np.sqrt(1.5)
    for t in range(T):
        if t:
            h[t] = om + a * e[t - 1] ** 2 + g * e[t - 1] ** 2 * (e[t - 1] < 0) + b * h[t - 1]
        e[t] = np.sqrt(h[t]) * z[t]
    return 0.05 + e


@pytest.mark.parametrize("model,dist,kw", [
    ("garch", "normal", dict(vol="GARCH", p=1, q=1)),
    ("tarch", "t", dict(vol="GARCH", p=1, o=1, q=1, dist="t")),
    ("egarch", "normal", dict(vol="EGARCH", p=1, o=1, q=1)),
    ("parch", "ged", dict(vol="APARCH", p=1, o=1, q=1, dist="ged")),
])
def test_native_garch_matches_arch_package(gjr_data, model, dist, kw):
    from arch import arch_model
    from finomania.models.garch import GarchResult
    wf, ev = _wf_u(len(gjr_data), r=gjr_data)
    r = GarchResult(wf, ev, "r c", model=model, dist=dist)
    ref = arch_model(gjr_data, mean="Constant", **kw).fit(disp="off")
    assert r.llf == pytest.approx(ref.loglikelihood, abs=0.05)
    if model != "egarch":     # EGARCH constant differs by the E|z| convention (EViews form)
        np.testing.assert_allclose(r.params, ref.params.values, rtol=2e-2, atol=2e-3)


def test_garch_m_and_forecast(gjr_data):
    from finomania.models.garch import GarchResult
    wf, ev = _wf_u(len(gjr_data), r=gjr_data)
    r = GarchResult(wf, ev, "r c ar(1)", model="tarch", archm="sd", dist="t")
    assert r.fit.converged
    out = r.proc_forecast(10)
    assert wf.nobs == len(gjr_data) + 10
    assert np.isfinite(wf.series("r_vf").iloc[-10:]).all()


def test_dcc_recovers_parameters():
    from finomania.models.mgarch import DCCResult
    rng = np.random.default_rng(3)
    T, k, a, b = 3000, 3, 0.04, 0.93
    Rbar = np.array([[1, .5, .3], [.5, 1, .4], [.3, .4, 1]])
    Q, Z = Rbar.copy(), np.zeros((T, k))
    h, e = np.ones((T, k)), np.zeros((T, k))
    for t in range(T):
        if t:
            Q = (1 - a - b) * Rbar + a * np.outer(Z[t - 1], Z[t - 1]) + b * Q
            h[t] = 0.05 + 0.08 * e[t - 1] ** 2 + 0.9 * h[t - 1]
        d = np.sqrt(np.diag(Q))
        Z[t] = np.linalg.cholesky(Q / np.outer(d, d)) @ rng.normal(size=k)
        e[t] = np.sqrt(h[t]) * Z[t]
    wf, ev = _wf_u(T, r1=e[:, 0], r2=e[:, 1], r3=e[:, 2])
    d = DCCResult(wf, ev, ["r1", "r2", "r3"])
    assert abs(d.p2[0] - a) < 0.015 and abs(d.p2[1] - b) < 0.03
    assert d.corr().shape == (T, 3, 3)


def test_panel_fe_matches_linearmodels():
    from linearmodels.panel import PanelOLS
    from finomania.models.panel import PanelResult
    rng = np.random.default_rng(1)
    rows = []
    for i in range(60):
        eta = rng.normal()
        for t in range(10):
            x = rng.normal() + eta
            rows.append((i, 2000 + t, 1 + 0.7 * x + eta + rng.normal(), x))
    df = pd.DataFrame(rows, columns=["id", "year", "y", "x"])
    wf = Workfile.from_dataframe(df, panel=("id", "year"))
    ev = Evaluator(wf)
    r = PanelResult(wf, ev, "y c x", "fe")
    d = df.set_index(["id", "year"])
    d.index = wf.data.index
    ref = PanelOLS(d["y"], pd.DataFrame({"C": 1.0, "X": d["x"]}), entity_effects=True).fit()
    assert r.params[1] == pytest.approx(float(ref.params["X"]))


def test_arellano_bond_recovers_rho():
    from finomania.models.panel import DPDResult
    rng = np.random.default_rng(5)
    rows = []
    for i in range(300):
        eta = rng.normal()
        y = eta / 0.5
        x = 0.0
        for t in range(18):
            x = 0.5 * x + rng.normal()
            y = 0.5 * y + 1.0 * x + eta + rng.normal()
            if t >= 10:
                rows.append((i, 2000 + t - 10, y, x))
    df = pd.DataFrame(rows, columns=["id", "year", "y", "x"])
    wf = Workfile.from_dataframe(df, panel=("id", "year"))
    g = DPDResult(wf, Evaluator(wf), "y x", lags=1, system=False, twostep=True)
    assert abs(g.theta[0] - 0.5) < 0.1 and abs(g.theta[1] - 1.0) < 0.1
    assert g.ar_tests[0][1] < 0.05          # AR(1) in differences expected
    assert g.ar_tests[1][1] > 0.01          # no AR(2)


def test_french_parser_and_grs(tmp_path):
    from finomania.models.factors import FactorResult, load_french
    rng = np.random.default_rng(2)
    dates = pd.date_range("2000-01-01", periods=240, freq="MS")
    f = rng.normal(0.5, 4, (240, 3))
    lines = ["header text", "", ",Mkt-RF,SMB,HML,RF"]
    lines += [f"{d.year}{d.month:02d}," + ",".join(f"{v:.2f}" for v in row) + ",0.20" for d, row in zip(dates, f)]
    lines += ["", " Annual Factors: January-December ", ",Mkt-RF,SMB,HML,RF", "2001,1,2,3,4"]
    p = tmp_path / "ff.csv"
    p.write_text("\n".join(lines))
    F = load_french("ff3", "m", str(p))
    assert F.shape == (240, 4) and F["rf"].iloc[0] == pytest.approx(0.002)
    wf = Workfile.create("m", "2000m1", "2019m12")
    for c in F.columns:
        wf.set_series(c, F[c].values)
    for i in range(5):
        wf.set_series(f"p{i}", F["rf"].values + F[["mkt_rf", "smb", "hml"]].values @ rng.normal(1, .3, 3)
                      + rng.normal(0, .01, 240))
    fr = FactorResult(wf, Evaluator(wf), [f"p{i}" for i in range(5)], ["mkt_rf", "smb", "hml"], "rf")
    g = fr.grs()
    assert 0 <= g["p"] <= 1 and g["N"] == 5
    assert "Lambda" in fr.view_fmb().to_text()


def test_numpy_lstm_gradients():
    from finomania.models.ml import NumpyLSTM
    rng = np.random.default_rng(0)
    net = NumpyLSTM(3, 5, 2, layers=2, seed=1)
    X, Y = rng.normal(size=(4, 6, 3)), rng.normal(size=(4, 2))

    def loss():
        return float(((net.forward(X) - Y) ** 2).mean())

    p = net.forward(X)
    grads = net.backward(2 * (p - Y) / (p - Y).size)
    for P, G in zip(net.params(), grads):
        idx = tuple(rng.integers(0, s) for s in P.shape)
        old = P[idx]
        P[idx] = old + 1e-6
        lp = loss()
        P[idx] = old - 1e-6
        lm = loss()
        P[idx] = old
        num = (lp - lm) / 2e-6
        assert abs(num - G[idx]) <= 1e-3 * max(1e-6, abs(num) + abs(G[idx]))


def test_lstm_learns_sine():
    from finomania.models.ml import MLResult
    t = np.arange(600)
    y = np.sin(t / 8) + 0.05 * np.random.default_rng(0).normal(size=600)
    wf, ev = _wf_u(600, y=y)
    m = MLResult(wf, ev, "y", method="lstm", lookback=16, epochs=30, hidden=16, backend="numpy", seed=1)
    assert m.metrics["Test"]["rmse"] < m.metrics["Test"]["rmse_naive"]


def test_var_backtest_statistics():
    from finomania.models.risk import backtest_report
    rng = np.random.default_rng(0)
    wf, ev = _wf_u(1200, r=rng.normal(0, 0.01, 1200))
    rep = backtest_report(wf, ev, "r", 0.99, 250, "gaussian")[0]
    txt = rep.to_text()
    assert "Kupiec" in txt and "Christoffersen" in txt


def test_unit_root_and_cointegration():
    from finomania.models.timeseries import engle_granger_test, johansen_test, unit_root_test
    rng = np.random.default_rng(4)
    x = np.cumsum(rng.normal(size=300))
    y = 0.8 * x + rng.normal(size=300)
    wf, ev = _wf_u(300, x=x, y=y)
    assert "Augmented Dickey-Fuller" in unit_root_test(wf, ev, "x").title
    assert "None *" in johansen_test(wf, ev, ["y", "x"], 1).to_text()
    assert "0.0000" in engle_granger_test(wf, ev, ["y", "x"]).to_text()
