"""Dialogs.  Every dialog produces a command string that is executed by the
interpreter (and echoed in the command window, so work stays reproducible)."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QPlainTextEdit, QSpinBox, QTabWidget, QVBoxLayout, QWidget)

from ..models.base import Param
from .widgets import ParamForm

P = Param


# ============================================================================ estimation methods catalogue
COV_CHOICES = ["ordinary", "white", "hac", "hc0", "hc2", "hc3"]
DIST_CHOICES = ["normal", "t", "ged", "skewt"]

METHODS: dict[str, dict] = {
    "ls": {"label": "LS - Least Squares (NLS and ARMA)", "group": "Single equation",
           "hint": "Dependent variable followed by regressors (C = constant), ARMA terms allowed:\n"
                   "    y c x1 x2(-1) log(z) ar(1) ma(1)\nor an explicit formula for NLS:\n    y = c(1) + c(2)*x^c(3)",
           "params": [P("cov", "Coefficient covariance", "choice", "ordinary", COV_CHOICES + ["cluster"]),
                      P("lag", "HAC lag truncation (blank = automatic)", "text", ""),
                      P("cluster", "Cluster series (cov=cluster)", "text", ""),
                      P("w", "Weight series (WLS)", "text", "")]},
    "tsls": {"label": "TSLS - Two-Stage Least Squares (IV)", "group": "Single equation",
             "hint": "y c x1 x2 @ c z1 z2   (instruments after @)",
             "params": [P("cov", "Coefficient covariance", "choice", "ordinary", ["ordinary", "white", "hac"])]},
    "gmm": {"label": "GMM - Generalized Method of Moments", "group": "Single equation",
            "hint": "y c x1 x2 @ c z1 z2 z3", "params": [P("w", "Weighting matrix", "choice", "robust",
                                                          ["robust", "hac"])]},
    "qreg": {"label": "QREG - Quantile regression", "group": "Single equation", "hint": "y c x1 x2",
             "params": [P("quant", "Quantile", "text", "0.5"), P("cov", "Covariance", "choice", "robust",
                                                                 ["robust", "iid"])]},
    "logit": {"label": "LOGIT - Binary logit", "group": "Single equation", "hint": "y01 c x1 x2",
              "params": [P("cov", "Covariance", "choice", "ordinary", ["ordinary", "white"])]},
    "probit": {"label": "PROBIT - Binary probit", "group": "Single equation", "hint": "y01 c x1 x2",
               "params": [P("cov", "Covariance", "choice", "ordinary", ["ordinary", "white"])]},
    "count": {"label": "COUNT - Poisson / negative binomial", "group": "Single equation", "hint": "counts c x1",
              "params": [P("d", "Distribution", "choice", "p", ["p", "n"])]},
    "arch": {"label": "ARCH - Autoregressive Conditional Heteroskedasticity", "group": "Volatility",
             "hint": "Mean equation:  r c [x1 ...] [ar(1)] [ma(1)]\nOptional variance regressors after @:   r c @ z1 z2",
             "params": [P("model", "Model", "choice", "garch",
                          ["garch", "tarch", "egarch", "parch", "cgarch", "igarch", "figarch", "harch", "ewma"]),
                        P("p", "ARCH order", "int", 1), P("q", "GARCH order", "int", 1),
                        P("thrsh", "Threshold / asymmetric order", "int", 0),
                        P("dist", "Error distribution", "choice", "normal", DIST_CHOICES),
                        P("archm", "ARCH-M term", "choice", "none", ["none", "sd", "var", "log"]),
                        P("power", "PARCH power (blank = estimate)", "text", ""),
                        P("cov", "Coefficient covariance", "choice", "bw", ["bw", "opg", "hessian"])]},
    "dcc": {"label": "DCC - Dynamic conditional correlation GARCH", "group": "Volatility",
            "hint": "List of return series:   r1 r2 r3",
            "params": [P("model", "Univariate model", "choice", "garch", ["garch", "tarch", "egarch", "parch", "cgarch"]),
                       P("dist", "Univariate distribution", "choice", "normal", DIST_CHOICES),
                       P("mvdist", "Correlation-step distribution", "choice", "normal", ["normal", "t"]),
                       P("mean", "Mean equation", "choice", "c", ["c", "zero", "ar1"])]},
    "adcc": {"label": "ADCC - Asymmetric DCC-GARCH", "group": "Volatility", "hint": "r1 r2 r3",
             "always": ["model", "dist"],
             "params": [P("model", "Univariate model", "choice", "tarch", ["garch", "tarch", "egarch"]),
                        P("dist", "Univariate distribution", "choice", "t", DIST_CHOICES),
                        P("mvdist", "Correlation-step distribution", "choice", "normal", ["normal", "t"])]},
    "ccc": {"label": "CCC - Constant conditional correlation GARCH", "group": "Volatility", "hint": "r1 r2 r3",
            "params": [P("model", "Univariate model", "choice", "garch", ["garch", "tarch", "egarch"]),
                       P("dist", "Univariate distribution", "choice", "normal", DIST_CHOICES)]},
    "bekk": {"label": "BEKK - BEKK(1,1) GARCH", "group": "Volatility", "hint": "r1 r2 [r3]",
             "params": [P("type", "BEKK type", "choice", "diagonal", ["diagonal", "scalar"])]},
    "arima": {"label": "ARIMA - (Seasonal) ARIMA", "group": "Time series",
              "hint": "y [c] [exogenous regressors]",
              "params": [P("p", "AR order p", "int", 1), P("d", "Differencing d", "int", 0), P("q", "MA order q", "int", 0),
                         P("sp", "Seasonal AR P", "int", 0), P("sd", "Seasonal diff D", "int", 0),
                         P("sq", "Seasonal MA Q", "int", 0), P("s", "Season length s", "int", 0)]},
    "autoarima": {"label": "AUTOARIMA - Automatic ARMA selection", "group": "Time series", "hint": "y c",
                  "params": [P("d", "Differencing d", "int", 0), P("maxp", "Max AR", "int", 4),
                             P("maxq", "Max MA", "int", 4), P("ic", "Criterion", "choice", "aic", ["aic", "bic", "hq"])]},
    "ets": {"label": "ETS - Exponential smoothing", "group": "Time series", "hint": "y",
            "params": [P("error", "Error", "choice", "add", ["add", "mul"]),
                       P("trend", "Trend", "choice", "none", ["none", "add", "mul"]),
                       P("damped", "Damped trend", "bool", False),
                       P("seasonal", "Seasonal", "choice", "none", ["none", "add", "mul"]),
                       P("periods", "Seasonal periods (blank = auto)", "text", "")]},
    "var": {"label": "VAR - Vector autoregression", "group": "Time series",
            "hint": "Endogenous series, optional exogenous after @:   y1 y2 y3 @ x",
            "params": [P("lags", "Lag order", "int", 2), P("trend", "Deterministic", "choice", "c", ["c", "ct", "n"])]},
    "vec": {"label": "VEC - Vector error correction", "group": "Time series", "hint": "y1 y2 y3",
            "params": [P("rank", "Cointegrating rank", "int", 1), P("lags", "Lags in differences", "int", 1),
                       P("det", "Deterministic (EViews case)", "choice", "co", ["n", "ci", "co", "li", "lo"])]},
    "ardl": {"label": "ARDL - Autoregressive distributed lag + bounds test", "group": "Time series", "hint": "y x1 x2",
             "params": [P("maxlag", "Max lags of y", "int", 4), P("maxdl", "Max lags of regressors", "int", 4),
                        P("ic", "Criterion", "choice", "aic", ["aic", "bic", "hqic"]),
                        P("case", "Bounds-test case", "int", 3)]},
    "switchreg": {"label": "SWITCHREG - Markov switching regression", "group": "Time series", "hint": "y c [x]",
                  "params": [P("k", "Regimes", "int", 2), P("ar", "AR order", "int", 0),
                             P("switchvar", "Regime-specific variance", "bool", True)]},
    "panel": {"label": "PANEL - Static panel regression", "group": "Panel", "always": ["method", "cov"],
              "hint": "y c x1 x2   (panel workfile required)",
              "params": [P("method", "Estimator", "choice", "fe", ["pooled", "fe", "te", "twoway", "re", "be", "fd", "fm"]),
                         P("cov", "Covariance", "choice", "cluster",
                           ["ordinary", "white", "cluster", "cluster_time", "cluster2", "dk"])]},
    "dpd": {"label": "DPD - Dynamic panel GMM (Arellano-Bond / Blundell-Bond)", "group": "Panel",
            "hint": "y x1 x2   (lags of y added automatically)",
            "params": [P("lags", "Lags of dependent variable", "int", 1), P("sys", "System GMM (Blundell-Bond)", "bool", False),
                       P("twostep", "Two-step (Windmeijer SE)", "bool", True),
                       P("endog", "Endogenous regressors", "text", ""), P("predet", "Predetermined regressors", "text", ""),
                       P("maxlag", "Max GMM instrument lag", "int", 99), P("collapse", "Collapse instruments", "bool", False),
                       P("timedum", "Time dummies", "bool", True)]},
    "ff": {"label": "FF - Fama-French factor model", "group": "Asset pricing",
           "hint": "Asset return series (same units as factors):   p1 p2 p3 ...",
           "params": [P("model", "Model", "choice", "ff3", ["capm", "ff3", "carhart", "ff5", "ff6"]),
                      P("freq", "Factor frequency (blank = workfile)", "text", ""),
                      P("units", "Factor units", "choice", "decimal", ["decimal", "pct"]),
                      P("cov", "Standard errors", "choice", "ordinary", ["ordinary", "white", "hac"]),
                      P("file", "Local factor file (optional)", "text", "")]},
    "factor": {"label": "FACTOR - Custom factor regressions", "group": "Asset pricing", "hint": "p1 p2 p3",
               "params": [P("factors", "Factor series", "text", ""), P("rf", "Risk-free series (optional)", "text", ""),
                          P("cov", "Standard errors", "choice", "ordinary", ["ordinary", "white", "hac"])]},
    "fmb": {"label": "FMB - Fama-MacBeth two-pass", "group": "Asset pricing",
            "hint": "Test-asset returns:  p1 p2 ... p10   (panel workfile: y c x1 x2)",
            "params": [P("model", "FF model (if no factors given)", "choice", "ff3", ["capm", "ff3", "carhart", "ff5", "ff6"]),
                       P("factors", "Factor series (optional)", "text", ""), P("rf", "Risk-free series", "text", ""),
                       P("nw", "Newey-West lags", "int", 0), P("window", "Rolling beta window (0 = full)", "int", 0)]},
    "lstm": {"label": "LSTM - Long short-term memory network", "group": "Machine learning",
             "hint": "Target series [feature series ...]",
             "params": [P("lookback", "Lookback window", "int", 20), P("horizon", "Forecast horizon", "int", 1),
                        P("hidden", "Hidden units", "int", 32), P("layers", "Layers", "int", 1),
                        P("epochs", "Max epochs", "int", 100), P("lr", "Learning rate", "text", "0.001"),
                        P("batch", "Batch size", "int", 32),
                        P("transform", "Target transform", "choice", "none", ["none", "diff", "dlog", "pct", "log"]),
                        P("test", "Test fraction", "text", "0.2"), P("seed", "Random seed", "int", 0),
                        P("backend", "Backend", "choice", "auto", ["auto", "numpy", "torch"])]},
    "ml": {"label": "ML - Machine-learning comparators", "group": "Machine learning",
           "hint": "Target series [feature series ...]",
           "params": [P("method", "Method", "choice", "rf", ["rf", "gbm", "mlp", "svr", "ridge"]),
                      P("lookback", "Lookback window", "int", 20), P("horizon", "Horizon", "int", 1),
                      P("transform", "Target transform", "choice", "none", ["none", "diff", "dlog", "pct", "log"]),
                      P("test", "Test fraction", "text", "0.2")]},
}
POSITIONAL = {}


class EstimationDialog(QDialog):
    """EViews 'Equation Estimation' dialog generalised to every Finomania estimator."""

    def __init__(self, parent=None, method="ls", spec="", name="", sample="", methods: list[str] | None = None):
        super().__init__(parent)
        self.setWindowTitle("Estimation")
        self.resize(640, 560)
        lay = QVBoxLayout(self)
        tabs = QTabWidget()
        lay.addWidget(tabs)
        page = QWidget()
        pl = QVBoxLayout(page)
        tabs.addTab(page, "Specification")
        self.hint = QLabel()
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color:#555")
        g1 = QGroupBox("Equation specification")
        g1l = QVBoxLayout(g1)
        g1l.addWidget(self.hint)
        self.spec = QPlainTextEdit(spec)
        self.spec.setFixedHeight(90)
        g1l.addWidget(self.spec)
        pl.addWidget(g1)
        g2 = QGroupBox("Estimation settings")
        fl = QFormLayout(g2)
        self.method = QComboBox()
        keys = methods or list(METHODS)
        for k in keys:
            self.method.addItem(METHODS[k]["label"], k)
        self.method.currentIndexChanged.connect(self._method_changed)
        fl.addRow("Method:", self.method)
        self.sample = QLineEdit(sample)
        fl.addRow("Sample:", self.sample)
        self.name = QLineEdit(name)
        self.name.setPlaceholderText("optional — store as a named object, e.g. eq1")
        fl.addRow("Object name:", self.name)
        pl.addWidget(g2)
        self.opt_page = QWidget()
        self.opt_layout = QVBoxLayout(self.opt_page)
        tabs.addTab(self.opt_page, "Options")
        self.form = None
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        if method in keys:
            self.method.setCurrentIndex(keys.index(method))
        self._method_changed()
        self.tabs = tabs
        self.preset = None

    def _method_changed(self):
        k = self.method.currentData()
        m = METHODS[k]
        self.hint.setText(m["hint"])
        if self.form is not None:
            self.form.setParent(None)
        self.form = ParamForm(m["params"])
        self.opt_layout.addWidget(self.form)
        self.opt_layout.addStretch()

    def set_options(self, kw: dict):
        if self.form is None:
            return
        for k, v in kw.items():
            w = self.form.widgets.get(k)
            if w is None:
                continue
            from PySide6.QtWidgets import QCheckBox, QComboBox as C, QSpinBox as S
            if isinstance(w, QCheckBox):
                w.setChecked(str(v).lower() in ("1", "true", "yes"))
            elif isinstance(w, C):
                w.setCurrentText(str(v))
            elif isinstance(w, S):
                try:
                    w.setValue(int(float(v)))
                except ValueError:
                    pass
            else:
                w.setText(str(v))

    def command(self) -> tuple[str, str]:
        k = self.method.currentData()
        opts = self.form.options_string(skip_defaults=True, always=METHODS[k].get("always"))
        if k == "arch":
            vals = self.form.values()
            o = [f"{vals['p']}", f"{vals['q']}"]
            if vals["model"] != "garch":
                o.append(vals["model"])
            if vals["thrsh"]:
                o.append(f"thrsh={vals['thrsh']}")
            if vals["dist"] != "normal":
                o.append(f"dist={vals['dist']}")
            if vals["archm"] != "none":
                o.append(f"archm={vals['archm']}")
            if vals["power"]:
                o.append(f"power={vals['power']}")
            if vals["cov"] != "bw":
                o.append(f"cov={vals['cov']}")
            opts = ", ".join(o)
        if k == "var":
            vals = self.form.values()
            opts = f"lags={vals['lags']}" + (f", trend={vals['trend']}" if vals["trend"] != "c" else "")
        spec = " ".join(self.spec.toPlainText().split())
        name = self.name.text().strip()
        head = f"{k}({opts})" if opts else k
        cmd = f"equation {name}.{head} {spec}" if name else f"{head} {spec}"
        smpl = self.sample.text().strip()
        return cmd, smpl


# ============================================================================ generic command dialog
class CommandDialog(QDialog):
    """Dialog for a command with a series list and options, e.g. unit-root tests."""

    def __init__(self, parent, title: str, command: str, params: list[Param], series: str = "",
                 series_label: str = "Series / group", positional: list[str] | None = None, note: str = ""):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.cmdname = command
        self.positional = positional or []
        lay = QVBoxLayout(self)
        fl = QFormLayout()
        self.series = QLineEdit(series)
        if series_label:
            fl.addRow(series_label + ":", self.series)
        lay.addLayout(fl)
        self.form = ParamForm(params)
        if params:
            g = QGroupBox("Options")
            gl = QVBoxLayout(g)
            gl.addWidget(self.form)
            lay.addWidget(g)
        if note:
            lab = QLabel(note)
            lab.setWordWrap(True)
            lab.setStyleSheet("color:#555")
            lay.addWidget(lab)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def command(self) -> str:
        opts = self.form.options_string(skip_defaults=False, positional=self.positional)
        head = f"{self.cmdname}({opts})" if opts else self.cmdname
        return f"{head} {self.series.text().strip()}".strip()


TEST_DIALOGS = {
    "uroot": ("Unit Root Test", [P("test", "Test type", "choice", "adf", ["adf", "pp", "kpss", "dfgls", "za"]),
                                 P("trend", "Include in test equation", "choice", "c", ["c", "ct", "n"]),
                                 P("dif", "Test for unit root in (0=level, 1=1st diff, 2=2nd diff)", "int", 0),
                                 P("ic", "Lag selection criterion", "choice", "sic", ["sic", "aic", "tstat"])],
              ["test", "trend"]),
    "urootsum": ("Unit Root Summary", [P("trend", "Deterministic", "choice", "c", ["c", "ct"])], []),
    "coint": ("Cointegration Test", [P("method", "Method", "choice", "johansen", ["johansen", "eg", "po"]),
                                     P("lags", "Lags (Johansen, in differences)", "int", 1),
                                     P("det", "Deterministic (Johansen)", "choice", "c", ["n", "c", "ct"])], ["method"]),
    "granger": ("Granger Causality", [P("lags", "Lags", "int", 2)], []),
    "dhcause": ("Dumitrescu-Hurlin Panel Causality", [P("lags", "Lags", "int", 1)], []),
    "correl": ("Correlogram", [P("lags", "Lags to include", "int", 24)], ["lags"]),
    "normtest": ("Normality Tests", [], []),
    "bds": ("BDS Independence Test", [P("dim", "Max embedding dimension", "int", 6)], []),
    "vratio": ("Variance Ratio Test", [P("lags", "Periods", "text", "2 4 8 16"), P("log", "Use log of series", "bool", False)],
               []),
    "lbq": ("Ljung-Box Q Test", [P("lags", "Lags", "int", 10)], []),
    "ttest": ("Test of Mean", [P("mu", "Hypothesised mean", "text", "0")], []),
    "eqtest": ("Equality Tests by Series", [], []),
    "stats": ("Descriptive Statistics", [], []),
    "cor": ("Correlation Matrix", [P("method", "Method", "choice", "pearson", ["pearson", "spearman", "kendall"]),
                                   P("tests", "Show t-tests", "bool", False)], []),
    "cov": ("Covariance Matrix", [], []),
    "pca": ("Principal Components", [P("cov", "Use covariance (not correlation)", "bool", False),
                                     P("scores", "Save scores with prefix", "text", "")], []),
    "hpf": ("Hodrick-Prescott Filter", [P("lambda", "Smoothing parameter (blank = default)", "text", "")], []),
    "risk": ("Value-at-Risk / Expected Shortfall", [P("levels", "Confidence levels", "text", "0.95 0.99"),
                                                    P("methods", "Methods", "text",
                                                      "historical gaussian cornish t ewma garch fhs"),
                                                    P("value", "Position value (optional)", "text", "")], []),
    "backtest": ("VaR Backtest", [P("level", "Confidence level", "text", "0.99"), P("window", "Rolling window", "int", 250),
                                  P("method", "Method", "choice", "historical",
                                    ["historical", "gaussian", "cornish", "t", "ewma", "garch", "fhs"]),
                                  P("refit", "GARCH refit frequency", "int", 20)], []),
    "perf": ("Performance Analytics", [P("rf", "Risk-free (series or number)", "text", ""),
                                       P("bench", "Benchmark series", "text", ""),
                                       P("periods", "Periods per year (blank = auto)", "text", "")], []),
}


def test_dialog(parent, key: str, series: str = "") -> CommandDialog:
    title, params, pos = TEST_DIALOGS[key]
    return CommandDialog(parent, title, key, params, series, positional=pos)


# ============================================================================ workfile dialogs
class NewWorkfileDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Workfile Create")
        lay = QVBoxLayout(self)
        fl = QFormLayout()
        self.freq = QComboBox()
        for k, lab in (("a", "Annual"), ("q", "Quarterly"), ("m", "Monthly"), ("w", "Weekly"),
                       ("d5", "Daily - 5 day week"), ("d7", "Daily - 7 day week"), ("u", "Undated / irregular")):
            self.freq.addItem(lab, k)
        self.freq.setCurrentIndex(2)
        self.start = QLineEdit("2000m01")
        self.end = QLineEdit("2024m12")
        self.freq.currentIndexChanged.connect(self._fr)
        self.name = QLineEdit("untitled")
        self.ncross = QSpinBox()
        self.ncross.setRange(0, 100000)
        fl.addRow("Frequency:", self.freq)
        fl.addRow("Start date / first obs:", self.start)
        fl.addRow("End date / last obs:", self.end)
        fl.addRow("Panel cross-sections (0 = none):", self.ncross)
        fl.addRow("Workfile name:", self.name)
        lay.addLayout(fl)
        lab = QLabel("Dates: 2010, 2010q1, 2010m3 or 2010-03-15.  Undated: start = 1, end = number of observations.")
        lab.setWordWrap(True)
        lab.setStyleSheet("color:#555")
        lay.addWidget(lab)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _fr(self):
        k = self.freq.currentData()
        ex = {"a": ("1980", "2024"), "q": ("1990q1", "2024q4"), "m": ("2000m01", "2024m12"),
              "w": ("2015-01-05", "2024-12-30"), "d5": ("2015-01-01", "2024-12-31"), "d7": ("2020-01-01", "2024-12-31"),
              "u": ("1", "100")}[k]
        self.start.setText(ex[0])
        self.end.setText(ex[1])

    def command(self) -> str:
        k = self.freq.currentData()
        opts = []
        if self.name.text().strip() and self.name.text().strip() != "untitled":
            opts.append(f"name={self.name.text().strip()}")
        if self.ncross.value():
            opts.append(f"panel={self.ncross.value()}")
        o = f"({', '.join(opts)})" if opts else ""
        return f"wfcreate{o} {k} {self.start.text().strip()} {self.end.text().strip()}"


class FetchDialog(QDialog):
    def __init__(self, parent=None, source="yahoo"):
        super().__init__(parent)
        self.setWindowTitle("Fetch data from the web")
        lay = QVBoxLayout(self)
        fl = QFormLayout()
        self.source = QComboBox()
        self.source.addItem("Yahoo Finance (prices)", "yahoo")
        self.source.addItem("FRED (St. Louis Fed)", "fred")
        self.source.addItem("Fama-French factors (Ken French library)", "ff")
        self.source.setCurrentIndex({"yahoo": 0, "fred": 1, "ff": 2}.get(source, 0))
        self.symbols = QLineEdit("AAPL MSFT ^GSPC")
        self.start = QLineEdit("2015-01-01")
        self.end = QLineEdit("")
        self.field = QComboBox()
        self.field.addItems(["adjclose", "close", "open", "high", "low", "volume"])
        self.model = QComboBox()
        self.model.addItems(["ff3", "ff5", "carhart", "ff6", "capm"])
        self.freq = QComboBox()
        self.freq.addItems(["m", "d", "w"])
        fl.addRow("Source:", self.source)
        fl.addRow("Symbols / series codes:", self.symbols)
        fl.addRow("Start date:", self.start)
        fl.addRow("End date (blank = today):", self.end)
        fl.addRow("Price field (Yahoo):", self.field)
        fl.addRow("Factor model (Fama-French):", self.model)
        fl.addRow("Factor frequency:", self.freq)
        lay.addLayout(fl)
        self.source.currentIndexChanged.connect(self._src)
        self._src()
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _src(self):
        s = self.source.currentData()
        if s == "fred" and self.symbols.text().startswith("AAPL"):
            self.symbols.setText("GDP CPIAUCSL FEDFUNDS")
        if s == "yahoo" and self.symbols.text().startswith("GDP"):
            self.symbols.setText("AAPL MSFT ^GSPC")
        self.symbols.setEnabled(s != "ff")
        self.field.setEnabled(s == "yahoo")
        self.model.setEnabled(s == "ff")
        self.freq.setEnabled(s == "ff")

    def command(self) -> str:
        s = self.source.currentData()
        if s == "ff":
            return f"ffget(model={self.model.currentText()}, freq={self.freq.currentText()})"
        opts = [f"source={s}"]
        if self.start.text().strip():
            opts.append(f"start={self.start.text().strip()}")
        if self.end.text().strip():
            opts.append(f"end={self.end.text().strip()}")
        if s == "yahoo" and self.field.currentText() != "adjclose":
            opts.append(f"field={self.field.currentText()}")
        return f"fetch({', '.join(opts)}) {self.symbols.text().strip()}"


class GenrDialog(QDialog):
    def __init__(self, parent=None, sample=""):
        super().__init__(parent)
        self.setWindowTitle("Generate Series by Equation")
        self.resize(520, 220)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Enter equation:"))
        self.eq = QLineEdit()
        self.eq.setPlaceholderText("e.g.  r = 100*dlog(price)     or     vol = @movstdev(r, 20)")
        lay.addWidget(self.eq)
        fl = QFormLayout()
        self.sample = QLineEdit(sample)
        fl.addRow("Sample:", self.sample)
        lay.addLayout(fl)
        lab = QLabel("Functions: log, exp, d(), dlog(), x(-1) lags, @mean, @stdev, @movav, @movstdev, @pch, @trend, "
                     "@recode(cond,a,b), nrnd, rnd, @year, @month ...")
        lab.setWordWrap(True)
        lab.setStyleSheet("color:#555")
        lay.addWidget(lab)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def command(self):
        return f"series {self.eq.text().strip()}", self.sample.text().strip()


class SampleDialog(QDialog):
    def __init__(self, parent=None, current="@all"):
        super().__init__(parent)
        self.setWindowTitle("Sample")
        lay = QVBoxLayout(self)
        fl = QFormLayout()
        rng, cond = current, ""
        if " if " in f" {current} ".lower():
            i = current.lower().index("if")
            rng, cond = current[:i].strip(), current[i + 2:].strip()
        self.rng = QLineEdit(rng or "@all")
        self.cond = QLineEdit(cond)
        fl.addRow("Sample range pairs (or @all):", self.rng)
        fl.addRow("IF condition (optional):", self.cond)
        lay.addLayout(fl)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def command(self):
        c = f"smpl {self.rng.text().strip() or '@all'}"
        if self.cond.text().strip():
            c += f" if {self.cond.text().strip()}"
        return c


class ParamDialog(QDialog):
    """Asks for the parameters of an object view / proc."""

    def __init__(self, parent, title: str, params: list[Param]):
        super().__init__(parent)
        self.setWindowTitle(title)
        lay = QVBoxLayout(self)
        self.form = ParamForm(params)
        lay.addWidget(self.form)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
