"""
Machine-learning forecasting for time series.

* LSTM (and GRU with PyTorch) networks.  A dependency-free NumPy LSTM
  (stacked layers, BPTT, Adam, gradient clipping, early stopping) is built in;
  if PyTorch is installed and backend=torch (or auto) it is used instead.
* Scikit-learn comparators on the same lag windows: MLP, random forest,
  gradient boosting, support-vector regression, ridge.
* Proper out-of-sample evaluation: chronological train / validation / test
  split, scaling fitted on the training block only, RMSE / MAE / MAPE / R²
  / directional accuracy against naive and mean benchmarks and a
  Diebold-Mariano test.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
from scipy import stats

from ..report import FigureOutput, Report, fmt_num
from . import plots
from .base import ModelError, ModelResult, Param, View, contiguous_check, series_full


# ============================================================================
# NumPy LSTM
# ============================================================================

def _sig(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


class LSTMLayer:
    def __init__(self, n_in, n_h, rng):
        s = 1.0 / math.sqrt(n_h)
        self.Wx = rng.uniform(-s, s, (n_in, 4 * n_h))
        self.Wh = rng.uniform(-s, s, (n_h, 4 * n_h))
        self.b = np.zeros(4 * n_h)
        self.b[n_h:2 * n_h] = 1.0  # forget-gate bias
        self.H = n_h

    def params(self):
        return [self.Wx, self.Wh, self.b]

    def forward(self, X):
        B, L, _ = X.shape
        H = self.H
        h = np.zeros((B, H))
        c = np.zeros((B, H))
        self.cache = []
        out = np.empty((B, L, H))
        for t in range(L):
            z = X[:, t] @ self.Wx + h @ self.Wh + self.b
            i = _sig(z[:, :H])
            f = _sig(z[:, H:2 * H])
            g = np.tanh(z[:, 2 * H:3 * H])
            o = _sig(z[:, 3 * H:])
            c_new = f * c + i * g
            tc = np.tanh(c_new)
            h_new = o * tc
            self.cache.append((X[:, t], h, c, i, f, g, o, tc))
            h, c = h_new, c_new
            out[:, t] = h
        return out

    def backward(self, dH):
        """dH: gradient wrt all hidden outputs (B, L, H). Returns dX and fills self.grads."""
        B, L, H = dH.shape
        dWx = np.zeros_like(self.Wx)
        dWh = np.zeros_like(self.Wh)
        db = np.zeros_like(self.b)
        dX = np.empty((B, L, self.Wx.shape[0]))
        dh_next = np.zeros((B, H))
        dc_next = np.zeros((B, H))
        for t in reversed(range(L)):
            x, h_prev, c_prev, i, f, g, o, tc = self.cache[t]
            dh = dH[:, t] + dh_next
            do = dh * tc
            dc = dc_next + dh * o * (1 - tc * tc)
            di = dc * g
            dg = dc * i
            df = dc * c_prev
            dz = np.concatenate([di * i * (1 - i), df * f * (1 - f), dg * (1 - g * g), do * o * (1 - o)], axis=1)
            dWx += x.T @ dz
            dWh += h_prev.T @ dz
            db += dz.sum(0)
            dX[:, t] = dz @ self.Wx.T
            dh_next = dz @ self.Wh.T
            dc_next = dc * f
        self.grads = [dWx, dWh, db]
        return dX


class NumpyLSTM:
    def __init__(self, n_in, n_h=32, n_out=1, layers=1, seed=0):
        rng = np.random.default_rng(seed)
        self.layers = [LSTMLayer(n_in if k == 0 else n_h, n_h, rng) for k in range(layers)]
        s = 1.0 / math.sqrt(n_h)
        self.Wy = rng.uniform(-s, s, (n_h, n_out))
        self.by = np.zeros(n_out)

    def params(self):
        p = []
        for l in self.layers:
            p += l.params()
        return p + [self.Wy, self.by]

    def forward(self, X):
        h = X
        for l in self.layers:
            h = l.forward(h)
        self.hlast = h[:, -1]
        self.hseq_shape = h.shape
        return self.hlast @ self.Wy + self.by

    def backward(self, dy):
        dWy = self.hlast.T @ dy
        dby = dy.sum(0)
        dH = np.zeros(self.hseq_shape)
        dH[:, -1] = dy @ self.Wy.T
        for l in reversed(self.layers):
            dH = l.backward(dH)
        g = []
        for l in self.layers:
            g += l.grads
        return g + [dWy, dby]

    def predict(self, X, batch=1024):
        out = [self.forward(X[i:i + batch]) for i in range(0, len(X), batch)]
        return np.concatenate(out, axis=0)

    def get_state(self):
        return [p.copy() for p in self.params()]

    def set_state(self, st):
        for p, s in zip(self.params(), st):
            p[...] = s


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.p, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = [np.zeros_like(x) for x in params]
        self.v = [np.zeros_like(x) for x in params]
        self.t = 0

    def step(self, grads, clip=1.0):
        self.t += 1
        gn = math.sqrt(sum(float((g * g).sum()) for g in grads))
        scale = clip / gn if clip and gn > clip else 1.0
        for p, g, m, v in zip(self.p, grads, self.m, self.v):
            g = g * scale
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            mh = m / (1 - self.b1 ** self.t)
            vh = v / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


def train_numpy_lstm(Xtr, ytr, Xva, yva, hidden=32, layers=1, epochs=100, batch=32, lr=1e-3, patience=15, seed=0,
                     progress=None):
    net = NumpyLSTM(Xtr.shape[2], hidden, ytr.shape[1], layers, seed)
    opt = Adam(net.params(), lr)
    rng = np.random.default_rng(seed)
    hist = []
    best, best_state, bad = np.inf, net.get_state(), 0
    n = len(Xtr)
    for ep in range(int(epochs)):
        perm = rng.permutation(n)
        tl = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            yb = ytr[idx]
            pr = net.forward(Xtr[idx])
            err = pr - yb
            tl += float((err ** 2).sum())
            grads = net.backward(2 * err / err.size)
            opt.step(grads)
        tl /= ytr.size
        vl = float(((net.predict(Xva) - yva) ** 2).mean()) if len(Xva) else tl
        hist.append((tl, vl))
        if progress:
            progress(ep + 1, epochs, tl, vl)
        if vl < best - 1e-7:
            best, best_state, bad = vl, net.get_state(), 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.set_state(best_state)
    return net, np.array(hist)


def train_torch(Xtr, ytr, Xva, yva, cell="lstm", hidden=32, layers=1, epochs=100, batch=32, lr=1e-3, patience=15,
                seed=0, dropout=0.0, progress=None):
    import torch
    from torch import nn
    torch.manual_seed(seed)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            R = nn.LSTM if cell == "lstm" else nn.GRU
            self.rnn = R(Xtr.shape[2], hidden, num_layers=layers, batch_first=True,
                         dropout=dropout if layers > 1 else 0.0)
            self.fc = nn.Linear(hidden, ytr.shape[1])

        def forward(self, x):
            o, _ = self.rnn(x)
            return self.fc(o[:, -1])

    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.MSELoss()
    Xt, yt = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32)
    Xv, yv = torch.tensor(Xva, dtype=torch.float32), torch.tensor(yva, dtype=torch.float32)
    best, best_state, bad = np.inf, None, 0
    hist = []
    g = torch.Generator().manual_seed(seed)
    for ep in range(int(epochs)):
        net.train()
        perm = torch.randperm(len(Xt), generator=g)
        tl = 0.0
        for i in range(0, len(Xt), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = lossf(net(Xt[idx]), yt[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tl += float(loss) * len(idx)
        tl /= len(Xt)
        net.eval()
        with torch.no_grad():
            vl = float(lossf(net(Xv), yv)) if len(Xv) else tl
        hist.append((tl, vl))
        if progress:
            progress(ep + 1, epochs, tl, vl)
        if vl < best - 1e-7:
            best, best_state, bad = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state:
        net.load_state_dict(best_state)
    net.eval()

    class Wrap:
        def predict(self, X):
            with torch.no_grad():
                return net(torch.tensor(X, dtype=torch.float32)).numpy()

    return Wrap(), np.array(hist)


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


# ============================================================================
# result object
# ============================================================================

SK_METHODS = {"mlp", "rf", "gbm", "svr", "ridge", "linear"}


class MLResult(ModelResult):
    object_type = "ml"

    def __init__(self, wf, ev, target: str, features: list[str] | None = None, method: str = "lstm",
                 lookback: int = 20, horizon: int = 1, transform: str = "none", test: float = 0.2, val: float = 0.15,
                 hidden: int = 32, layers: int = 1, epochs: int = 100, batch: int = 32, lr: float = 1e-3,
                 patience: int = 15, seed: int = 0, backend: str = "auto", name: str = "", progress=None):
        self.wf, self.ev, self.name = wf, ev, name
        self.method = method.lower()
        self.target = target
        self.features = [f.lower() for f in (features or [])]
        self.L, self.Hz = int(lookback), int(horizon)
        self.transform = transform.lower()
        self.seed = int(seed)
        y = ev.eval(target, ignore_sample=True, as_series=True)
        cols = {"__y": y}
        for f in self.features:
            cols[f] = ev.eval(f, ignore_sample=True, as_series=True)
        df = pd.DataFrame(cols, index=wf.data.index)
        mask = wf.sample & df.notna().all(axis=1).values
        if not contiguous_check(mask):
            raise ModelError("ML forecasting requires a contiguous sample")
        self.mask = mask
        level = df["__y"].values[mask].astype(float)
        self.level = level
        z = self._fwd(level)
        feats = df[self.features].values[mask].astype(float) if self.features else np.zeros((len(level), 0))
        if self.transform != "none":
            feats = feats[1:]
        self.z = z
        n = len(z)
        L, Hz = self.L, self.Hz
        N = n - L - Hz + 1
        if N < 30:
            raise ModelError(f"Too few observations ({n}) for lookback={L} and horizon={Hz}")
        ntest = max(int(round(N * float(test))), 1) if float(test) > 0 else 0
        ntrv = N - ntest
        nval = max(int(round(ntrv * float(val))), 1) if float(val) > 0 else 0
        ntr = ntrv - nval
        # scaling from training block only (positions < ntr + L)
        tr_end = ntr + L
        self.mu, self.sd = float(z[:tr_end].mean()), float(z[:tr_end].std() or 1.0)
        if feats.shape[1]:
            self.fmu, self.fsd = feats[:tr_end].mean(0), feats[:tr_end].std(0)
            self.fsd[self.fsd == 0] = 1
            fz = (feats - self.fmu) / self.fsd
        else:
            fz = feats
        zs = (z - self.mu) / self.sd
        inp = np.column_stack([zs, fz])
        X = np.stack([inp[i:i + L] for i in range(N)])
        Y = np.stack([zs[i + L:i + L + Hz] for i in range(N)])
        self.X, self.Y = X, Y
        self.split = (ntr, nval, ntest)
        Xtr, Ytr = X[:ntr], Y[:ntr]
        Xva, Yva = X[ntr:ntr + nval], Y[ntr:ntr + nval]
        t0 = time.time()
        self.backend = "numpy"
        if self.method in ("lstm", "gru"):
            use_torch = (backend == "torch") or (backend == "auto" and torch_available())
            if self.method == "gru" and not use_torch:
                raise ModelError("GRU requires PyTorch (pip install torch); use method=lstm for the built-in engine")
            if use_torch:
                try:
                    self.model, self.hist = train_torch(Xtr, Ytr, Xva, Yva, self.method, hidden, layers, epochs, batch,
                                                        lr, patience, self.seed, progress=progress)
                    self.backend = "torch"
                except ImportError:
                    use_torch = False
            if not use_torch:
                self.model, self.hist = train_numpy_lstm(Xtr, Ytr, Xva, Yva, hidden, layers, epochs, batch, lr,
                                                         patience, self.seed, progress)
        elif self.method in SK_METHODS:
            self.model = self._sk_model(hidden)
            Xf = X.reshape(len(X), -1)
            self.model.fit(Xf[:ntr], Y[:ntr] if Hz > 1 else Y[:ntr].ravel())
            self.backend = "scikit-learn"
            self.hist = None
        else:
            raise ModelError(f"Unknown ML method '{method}' (lstm, gru, mlp, rf, gbm, svr, ridge)")
        self.train_time = time.time() - t0
        self.config = dict(hidden=hidden, layers=layers, epochs=epochs, batch=batch, lr=lr, patience=patience)
        self.pred_z = self._predict(X)
        self._evaluate()

    # ---- transforms
    def _fwd(self, level):
        t = self.transform
        if t == "none":
            return level.copy()
        if t == "diff":
            return np.diff(level)
        if t in ("dlog", "logdiff"):
            if (level <= 0).any():
                raise ModelError("dlog transform needs positive data")
            return np.diff(np.log(level))
        if t in ("pct", "return"):
            return level[1:] / level[:-1] - 1
        if t == "log":
            return np.log(level)
        raise ModelError("transform must be none, diff, dlog, pct or log")

    def _to_level(self, zpath, last_level):
        t = self.transform
        if t == "none":
            return zpath
        if t == "log":
            return np.exp(zpath)
        out = []
        lv = last_level
        for v in zpath:
            lv = lv + v if t == "diff" else lv * math.exp(v) if t in ("dlog", "logdiff") else lv * (1 + v)
            out.append(lv)
        return np.array(out)

    def _sk_model(self, hidden):
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from sklearn.linear_model import Ridge
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.neural_network import MLPRegressor
        from sklearn.svm import SVR
        m = self.method
        if m == "mlp":
            mod = MLPRegressor(hidden_layer_sizes=(hidden, hidden), max_iter=2000, early_stopping=True,
                               random_state=self.seed)
        elif m == "rf":
            mod = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=self.seed, n_jobs=1)
        elif m == "gbm":
            mod = GradientBoostingRegressor(random_state=self.seed)
        elif m == "svr":
            mod = SVR(C=1.0, epsilon=0.05)
        else:
            mod = Ridge(alpha=1.0)
        if self.Hz > 1 and m in ("gbm", "svr"):
            mod = MultiOutputRegressor(mod)
        return mod

    def _predict(self, X):
        if self.method in SK_METHODS:
            p = self.model.predict(X.reshape(len(X), -1))
            return p.reshape(len(X), -1)
        return np.asarray(self.model.predict(X)).reshape(len(X), -1)

    # ---- evaluation
    def _evaluate(self):
        ntr, nval, ntest = self.split
        P = self.pred_z * self.sd + self.mu       # transformed scale, (N, Hz)
        A = self.Y * self.sd + self.mu
        self.P, self.A = P, A
        res = {}
        for lab, sl in (("Train", slice(0, ntr)), ("Validation", slice(ntr, ntr + nval)),
                        ("Test", slice(ntr + nval, ntr + nval + ntest))):
            a, p = A[sl, 0], P[sl, 0]
            if len(a) < 2:
                continue
            e = a - p
            # naive benchmark: last observed value of the transformed series (random walk / no-change)
            Xl = self.X[sl][:, -1, 0] * self.sd + self.mu
            naive = Xl if self.transform == "none" or self.transform == "log" else np.zeros_like(a)
            mean_b = np.full_like(a, self.mu)
            en, em = a - naive, a - mean_b
            dirs = np.mean(np.sign(p - (Xl if self.transform in ("none", "log") else 0)) ==
                           np.sign(a - (Xl if self.transform in ("none", "log") else 0)))
            res[lab] = {"rmse": math.sqrt(np.mean(e ** 2)), "mae": float(np.mean(np.abs(e))),
                        "mape": float(np.mean(np.abs(e / a)) * 100) if np.all(a != 0) else np.nan,
                        "r2": 1 - np.sum(e ** 2) / np.sum((a - a.mean()) ** 2),
                        "dir": float(dirs), "rmse_naive": math.sqrt(np.mean(en ** 2)),
                        "rmse_mean": math.sqrt(np.mean(em ** 2)), "dm": _dm_test(e, en), "n": len(a)}
        self.metrics = res

    def report(self) -> Report:
        ntr, nval, ntest = self.split
        r = Report(f"ML Model: {self.name.upper()}" if self.name else "Machine-learning forecast model")
        meth = {"lstm": "LSTM recurrent neural network", "gru": "GRU recurrent neural network",
                "mlp": "Multi-layer perceptron", "rf": "Random forest", "gbm": "Gradient boosting",
                "svr": "Support vector regression", "ridge": "Ridge regression", "linear": "Ridge regression"}
        hdr = [f"Target: {self.target.upper()}" + (f" (transform: {self.transform})" if self.transform != "none" else ""),
               f"Method: {meth[self.method]}  [backend: {self.backend}]",
               f"Inputs: {self.L} lags of target" + (f" + {', '.join(f.upper() for f in self.features)}" if self.features else ""),
               f"Forecast horizon: {self.Hz} step(s) (direct multi-output)",
               f"Sample: {self.wf.sample_label(self.mask)}",
               f"Windows: train {ntr}, validation {nval}, test {ntest} (chronological split)",
               f"Training time: {self.train_time:.1f}s"]
        if self.method in ("lstm", "gru"):
            c = self.config
            hdr.append(f"Architecture: {c['layers']} layer(s) x {c['hidden']} units; Adam lr={c['lr']}, batch={c['batch']}, "
                       f"epochs run={len(self.hist)} (max {c['epochs']}, early-stopping patience {c['patience']})")
        r.header(hdr, stamp=True)
        rows = []
        for lab, m in self.metrics.items():
            rows.append([lab, m["n"], m["rmse"], m["mae"], m["mape"], m["r2"], m["dir"] * 100, m["rmse_naive"],
                         m["rmse_mean"], m["dm"][0], m["dm"][1]])
        r.table(["Block", "Obs", "RMSE", "MAE", "MAPE %", "R²", "Dir. acc. %", "RMSE naive", "RMSE mean", "DM stat",
                 "DM prob."], rows, ["str", "int", "num", "num", "f2", "f4", "f2", "num", "num", "f3", "prob"],
                title="One-step-ahead forecast accuracy (transformed scale)")
        r.note("Naive benchmark = no-change forecast (random walk on levels; zero change for differenced targets). "
               "DM: Diebold-Mariano test of equal squared-error loss vs the naive benchmark (negative = model better).")
        if "Test" in self.metrics:
            t = self.metrics["Test"]
            better = t["rmse"] < t["rmse_naive"]
            r.text(f"Out-of-sample: model RMSE {fmt_num(t['rmse'])} vs naive {fmt_num(t['rmse_naive'])} — "
                   + ("the model beats the naive benchmark." if better else "the model does NOT beat the naive benchmark."))
        return r

    def views(self):
        v = [View("output", "Model Summary", self.report),
             View("testplot", "Test-set Forecast vs Actual", self.view_test),
             View("fitplot", "Fitted vs Actual (all windows)", self.view_fit)]
        if self.hist is not None:
            v.append(View("loss", "Training / Validation Loss", self.view_loss))
        return v

    def procs(self):
        return [View("forecast", "Forecast Beyond Sample...", self.proc_forecast,
                     [Param("h", "Horizon", "int", max(self.Hz, 10)), Param("name", "Forecast name", "text",
                                                                           f"{_clean(self.target)}_ml")], kind="proc"),
                View("makefit", "Make Fitted Series", self.proc_makefit,
                     [Param("name", "Name", "text", f"{_clean(self.target)}_fit")], kind="proc")]

    def _window_index(self):
        idx = self.wf.data.index[self.mask]
        off = 1 if self.transform not in ("none", "log") else 0
        return idx[off + self.L: off + self.L + len(self.P)]

    def view_test(self):
        ntr, nval, ntest = self.split
        ix = self._window_index()
        sl = slice(ntr + nval, ntr + nval + ntest)
        fig, ax = plots.new_figure()
        ax.plot(ix[sl], self.A[sl, 0], color="#333", lw=1.1, label="Actual")
        ax.plot(ix[sl], self.P[sl, 0], color=plots.PALETTE[0], lw=1.3, label=f"{self.method.upper()} forecast")
        ax.legend(frameon=False)
        ax.set_title(f"Out-of-sample one-step forecasts: {self.target.upper()}")
        fig.tight_layout()
        return FigureOutput(fig, "Test forecasts")

    def view_fit(self):
        ntr, nval, ntest = self.split
        ix = self._window_index()
        fig, ax = plots.new_figure(7.5, 4.2)
        ax.plot(ix, self.A[:, 0], color="#333", lw=0.9, label="Actual")
        ax.plot(ix, self.P[:, 0], color=plots.PALETTE[0], lw=1.0, label="Fitted / forecast")
        for k, c in ((ntr, "#2ca02c"), (ntr + nval, "#d62728")):
            if 0 < k < len(ix):
                ax.axvline(ix[k], color=c, ls="--", lw=1)
        ax.legend(frameon=False)
        ax.set_title("Train | validation | test (dashed lines mark the splits)")
        fig.tight_layout()
        return FigureOutput(fig, "Fit")

    def view_loss(self):
        fig, ax = plots.new_figure(6.5, 4)
        ax.plot(self.hist[:, 0], label="Training MSE", color=plots.PALETTE[0])
        ax.plot(self.hist[:, 1], label="Validation MSE", color=plots.PALETTE[1])
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.legend(frameon=False)
        ax.set_title("Learning curve (standardised units)")
        fig.tight_layout()
        return FigureOutput(fig, "Loss")

    def proc_makefit(self, name=""):
        name = name or f"{_clean(self.target)}_fit"
        ix = self._window_index()
        s = pd.Series(np.nan, index=self.wf.data.index)
        pos = self.wf.data.index.get_indexer(ix)
        s.iloc[pos] = self.P[:, 0]
        self.wf.set_series(name, s, respect_sample=False)
        return f"One-step fitted values (transformed scale) saved to {name.upper()}"

    def proc_forecast(self, h=10, name=""):
        from .timeseries import auto_expand
        if self.features:
            raise ModelError("Out-of-sample forecasting beyond the data needs future feature values; "
                             "re-estimate with the target only for recursive forecasts")
        h = int(h)
        name = name or f"{_clean(self.target)}_ml"
        zs = list((self.z - self.mu) / self.sd)
        path = []
        while len(path) < h:
            w = np.array(zs[-self.L:])[None, :, None]
            p = self._predict(w)[0]
            for v in p:
                if len(path) < h:
                    path.append(v)
                    zs.append(v)
        zpath = np.array(path) * self.sd + self.mu
        lev = self._to_level(zpath, self.level[-1])
        fm = auto_expand(self.wf, h, self.mask)
        s = pd.Series(np.nan, index=self.wf.data.index)
        s.iloc[np.flatnonzero(fm)] = lev
        self.wf.set_series(name, s, respect_sample=False)
        r = Report(f"{self.method.upper()} forecast: {name.upper()}")
        labs = np.array(self.wf.labels())[fm]
        r.table(["Period", "Forecast (level)"] + (["Forecast (transformed)"] if self.transform != "none" else []),
                [[labs[i], lev[i]] + ([zpath[i]] if self.transform != "none" else []) for i in range(h)])
        r.note("Recursive multi-step forecast (predictions fed back as inputs)" if self.Hz == 1 else
               f"Direct {self.Hz}-step outputs, iterated beyond {self.Hz} steps")
        fig, ax = plots.new_figure()
        hist_idx = self.wf.data.index[self.mask][-min(len(self.level), 5 * h):]
        ax.plot(hist_idx, self.level[-len(hist_idx):], color="#333", lw=1, label=self.target.upper())
        ax.plot(self.wf.data.index[fm], lev, color=plots.PALETTE[0], lw=1.5, label="Forecast")
        ax.legend(frameon=False)
        ax.set_title(f"{self.method.upper()} out-of-sample forecast")
        fig.tight_layout()
        return [r, FigureOutput(fig, "ML forecast")]


def _clean(s):
    from ..core.workfile import sanitize_name
    return sanitize_name(s)


def _dm_test(e1, e2, h=1):
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    if n < 5 or np.allclose(d, 0):
        return (np.nan, np.nan)
    dbar = d.mean()
    g0 = np.var(d)
    v = g0
    for k in range(1, h):
        v += 2 * np.cov(d[k:], d[:-k])[0, 1]
    dm = dbar / math.sqrt(v / n)
    dm *= math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return (float(dm), float(2 * stats.t.sf(abs(dm), n - 1)))
