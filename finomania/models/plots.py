"""Matplotlib helpers producing clean, EViews-like graphs (thread-safe: they
never touch pyplot)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

PALETTE = ["#1f4e9c", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#17becf", "#e377c2",
           "#7f7f7f", "#bcbd22"]

RC = {
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
}


def new_figure(w: float = 7.0, h: float = 4.2, nrows: int = 1, ncols: int = 1, sharex=False, squeeze=True):
    import matplotlib as mpl
    with mpl.rc_context(RC):
        fig = Figure(figsize=(w, h), dpi=100)
        FigureCanvasAgg(fig)
        axes = fig.subplots(nrows, ncols, sharex=sharex, squeeze=squeeze)
    for ax in np.atleast_1d(axes).ravel():
        style_axes(ax)
    return fig, axes


def style_axes(ax):
    ax.grid(True, color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=8)


def _x(index):
    if isinstance(index, pd.MultiIndex):
        return np.arange(len(index))
    return index


def line_plot(data: dict[str, pd.Series] | pd.DataFrame, title: str = "", ylabel: str = "",
              multiple: bool = False, bands: dict | None = None, zero: bool = False):
    if isinstance(data, pd.DataFrame):
        data = {c: data[c] for c in data.columns}
    n = len(data)
    if multiple and n > 1:
        ncols = 2 if n > 1 else 1
        nrows = math.ceil(n / ncols)
        fig, axes = new_figure(7.5, 2.4 * nrows, nrows, ncols, squeeze=False)
        for k, (name, s) in enumerate(data.items()):
            ax = axes.ravel()[k]
            ax.plot(_x(s.index), s.values, color=PALETTE[0], lw=1.1)
            ax.set_title(name.upper(), fontsize=9)
            if zero:
                ax.axhline(0, color="#555", lw=0.6)
        for k in range(n, nrows * ncols):
            axes.ravel()[k].set_visible(False)
        fig.tight_layout()
        return fig
    fig, ax = new_figure()
    for k, (name, s) in enumerate(data.items()):
        ax.plot(_x(s.index), s.values, lw=1.2, color=PALETTE[k % len(PALETTE)], label=name.upper())
    if bands:
        for name, (lo, hi) in bands.items():
            ax.fill_between(_x(lo.index), lo.values, hi.values, color="#9fb8e3", alpha=0.35, lw=0, label=name)
    if zero:
        ax.axhline(0, color="#555", lw=0.6)
    if n > 1 or bands:
        ax.legend(frameon=False, ncol=min(n + (1 if bands else 0), 4), loc="best")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    fig.autofmt_xdate() if isinstance(next(iter(data.values())).index, pd.DatetimeIndex) else None
    fig.tight_layout()
    return fig


def bar_plot(data: dict[str, pd.Series], title=""):
    fig, ax = new_figure()
    n = len(data)
    first = next(iter(data.values()))
    x = np.arange(len(first))
    w = 0.8 / n
    for k, (name, s) in enumerate(data.items()):
        ax.bar(x + k * w, s.values, w, color=PALETTE[k % len(PALETTE)], label=name.upper())
    step = max(1, len(x) // 12)
    labels = [str(i.date()) if hasattr(i, "date") else str(i) for i in first.index]
    ax.set_xticks(x[::step])
    ax.set_xticklabels(labels[::step], rotation=45, ha="right")
    if n > 1:
        ax.legend(frameon=False)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def scatter_plot(x: pd.Series, y: pd.Series, title="", fit: bool = True, xname="", yname=""):
    fig, ax = new_figure(6, 4.5)
    ok = x.notna() & y.notna()
    ax.scatter(x[ok], y[ok], s=12, color=PALETTE[0], alpha=0.7, edgecolor="none")
    if fit and ok.sum() > 2:
        b = np.polyfit(x[ok].values, y[ok].values, 1)
        xs = np.linspace(x[ok].min(), x[ok].max(), 50)
        ax.plot(xs, b[0] * xs + b[1], color=PALETTE[1], lw=1.4,
                label=f"{yname.upper()} = {b[1]:.4g} + {b[0]:.4g}·{xname.upper()}")
        ax.legend(frameon=False)
    ax.set_xlabel(xname.upper())
    ax.set_ylabel(yname.upper())
    ax.set_title(title)
    fig.tight_layout()
    return fig


def hist_plot(v: np.ndarray, name: str, stats_text: str = "", bins: int | None = None, normal: bool = True):
    fig = Figure(figsize=(7.2, 4.0), dpi=100)
    FigureCanvasAgg(fig)
    gs = fig.add_gridspec(1, 3)
    ax = fig.add_subplot(gs[0, :2])
    style_axes(ax)
    v = v[~np.isnan(v)]
    bins = bins or int(np.clip(np.sqrt(len(v)) * 1.2, 8, 60))
    ax.hist(v, bins=bins, color="#6f93cf", edgecolor="white", linewidth=0.5, density=normal)
    if normal and len(v) > 2 and v.std() > 0:
        from scipy.stats import norm
        xs = np.linspace(v.min(), v.max(), 200)
        ax.plot(xs, norm.pdf(xs, v.mean(), v.std(ddof=1)), color=PALETTE[1], lw=1.3, label="Normal")
        ax.legend(frameon=False)
    ax.set_title(f"Histogram: {name.upper()}")
    axt = fig.add_subplot(gs[0, 2])
    axt.axis("off")
    axt.text(0.02, 0.98, stats_text, va="top", ha="left", family="monospace", fontsize=8.2,
             transform=axt.transAxes)
    fig.tight_layout()
    return fig


def afr_plot(actual: pd.Series, fitted: pd.Series, resid: pd.Series, title="Actual, Fitted, Residual"):
    """EViews' signature Actual/Fitted/Residual graph."""
    fig, ax = new_figure(7.5, 4.4)
    x = _x(actual.index)
    ax2 = ax.twinx()
    ax2.plot(x, resid.values, color=PALETTE[0], lw=0.9, label="Residual")
    sd = np.nanstd(resid.values)
    for k in (-1, 0, 1):
        ax2.axhline(k * sd, color="#888", lw=0.6, ls="--" if k else "-")
    rmin, rmax = np.nanmin(resid.values), np.nanmax(resid.values)
    span = max(abs(rmin), abs(rmax)) or 1
    ax2.set_ylim(-span * 1.1, span * 3.2)
    lo, hi = np.nanmin([actual.min(), fitted.min()]), np.nanmax([actual.max(), fitted.max()])
    rg = (hi - lo) or 1
    ax.set_ylim(lo - rg * 1.3, hi + rg * 0.1)
    ax.plot(x, actual.values, color=PALETTE[1], lw=1.0, label="Actual")
    ax.plot(x, fitted.values, color=PALETTE[2], lw=1.0, label="Fitted")
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h2 + h1, l2 + l1, frameon=False, loc="upper left", ncol=3)
    ax.set_title(title)
    ax2.spines["top"].set_visible(False)
    fig.tight_layout()
    return fig


def grid_plot(panels: list[tuple[str, np.ndarray, np.ndarray | None, np.ndarray | None]], ncols: int,
              title: str = "", zero=True, x=None):
    """Small-multiples (e.g. impulse responses): each panel = (title, y, lo, hi)."""
    n = len(panels)
    nrows = math.ceil(n / ncols)
    fig, axes = new_figure(2.6 * ncols + 0.6, 2.1 * nrows + 0.6, nrows, ncols, squeeze=False)
    for k, (t, y, lo, hi) in enumerate(panels):
        ax = axes.ravel()[k]
        xx = np.arange(1, len(y) + 1) if x is None else x
        ax.plot(xx, y, color=PALETTE[0], lw=1.3)
        if lo is not None and hi is not None:
            ax.plot(xx, lo, color=PALETTE[1], lw=0.9, ls="--")
            ax.plot(xx, hi, color=PALETTE[1], lw=0.9, ls="--")
        if zero:
            ax.axhline(0, color="#444", lw=0.6)
        ax.set_title(t, fontsize=8)
    for k in range(n, nrows * ncols):
        axes.ravel()[k].set_visible(False)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def correlogram_plot(acf, pacf, n, name=""):
    fig, axes = new_figure(7.5, 4.2, 2, 1, sharex=True)
    lags = np.arange(1, len(acf) + 1)
    band = 2 / math.sqrt(n)
    for ax, vals, t in ((axes[0], acf, "Autocorrelation"), (axes[1], pacf, "Partial autocorrelation")):
        ax.bar(lags, vals, color=PALETTE[0], width=0.6)
        ax.axhline(band, color=PALETTE[1], ls="--", lw=0.8)
        ax.axhline(-band, color=PALETTE[1], ls="--", lw=0.8)
        ax.axhline(0, color="#333", lw=0.6)
        ax.set_title(f"{t}{': ' + name.upper() if name else ''}")
    axes[1].set_xlabel("Lag")
    fig.tight_layout()
    return fig


def qq_plot(v: np.ndarray, name: str, dist: str = "normal"):
    from scipy import stats
    v = np.sort(v[~np.isnan(v)])
    n = len(v)
    p = (np.arange(1, n + 1) - 0.5) / n
    q = stats.norm.ppf(p)
    fig, ax = new_figure(5.5, 4.5)
    ax.scatter(q, v, s=10, color=PALETTE[0])
    b = np.polyfit(q, v, 1)
    ax.plot(q, b[0] * q + b[1], color=PALETTE[1], lw=1.2)
    ax.set_xlabel("Normal quantile")
    ax.set_ylabel(f"Quantiles of {name.upper()}")
    ax.set_title("Theoretical quantile-quantile")
    fig.tight_layout()
    return fig


def kde_plot(v: np.ndarray, name: str):
    from scipy import stats
    v = v[~np.isnan(v)]
    fig, ax = new_figure(6, 4)
    xs = np.linspace(v.min(), v.max(), 300)
    ax.plot(xs, stats.gaussian_kde(v)(xs), color=PALETTE[0], lw=1.4)
    ax.set_title(f"Kernel density: {name.upper()}")
    fig.tight_layout()
    return fig


def boxplot(data: dict[str, np.ndarray], title=""):
    fig, ax = new_figure(6.5, 4.2)
    ax.boxplot([v[~np.isnan(v)] for v in data.values()], tick_labels=[k.upper() for k in data])
    ax.set_title(title)
    fig.tight_layout()
    return fig


def heatmap(mat: np.ndarray, labels: list[str], title=""):
    fig, ax = new_figure(5.8, 5.0)
    ax.grid(False)
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.upper() for l in labels], rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([l.upper() for l in labels])
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(mat[i, j]) > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title)
    fig.tight_layout()
    return fig
