"""Frequencies, observation labels and EViews-style date parsing."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

FREQ_NAMES = {
    "a": "Annual", "s": "Semi-annual", "q": "Quarterly", "m": "Monthly",
    "w": "Weekly", "d5": "Daily - 5 day week", "d7": "Daily - 7 day week", "u": "Undated",
}
# pandas offset aliases (period start, like EViews)
PANDAS_FREQ = {"a": "YS", "s": "6MS", "q": "QS", "m": "MS", "w": "W-MON", "d5": "B", "d7": "D"}
PERIODS_PER_YEAR = {"a": 1, "s": 2, "q": 4, "m": 12, "w": 52, "d5": 252, "d7": 365, "u": 1}

FREQ_ALIASES = {
    "a": "a", "y": "a", "annual": "a", "yearly": "a",
    "s": "s", "semi": "s",
    "q": "q", "quarterly": "q",
    "m": "m", "monthly": "m",
    "w": "w", "weekly": "w",
    "d": "d5", "d5": "d5", "b": "d5", "daily": "d5", "business": "d5",
    "d7": "d7", "cal": "d7",
    "u": "u", "undated": "u",
}


def norm_freq(f: str) -> str:
    f = (f or "u").lower()
    if f not in FREQ_ALIASES:
        raise ValueError(f"Unknown frequency '{f}'. Use a, s, q, m, w, d5, d7 or u.")
    return FREQ_ALIASES[f]


def infer_freq(idx: pd.DatetimeIndex) -> str:
    if len(idx) < 3:
        return "d7"
    d = np.diff(idx.values.astype("datetime64[D]").astype(np.int64))
    d = d[d > 0]
    if len(d) == 0:
        return "d7"
    med = float(np.median(d))
    if med >= 300:
        return "a"
    if med >= 150:
        return "s"
    if med >= 80:
        return "q"
    if med >= 26:
        return "m"
    if med >= 6:
        return "w"
    # daily: weekends present?
    wd = pd.DatetimeIndex(idx).dayofweek
    return "d7" if np.mean(wd >= 5) > 0.05 else "d5"


def obs_label(ts, freq: str) -> str:
    if freq == "u":
        return str(ts)
    ts = pd.Timestamp(ts)
    if freq == "a":
        return f"{ts.year}"
    if freq == "s":
        return f"{ts.year}S{1 if ts.month <= 6 else 2}"
    if freq == "q":
        return f"{ts.year}Q{(ts.month - 1) // 3 + 1}"
    if freq == "m":
        return f"{ts.year}M{ts.month:02d}"
    return ts.strftime("%Y-%m-%d")


def obs_labels(idx, freq: str) -> list[str]:
    if isinstance(idx, pd.MultiIndex):
        ids = idx.get_level_values(0)
        tt = idx.get_level_values(1)
        return [f"{a} - {obs_label(b, freq) if freq != 'u' else b}" for a, b in zip(ids, tt)]
    if freq == "u":
        return [str(i) for i in range(1, len(idx) + 1)] if not _is_int_index(idx) else [str(x) for x in idx]
    idx = pd.DatetimeIndex(idx)
    if freq == "a":
        return [str(y) for y in idx.year]
    if freq == "q":
        return [f"{y}Q{q}" for y, q in zip(idx.year, idx.quarter)]
    if freq == "m":
        return [f"{y}M{m:02d}" for y, m in zip(idx.year, idx.month)]
    if freq == "s":
        return [f"{y}S{1 if m <= 6 else 2}" for y, m in zip(idx.year, idx.month)]
    return [d.strftime("%Y-%m-%d") for d in idx]


def _is_int_index(idx) -> bool:
    return pd.api.types.is_integer_dtype(getattr(idx, "dtype", None))


_RX_Q = re.compile(r"^(\d{4})\s*[qQ:]\s*([1-4])$")
_RX_M = re.compile(r"^(\d{4})\s*[mM:]\s*(\d{1,2})$")
_RX_S = re.compile(r"^(\d{4})\s*[sS]\s*([12])$")
_RX_Y = re.compile(r"^(\d{4})$")


def parse_date(tok: str, freq: str, end: bool = False) -> pd.Timestamp:
    """Parse an EViews-style date token ('2010', '2010q2', '2010m3', '2010:2',
    '2010-03-15', '3/15/2010').  With end=True the *end* of the period is
    returned so that ``index <= result`` includes the whole period."""
    tok = str(tok).strip().strip('"').strip("'")
    m = _RX_Q.match(tok)
    if m and (freq in ("q", "a", "s") or "q" in tok.lower()):
        y, q = int(m.group(1)), int(m.group(2))
        start = pd.Timestamp(year=y, month=3 * (q - 1) + 1, day=1)
        return (start + pd.offsets.QuarterEnd(0)).normalize() + pd.Timedelta(hours=23, minutes=59) if end else start
    m = _RX_M.match(tok)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        if freq == "q" and ":" in tok and mo <= 4:
            start = pd.Timestamp(year=y, month=3 * (mo - 1) + 1, day=1)
            return (start + pd.offsets.QuarterEnd(0)) + pd.Timedelta(hours=23) if end else start
        start = pd.Timestamp(year=y, month=mo, day=1)
        return (start + pd.offsets.MonthEnd(0)) + pd.Timedelta(hours=23) if end else start
    m = _RX_S.match(tok)
    if m:
        y, h = int(m.group(1)), int(m.group(2))
        start = pd.Timestamp(year=y, month=1 if h == 1 else 7, day=1)
        return (start + pd.DateOffset(months=6) - pd.Timedelta(minutes=1)) if end else start
    m = _RX_Y.match(tok)
    if m:
        y = int(m.group(1))
        return pd.Timestamp(year=y, month=12, day=31, hour=23) if end else pd.Timestamp(year=y, month=1, day=1)
    ts = pd.Timestamp(tok)
    if end:
        ts = ts.normalize() + pd.Timedelta(hours=23, minutes=59)
    return ts


def make_range(freq: str, start: str, end: str) -> pd.DatetimeIndex:
    freq = norm_freq(freq)
    s = parse_date(start, freq)
    e = parse_date(end, freq, end=True)
    idx = pd.date_range(s, e, freq=PANDAS_FREQ[freq])
    if freq == "w":
        # align weekly to the first Monday on/after start
        idx = pd.date_range(s, e, freq="W-MON")
    return idx


def extend_index(idx: pd.DatetimeIndex, freq: str, n: int) -> pd.DatetimeIndex:
    if n <= 0:
        return idx
    off = pd.tseries.frequencies.to_offset(PANDAS_FREQ[freq])
    extra = pd.date_range(idx[-1] + off, periods=n, freq=off)
    return idx.append(extra)
