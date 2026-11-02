"""Data import / export and web data sources (Yahoo Finance, FRED, Fama-French)."""
from __future__ import annotations

import datetime as _dt
import io
import json
import os
import re
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from .workfile import Workfile, WorkfileError, sanitize_name

DATE_NAMES = {"date", "dates", "time", "period", "obs", "observation_date", "datetime", "day", "month", "year",
              "timestamp", "dateid", "yyyymm", "yyyymmdd", "quarter"}
ID_NAMES = {"id", "firm", "ticker", "symbol", "country", "permno", "gvkey", "company", "crossid", "code", "isin", "cusip",
            "state", "region", "bank", "fund", "stock", "name", "entity", "unit", "group"}


def read_table(path: str, sheet: str | int | None = None, **kw) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".txt", ".tsv", ".dat"):
        sep = kw.get("sep")
        if sep is None:
            with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
                head = f.read(4096)
            sep = "\t" if head.count("\t") > head.count(",") else ";" if head.count(";") > head.count(",") else ","
        return pd.read_csv(path, sep=sep, encoding="utf-8-sig", low_memory=False)
    if ext in (".xlsx", ".xlsm", ".xls"):
        return pd.read_excel(path, sheet_name=sheet if sheet is not None else 0)
    if ext == ".dta":
        return pd.read_stata(path)
    if ext in (".sav",):
        return pd.read_spss(path)
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if ext == ".json":
        return pd.read_json(path)
    if ext in (".pkl", ".pickle"):
        raise WorkfileError("Pickle files are not imported for security reasons; use CSV, Excel, Stata or Parquet")
    raise WorkfileError(f"Unsupported file type '{ext}'")


def _parse_date_col(col: pd.Series):
    s = col.dropna()
    if len(s) == 0:
        return None
    if pd.api.types.is_datetime64_any_dtype(col):
        return pd.to_datetime(col)
    if pd.api.types.is_numeric_dtype(col):
        v = s.astype(float)
        if (v == v.round()).all():
            iv = v.astype(np.int64)
            if iv.between(1000, 2999).all():
                return pd.to_datetime(col.astype("Int64").astype(str), format="%Y", errors="coerce")
            if iv.between(100001, 299912).all():
                return pd.to_datetime(col.astype("Int64").astype(str), format="%Y%m", errors="coerce")
            if iv.between(10000101, 29991231).all():
                return pd.to_datetime(col.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
        return None
    st = s.astype(str).str.strip()
    # EViews style 2010Q1 / 2010M01 / 2010:1
    if st.str.match(r"^\d{4}\s*[qQ]\s*[1-4]$").all():
        return pd.PeriodIndex(col.astype(str).str.replace(" ", "").str.upper(), freq="Q").to_timestamp() \
            if not col.isna().any() else None
    if st.str.match(r"^\d{4}\s*[mM]\s*\d{1,2}$").all():
        return pd.to_datetime(col.astype(str).str.upper().str.replace(r"\s*M\s*", "-", regex=True), format="%Y-%m",
                              errors="coerce")
    try:
        d = pd.to_datetime(col, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        d = pd.to_datetime(col, errors="coerce")
    if d.notna().sum() >= 0.9 * col.notna().sum():
        return d
    return None


def detect_structure(df: pd.DataFrame, date: str | None = None, ident: str | None = None):
    cols = list(df.columns)
    low = {str(c).lower(): c for c in cols}
    date_col = None
    if date:
        date_col = low.get(date.lower(), date)
    else:
        for c in cols:
            if str(c).lower() in DATE_NAMES:
                date_col = c
                break
        if date_col is None and cols:
            c0 = cols[0]
            if not pd.api.types.is_float_dtype(df[c0]) and _parse_date_col(df[c0]) is not None:
                date_col = c0
    id_col = None
    if ident:
        id_col = low.get(ident.lower(), ident)
    elif date_col is not None:
        dates = df[date_col]
        if dates.duplicated().any():
            for c in cols:
                if c != date_col and str(c).lower() in ID_NAMES:
                    id_col = c
                    break
            if id_col is None:
                for c in cols:
                    if c != date_col and not pd.api.types.is_float_dtype(df[c]) and \
                            not df[[c, date_col]].duplicated().any():
                        id_col = c
                        break
    return date_col, id_col


def import_file(path: str, wf: Workfile | None = None, date: str | None = None, ident: str | None = None,
                freq: str | None = None, sheet=None, names: list[str] | None = None, start: str | None = None,
                undated: bool = False) -> tuple[Workfile, list[str], bool]:
    """Import a data file.  Returns (workfile, imported names, created_new_workfile)."""
    df = read_table(path, sheet)
    if names:
        keep = [c for c in df.columns if str(c).lower() in [n.lower() for n in names]]
        dc, ic = detect_structure(df, date, ident)
        keep = [c for c in [dc, ic] if c is not None and c not in keep] + keep
        df = df[keep]
    date_col, id_col = (None, None) if undated else detect_structure(df, date, ident)
    base = os.path.splitext(os.path.basename(path))[0]
    if id_col is not None and date_col is not None:
        df[date_col] = _parse_date_col(df[date_col]) if _parse_date_col(df[date_col]) is not None else df[date_col]
        new = Workfile.from_dataframe(df, name=sanitize_name(base), freq=freq, panel=(id_col, date_col))
    elif date_col is not None:
        dts = _parse_date_col(df[date_col])
        if dts is None:
            raise WorkfileError(f"Could not parse dates in column '{date_col}'")
        df = df.drop(columns=[date_col])
        df.index = pd.DatetimeIndex(dts)
        df = df[~df.index.isna()]
        if df.index.duplicated().any():
            df = df[~df.index.duplicated(keep="last")]
        new = Workfile.from_dataframe(df, name=sanitize_name(base), freq=freq)
    else:
        if start and freq:
            from .dates import make_range, norm_freq, PANDAS_FREQ
            f = norm_freq(freq)
            df.index = pd.date_range(pd.Timestamp(start), periods=len(df), freq=PANDAS_FREQ[f])
            new = Workfile.from_dataframe(df, name=sanitize_name(base), freq=f)
        else:
            new = Workfile.from_dataframe(df.reset_index(drop=True), name=sanitize_name(base))
    imported = new.series_names()
    if wf is None or (wf.nobs <= 1 and not wf.series_names()) or (not wf.series_names() and not wf.objects):
        new.name = new.name if wf is None or wf.name == "untitled" else wf.name
        return new, imported, True
    # merge into the existing workfile
    merge_into(wf, new)
    return wf, imported, False


def merge_into(wf: Workfile, new: Workfile, overwrite: bool = True):
    if wf.is_panel != new.is_panel:
        raise WorkfileError("Cannot merge panel and non-panel data; open the file as a new workfile instead")
    if wf.is_dated and new.is_dated and not wf.is_panel:
        # align on periods
        idx = wf.data.index
        src = new.data
        for c in src.columns:
            s = src[c]
            if wf.freq != new.freq:
                s = _convert_freq(s, new.freq, wf.freq)
            s = s.reindex(idx)
            if c in wf.data.columns and not overwrite:
                continue
            wf.data[c] = s.values
    elif wf.is_panel:
        for c in new.data.columns:
            wf.data[c] = new.data[c].reindex(wf.data.index).values
    else:
        n = min(wf.nobs, new.nobs)
        for c in new.data.columns:
            v = np.full(wf.nobs, np.nan, dtype=object if new.data[c].dtype == object else float)
            v[:n] = new.data[c].values[:n]
            wf.data[c] = v
    wf.notify("series")


def _convert_freq(s: pd.Series, f_from: str, f_to: str, how: str = "last") -> pd.Series:
    order = ["d7", "d5", "w", "m", "q", "s", "a"]
    rule = {"a": "YS", "s": "6MS", "q": "QS", "m": "MS", "w": "W-MON", "d5": "B", "d7": "D"}[f_to]
    if order.index(f_from) < order.index(f_to):
        r = s.resample(rule)
        return getattr(r, how)()
    return s.resample(rule).ffill()


def export_workfile(wf: Workfile, path: str, names: list[str] | None = None):
    names = wf.expand_names(names) if names else wf.series_names()
    df = wf.data[names].copy()
    if wf.is_dated and not wf.is_panel:
        df.index = wf.labels()
        df.index.name = "date"
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        df.to_excel(path)
    elif ext == ".dta":
        df.reset_index().to_stata(path, write_index=False)
    elif ext in (".parquet", ".pq"):
        df.to_parquet(path)
    else:
        df.to_csv(path)
    return path


# ---------------------------------------------------------------------------------------------------- web
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Finomania/1.0"}


def _get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch_fred(codes: list[str], start: str | None = None, end: str | None = None) -> pd.DataFrame:
    out = {}
    for c in codes:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={urllib.parse.quote(c.upper())}"
        if start:
            url += f"&cosd={start}"
        if end:
            url += f"&coed={end}"
        try:
            raw = _get(url).decode("utf-8")
        except Exception as e:
            raise WorkfileError(f"FRED download failed for {c}: {e}") from e
        df = pd.read_csv(io.StringIO(raw))
        dcol = df.columns[0]
        df[dcol] = pd.to_datetime(df[dcol])
        s = pd.to_numeric(df.iloc[:, 1].replace(".", np.nan), errors="coerce")
        s.index = df[dcol]
        out[sanitize_name(c)] = s
    return pd.DataFrame(out)


def fetch_yahoo(symbols: list[str], start: str | None = None, end: str | None = None, interval: str = "1d",
                field: str = "adjclose") -> pd.DataFrame:
    """Download prices from Yahoo Finance (uses yfinance when installed, else the public chart API)."""
    start = start or "2000-01-01"
    end = end or _dt.date.today().isoformat()
    field = field.lower().replace(" ", "")
    try:
        import yfinance as yf  # type: ignore
        data = yf.download(symbols, start=start, end=end, interval=interval, auto_adjust=False, progress=False,
                           group_by="column")
        colmap = {"adjclose": "Adj Close", "close": "Close", "open": "Open", "high": "High", "low": "Low",
                  "volume": "Volume"}
        sub = data[colmap.get(field, "Adj Close")]
        if isinstance(sub, pd.Series):
            sub = sub.to_frame(symbols[0])
        sub.columns = [sanitize_name(str(c)) for c in sub.columns]
        sub.index = pd.DatetimeIndex(sub.index).tz_localize(None)
        return sub
    except ImportError:
        pass
    out = {}
    p1 = int(pd.Timestamp(start).timestamp())
    p2 = int(pd.Timestamp(end).timestamp()) + 86400
    for s in symbols:
        url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(s)}?period1={p1}&period2={p2}"
               f"&interval={interval}&events=div%2Csplits&includeAdjustedClose=true")
        try:
            js = json.loads(_get(url))
            res = js["chart"]["result"][0]
            ts = pd.to_datetime(res["timestamp"], unit="s").normalize()
            q = res["indicators"]["quote"][0]
            if field == "adjclose":
                vals = res["indicators"].get("adjclose", [{}])[0].get("adjclose") or q["close"]
            else:
                vals = q[field]
            out[sanitize_name(s)] = pd.Series(vals, index=ts, dtype=float)
        except Exception as e:
            raise WorkfileError(f"Yahoo Finance download failed for {s}: {e}. Install yfinance (pip install yfinance) "
                                "for a more robust connection.") from e
    df = pd.DataFrame(out)
    df = df[~df.index.duplicated(keep="last")]
    return df
