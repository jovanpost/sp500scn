# data_lake/ingest.py
"""Download and store price data for S&P 500 members."""

from __future__ import annotations

import io
import json
import logging
import time
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from supabase import Client
from urllib3.util.retry import Retry

from .schemas import IngestJob, IngestResult
from .storage import Storage, _tidy_prices, validate_prices_schema

log = logging.getLogger(__name__)

SCHEMA = [
    "date",
    "Ticker",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
    "Dividends",
    "Stock Splits",
]


# -------------------------- Yahoo RAW helpers -------------------------- #

def _make_yf_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            ),
            "Accept": "application/json,text/plain,*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
        }
    )
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def _yahoo_symbol(ticker: str) -> str:
    return ticker.replace(".", "-").upper()


def _download_raw_yahoo(ticker: str, start: str | None, end: str | None) -> pd.DataFrame:
    import yfinance as yf  # optional dependency, imported lazily

    session = _make_yf_session()
    symbol = _yahoo_symbol(ticker)
    df = yf.download(
        symbol,
        start=start,
        end=end,
        auto_adjust=False,
        actions=True,
        progress=False,
        threads=False,
        session=session,
    )

    if df is None or df.empty:
        df = yf.download(
            symbol,
            period="30d",
            auto_adjust=False,
            actions=True,
            progress=False,
            threads=False,
            session=session,
        )

    if df is None or df.empty:
        raise RuntimeError(f"Yahoo returned no rows for {symbol} [{start or 'period'}..{end or 'now'}]")

    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(1, axis=1)

    df = df.reset_index().rename(columns={"Date": "date"})

    out = pd.DataFrame(index=range(len(df)))
    out["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    out["Ticker"] = ticker.upper()
    out["Open"] = df.get("Open", pd.NA)
    out["High"] = df.get("High", pd.NA)
    out["Low"] = df.get("Low", pd.NA)
    out["Close"] = df.get("Close", pd.NA)
    out["Adj Close"] = df.get("Adj Close", pd.NA)
    out["Volume"] = df.get("Volume", pd.NA)
    out["Dividends"] = df.get("Dividends", 0.0 if "Dividends" in df else 0.0)
    out["Stock Splits"] = df.get("Stock Splits", 0.0 if "Stock Splits" in df else 0.0)
    return out[SCHEMA].sort_values("date")


def _write_prices_with_backup(storage: Storage, ticker: str, df: pd.DataFrame) -> None:
    dest = f"prices/{ticker.upper()}.parquet"
    ts = pd.Timestamp.utcnow().strftime("%Y%m%d%H%M%S")
    try:
        if getattr(storage, "exists", None) and storage.exists(dest):
            prev = storage.read_bytes(dest)
            storage.write_bytes(f"backups/prices/{ticker.upper()}.{ts}.parquet", prev)
    except Exception:
        # best-effort backup
        pass

    buf = io.BytesIO()
    df.to_parquet(buf, index=False, compression="snappy")
    storage.write_bytes(dest, buf.getvalue())


def ingest_raw_yahoo_batch(
    storage: Storage,
    jobs: List[dict],
    progress_cb=None,
    pause_s: float = 0.5,
) -> dict:
    """Fetch RAW prices from Yahoo and write parquet with backup; emit a manifest."""
    ok = failed = 0
    results: list[dict] = []
    total = max(len(jobs), 1)

    for i, job in enumerate(jobs, start=1):
        tkr = job["ticker"].upper()
        start = job.get("start") or "1990-01-01"
        end = job.get("end")
        try:
            frame = _download_raw_yahoo(tkr, start, end)
            validate_prices_schema(frame, strict=False)
            _write_prices_with_backup(storage, tkr, frame)
            ok += 1
            results.append({"ticker": tkr, "rows": int(len(frame))})
        except Exception as exc:
            failed += 1
            results.append({"ticker": tkr, "error": str(exc)})
        if progress_cb:
            progress_cb(i, total)
        time.sleep(max(pause_s, 0.0))

    manifest = {"ts": datetime.utcnow().isoformat(), "ok": ok, "failed": failed, "results": results}
    mpath = f"manifests/ingest_raw/{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.json"
    try:
        storage.write_bytes(mpath, json.dumps(manifest, separators=(",", ":")).encode())
    except Exception:
        mpath = "-"

    return {"ok": ok, "failed": failed, "results": results, "manifest_path": mpath}


# -------------------------- RAW detector / coverage helpers -------------------------- #

def _looks_raw_prices(df: pd.DataFrame) -> bool:
    """Heuristic RAW detector.

    RAW if on any action day (dividend or split) Close != Adj Close.
    If there are no action days in the slice, accept as RAW to avoid endless rebuilds.
    """
    needed = {"Close", "Adj Close", "Dividends", "Stock Splits"}
    if not needed.issubset(set(df.columns)):
        return False

    d = df.copy()
    d["Dividends"] = pd.to_numeric(d["Dividends"], errors="coerce").fillna(0)
    d["Stock Splits"] = pd.to_numeric(d["Stock Splits"], errors="coerce").fillna(0)

    actions = (d["Dividends"].ne(0)) | (d["Stock Splits"].ne(0))
    if not actions.any():
        return True

    sub = d.loc[actions, ["Close", "Adj Close"]].dropna()
    if sub.empty:
        return True

    same = np.isclose(sub["Close"], sub["Adj Close"], rtol=0, atol=1e-6)
    return not bool(same.all())


def lake_file_is_raw(storage: Storage, ticker: str) -> bool:
    """Return True if the stored prices parquet for ticker appears RAW."""
    path = f"prices/{ticker.upper()}.parquet"
    if not storage.exists(path):
        return False
    try:
        try:
            df = storage.read_parquet_df(path, columns=["date", "Close", "Adj Close", "Dividends", "Stock Splits"])
        except TypeError:
            df = storage.read_parquet_df(path)
        return _looks_raw_prices(df)
    except Exception:
        return False


# -------------------------- Existing Supabase ingest -------------------------- #

def _fetch(storage: Storage, job: IngestJob) -> pd.DataFrame:
    ticker = job["ticker"]
    start = job.get("start") or "1990-01-01"
    end = job.get("end") or datetime.utcnow().date().isoformat()

    supabase: Client | None = getattr(storage, "supabase_client", None)
    if not supabase:
        return pd.DataFrame()

    page_size, offset = 1000, 0
    rows: list[dict] = []
    select_cols = "ticker, date, open, high, low, close, adj_close, volume, dividends, stock_splits"
    use_wildcard = False

    while True:
        query = (
            supabase.table("sp500_ohlcv")
            .select(select_cols if not use_wildcard else "*")
            .eq("ticker", ticker)
            .gte("date", start)
            .lte("date", end)
            .order("date")
            .range(offset, offset + page_size - 1)
        )
        try:
            resp = query.execute()
        except Exception as exc:
            if not use_wildcard:
                log.warning("ingest: explicit select failed for %s; retrying with '*': %s", ticker, exc)
                use_wildcard, offset, rows = True, 0, []
                continue
            raise

        data = resp.data or []
        if (not use_wildcard) and data and all("adj_close" not in r for r in data if isinstance(r, dict)):
            log.warning("ingest: rows lack 'adj_close' for %s; retrying with wildcard", ticker)
            use_wildcard, offset, rows = True, 0, []
            continue

        if not data:
            break

        rows.extend(data)
        if len(data) < page_size:
            break
        offset += page_size

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "adj_close" not in df.columns:
        df["adj_close"] = pd.NA
        log.warning("ingest: 'adj_close' missing; writing RAW OHLC with null Adj Close for %s", ticker)

    df["date"] = pd.to_datetime(df.get("date"), errors="coerce")
    df["ticker"] = ticker
    cols = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "dividends",
        "stock_splits",
        "ticker",
    ]
    return df[[c for c in cols if c in df.columns]]


def ingest_batch(storage: Storage, jobs: List[IngestJob], progress_cb=None) -> dict:
    """Legacy Supabase ingest (kept for 'Files present' target and fallback)."""
    results: List[IngestResult] = []
    all_dates = []

    for idx, job in enumerate(jobs):
        path = f"prices/{job['ticker']}.parquet"
        error = None
        rows = 0
        try:
            df_raw = _fetch(storage, job)
            tidy = _tidy_prices(df_raw, ticker=job["ticker"]).reset_index()
            validate_prices_schema(tidy, strict=False)

            buf = io.BytesIO()
            tidy.to_parquet(buf, index=False)
            storage.write_bytes(path, buf.getvalue())

            rows = len(tidy)
            if rows:
                all_dates.append(tidy["date"])
        except Exception as e:
            error = str(e)

        results.append({"ticker": job["ticker"], "rows_written": rows, "path": path, "error": error})
        if progress_cb:
            progress_cb(idx + 1, len(jobs))

    ok = sum(1 for r in results if not r["error"])
    failed = len(results) - ok
    min_date = max_date = None
    if all_dates:
        series = pd.concat(all_dates)
        min_date = str(pd.to_datetime(series.min()).date())
        max_date = str(pd.to_datetime(series.max()).date())

    manifest = {
        "generated_at_utc": datetime.utcnow().isoformat(),
        "storage_backend": storage.mode,
        "provider": "supabase",
        "ok": ok,
        "failed": failed,
        "min_date": min_date,
        "max_date": max_date,
        "tickers": [j["ticker"] for j in jobs],
    }
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    manifest_path = f"manifests/ingest_{ts}.json"
    storage.write_bytes(manifest_path, json.dumps(manifest).encode("utf-8"))
    return {"ok": ok, "failed": failed, "results": results, "manifest_path": manifest_path}
