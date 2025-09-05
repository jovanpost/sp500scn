"""Download and store price data for S&P 500 members."""

from __future__ import annotations

import io
import json
import time
from datetime import datetime
from typing import List

import pandas as pd
import yfinance as yf

from .schemas import IngestJob, IngestResult
from .storage import Storage

RATE_LIMIT_SECONDS = 0.5
MAX_RETRIES = 3


def _fetch(job: IngestJob) -> pd.DataFrame:
    ticker = job["ticker"]
    start, end = job["start"], job["end"]
    for attempt in range(MAX_RETRIES):
        try:
            df = yf.Ticker(ticker).history(
                interval="1d", start=start, end=end, auto_adjust=True, actions=True
            )
            break
        except Exception:
            if attempt + 1 == MAX_RETRIES:
                raise
            time.sleep(RATE_LIMIT_SECONDS * (attempt + 1))
    if df.empty:
        df = pd.DataFrame(columns=[
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Dividends",
            "Stock Splits",
        ])
    for col in ["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"]:
        if col not in df.columns:
            df[col] = 0.0 if col in {"Dividends", "Stock Splits"} else 0
    df = df[["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"]]
    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Dividends": "dividends",
            "Stock Splits": "stock_splits",
        }
    )
    df = df.reset_index().rename(columns={"Date": "date"})
    df["ticker"] = ticker
    return df


def ingest_batch(storage: Storage, jobs: List[IngestJob], progress_cb=None) -> dict:
    """Run batch; returns dict with summary and results."""

    results: List[IngestResult] = []
    all_dates = []
    for idx, job in enumerate(jobs):
        path = f"prices/{job['ticker']}.parquet"
        error = None
        rows = 0
        try:
            df = _fetch(job)
            buffer = io.BytesIO()
            df.to_parquet(buffer, index=False)
            storage.write_bytes(path, buffer.getvalue())
            rows = len(df)
            if rows:
                all_dates.append(df["date"])
        except Exception as e:
            error = str(e)
        results.append({"ticker": job["ticker"], "rows_written": rows, "path": path, "error": error})
        if progress_cb:
            progress_cb(idx + 1, len(jobs))
        time.sleep(RATE_LIMIT_SECONDS)

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
        "provider": "yfinance",
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
