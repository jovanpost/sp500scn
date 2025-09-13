"""Download and store price data for S&P 500 members."""

from __future__ import annotations

import io
import json
from datetime import datetime
from typing import List

import pandas as pd
from supabase import Client

from .schemas import IngestJob, IngestResult
from .storage import Storage


def _fetch(storage: Storage, job: IngestJob) -> pd.DataFrame:
    ticker = job["ticker"]
    start, end = job["start"], job["end"]
    supabase: Client | None = getattr(storage, "supabase_client", None)
    if not supabase:
        return pd.DataFrame()

    page_size = 1000
    offset = 0
    rows: list[dict] = []
    while True:
        resp = (
            supabase.table("sp500_ohlcv")
            .select(
                "ticker, date, open, high, low, close, volume, dividends, stock_splits"
            )
            .eq("ticker", ticker)
            .gte("date", start)
            .lte("date", end)
            .order("date")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        if not resp.data:
            break
        rows.extend(resp.data)
        if len(resp.data) < page_size:
            break
        offset += page_size

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ticker"] = ticker
    return df[
        [
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "dividends",
            "stock_splits",
            "ticker",
        ]
    ]


def ingest_batch(storage: Storage, jobs: List[IngestJob], progress_cb=None) -> dict:
    """Run batch; returns dict with summary and results."""

    results: List[IngestResult] = []
    all_dates = []
    for idx, job in enumerate(jobs):
        path = f"prices/{job['ticker']}.parquet"
        error = None
        rows = 0
        try:
            df = _fetch(storage, job)
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
