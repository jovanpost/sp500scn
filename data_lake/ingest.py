"""Download and store price data for S&P 500 members."""

from __future__ import annotations

import io
import json
import logging
from datetime import datetime
from typing import List

import pandas as pd
from supabase import Client

from .schemas import IngestJob, IngestResult
from .storage import Storage, _tidy_prices, validate_prices_schema


log = logging.getLogger(__name__)


def _fetch(storage: Storage, job: IngestJob) -> pd.DataFrame:
    ticker = job["ticker"]
    start, end = job["start"], job["end"]
    supabase: Client | None = getattr(storage, "supabase_client", None)
    if not supabase:
        return pd.DataFrame()

    page_size, offset = 1000, 0
    rows: list[dict] = []
    select_cols = (
        "ticker, date, open, high, low, close, adj_close, volume, dividends, stock_splits"
    )
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
                # Fall back to wildcard selection when explicit columns fail (older schemas).
                log.warning(
                    "ingest: explicit select failed for %s; retrying with '*': %s",
                    ticker,
                    exc,
                )
                use_wildcard, offset, rows = True, 0, []
                continue
            raise

        data = resp.data or []
        if (
            not use_wildcard
            and data
            and all("adj_close" not in row for row in data if isinstance(row, dict))
        ):
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
        log.warning(
            "ingest: 'adj_close' missing; writing RAW OHLC with null Adj Close for %s",
            ticker,
        )

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
    """Run batch; returns dict with summary and results."""

    results: List[IngestResult] = []
    all_dates = []
    for idx, job in enumerate(jobs):
        path = f"prices/{job['ticker']}.parquet"
        error = None
        rows = 0
        try:
            df_raw = _fetch(storage, job)
            tidy = _tidy_prices(df_raw, ticker=job["ticker"]).reset_index()
            # Warn-only during ingest so legacy data can be migrated without failing jobs.
            validate_prices_schema(tidy, strict=False)

            buffer = io.BytesIO()
            tidy.to_parquet(buffer, index=False)
            storage.write_bytes(path, buffer.getvalue())

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
