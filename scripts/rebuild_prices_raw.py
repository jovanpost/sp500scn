#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import sys
import time
from typing import Sequence

import pandas as pd
import yfinance as yf

from data_lake.membership import historical_tickers
from data_lake.storage import Storage, _tidy_prices, validate_prices_schema


def _norm_date(value: str | None) -> str | None:
    if not value:
        return None
    return pd.Timestamp(value).normalize().strftime("%Y-%m-%d")


def _download_one(ticker: str, start: str | None, end: str | None) -> pd.DataFrame:
    raw = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        actions=True,
        progress=False,
        threads=False,
    )
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.droplevel(1, axis=1)
    raw = raw.reset_index()
    raw["Ticker"] = ticker
    tidy = _tidy_prices(raw, ticker=ticker).reset_index()
    validate_prices_schema(tidy)
    return tidy


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="*")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    storage = Storage.from_env()
    start = _norm_date(args.start)
    end = _norm_date(args.end)
    tickers = [t.upper() for t in (args.tickers or [])] or historical_tickers(
        storage, limit=args.limit
    )
    if not tickers:
        print("No tickers", file=sys.stderr)
        return 1

    for ticker in tickers:
        df = _download_one(ticker, start, end)
        if not args.dry_run:
            buf = io.BytesIO()
            df.to_parquet(buf, index=False)
            storage.write_bytes(f"prices/{ticker}.parquet", buf.getvalue())
        if not df.empty:
            preview = df.iloc[[0, -1]][["date", "Open", "Close", "Adj Close"]]
            print(
                f"{ticker}: {len(df)} rows\n{preview.to_string(index=False)}",
                file=sys.stderr,
            )
        else:
            print(f"{ticker}: no data", file=sys.stderr)
        time.sleep(args.sleep)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
