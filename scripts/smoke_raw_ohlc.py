#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import sys

import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_lake.storage import Storage, load_prices_cached, validate_prices_schema


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-check RAW OHLC vs Adj Close")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--lake-root", default=None)
    args = parser.parse_args()

    storage = Storage.from_env()
    if args.lake_root:
        storage.local_root = args.lake_root

    start = pd.Timestamp(args.start or pd.Timestamp(args.date) - pd.Timedelta(days=3))
    end = pd.Timestamp(args.end or pd.Timestamp(args.date) + pd.Timedelta(days=3))

    df = load_prices_cached(
        storage,
        cache_salt=storage.cache_salt(),
        tickers=[args.ticker],
        start=start,
        end=end,
    )
    if df.empty:
        print("NO ROWS LOADED", file=sys.stderr)
        return 2

    validate_prices_schema(df)

    row = df[df["date"] == pd.Timestamp(args.date)]
    if row.empty:
        print(f"NO ROW FOR {args.ticker} @ {args.date}", file=sys.stderr)
        print(df.tail(5).to_string(index=False))
        return 3

    r = row.iloc[0]
    out = {
        "ticker": r["Ticker"],
        "date": str(pd.to_datetime(r["date"]).date()),
        "open": float(r["Open"]),
        "high": float(r["High"]),
        "low": float(r["Low"]),
        "close": float(r["Close"]),
        "adj_close": float(r["Adj Close"]) if pd.notna(r["Adj Close"]) else None,
        "dividends": float(r.get("Dividends", 0) or 0),
        "stock_splits": float(r.get("Stock Splits", 0) or 0),
        "volume": int(r["Volume"]) if pd.notna(r["Volume"]) else None,
    }
    print(out)

    if out["adj_close"] is not None and out["dividends"] > 0:
        if abs(out["adj_close"] - out["close"]) < 1e-9:
            print(
                "FAIL: Adj Close equals Close on a dividend day (looks adjusted).",
                file=sys.stderr,
            )
            return 4

    print("OK: RAW OHLC with separate Adj Close.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
