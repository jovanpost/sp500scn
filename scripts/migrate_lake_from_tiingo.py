#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import os
import sys
import time
import typing as t

import pandas as pd
import pandas_datareader as pdr

from data_lake.storage import Storage


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


def _validate(df: pd.DataFrame, ticker: str) -> None:
    if df is None or df.empty:
        raise ValueError(f"{ticker}: empty frame")

    missing = [c for c in SCHEMA if c not in df.columns]
    if missing:
        raise ValueError(f"{ticker}: missing columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
    if df["date"].isna().any():
        raise ValueError(f"{ticker}: bad dates present")


def _to_schema(df: pd.DataFrame, tkr: str) -> pd.DataFrame:
    df = df.reset_index()  # has columns: ['symbol','date', ...]
    df = df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "adjClose": "Adj Close",
            "volume": "Volume",
            "divCash": "Dividends",
            "splitFactor": "_split",
        }
    )

    out = pd.DataFrame(index=range(len(df)))
    out["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    out["Ticker"] = tkr.upper()
    out["Open"] = df.get("Open", pd.NA)
    out["High"] = df.get("High", pd.NA)
    out["Low"] = df.get("Low", pd.NA)
    out["Close"] = df.get("Close", pd.NA)
    out["Adj Close"] = df.get("Adj Close", pd.NA)
    out["Volume"] = df.get("Volume", pd.NA)
    out["Dividends"] = df.get("Dividends", 0.0)

    # Map Tiingo splitFactor (1.0=no split) to our "Stock Splits" (0.0=no split)
    sp = df.get("_split", 1.0)
    out["Stock Splits"] = (
        pd.to_numeric(sp, errors="coerce")
        .fillna(1.0)
        .apply(lambda x: 0.0 if x == 1.0 else float(x))
    )
    return out[SCHEMA].sort_values("date")


def _write_with_backup(storage: Storage, tkr: str, df: pd.DataFrame) -> None:
    dest = f"prices/{tkr.upper()}.parquet"
    ts = pd.Timestamp.utcnow().strftime("%Y%m%d%H%M%S")

    # best-effort backup
    try:
        if getattr(storage, "exists", None) and storage.exists(dest):
            prev = storage.read_bytes(dest)
            storage.write_bytes(f"backups/prices/{tkr.upper()}.{ts}.parquet", prev)
    except Exception:
        pass

    buf = io.BytesIO()
    df.to_parquet(buf, index=False, compression="snappy")
    storage.write_bytes(dest, buf.getvalue())


def fetch_tiingo(tkr: str, start: str | None, end: str | None, api_key: str) -> pd.DataFrame:
    # Tiingo expects BRK.B style tickers as-is; keep your filename/column as original uppercase
    df = pdr.get_data_tiingo(tkr, start=start, end=end, api_key=api_key)
    return _to_schema(df, tkr)


def main(argv: t.Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Migrate RAW OHLC from Tiingo into lake.")
    ap.add_argument(
        "--ticker",
        required=True,
        help="Single ticker (repeat step for many) or comma-separated list",
    )
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--sleep", type=float, default=0.4)
    args = ap.parse_args(argv)

    api_key = os.getenv("TIINGO_API_KEY", "")
    if not api_key:
        print("TIINGO_API_KEY not set", file=sys.stderr)
        return 2

    storage = Storage.from_env()

    tickers = [t for t in args.ticker.replace(",", " ").upper().split() if t]
    ok = fail = 0
    for tkr in tickers:
        try:
            df = fetch_tiingo(tkr, args.start, args.end, api_key)
            _validate(df, tkr)
            _write_with_backup(storage, tkr, df)
            print(
                f"{tkr}: wrote {len(df)} rows [{df['date'].min().date()} → {df['date'].max().date()}]"
            )
            ok += 1
            time.sleep(max(args.sleep, 0))
        except Exception as e:  # noqa: PERF203 - simple CLI tool
            print(f"{tkr}: ERROR {e}", file=sys.stderr)
            fail += 1
    print(f"Done. ok={ok}, failed={fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
