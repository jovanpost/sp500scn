#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import random
import sys
import time
import typing as t
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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


def make_yf_session() -> requests.Session:
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
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def _yahoo_symbol(tkr: str) -> str:
    """Return the Yahoo-compatible ticker symbol."""

    return tkr.replace(".", "-").upper()


def _validate_prices_schema(df: pd.DataFrame, ticker: str) -> None:
    if df is None or df.empty:
        raise ValueError(f"{ticker}: empty frame")

    missing = [c for c in SCHEMA if c not in df.columns]
    if missing:
        raise ValueError(f"{ticker}: missing columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
    if df["date"].isna().any():
        raise ValueError(f"{ticker}: bad dates present")

    for col in [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Dividends",
        "Stock Splits",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

    if not df["Ticker"].astype(str).str.upper().eq(df["Ticker"]).all():
        raise ValueError(f"{ticker}: Ticker must be uppercase")

    actions = (df["Dividends"].fillna(0).ne(0)) | (df["Stock Splits"].fillna(0).ne(0))
    if actions.any():
        sub = df.loc[actions, ["Close", "Adj Close"]].dropna()
        if not sub.empty:
            same = np.isclose(sub["Close"], sub["Adj Close"], rtol=0, atol=1e-6)
            if same.all():
                raise ValueError(f"{ticker}: Close == Adj Close on action days (looks adjusted)")


def _download_raw_yahoo(tkr: str, start: str | None, end: str | None) -> pd.DataFrame:
    sess = make_yf_session()
    symbol = _yahoo_symbol(tkr)
    y = yf.download(
        symbol,
        start=start,
        end=end,
        auto_adjust=False,
        actions=True,
        progress=False,
        threads=False,
        session=sess,
    )

    if y is None or y.empty:
        y = yf.download(
            symbol,
            period="30d",
            auto_adjust=False,
            actions=True,
            progress=False,
            threads=False,
            session=sess,
        )

    if y is None or y.empty:
        raise RuntimeError(
            f"Yahoo returned no rows for {symbol} "
            f"[{start or 'period'}..{end or 'now'}] after retries."
        )

    if isinstance(y.columns, pd.MultiIndex):
        y = y.droplevel(1, axis=1)

    y = y.reset_index().rename(columns={"Date": "date"})

    out = pd.DataFrame(index=range(len(y)))
    out["date"] = pd.to_datetime(y["date"]).dt.tz_localize(None)
    out["Ticker"] = tkr.upper()
    out["Open"] = y.get("Open", pd.NA)
    out["High"] = y.get("High", pd.NA)
    out["Low"] = y.get("Low", pd.NA)
    out["Close"] = y.get("Close", pd.NA)
    out["Adj Close"] = y.get("Adj Close", pd.NA)
    out["Volume"] = y.get("Volume", pd.NA)
    out["Dividends"] = y.get("Dividends", 0.0 if "Dividends" in y else 0.0)
    out["Stock Splits"] = y.get("Stock Splits", 0.0 if "Stock Splits" in y else 0.0)

    out = out[SCHEMA].sort_values("date")
    return out


def _list_existing_tickers(storage: Storage, prefix: str = "prices") -> list[str]:
    items = storage.list_prefix(prefix)
    tickers: list[str] = []
    for key in items:
        name = key.split("/")[-1]
        if name.lower().endswith(".parquet"):
            tickers.append(name[:-8].upper())
    return sorted(set(tickers))


def _backup_and_write(
    storage: Storage,
    ticker: str,
    df: pd.DataFrame,
    *,
    dest_prefix: str = "prices",
    backup_prefix: str = "backups/prices",
) -> None:
    target = f"{dest_prefix}/{ticker}.parquet"
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    if storage.exists(target):
        try:
            old = storage.read_bytes(target)
            backup_key = f"{backup_prefix}/{ticker}.parquet.{timestamp}.bak"
            storage.write_bytes(backup_key, old)
        except Exception as exc:  # pragma: no cover - best effort logging
            print(f"[WARN] {ticker}: backup failed: {exc}", file=sys.stderr)

    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    storage.write_bytes(target, buffer.getvalue())


def main(argv: t.Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Migrate lake to canonical RAW OHLC using Yahoo",
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        help="Tickers to migrate (default: all found under prices/)",
    )
    parser.add_argument("--start", help="Start date (YYYY-MM-DD), optional")
    parser.add_argument("--end", help="End date (YYYY-MM-DD), optional")
    parser.add_argument("--sleep", type=float, default=0.25)
    parser.add_argument("--limit", type=int, help="Limit number of tickers")

    args = parser.parse_args(argv)

    storage = Storage.from_env()
    tickers = [t.upper() for t in (args.tickers or [])]

    if not tickers:
        tickers = _list_existing_tickers(storage, prefix="prices")

    if args.limit:
        tickers = tickers[: args.limit]

    if not tickers:
        print(
            "No tickers found. Put some files in prices/ or pass --tickers.",
            file=sys.stderr,
        )
        return 2

    ok = 0
    failed = 0

    for ticker in tickers:
        try:
            df = _download_raw_yahoo(ticker, args.start, args.end)
            if df.empty:
                print(
                    f"{ticker}: no rows from Yahoo in [{args.start}..{args.end}]",
                    file=sys.stderr,
                )
                failed += 1
                continue

            _validate_prices_schema(df, ticker)
            _backup_and_write(
                storage,
                ticker,
                df,
                dest_prefix="prices",
                backup_prefix="backups/prices",
            )
            print(
                f"{ticker}: wrote {len(df)} rows"
                f" [{df['date'].min().date()} → {df['date'].max().date()}]"
            )
            ok += 1
            sleep_for = max(args.sleep, 0) + 0.3 + random.random() * 0.7
            time.sleep(sleep_for)
        except Exception as exc:  # pragma: no cover - top-level logging
            print(f"{ticker}: ERROR {exc}", file=sys.stderr)
            failed += 1

    print(f"Done. ok={ok}, failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
