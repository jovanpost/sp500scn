from __future__ import annotations

import pandas as pd

from data_lake.storage import Storage, load_prices_cached


def fetch_history(
    ticker: str,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return daily OHLCV (+ Adj Close) for ``ticker``.

    Data is sourced from Supabase via :func:`data_lake.storage.load_prices_cached`
    and returned with a ``DatetimeIndex`` and title-cased columns similar to
    previous APIs.
    """
    if not ticker:
        return pd.DataFrame()
    storage = Storage()
    start_ts = pd.to_datetime(start or "1990-01-01")
    end_ts = pd.to_datetime(end) if end else pd.Timestamp.today()
    df = load_prices_cached(storage, [ticker], start_ts, end_ts)
    if df.empty:
        cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Ticker"]
        return pd.DataFrame(columns=cols)
    df = df[df.get("ticker") == ticker]
    df = df.drop(columns=["ticker"], errors="ignore")
    df["Adj Close"] = df.get("Close")
    cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    df = df[cols]
    df["Ticker"] = ticker
    return df
