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
    cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    if not ticker:
        return pd.DataFrame(columns=cols)
    s = Storage.from_env()
    df = load_prices_cached(s, [ticker], start, end, cache_salt=s.cache_salt())
    if "Ticker" in df.columns:
        df = df[df["Ticker"] == ticker]
    if df.empty:
        return pd.DataFrame(columns=cols)
    return df[cols].copy()
