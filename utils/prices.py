from __future__ import annotations

import pandas as pd

from data_lake.provider import get_daily_adjusted


def fetch_history(
    ticker: str,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return daily OHLCV (+ Adj Close) for ``ticker``.

    Data is sourced from :func:`data_lake.provider.get_daily_adjusted` and
    returned with a ``DatetimeIndex`` and title-cased columns matching the
    previous ``yfinance`` output.
    """
    if not ticker:
        return pd.DataFrame()
    df = get_daily_adjusted(ticker, start=start or "1990-01-01", end=end)
    if df.empty:
        cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Ticker"]
        return pd.DataFrame(columns=cols)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date").sort_index()
    else:
        df = df.copy()
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df.dropna().sort_index()
    rename = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "adj_close": "Adj Close",
        "volume": "Volume",
        "ticker": "Ticker",
    }
    df = df.rename(columns=rename)
    # Ensure canonical column order when available
    cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Ticker"]
    df = df[[c for c in cols if c in df.columns]]
    return df
