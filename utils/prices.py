from __future__ import annotations

import pandas as pd
import yfinance as yf


def fetch_history(
    ticker: str,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    interval: str = "1d",
    auto_adjust: bool = False,
):
    """Fetch price history for *ticker* with normalized columns.

    Returns a DataFrame with title-cased columns and numeric price/volume fields
    or ``None`` if data cannot be retrieved. The helper removes timezone
    information from the index and gracefully handles any errors.
    """
    try:
        if not ticker:
            return None
        df = yf.Ticker(ticker).history(
            start=start,
            end=end,
            interval=interval,
            auto_adjust=auto_adjust,
            actions=False,
        )
        if df is None or df.empty:
            return None
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)
        df.columns = [c.title() for c in df.columns]
        for col in ("Open", "High", "Low", "Close", "Adj Close", "Volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception:
        return None
