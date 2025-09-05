"""Price data provider built on top of yfinance."""

from __future__ import annotations

from datetime import date

import pandas as pd
import yfinance as yf


def get_daily_adjusted(ticker: str, start: date | str = "1990-01-01", end: date | None = None) -> pd.DataFrame:
    """Fetch daily OHLCV with adjustments using yfinance.

    Returns a DataFrame with columns:
    ``date, open, high, low, close, adj_close, volume, ticker``.
    """

    if end is None:
        end = date.today()
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Adj Close", "Volume"])
    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    df = df.reset_index().rename(columns={"Date": "date"})
    df["ticker"] = ticker
    return df[["date", "open", "high", "low", "close", "adj_close", "volume", "ticker"]]
