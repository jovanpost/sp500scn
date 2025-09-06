"""Price data provider built on top of yfinance."""

from __future__ import annotations

from datetime import date

import pandas as pd
import yfinance as yf


def get_daily_adjusted(
    ticker: str, start: date | str = "1990-01-01", end: date | None = None
) -> pd.DataFrame:
    """Fetch daily OHLCV with adjustments using yfinance.

    Returns a DataFrame with columns:
    ``date, open, high, low, close, adj_close, volume, ticker``.
    """

    if end is None:
        end = date.today()
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        df = pd.DataFrame(
            {
                "date": pd.Series(dtype="datetime64[ns]"),
                "open": pd.Series(dtype="float64"),
                "high": pd.Series(dtype="float64"),
                "low": pd.Series(dtype="float64"),
                "close": pd.Series(dtype="float64"),
                "adj_close": pd.Series(dtype="float64"),
                "volume": pd.Series(dtype="int64"),
                "ticker": pd.Series(dtype="object"),
            }
        )
        df["ticker"] = ticker
        return df[["date", "open", "high", "low", "close", "adj_close", "volume", "ticker"]]
    df = df.reset_index()
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    if "date" not in df.columns:
        df["date"] = pd.to_datetime(pd.Series([], dtype="datetime64[ns]"))
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
    df["ticker"] = ticker
    return df[["date", "open", "high", "low", "close", "adj_close", "volume", "ticker"]]
