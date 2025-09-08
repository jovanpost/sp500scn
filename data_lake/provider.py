"""Price data provider built on top of yfinance."""

from __future__ import annotations

from datetime import date
from typing import Iterable

import pandas as pd
import yfinance as yf

from .schemas import IngestJob
from .ingest import ingest_batch
from .storage import Storage


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


def ingest(
    storage: Storage,
    tickers: Iterable[str],
    start: date | str = "1990-01-01",
    end: date | None = None,
    progress_cb=None,
):
    """Ingest a collection of tickers into storage."""
    if end is None:
        end = date.today()
    jobs: list[IngestJob] = [
        {"ticker": t, "start": str(start), "end": str(end)} for t in tickers
    ]
    return ingest_batch(storage, jobs, progress_cb=progress_cb)
