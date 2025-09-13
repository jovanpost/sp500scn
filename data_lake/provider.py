from __future__ import annotations

from datetime import date

import pandas as pd

from .storage import Storage


def get_daily_adjusted(
    ticker: str,
    start: date | str = "1990-01-01",
    end: date | None = None,
) -> pd.DataFrame:
    """Fetch daily OHLCV for ``ticker`` from Supabase.

    Returns columns ``date, open, high, low, close, adj_close, volume, ticker``.
    ``adj_close`` mirrors ``close`` as adjusted values are not provided.
    """
    storage = Storage()
    supabase = storage.supabase_client
    if end is None:
        end = date.today()
    response = (
        supabase.table("sp500_ohlcv")
        .select("date, open, high, low, close, volume")
        .eq("ticker", ticker)
        .gte("date", str(start))
        .lte("date", str(end))
        .order("date")
        .limit(None)
        .execute()
    )
    data = response.data or []
    df = pd.DataFrame(data)
    if df.empty:
        df = pd.DataFrame(
            columns=["date", "open", "high", "low", "close", "volume"]
        )
    df["adj_close"] = df.get("close")
    df["ticker"] = ticker
    return df[["date", "open", "high", "low", "close", "adj_close", "volume", "ticker"]]
