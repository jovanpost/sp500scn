import pandas as pd
import yfinance as yf


def fetch_history(
    ticker: str,
    *,
    start=None,
    end=None,
    period: str | None = None,
    auto_adjust: bool = False,
    actions: bool = False,
):
    """Fetch historical price data for ``ticker`` using yfinance.

    Parameters mirror ``yfinance.Ticker.history`` with sensible defaults and
    normalization applied to the resulting frame.  ``auto_adjust`` controls
    whether adjusted prices are returned.

    Returns ``pandas.DataFrame`` with a ``DatetimeIndex`` (tz-naive) and title-
    cased columns converted to numeric where possible.  ``None`` is returned on
    any fetch error or empty result.
    """
    try:
        df = yf.Ticker(ticker).history(
            start=start,
            end=end,
            period=period,
            auto_adjust=auto_adjust,
            actions=actions,
        )
    except Exception:
        return None

    if df is None or df.empty:
        return None

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)

    df.columns = [str(c).title() for c in df.columns]
    numeric_cols = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
    for col in numeric_cols & set(df.columns):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df
