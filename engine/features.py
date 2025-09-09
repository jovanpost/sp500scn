from __future__ import annotations
import pandas as pd


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df['close'].shift(1)
    return pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_close).abs(),
        (df['low']  - prev_close).abs()
    ], axis=1).max(axis=1)


def atr(df: pd.DataFrame, window: int = 21, method: str = "sma") -> pd.Series:
    tr = true_range(df)
    if method == "ema":
        return tr.ewm(span=window, adjust=False, min_periods=window).mean()
    return tr.rolling(window, min_periods=window).mean()


def up_days_ratio(close: pd.Series, window: int = 21) -> pd.Series:
    up = (close > close.shift(1)).astype(float)
    return up.rolling(window, min_periods=window).mean()


def rolling_mean(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def pct_change(series: pd.Series, n: int) -> pd.Series:
    return series.pct_change(n)


def gap_from_prev_close(df: pd.DataFrame) -> pd.Series:
    """(Open_t / Close_{t-1} - 1), aligned on t."""
    return df['open'] / df['close'].shift(1) - 1
