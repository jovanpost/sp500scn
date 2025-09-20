from __future__ import annotations
import numpy as np
import pandas as pd


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def _wilder_rma(tr: pd.Series, window: int) -> pd.Series:
    """Return Wilder's moving average (RMA) seeded with the SMA of the first window."""

    if window <= 0:
        raise ValueError("window must be positive for Wilder ATR")

    values = tr.astype(float).to_numpy()
    out = np.full_like(values, fill_value=np.nan, dtype=float)
    if len(values) < window:
        return pd.Series(out, index=tr.index)

    # Seed with the simple average of the first `window` true ranges.
    seed = float(np.nanmean(values[:window]))
    out[window - 1] = seed
    prev = seed

    for i in range(window, len(values)):
        val = values[i]
        if np.isnan(val):
            out[i] = prev
            continue
        prev = ((window - 1) * prev + val) / window
        out[i] = prev

    return pd.Series(out, index=tr.index)


def atr(df: pd.DataFrame, window: int = 14, method: str = "wilder") -> pd.Series:
    """Average True Range supporting Wilder (RMA), SMA, and EMA variants."""

    method = (method or "wilder").strip().lower()
    tr = true_range(df)

    if method in ("wilder", "rma"):
        return _wilder_rma(tr, window)
    if method == "sma":
        return tr.rolling(window, min_periods=window).mean()
    if method == "ema":
        return tr.ewm(span=window, adjust=False, min_periods=window).mean()
    raise ValueError(f"Unsupported ATR method: {method}")


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
