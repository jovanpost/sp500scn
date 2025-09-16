import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from .replay import _col

def _calc_atr(series_h, series_l, series_c, window: int) -> pd.Series:
    cprev = series_c.shift(1)
    tr = pd.concat([
        (series_h - series_l).abs(),
        (series_h - cprev).abs(),
        (series_l - cprev).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=window).mean()


def has_21d_precedent(df: pd.DataFrame, asof_idx: int, required_pct: float,
                      lookback_days: int = 252, window: int = 21) -> bool:
    """
    df must have columns: 'close','high'. asof_idx corresponds to D-1.
    For each t in [asof_idx - lookback_days, asof_idx], check if the next `window`
    days ever achieve >= required_pct gain vs close[t].
    """
    if df is None or df.empty or pd.isna(required_pct):
        return False
    start = max(0, asof_idx - lookback_days)
    highs = df["high"].to_numpy()
    closes = df["close"].to_numpy()
    n = len(df)
    for t in range(start, asof_idx + 1):
        base = closes[t]
        end = min(t + 1 + window, n)
        if t + 1 >= end:
            continue
        max_fwd_high = np.max(highs[t+1:end])
        if base > 0 and (max_fwd_high / base - 1.0) >= required_pct:
            return True
    return False


def precedent_ok_pct_target(
    prices_one_ticker: pd.DataFrame,
    asof_date: pd.Timestamp,
    tp_pct: float,
    lookback_bdays: int,
    window_bdays: int,
) -> bool:
    """Check if a percent target was hit in the recent past."""

    if pd.isna(tp_pct):
        return False

    prices = prices_one_ticker.copy()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = prices.sort_index()

    asof_date = pd.Timestamp(asof_date).tz_localize(None)

    hc = _col(prices, "high")
    oc = _col(prices, "open")

    start_lb = asof_date - BDay(lookback_bdays + window_bdays)
    past = prices.loc[(prices.index >= start_lb) & (prices.index < asof_date)]
    if past.empty:
        return False

    bdays = past.index.unique()
    if len(bdays) < window_bdays + 1:
        return False

    starts = pd.bdate_range(asof_date - BDay(lookback_bdays), asof_date - BDay(1))
    starts = [s for s in starts if s in bdays]

    for S in starts:
        entry_vals = past.loc[S, oc]
        if isinstance(entry_vals, pd.Series):
            entry_open = float(entry_vals.iloc[-1])
        else:
            entry_open = float(entry_vals)
        target_abs = entry_open * (1.0 + tp_pct / 100.0)
        endS = S + BDay(window_bdays)
        fwd = prices.loc[(prices.index > S) & (prices.index <= endS)]
        if not fwd.empty and (fwd[hc] >= target_abs).any():
            return True
    return False


def atr_feasible(df: pd.DataFrame, asof_idx: int, required_pct: float, atr_window: int) -> bool:
    """
    df must have: high, low, close, open. asof_idx = D-1; entry is open at asof_idx+1 (if exists).
    Checks: ATR(at D-1) * atr_window >= entry_price * required_pct
    """
    if df is None or df.empty or pd.isna(required_pct):
        return False
    if asof_idx + 1 >= len(df):
        return False
    entry_price = float(df["open"].iloc[asof_idx + 1])
    atr = _calc_atr(df["high"], df["low"], df["close"], atr_window).iloc[asof_idx]
    if pd.isna(atr):
        return False
    required_dollars = entry_price * required_pct
    return float(atr) * atr_window >= required_dollars
