import numpy as np
import pandas as pd

from .utils_precedent import compute_precedent_hit_details

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
def precedent_hits_pct_target(
    prices_one_ticker: pd.DataFrame,
    asof_date: pd.Timestamp,
    tp_value: float,
    lookback_bdays: int = 252,
    window_bdays: int = 21,
) -> int:
    """Count precedent hits for a percent target without peeking beyond D-1."""

    hits, _ = compute_precedent_hit_details(
        prices_one_ticker=prices_one_ticker,
        asof_date=asof_date,
        tp_value=tp_value,
        lookback_bdays=lookback_bdays,
        window_bdays=window_bdays,
    )
    return int(hits)


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
