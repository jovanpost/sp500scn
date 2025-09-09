from __future__ import annotations
import pandas as pd
from typing import Optional

ATR_TP_MULT = 3.0  # target must be <= ATR_TP_MULT * ATR%

def last_pivot_low(lows: pd.Series, radius: int = 3, max_lookback: int = 21) -> Optional[float]:
    """Return the most recent pivot low within lookback or None.

    A pivot low is defined as a bar whose low is less than or equal to the lows
    of ``radius`` bars on both sides. Only pivots within ``max_lookback`` bars
    from the end of ``lows`` are considered.
    """
    n = len(lows)
    for i in range(n - radius - 1, radius - 1, -1):
        left = lows.iloc[i - radius:i]
        right = lows.iloc[i + 1:i + 1 + radius]
        if lows.iloc[i] <= left.min() and lows.iloc[i] <= right.min():
            if i >= n - max_lookback - 1:
                return float(lows.iloc[i])
    return None

def support_resistance(
    df: pd.DataFrame, D: pd.Timestamp, entry_price: float
) -> tuple[Optional[float], Optional[float]]:
    """Compute support and resistance as of D-1.

    Returns (support_S, resistance_R). ``df`` must contain columns 'high' and
    'low' indexed by date. ``entry_price`` is used to filter supports above the
    entry.
    """
    if D not in df.index:
        return None, None
    s_loc = df.index.get_loc(D) - 1
    if s_loc < 0:
        return None, None
    highs = df['high'].iloc[:s_loc + 1]
    lows = df['low'].iloc[:s_loc + 1]
    R = highs.rolling(21, min_periods=1).max().iloc[-1]
    pivot = last_pivot_low(lows, radius=3, max_lookback=21)
    roll_min10 = lows.rolling(10, min_periods=1).min().iloc[-1]
    low_dm1 = lows.iloc[-1]
    candidates = [v for v in [pivot, roll_min10, low_dm1] if v is not None and v <= entry_price]
    S = max(candidates) if candidates else None
    return S, float(R)

def has_21d_forward_hit(
    df: pd.DataFrame, s_loc: int, tp_pct: float, lookback: int = 251
) -> bool:
    """Check if within the past ``lookback`` sessions a 21d forward move hit ``tp_pct``.

    Uses only data strictly prior to D to avoid look-ahead. ``s_loc`` is the
    integer location of D-1 in ``df``.
    """
    if tp_pct is None or tp_pct <= 0:
        return False
    start = max(0, (s_loc - 21) - lookback + 1)
    seg = df.iloc[start:s_loc + 1]
    if len(seg) <= 21:
        return False
    fwd_max = seg['high'].shift(-1).rolling(21, min_periods=21).max()
    rr = (fwd_max / seg['close']) - 1
    rr = rr.dropna()
    return bool((rr >= tp_pct).any())


def is_atr_feasible(tp_pct: float, atr_pct: float, mult: float = ATR_TP_MULT) -> bool:
    """Return True if tp_pct is <= mult * atr_pct."""
    if pd.isna(atr_pct) or atr_pct <= 0:
        return False
    return tp_pct <= mult * atr_pct
