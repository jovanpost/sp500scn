import pandas as pd
import numpy as np


def has_21d_runup_precedent(
    df: pd.DataFrame,
    lookback_days: int,
    horizon_days: int,
    target_move_pct: float,
) -> bool:
    """Return True if any window meets the target move.

    df: sorted by date asc; must include columns ['date','close','high'] and end at D-1.
    Look back `lookback_days` from D-1. Return True if any rolling window of
    `horizon_days` has max(high)/close0 - 1 >= target_move_pct.
    """
    if df is None or df.empty or pd.isna(target_move_pct):
        return False

    tail = df.tail(int(lookback_days))
    if len(tail) < horizon_days:
        return False

    highs = tail["high"].to_numpy()
    closes = tail["close"].to_numpy()
    n = len(tail)
    for i in range(n - horizon_days + 1):
        close0 = closes[i]
        if close0 and not np.isnan(close0):
            window_max = np.nanmax(highs[i : i + horizon_days])
            if close0 > 0 and (window_max / close0 - 1.0) >= target_move_pct:
                return True
    return False

