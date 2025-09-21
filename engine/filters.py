import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from .features import atr as _atr
from .replay import _col


def _calc_atr(
    series_h: pd.Series,
    series_l: pd.Series,
    series_c: pd.Series,
    window: int,
    method: str = "wilder",
) -> pd.Series:
    """Internal ATR helper supporting Wilder and SMA modes."""

    df = pd.DataFrame({"high": series_h, "low": series_l, "close": series_c})
    method = (method or "wilder").strip().lower()
    if method in ("wilder", "rma", "sma", "ema"):
        return _atr(df, window=window, method=method)

    raise ValueError(f"Unsupported ATR method: {method}")


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


def _normalize_tp_pct(tp_value: float) -> float:
    """Normalize TP% values supplied as either fraction or percent."""

    if tp_value is None or pd.isna(tp_value):
        return float("nan")
    return float(tp_value / 100.0) if tp_value > 1.5 else float(tp_value)


def precedent_hits_pct_target(
    prices_one_ticker: pd.DataFrame,
    asof_date: pd.Timestamp,
    tp_value: float,
    lookback_bdays: int = 252,
    window_bdays: int = 21,
) -> int:
    """Count precedent hits for a percent target without peeking beyond D-1."""

    prices = prices_one_ticker.copy()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = prices.sort_index()

    asof_date = pd.Timestamp(asof_date).tz_localize(None)

    oc = _col(prices, "open")
    hc = _col(prices, "high")

    tp_frac = _normalize_tp_pct(tp_value)
    if pd.isna(tp_frac) or tp_frac <= 0:
        return 0

    hist = prices.loc[prices.index < asof_date]
    if hist.empty:
        return 0

    start_min = asof_date - BDay(lookback_bdays)
    start_lb = max(start_min, hist.index.min())
    start_ub = asof_date - BDay(1)
    if start_lb > start_ub:
        return 0

    starts = pd.bdate_range(start_lb, start_ub)
    starts = [s for s in starts if s in hist.index]
    if not starts:
        return 0

    hits = 0
    for S in starts:
        entry_vals = hist.loc[S, oc]
        if isinstance(entry_vals, pd.Series):
            entry_open = float(entry_vals.iloc[-1])
        else:
            entry_open = float(entry_vals)
        target_abs = entry_open * (1.0 + tp_frac)
        endS = min(S + BDay(window_bdays), asof_date - BDay(1))
        fwd = hist.loc[(hist.index > S) & (hist.index <= endS)]
        if not fwd.empty and (fwd[hc] >= target_abs).any():
            hits += 1
    return int(hits)


def atr_feasible(
    df: pd.DataFrame,
    asof_idx: int,
    required_pct: float,
    atr_window: int,
    atr_method: str = "wilder",
) -> bool:
    """
    df must have: high, low, close, open. asof_idx = D-1; entry is open at asof_idx+1 (if exists).
    Checks: ATR(at D-1) * atr_window >= entry_price * required_pct
    """
    if df is None or df.empty or pd.isna(required_pct):
        return False
    df = df.reset_index(drop=True)
    if asof_idx + 1 >= len(df):
        return False
    entry_price = float(df["open"].iloc[asof_idx + 1])
    atr = _calc_atr(
        df["high"], df["low"], df["close"], atr_window, method=atr_method
    ).iloc[asof_idx]
    if pd.isna(atr):
        return False
    required_dollars = entry_price * required_pct
    return float(atr) * atr_window >= required_dollars
