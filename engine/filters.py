import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

PRICE_COLS = {
    "open": ["Open", "open", "OPEN", "o"],
    "high": ["High", "high", "HIGH", "h"],
}


def _col(df: pd.DataFrame, kind: str) -> str:
    for k in PRICE_COLS[kind]:
        if k in df.columns:
            return k
    raise KeyError(f"Missing price column for {kind}")


def _tp_to_frac(tp_value: float) -> float:
    """Accept fraction (0.32) or percent (32.0) and normalize to fraction."""

    if tp_value is None or pd.isna(tp_value):
        return float("nan")
    return float(tp_value / 100.0) if tp_value > 1.5 else float(tp_value)


def compute_precedent_events(
    prices_one_ticker: pd.DataFrame,
    asof_date: pd.Timestamp,
    tp_value,
    lookback_bdays: int,
    window_bdays: int,
    include_misses: bool = True,
    limit: int = 300,
):
    """Return (hits_count, events_list). No peeking past D-1. BDay aware."""

    if prices_one_ticker is None or prices_one_ticker.empty:
        return 0, []

    prices = prices_one_ticker.copy()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]

    asof = pd.Timestamp(asof_date).tz_localize(None)

    oc = _col(prices, "open")
    hc = _col(prices, "high")

    tp_frac = _tp_to_frac(float(tp_value) if tp_value is not None else float("nan"))
    if pd.isna(tp_frac) or tp_frac <= 0:
        return 0, []

    hist = prices.loc[prices.index < asof]
    if hist.empty:
        return 0, []

    start_min = asof - BDay(int(lookback_bdays))
    starts = pd.bdate_range(max(start_min, hist.index.min()), asof - BDay(1))
    starts = [s for s in starts if s in hist.index]

    hits = 0
    events = []

    for S in starts:
        entry_vals = hist.loc[S, oc]
        if isinstance(entry_vals, pd.Series):
            entry_open = float(entry_vals.iloc[-1])
        else:
            entry_open = float(entry_vals)

        target_abs = entry_open * (1.0 + tp_frac)
        endS = min(S + BDay(int(window_bdays)), asof - BDay(1))
        fwd = hist.loc[(hist.index > S) & (hist.index <= endS)]
        if fwd.empty:
            continue

        hit_mask = fwd[hc] >= target_abs
        if hit_mask.any():
            first_hit = hit_mask.idxmax()
            days_to_hit = pd.bdate_range(S, first_hit).size - 1
            hit = True
            hits += 1
        else:
            days_to_hit = None
            hit = False

        max_high = float(fwd[hc].max())
        max_gain_pct = (max_high / entry_open - 1.0) * 100.0

        if include_misses or hit:
            events.append(
                {
                    "date": str(S.date()),
                    "entry_price": round(entry_open, 6),
                    "target_pct": round(tp_frac * 100.0, 6),
                    "max_gain_pct": round(max_gain_pct, 6),
                    "days_to_hit": int(days_to_hit) if days_to_hit is not None else None,
                    "hit": bool(hit),
                }
            )

        if len(events) >= int(limit):
            break

    return hits, events

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
    hits, _events = compute_precedent_events(
        prices_one_ticker,
        asof_date,
        tp_value,
        lookback_bdays=lookback_bdays,
        window_bdays=window_bdays,
        include_misses=True,
        limit=max(int(lookback_bdays) + 10, 1000),
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
