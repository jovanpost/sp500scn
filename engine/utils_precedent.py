from typing import Tuple, List, Dict

import pandas as pd
from pandas.tseries.offsets import BDay

PRICE_COLS = {
    "open": ["Open", "open", "OPEN", "o"],
    "high": ["High", "high", "HIGH", "h"],
}


def _col(df: pd.DataFrame, kind: str) -> str:
    for candidate in PRICE_COLS[kind]:
        if candidate in df.columns:
            return candidate
    raise KeyError(f"Missing price column for {kind}")


def _tp_to_frac(tp_value: float) -> float:
    """Normalize target percent to a fraction."""

    if tp_value is None or pd.isna(tp_value):
        return float("nan")
    return float(tp_value / 100.0) if tp_value > 1.5 else float(tp_value)


def compute_precedent_hit_details(
    prices_one_ticker: pd.DataFrame,
    asof_date: pd.Timestamp,
    tp_value,
    lookback_bdays: int = 252,
    window_bdays: int = 21,
    limit: int = 500,
) -> Tuple[int, List[Dict[str, object]]]:
    """Return count and details of precedent hits for a given target move."""

    oc = _col(prices_one_ticker, "open")
    hc = _col(prices_one_ticker, "high")

    tp_frac = _tp_to_frac(tp_value)
    if pd.isna(tp_frac) or tp_frac <= 0:
        return 0, []

    prices = prices_one_ticker.copy()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = prices.sort_index()

    D = pd.Timestamp(asof_date).tz_localize(None)

    hist = prices.loc[prices.index < D]
    if hist.empty:
        return 0, []

    start_min = D - BDay(lookback_bdays)
    starts = pd.bdate_range(max(start_min, hist.index.min()), D - BDay(1))
    starts = [s for s in starts if s in hist.index]

    hits: List[Dict[str, object]] = []
    for S in starts:
        entry_vals = hist.loc[S, oc]
        oS = (
            float(entry_vals.iloc[-1])
            if isinstance(entry_vals, pd.Series)
            else float(entry_vals)
        )
        target_abs = oS * (1.0 + tp_frac)
        endS = min(S + BDay(window_bdays), D - BDay(1))
        fwd = hist.loc[(hist.index > S) & (hist.index <= endS)]
        if fwd.empty:
            continue

        hit_mask = fwd[hc] >= target_abs
        if hit_mask.any():
            first_hit = hit_mask.idxmax()
            days_to_hit = pd.bdate_range(S, first_hit).size - 1
            max_high = float(fwd[hc].max())
            max_gain_pct = (max_high / oS - 1.0) * 100.0
            hits.append(
                {
                    "date": str(S.date()),
                    "entry_price": round(oS, 6),
                    "target_pct": round(tp_frac * 100.0, 6),
                    "days_to_hit": int(days_to_hit),
                    "max_gain_pct": round(max_gain_pct, 6),
                }
            )
            if len(hits) >= limit:
                break

    return len(hits), hits
