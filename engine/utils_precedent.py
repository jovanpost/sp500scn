from __future__ import annotations

from typing import Tuple, List, Dict, Optional

import pandas as pd

_PRICE_COLS = {
    "open": ["Open", "open", "OPEN", "o"],
    "high": ["High", "high", "HIGH", "h"],
}


def _col(df: pd.DataFrame, kind: str) -> str:
    for c in _PRICE_COLS[kind]:
        if c in df.columns:
            return c
    raise KeyError(f"Missing price column for {kind}")


def tp_fraction_from_row(
    entry_open: Optional[float],
    tp_price_abs_target: Optional[float],
    tp_halfway_pct: Optional[float],
    tp_price_pct_target: Optional[float],
) -> float:
    """Derive the take-profit fraction from available row fields."""

    if entry_open is not None and tp_price_abs_target is not None:
        try:
            if entry_open > 0 and tp_price_abs_target > 0:
                return float(tp_price_abs_target) / float(entry_open) - 1.0
        except Exception:
            pass
    if tp_halfway_pct is not None and not pd.isna(tp_halfway_pct) and tp_halfway_pct > 0:
        return float(tp_halfway_pct)
    if (
        tp_price_pct_target is not None
        and not pd.isna(tp_price_pct_target)
        and tp_price_pct_target > 0
    ):
        return float(tp_price_pct_target) / 100.0
    return float("nan")


def compute_precedent_hit_details(
    prices_one_ticker: pd.DataFrame,
    asof_date: pd.Timestamp,
    tp_frac: float,
    lookback_bdays: int = 252,
    window_bdays: int = 21,
    limit: int = 500,
) -> Tuple[int, List[Dict[str, object]]]:
    """Trading-day precise, no peeking, first-touch-per-start, NO cross-start dedupe."""

    if tp_frac is None or pd.isna(tp_frac) or tp_frac <= 0:
        return 0, []

    df = prices_one_ticker.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index()

    D = pd.Timestamp(asof_date).tz_localize(None)
    oc = _col(df, "open")
    hc = _col(df, "high")

    hist = df.loc[df.index < D]
    if hist.empty:
        return 0, []

    idx = hist.index
    last_pre_D_pos = len(idx) - 1
    if last_pre_D_pos < 0:
        return 0, []

    earliest_pos = max(0, last_pre_D_pos - int(lookback_bdays))
    starts = idx[earliest_pos : last_pre_D_pos + 1]

    hits: List[Dict[str, object]] = []
    EPS = 1e-9

    for S in starts:
        oS_val = hist.loc[S, oc]
        oS = float(oS_val.iloc[-1]) if isinstance(oS_val, pd.Series) else float(oS_val)
        target_abs = oS * (1.0 + float(tp_frac))

        s_pos = idx.get_loc(S)
        end_pos = min(s_pos + int(window_bdays), last_pre_D_pos)
        if end_pos <= s_pos:
            continue

        fwd = hist.iloc[s_pos + 1 : end_pos + 1]
        if fwd.empty:
            continue

        hit_mask = fwd[hc] >= (target_abs - EPS)
        if hit_mask.any():
            first_hit = hit_mask.idxmax()
            hit_pos = idx.get_loc(first_hit)
            days_to_hit = int(hit_pos - s_pos)
            max_high = float(fwd[hc].max())
            max_gain_pct = (max_high / oS - 1.0) * 100.0

            hits.append(
                {
                    "date": str(pd.Timestamp(S).date()),
                    "entry_price": round(oS, 6),
                    "target_pct": round(float(tp_frac) * 100.0, 6),
                    "days_to_hit": days_to_hit,
                    "max_gain_pct": round(max_gain_pct, 6),
                }
            )
            if len(hits) >= limit:
                break

    return len(hits), hits
