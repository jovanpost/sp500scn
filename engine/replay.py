from __future__ import annotations

import pandas as pd
from pandas.tseries.offsets import BDay


PRICE_COLS = {
    "open": ["Open", "open", "OPEN", "o"],
    "high": ["High", "high", "HIGH", "h"],
    "low": ["Low", "low", "LOW", "l"],
    "close": ["Close", "close", "CLOSE", "c"],
}


def _col(df: pd.DataFrame, kind: str) -> str:
    for candidate in PRICE_COLS[kind]:
        if candidate in df.columns:
            return candidate
    raise KeyError(f"Missing price column for {kind}: tried {PRICE_COLS[kind]}")


def simulate_pct_target_only(
    prices_one_ticker: pd.DataFrame,
    entry_date: pd.Timestamp,
    entry_open: float,
    tp_pct: float,
    horizon_bdays: int,
) -> dict | None:
    """Simulate an exit path with a percent target only."""

    if pd.isna(entry_open) or pd.isna(tp_pct):
        return None

    entry_date = pd.Timestamp(entry_date).tz_localize(None)

    oc = _col(prices_one_ticker, "open")
    hc = _col(prices_one_ticker, "high")
    lc = _col(prices_one_ticker, "low")
    cc = _col(prices_one_ticker, "close")

    prices = prices_one_ticker.copy()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = prices.sort_index()

    end_date = entry_date + BDay(horizon_bdays)
    fwd = prices.loc[(prices.index >= entry_date) & (prices.index <= end_date)]
    if fwd.empty:
        return None

    target_abs = entry_open * (1.0 + tp_pct / 100.0)

    hit_mask = fwd[hc] >= target_abs
    tp_touch_date = pd.NaT
    if hit_mask.any():
        first_hit_idx = hit_mask[hit_mask].index[0]
        exit_pos = int(fwd.index.get_loc(first_hit_idx))
        hit = True
        exit_reason = "tp"
        exit_price = float(target_abs)
        tp_touch_date = pd.Timestamp(first_hit_idx).tz_localize(None)
    else:
        exit_pos = len(fwd) - 1
        hit = False
        exit_reason = "timeout"
        exit_price = float(fwd[cc].iloc[-1])

    exit_slice = fwd.iloc[: exit_pos + 1].copy()
    exit_row = exit_slice.iloc[-1]
    exit_date = pd.Timestamp(exit_row.name).tz_localize(None)

    max_high = exit_slice[hc].max()
    min_low = exit_slice[lc].min()

    mfe_idx = exit_slice[hc].idxmax()
    mae_idx = exit_slice[lc].idxmin()

    mfe_pct = float(((max_high / entry_open) - 1.0) * 100.0)
    mae_pct = float(((min_low / entry_open) - 1.0) * 100.0)

    days_to_exit = int(exit_pos + 1)

    return {
        "hit": bool(hit),
        "exit_reason": exit_reason,
        "exit_price": float(exit_price),
        "exit_date": exit_date,
        "exit_bar_high": float(exit_row[hc]) if pd.notna(exit_row[hc]) else float("nan"),
        "exit_bar_low": float(exit_row[lc]) if pd.notna(exit_row[lc]) else float("nan"),
        "tp_touch_date": tp_touch_date,
        "days_to_exit": days_to_exit,
        "mae_pct": mae_pct,
        "mae_date": pd.Timestamp(mae_idx).tz_localize(None) if pd.notna(mae_idx) else pd.NaT,
        "mfe_pct": mfe_pct,
        "mfe_date": pd.Timestamp(mfe_idx).tz_localize(None) if pd.notna(mfe_idx) else pd.NaT,
        "tp_price_pct_target": float(tp_pct),
        "tp_price_abs_target": float(target_abs),
    }


def replay_trade(
    bars: pd.DataFrame,  # columns: date, open, high, low, close
    entry_ts: pd.Timestamp,  # D (midnight) or normalized date
    entry_price: float,
    tp_price: float,
    stop_price: float | None,
    horizon_days: int = 30,
) -> dict:
    """Simulate a trade from entry_ts for up to horizon_days sessions."""

    def _empty_result(reason: str) -> dict:
        return {
            "hit": False,
            "exit_reason": reason,
            "exit_price": float("nan"),
            "exit_date": pd.NaT,
            "exit_bar_high": float("nan"),
            "exit_bar_low": float("nan"),
            "tp_touch_date": pd.NaT,
            "days_to_exit": 0,
            "mae_pct": float("nan"),
            "mae_date": pd.NaT,
            "mfe_pct": float("nan"),
            "mfe_date": pd.NaT,
        }

    if "date" not in bars.columns:
        return _empty_result("no_date_col")

    entry_ts = pd.Timestamp(entry_ts).tz_localize(None)

    # Dates should already be tz-naive normalized; make it idempotent
    d = pd.to_datetime(bars["date"], errors="coerce")
    # Drop timezone without shifting wall-clock time so comparisons remain valid
    d = d.dt.tz_localize(None)
    d = d.dt.normalize()
    bars = bars.assign(date=d).dropna(subset=["date"]).sort_values("date")

    # Slice D..D+H
    window = bars.loc[bars["date"] >= entry_ts].head(horizon_days + 1).copy()
    if window.empty:
        return _empty_result("no_window")

    exit_idx = len(window) - 1
    exit_reason = "timeout"
    exit_price = float(window.iloc[exit_idx]["close"])
    hit = False
    tp_touch_idx: int | None = None

    for i, row in enumerate(window.itertuples(index=False)):
        if (stop_price is not None) and (row.low <= stop_price):
            exit_idx = i
            exit_reason = "stop"
            exit_price = float(stop_price)
            hit = False
            break
        if row.high >= tp_price:
            exit_idx = i
            exit_reason = "tp"
            exit_price = float(tp_price)
            hit = True
            tp_touch_idx = i
            break

    exit_slice = window.iloc[: exit_idx + 1].copy()
    exit_row = exit_slice.iloc[-1]
    exit_date = pd.Timestamp(exit_row["date"]).tz_localize(None)

    max_high = exit_slice["high"].max()
    min_low = exit_slice["low"].min()
    mfe_idx = exit_slice["high"].idxmax()
    mae_idx = exit_slice["low"].idxmin()

    mfe_pct = float(((max_high / entry_price) - 1.0) * 100.0)
    mae_pct = float(((min_low / entry_price) - 1.0) * 100.0)

    mae_date = (
        pd.Timestamp(exit_slice.loc[mae_idx, "date"]).tz_localize(None)
        if pd.notna(mae_idx)
        else pd.NaT
    )
    mfe_date = (
        pd.Timestamp(exit_slice.loc[mfe_idx, "date"]).tz_localize(None)
        if pd.notna(mfe_idx)
        else pd.NaT
    )

    tp_touch_ts = (
        pd.Timestamp(window.iloc[tp_touch_idx]["date"]).tz_localize(None)
        if (hit and tp_touch_idx is not None)
        else pd.NaT
    )

    days_to_exit = int(exit_idx + 1)

    return {
        "hit": bool(hit),
        "exit_reason": exit_reason,
        "exit_price": float(exit_price),
        "exit_date": exit_date,
        "exit_bar_high": float(exit_row["high"]) if pd.notna(exit_row["high"]) else float("nan"),
        "exit_bar_low": float(exit_row["low"]) if pd.notna(exit_row["low"]) else float("nan"),
        "tp_touch_date": tp_touch_ts if hit else pd.NaT,
        "days_to_exit": days_to_exit,
        "mae_pct": mae_pct,
        "mae_date": mae_date,
        "mfe_pct": mfe_pct,
        "mfe_date": mfe_date,
    }

