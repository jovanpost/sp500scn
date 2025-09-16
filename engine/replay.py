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
    if hit_mask.any():
        hit_idx = hit_mask.idxmax()
        days_to_exit = pd.bdate_range(entry_date, hit_idx).size - 1
        hit = True
        exit_reason = "tp"
        exit_price = float(target_abs)
    else:
        days_to_exit = horizon_bdays
        hit = False
        exit_reason = "timeout"
        exit_price = float(fwd[cc].iloc[-1])

    mae_pct = float(((fwd[lc].min() / entry_open) - 1.0) * 100.0)
    mfe_pct = float(((fwd[hc].max() / entry_open) - 1.0) * 100.0)

    return {
        "hit": bool(hit),
        "exit_reason": exit_reason,
        "exit_price": float(exit_price),
        "days_to_exit": int(days_to_exit),
        "mae_pct": mae_pct,
        "mfe_pct": mfe_pct,
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

    # Defensive checks
    if "date" not in bars.columns:
        return {
            "hit": False,
            "exit_reason": "no_date_col",
            "exit_price": float("nan"),
            "days_to_exit": 0,
            "mae_pct": float("nan"),
            "mfe_pct": float("nan"),
        }

    # Dates should already be tz-naive normalized; make it idempotent
    d = pd.to_datetime(bars["date"], errors="coerce")
    # Drop timezone without shifting wall-clock time so comparisons remain valid
    d = d.dt.tz_localize(None)
    d = d.dt.normalize()
    bars = bars.assign(date=d).dropna(subset=["date"]).sort_values("date")

    # Slice D..D+H
    window = bars.loc[bars["date"] >= entry_ts].head(horizon_days + 1).copy()
    if window.empty:
        return {
            "hit": False,
            "exit_reason": "no_window",
            "exit_price": float("nan"),
            "days_to_exit": 0,
            "mae_pct": float("nan"),
            "mfe_pct": float("nan"),
        }

    # Track MFE/MAE vs entry using intraday ranges
    mfe = ((window["high"].max() - entry_price) / entry_price) * 100.0
    mae = ((window["low"].min() - entry_price) / entry_price) * 100.0

    # Iterate session by session checking intraday hit/stop
    for i, row in enumerate(window.itertuples(index=False)):
        # Stop first (conservative): intraday low breaches support
        if (stop_price is not None) and (row.low <= stop_price):
            return {
                "hit": False,
                "exit_reason": "stop",
                "exit_price": float(stop_price),
                "days_to_exit": i,
                "mae_pct": mae,
                "mfe_pct": mfe,
            }
        # TP: intraday high reaches tp
        if row.high >= tp_price:
            return {
                "hit": True,
                "exit_reason": "tp",
                "exit_price": float(tp_price),
                "days_to_exit": i,
                "mae_pct": mae,
                "mfe_pct": mfe,
            }

    # Timeout: exit at last close
    last = window.iloc[-1]
    return {
        "hit": False,
        "exit_reason": "timeout",
        "exit_price": float(last["close"]),
        "days_to_exit": len(window) - 1,
        "mae_pct": mae,
        "mfe_pct": mfe,
    }
