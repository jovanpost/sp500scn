from __future__ import annotations

import pandas as pd


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
    # Strip timezone info to ensure comparisons work regardless of input
    if d.dt.tz is not None:
        d = d.dt.tz_convert(None)
    else:
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
