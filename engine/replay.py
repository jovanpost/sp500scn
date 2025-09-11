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
    """
    Walks forward starting D (inclusive) for horizon_days trading sessions.
    Returns dict(hit:bool, exit_reason:str, exit_price:float, days_to_exit:int,
                 mae_pct:float, mfe_pct:float).
    """
    # Restrict to D..D+H (trading sessions)
    b = bars.loc[bars["date"] >= entry_ts].head(horizon_days + 1).copy()
    if b.empty:
        return {
            "hit": False,
            "exit_reason": "no_data",
            "exit_price": float("nan"),
            "days_to_exit": 0,
            "mae_pct": float("nan"),
            "mfe_pct": float("nan"),
        }
    # Track MFE/MAE vs entry using intraday ranges
    mfe = ((b["high"].max() - entry_price) / entry_price) * 100.0
    mae = ((b["low"].min() - entry_price) / entry_price) * 100.0

    # Iterate session by session checking intraday hit/stop
    for i, row in enumerate(b.itertuples(index=False)):
        # Stop first (conservative): intraday low breaches support
        if stop_price is not None and row.low <= stop_price:
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
    last = b.iloc[-1]
    return {
        "hit": False,
        "exit_reason": "timeout",
        "exit_price": float(last["close"]),
        "days_to_exit": len(b) - 1,
        "mae_pct": mae,
        "mfe_pct": mfe,
    }

