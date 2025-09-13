from __future__ import annotations

import time
from typing import Callable, Tuple

import pandas as pd

from data_lake.storage import Storage
from engine.signal_scan import scan_day, ScanParams


def trading_days(storage: Storage, start: str, end: str) -> pd.DatetimeIndex:
    """Derive trading days from AAPL parquet dates."""
    df = storage.read_parquet("prices/AAPL.parquet")
    if "date" not in df.columns:
        df["date"] = pd.to_datetime(df.get("index") or df.get("Date"))
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    mask = (df["date"] >= pd.to_datetime(start)) & (df["date"] <= pd.to_datetime(end))
    days = pd.DatetimeIndex(df.loc[mask, "date"].dropna().unique()).sort_values()
    return days


def run_range(
    storage: Storage,
    start: str,
    end: str,
    params: ScanParams,
    progress_cb: Callable[[int, int, pd.Timestamp, int, int], None] | None = None,
) -> Tuple[pd.DataFrame, dict]:
    days = trading_days(storage, start, end)
    all_trades = []
    total_days = len(days)
    days_with_candidates = 0
    for idx, D in enumerate(days, 1):
        cands, out, _fails, _dbg = scan_day(storage, D, params)
        cand_count = int(len(cands))
        hit_count = int(out["hit"].sum()) if not out.empty else 0
        if cand_count:
            days_with_candidates += 1
        if not out.empty:
            tmp = out.copy()
            tmp["date"] = D.normalize()
            all_trades.append(tmp)
        if progress_cb:
            progress_cb(idx, total_days, D, cand_count, hit_count)
        time.sleep(0.25)

    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    hits = int(trades_df["hit"].sum()) if not trades_df.empty else 0
    summary = {
        "total_days": int(total_days),
        "days_with_candidates": int(days_with_candidates),
        "trades": int(len(trades_df)),
        "hits": hits,
        "hit_rate": float(hits) / max(1, len(trades_df)),
        "median_days_to_exit": float(trades_df["days_to_exit"].median()) if not trades_df.empty else float("nan"),
        "avg_MAE_pct": float(trades_df["mae_pct"].mean()) if not trades_df.empty else float("nan"),
        "avg_MFE_pct": float(trades_df["mfe_pct"].mean()) if not trades_df.empty else float("nan"),
    }

    needed = [
        "date",
        "ticker",
        "entry_open",
        "tp_price",
        "hit",
        "exit_reason",
        "exit_price",
        "days_to_exit",
        "mae_pct",
        "mfe_pct",
        "sr_ratio",
        "tp_halfway_pct",
        "precedent_ok",
        "atr_ok",
        "reasons",
    ]
    trades_df = trades_df.reindex(columns=needed)

    return trades_df, summary
