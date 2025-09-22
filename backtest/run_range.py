from __future__ import annotations

import os
import time
from typing import Callable, Tuple

import pandas as pd

from data_lake.storage import Storage
from engine.signal_scan import scan_day, ScanParams


def trading_days(storage: Storage, start: str, end: str) -> pd.DatetimeIndex:
    """Derive trading days from AAPL parquet dates."""
    df = storage.read_parquet_df("prices/AAPL.parquet")
    if "date" not in df.columns:
        # Try common fallbacks produced by older pipelines
        fallback = None
        for cand in ("index", "Date", "DATE"):
            if cand in df.columns:
                fallback = cand
                break
        if fallback is None:
            raise KeyError("prices/AAPL.parquet missing a usable date column")
        df["date"] = pd.to_datetime(df[fallback])
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    mask = (df["date"] >= start_ts) & (df["date"] <= end_ts)
    days = pd.DatetimeIndex(df.loc[mask, "date"].dropna().unique()).sort_values()
    return days


def compute_spread_summary(trades_df: pd.DataFrame) -> dict:
    summary = {
        "trades_spread": 0,
        "wins_spread": 0,
        "losses_spread": 0,
        "win_rate_spread": float("nan"),
        "gross_profit_spread": 0.0,
        "gross_loss_spread": 0.0,
        "net_pnl_spread": 0.0,
        "avg_pnl_per_trade": float("nan"),
        "max_drawdown_spread": float("nan"),
        "trades_executed_spread": 0,
        "invested_dollars": 0.0,
        "gross_revenue_dollars": 0.0,
        "end_value_dollars": 0.0,
        "avg_cost_per_trade": float("nan"),
        "dollar_summary_str": "",
    }

    if trades_df is None or trades_df.empty or "contracts" not in trades_df.columns:
        return summary

    executed = trades_df.copy()
    executed["contracts"] = pd.to_numeric(executed["contracts"], errors="coerce")
    executed = executed.loc[executed["contracts"].fillna(0) > 0]
    if executed.empty or "pnl_dollars" not in executed.columns:
        return summary

    executed["pnl_dollars"] = pd.to_numeric(executed["pnl_dollars"], errors="coerce")
    executed = executed.dropna(subset=["pnl_dollars"])
    if executed.empty:
        return summary

    trades_count = int(len(executed))
    pnl = executed["pnl_dollars"]
    wins = int((pnl > 0).sum())
    losses = int((pnl <= 0).sum())
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = float(pnl[pnl < 0].sum())
    net = float(pnl.sum())
    avg = net / trades_count if trades_count else float("nan")
    win_rate = wins / trades_count if trades_count else float("nan")

    cum = pnl.cumsum()
    peaks = cum.cummax()
    drawdown = float((cum - peaks).min()) if not cum.empty else float("nan")

    invested = 0.0
    if "cash_outlay" in executed.columns:
        cash_outlay = pd.to_numeric(executed["cash_outlay"], errors="coerce")
        invested = float(cash_outlay.dropna().sum())

    gross_revenue = 0.0
    if "revenue" in executed.columns:
        revenue_series = pd.to_numeric(executed["revenue"], errors="coerce")
        gross_revenue = float(revenue_series.dropna().sum())

    end_value = invested + net
    avg_cost = invested / trades_count if trades_count else float("nan")

    dollar_summary = ""
    if trades_count:
        result_word = "profit" if net >= 0 else "loss"
        dollar_summary = (
            f"{trades_count} trades — ${invested:,.0f} invested → ${end_value:,.0f} end value ⇒ ${net:,.0f} net {result_word}"
        )

    summary.update(
        {
            "trades_spread": trades_count,
            "wins_spread": wins,
            "losses_spread": losses,
            "win_rate_spread": float(win_rate),
            "gross_profit_spread": gross_profit,
            "gross_loss_spread": gross_loss,
            "net_pnl_spread": net,
            "avg_pnl_per_trade": float(avg),
            "max_drawdown_spread": drawdown,
            "trades_executed_spread": trades_count,
            "invested_dollars": invested,
            "gross_revenue_dollars": gross_revenue,
            "end_value_dollars": end_value,
            "avg_cost_per_trade": float(avg_cost),
            "dollar_summary_str": dollar_summary,
        }
    )

    return summary


def run_range(
    storage: Storage,
    start: str,
    end: str,
    params: ScanParams,
    progress_cb: Callable[[int, int, pd.Timestamp, int, int], None] | None = None,
) -> Tuple[pd.DataFrame, dict]:
    params = dict(params)
    if "debug_include_all" not in params:
        env_value = os.getenv("BACKTEST_DEBUG_INCLUDE_ALL", "0")
        params["debug_include_all"] = str(env_value).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

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
            # normalize to date at midnight, tz-naive
            tmp["date"] = pd.Timestamp(D).normalize()
            all_trades.append(tmp)

        if progress_cb:
            try:
                progress_cb(idx, total_days, D, cand_count, hit_count)
            except Exception:
                pass

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

    summary.update(compute_spread_summary(trades_df))

    def _round2(s: pd.Series) -> pd.Series:
        try:
            return s.astype(float).round(2)
        except Exception:
            return s

    if not trades_df.empty:
        if "width_frac" in trades_df.columns and "width_pct" not in trades_df.columns:
            trades_df["width_pct"] = trades_df["width_frac"].astype(float) * 100.0

        price_cols = [
            "entry_open",
            "tp_price",
            "tp_price_abs_target",
            "exit_price",
            "exit_bar_high",
            "exit_bar_low",
            "support",
            "resistance",
            "sr_support",
            "sr_resistance",
            "atr_value_dm1",
            "atr_dminus1",
            "atr_budget_dollars",
            "tp_required_dollars",
        ]
        pct_cols = [
            "close_up_pct",
            "gap_open_pct",
            "tp_pct_used",
            "mae_pct",
            "mfe_pct",
            "atr21_pct",
            "ret21_pct",
            "width_pct",
        ]
        ratio_cols = ["sr_ratio", "vol_multiple", "tp_sr_fraction", "tp_atr_multiple"]

        for c in price_cols:
            if c in trades_df.columns:
                trades_df[c + "_2dp"] = _round2(trades_df[c])

        for c in pct_cols:
            if c in trades_df.columns:
                trades_df[c + "_2dp"] = _round2(trades_df[c])

        for c in ratio_cols:
            if c in trades_df.columns:
                trades_df[c + "_2dp"] = _round2(trades_df[c])

    # Columns expected by the UI/CSV
    needed = [
        "date",
        "ticker",
        "entry_open",
        "entry_open_2dp",
        "tp_price",
        "tp_price_2dp",
        "tp_price_abs_target",
        "tp_price_abs_target_2dp",
        "tp_price_pct_target",
        "exit_model",
        "exit_date",
        "tp_touch_date",
        "hit",
        "exit_reason",
        "exit_price",
        "exit_price_2dp",
        "exit_bar_high",
        "exit_bar_high_2dp",
        "exit_bar_low",
        "exit_bar_low_2dp",
        "days_to_exit",
        "mae_pct",
        "mae_pct_2dp",
        "mae_date",
        "mfe_pct",
        "mfe_pct_2dp",
        "mfe_date",
        "close_up_pct",
        "close_up_pct_2dp",
        "vol_multiple",
        "vol_multiple_2dp",
        "gap_open_pct",
        "gap_open_pct_2dp",
        "support",
        "support_2dp",
        "resistance",
        "resistance_2dp",
        "sr_support",
        "sr_support_2dp",
        "sr_resistance",
        "sr_resistance_2dp",
        "sr_ratio",
        "sr_ratio_2dp",
        "sr_window_used",
        "sr_ok",
        "tp_frac_used",
        "tp_pct_used",
        "tp_pct_used_2dp",
        "tp_mode",
        "tp_sr_fraction",
        "tp_atr_multiple",
        "tp_halfway_pct",
        "precedent_hits",
        "precedent_ok",
        "precedent_details_hits",
        "precedent_hit_start_dates",
        "precedent_max_hit_date",
        "atr_ok",
        "atr_window",
        "atr_method",
        "atr_value_dm1",
        "atr_value_dm1_2dp",
        "atr_dminus1",
        "atr_dminus1_2dp",
        "atr_budget_dollars",
        "atr_budget_dollars_2dp",
        "tp_required_dollars",
        "tp_required_dollars_2dp",
        "reasons",
        "passes_all_rules",
        "rule_fail_reasons",
        "opt_structure",
        "K1",
        "K2",
        "width_frac",
        "width_pct",
        "T_entry_days",
        "sigma_entry",
        "debit_entry",
        "contracts",
        "cash_outlay",
        "fees_entry",
        "S_exit",
        "T_exit_days",
        "sigma_exit",
        "debit_exit",
        "revenue",
        "fees_exit",
        "pnl_dollars",
        "opt_reason",
        "win",
    ]

    # Guard & reindex for UI/CSV
    if trades_df is None or trades_df.empty:
        # produce an empty table with the expected schema so UI/CSV don't crash
        trades_df = pd.DataFrame(columns=needed)
    else:
        # fail loudly if the precedent columns were dropped upstream
        missing = [
            c
            for c in (
                "precedent_details_hits",
                "precedent_hit_start_dates",
                "precedent_hits",
                "precedent_ok",
            )
            if c not in trades_df.columns
        ]
        if missing:
            raise RuntimeError(f"precedent columns missing at export: {missing}")
        trades_df = trades_df.reindex(columns=needed)

    return trades_df, summary

