from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from engine.scan_runner import StocksOnlyScanParams, run_scan


@dataclass(frozen=True)
class EvaluationResult:
    diagnostics: pd.DataFrame
    metrics: dict[str, Any]


def _normalise_flags(value: Any) -> list[str]:
    if isinstance(value, str):
        parts = [segment.strip() for segment in value.split(",") if segment.strip()]
        return parts
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    return []


def evaluate_precursors_naive(
    events_df: pd.DataFrame, within_days: int, logic: str
) -> EvaluationResult:
    if events_df is None or events_df.empty:
        empty = pd.DataFrame(columns=["ticker", "signal_date", "spike_date", "flags_fired", "naive_spike_hit"])
        return EvaluationResult(empty, {"total": 0, "spike_hits": 0, "precision": 0.0, "by_flag": {}})

    working = events_df.copy()
    if "signal_date" in working.columns:
        working["signal_date"] = pd.to_datetime(working["signal_date"])
    elif "date" in working.columns:
        working["signal_date"] = pd.to_datetime(working["date"])
    else:
        working["signal_date"] = pd.NaT
    if "spike_date" in working.columns:
        working["spike_date"] = pd.to_datetime(working["spike_date"])
    else:
        working["spike_date"] = pd.NaT

    if "flags_fired" not in working.columns:
        working["flags_fired"] = [[] for _ in range(len(working))]
    working["flags_fired"] = working["flags_fired"].apply(_normalise_flags)

    within_days = max(int(within_days or 1), 1)
    logic = str(logic or "ANY").upper()
    if logic not in {"ANY", "ALL"}:
        logic = "ANY"

    def _spike_hit(row: pd.Series) -> bool:
        spike_date = row.get("spike_date")
        if pd.isna(spike_date):
            return False
        signal_date = row.get("signal_date")
        if pd.isna(signal_date):
            return False
        delta = (pd.Timestamp(spike_date) - pd.Timestamp(signal_date)).days
        return 0 <= delta <= within_days

    working["naive_spike_hit"] = working.apply(_spike_hit, axis=1)

    by_flag: dict[str, dict[str, Any]] = {}
    for _, row in working.iterrows():
        flags = row["flags_fired"] or []
        if not flags:
            continue
        for flag in flags:
            stats = by_flag.setdefault(flag, {"count": 0, "hits": 0})
            stats["count"] += 1
            if row["naive_spike_hit"]:
                stats["hits"] += 1

    total = int(len(working))
    spike_hits = int(working["naive_spike_hit"].sum())
    precision = float(spike_hits / total) if total else 0.0

    for stats in by_flag.values():
        count = max(1, stats["count"])
        stats["precision"] = stats["hits"] / count

    metrics = {
        "total": total,
        "spike_hits": spike_hits,
        "precision": precision,
        "logic": logic,
        "within_days": within_days,
        "by_flag": by_flag,
    }

    diagnostics = working[["ticker", "signal_date", "spike_date", "flags_fired", "naive_spike_hit"]]
    return EvaluationResult(diagnostics, metrics)


def evaluate_precursors_scanner_aligned(
    events_df: pd.DataFrame,
    scan_params: StocksOnlyScanParams,
    *,
    scan_result: dict[str, Any] | None = None,
) -> EvaluationResult:
    if scan_result is None:
        scan_result = run_scan(scan_params)

    trades = scan_result.get("trades") if isinstance(scan_result, dict) else None
    if isinstance(trades, pd.DataFrame):
        trades = trades.copy()
    else:
        trades = pd.DataFrame()
    summary = scan_result.get("summary", {}) if isinstance(scan_result, dict) else {}

    diagnostics = pd.DataFrame(columns=["ticker", "signal_date", "entered_trade", "exit_reason", "pnl"])
    if events_df is not None and not events_df.empty:
        diagnostics = events_df.copy()
        if "signal_date" in diagnostics.columns:
            diagnostics["signal_date"] = pd.to_datetime(diagnostics["signal_date"])
        elif "date" in diagnostics.columns:
            diagnostics["signal_date"] = pd.to_datetime(diagnostics["date"])
        else:
            diagnostics["signal_date"] = pd.NaT
        diagnostics["entered_trade"] = False
        diagnostics["exit_reason"] = pd.NA
        diagnostics["pnl"] = pd.NA
        if "flags_fired" in diagnostics.columns:
            diagnostics["flags_fired"] = diagnostics["flags_fired"].apply(_normalise_flags)
        else:
            diagnostics["flags_fired"] = [[] for _ in range(len(diagnostics))]
        if isinstance(trades, pd.DataFrame) and not trades.empty:
            trades = trades.copy()
            trades["entry_date"] = pd.to_datetime(trades.get("entry_date"))
            trades["ticker"] = trades.get("ticker").astype(str)
            diagnostics["ticker"] = diagnostics.get("ticker").astype(str)
            merged = diagnostics.merge(
                trades[["ticker", "entry_date", "exit_reason", "pnl"]],
                left_on=["ticker", "signal_date"],
                right_on=["ticker", "entry_date"],
                how="left",
                suffixes=("", "_trade"),
            )
            diagnostics["entered_trade"] = merged["exit_reason_trade"].notna()
            diagnostics["exit_reason"] = merged["exit_reason_trade"]
            diagnostics["pnl"] = merged["pnl_trade"]

    trades_taken = int(summary.get("trades", 0)) if isinstance(summary, dict) else 0
    wins = int(summary.get("wins", 0)) if isinstance(summary, dict) else 0
    win_rate = float(wins / trades_taken) if trades_taken else 0.0
    candidates = int(summary.get("candidates", 0)) if isinstance(summary, dict) else 0
    denom = candidates if candidates else (trades_taken or 1)
    precision = float(trades_taken / denom) if denom else 0.0

    metrics = {
        "summary": summary,
        "trades": trades,
        "precision": precision,
        "win_rate": win_rate,
    }

    return EvaluationResult(diagnostics, metrics)


def build_diagnostic_table(naive: EvaluationResult, aligned: EvaluationResult) -> pd.DataFrame:
    naive_precision = naive.metrics.get("precision", 0.0)
    aligned_metrics = aligned.metrics.get("summary", {})
    aligned_precision = aligned.metrics.get("precision", 0.0)
    win_rate = aligned.metrics.get("win_rate", 0.0)
    trades = aligned_metrics.get("trades", 0)
    total = naive.metrics.get("total", 0)
    trades_df = aligned.metrics.get("trades")
    if isinstance(trades_df, pd.DataFrame) and "pnl" in trades_df.columns:
        avg_pnl = float(trades_df["pnl"].mean())
    else:
        avg_pnl = float("nan")

    rows = [
        {"Metric": "Total precursor hits", "Naive": total, "Scanner-aligned": aligned_metrics.get("candidates", trades)},
        {"Metric": "Spike precision", "Naive": round(naive_precision, 4), "Scanner-aligned": round(aligned_precision, 4)},
        {"Metric": "Trade win rate", "Naive": "N/A", "Scanner-aligned": round(win_rate, 4)},
        {"Metric": "Avg P/L per trade", "Naive": "N/A", "Scanner-aligned": avg_pnl},
    ]
    return pd.DataFrame(rows)


__all__ = [
    "EvaluationResult",
    "evaluate_precursors_naive",
    "evaluate_precursors_scanner_aligned",
    "build_diagnostic_table",
]
