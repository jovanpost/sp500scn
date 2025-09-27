from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

import backtest.precursor_eval as eval_mod
from backtest.precursor_eval import (
    build_diagnostic_table,
    evaluate_precursors_naive,
    evaluate_precursors_scanner_aligned,
)
from engine.scan_runner import StocksOnlyScanParams


def _sample_params() -> StocksOnlyScanParams:
    return StocksOnlyScanParams(
        start=pd.Timestamp("2023-01-01"),
        end=pd.Timestamp("2023-01-10"),
        horizon_days=5,
        sr_lookback=20,
        sr_min_ratio=2.0,
        min_yup_pct=0.0,
        min_gap_pct=0.0,
        min_volume_multiple=1.0,
        volume_lookback=20,
        exit_model="atr",
        atr_window=14,
        atr_method="wilder",
        tp_atr_multiple=1.0,
        sl_atr_multiple=1.0,
        use_sp_filter=False,
        precursors=None,
    )


def test_scanner_aligned_invokes_run_scan(monkeypatch):
    events_df = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "signal_date": ["2023-01-03", "2023-01-04"],
            "spike_date": ["2023-01-05", pd.NaT],
            "flags_fired": [["atr_squeeze_pct"], ["gap_up_ge_gpct_prev"]],
        }
    )

    called = {}

    def stub_run_scan(params: StocksOnlyScanParams, **kwargs):
        called["params"] = params
        trades = pd.DataFrame(
            {
                "ticker": ["AAA"],
                "entry_date": [pd.Timestamp("2023-01-03")],
                "exit_reason": ["tp"],
                "pnl": [0.1],
            }
        )
        summary = {"trades": 1, "wins": 0, "candidates": 4, "start": params.start, "end": params.end}
        return {"summary": summary, "trades": trades, "debug": {}}

    monkeypatch.setattr(eval_mod, "run_scan", stub_run_scan)

    naive = evaluate_precursors_naive(events_df, within_days=5, logic="ANY")
    aligned = evaluate_precursors_scanner_aligned(events_df, _sample_params())

    assert isinstance(called["params"], StocksOnlyScanParams)
    assert naive.metrics["precision"] > aligned.metrics["precision"]

    table = build_diagnostic_table(naive, aligned)
    assert "Metric" in table.columns
    assert len(table) >= 3


def test_scan_runner_deterministic(monkeypatch):
    def stub_run_scan(payload, **kwargs):
        value = float(np.random.random())
        trades = pd.DataFrame({"ticker": ["AAA"], "entry_date": [pd.Timestamp("2023-01-03")], "pnl": [value]})
        summary = {
            "start": payload["start"],
            "end": payload["end"],
            "trades": 1,
            "wins": 1,
            "total_pnl": value,
            "candidates": 1,
        }
        return trades, summary

    from engine import scan_runner

    monkeypatch.setattr(scan_runner, "_legacy_run_scan", stub_run_scan)

    params = _sample_params()

    result1 = scan_runner.run_scan(params)
    result2 = scan_runner.run_scan(params)

    assert result1["trades"].equals(result2["trades"])
    assert result1["summary"]["total_pnl"] == result2["summary"]["total_pnl"]

    params_b = replace(params, seed=123)
    result3 = scan_runner.run_scan(params_b)
    assert result3["summary"]["total_pnl"] != result1["summary"]["total_pnl"]
