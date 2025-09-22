import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.rules import RuleConfig, passes_all_rules
from engine import signal_scan as sigscan


class DummyStorage:
    def cache_salt(self) -> str:
        return "dummy"


def test_passes_all_rules_detects_failure():
    cfg = RuleConfig(min_rr_required=2.5)
    base_row = {
        "atr_ok": 1,
        "sr_ok": 1,
        "precedent_ok": 1,
        "rsi_1h": 40.0,
        "rsi_d": 55.0,
        "earnings_days": 12.0,
        "vwap_hold": 1,
        "setup_valid": 1,
    }

    ok, reasons = passes_all_rules(base_row, cfg)
    assert ok is True
    assert reasons == []

    failing = dict(base_row)
    failing.update({"rsi_1h": 80.0, "earnings_days": 2, "vwap_hold": 0})
    ok_fail, reasons_fail = passes_all_rules(failing, cfg)
    assert ok_fail is False
    assert "rsi_1h_high" in reasons_fail
    assert "earnings_window_fail" in reasons_fail
    assert "vwap_hold_fail" in reasons_fail


def test_passes_all_rules_handles_missing_rr():
    base_row = {
        "atr_ok": 1,
        "sr_ok": 1,
        "precedent_ok": 1,
        "rsi_1h": 45.0,
        "rsi_d": 55.0,
        "earnings_days": 15.0,
        "vwap_hold": 1,
        "setup_valid": 1,
        "entry_open": 100.0,
        "support": None,
        "tp_price_abs_target": None,
    }

    allow_cfg = RuleConfig(min_rr_required=2.0, allow_missing_rr=True)
    ok_allow, reasons_allow = passes_all_rules(base_row, allow_cfg)
    assert ok_allow is True
    assert "rr_fail" not in reasons_allow

    block_cfg = RuleConfig(min_rr_required=2.0, allow_missing_rr=False)
    ok_block, reasons_block = passes_all_rules(base_row, block_cfg)
    assert ok_block is False
    assert "rr_fail" in reasons_block

    rr_row = dict(base_row)
    rr_row["rr_ratio"] = 1.2
    rr_row["support"] = 90.0
    rr_row["tp_price_abs_target"] = 108.0

    rr_cfg = RuleConfig(min_rr_required=2.0, allow_missing_rr=True)
    ok_low_rr, reasons_low_rr = passes_all_rules(rr_row, rr_cfg)
    assert ok_low_rr is False
    assert reasons_low_rr == ["rr_fail"]


def test_scan_day_respects_rule_gate(monkeypatch):
    dates = pd.bdate_range("2022-01-03", periods=6)
    df_prices = pd.DataFrame(
        {
            "date": dates,
            "open": [90, 95, 110, 115, 120, 125],
            "high": [92, 110, 120, 140, 130, 135],
            "low": [88, 95, 108, 110, 118, 120],
            "close": [91, 105, 112, 130, 125, 130],
            "volume": [1_000_000, 1_200_000, 1_400_000, 1_600_000, 1_800_000, 2_000_000],
        }
    )

    membership = pd.DataFrame(
        {
            "ticker": ["PASS", "FAIL"],
            "start_date": [dates.min(), dates.min()],
            "end_date": [pd.NaT, pd.NaT],
        }
    )

    monkeypatch.setattr(sigscan, "_load_members", lambda storage, cache_salt=None: membership.copy())

    def _fake_load_prices(_storage, ticker):
        return df_prices.copy()

    monkeypatch.setattr(sigscan, "_load_prices", _fake_load_prices)

    def _fake_compute_metrics(df, D_ts, vol_lookback, atr_window, atr_method, sr_lookback):
        return {
            "close_up_pct": 5.0,
            "vol_multiple": 2.0,
            "gap_open_pct": 0.5,
            "atr21": 2.5,
            "support": 90.0,
            "resistance": 120.0,
            "sr_ratio": 3.0,
            "sr_support": 90.0,
            "sr_resistance": 120.0,
            "sr_window_len": int(sr_lookback),
            "entry_open": 100.0,
            "atr_method": atr_method,
        }

    monkeypatch.setattr(sigscan, "_compute_metrics", _fake_compute_metrics)
    monkeypatch.setattr(sigscan, "atr_feasible", lambda *args, **kwargs: True)

    gate_calls: list[str] = []

    def _fake_passes(row, cfg):
        gate_calls.append(row.get("ticker"))
        if row.get("ticker") == "FAIL":
            return False, ["forced_fail"]
        return True, []

    monkeypatch.setattr(sigscan, "passes_all_rules", _fake_passes)

    option_calls: list[str] = []

    def _fake_compute_vertical_spread_trade(**kwargs):
        option_calls.append(kwargs.get("direction"))
        return {
            "opt_structure": "CALL_VERTICAL_DEBIT",
            "K1": kwargs.get("entry_price", 100.0) - 1.0,
            "K2": kwargs.get("entry_price", 100.0),
            "width_frac": 0.01,
            "debit_entry": 1.0,
            "contracts": 1,
            "cash_outlay": 100.0,
            "fees_entry": 1.3,
            "chain_tick": 1.0,
            "opt_reason": "",
        }

    monkeypatch.setattr(sigscan, "compute_vertical_spread_trade", _fake_compute_vertical_spread_trade)

    params = {
        "min_close_up_pct": 0.0,
        "min_vol_multiple": 0.0,
        "min_gap_open_pct": -10.0,
        "atr_window": 3,
        "atr_method": "wilder",
        "lookback_days": 3,
        "horizon_days": 5,
        "sr_min_ratio": 0.0,
        "sr_lookback": 3,
        "use_precedent": False,
        "use_atr_feasible": False,
        "exit_model": "pct_tp_only",
        "min_rr_required": 0.5,
        "rule_defaults": {
            "rsi_1h": 40.0,
            "rsi_d": 50.0,
            "earnings_days": 10.0,
            "vwap_hold": 1,
            "setup_valid": 1,
        },
        "entry_model_default": "sr_breakout",
    }

    storage = DummyStorage()
    D = dates[-1]

    cand_df, out_df, fail_count, _ = sigscan.scan_day(storage, D, params)

    assert fail_count == 0
    assert set(gate_calls) == {"PASS", "FAIL"}
    assert len(option_calls) == 1  # only PASS ticker processed
    assert list(out_df["ticker"]) == ["PASS"]
    assert out_df.iloc[0]["passes_all_rules"] == 1
    assert out_df.iloc[0]["entry_model"]
    assert (out_df["rule_fail_reasons"] == "").all()


def test_scan_day_injects_rule_and_entry_defaults(monkeypatch):
    dates = pd.bdate_range("2022-01-03", periods=6)
    df_prices = pd.DataFrame(
        {
            "date": dates,
            "open": [90, 95, 110, 115, 120, 125],
            "high": [92, 110, 120, 140, 130, 135],
            "low": [88, 95, 108, 110, 118, 120],
            "close": [91, 105, 112, 130, 125, 130],
            "volume": [
                1_000_000,
                1_200_000,
                1_400_000,
                1_600_000,
                1_800_000,
                2_000_000,
            ],
        }
    )

    membership = pd.DataFrame(
        {
            "ticker": ["PASS", "FAIL"],
            "start_date": [dates.min(), dates.min()],
            "end_date": [pd.NaT, pd.NaT],
        }
    )

    monkeypatch.setattr(
        sigscan,
        "_load_members",
        lambda storage, cache_salt=None: membership.copy(),
    )

    def _fake_load_prices(_storage, ticker):
        return df_prices.copy()

    monkeypatch.setattr(sigscan, "_load_prices", _fake_load_prices)

    def _fake_compute_metrics(df, D_ts, vol_lookback, atr_window, atr_method, sr_lookback):
        return {
            "close_up_pct": 5.0,
            "vol_multiple": 2.0,
            "gap_open_pct": 0.5,
            "atr21": 2.5,
            "support": 90.0,
            "resistance": 120.0,
            "sr_ratio": 3.0,
            "sr_support": 90.0,
            "sr_resistance": 120.0,
            "sr_window_len": int(sr_lookback),
            "entry_open": 100.0,
            "atr_method": atr_method,
        }

    monkeypatch.setattr(sigscan, "_compute_metrics", _fake_compute_metrics)
    monkeypatch.setattr(sigscan, "atr_feasible", lambda *args, **kwargs: True)

    gate_calls: list[str] = []

    def _fake_passes(row, cfg):
        gate_calls.append(row.get("ticker"))
        if row.get("ticker") == "FAIL":
            return False, ["forced_fail"]
        return True, []

    monkeypatch.setattr(sigscan, "passes_all_rules", _fake_passes)

    option_calls: list[str] = []

    def _fake_compute_vertical_spread_trade(**kwargs):
        option_calls.append(kwargs.get("direction"))
        return {
            "opt_structure": "CALL_VERTICAL_DEBIT",
            "K1": kwargs.get("entry_price", 100.0) - 1.0,
            "K2": kwargs.get("entry_price", 100.0),
            "width_frac": 0.01,
            "debit_entry": 1.0,
            "contracts": 1,
            "cash_outlay": 100.0,
            "fees_entry": 1.3,
            "chain_tick": 1.0,
            "opt_reason": "",
        }

    monkeypatch.setattr(sigscan, "compute_vertical_spread_trade", _fake_compute_vertical_spread_trade)

    params = {
        "min_close_up_pct": 0.0,
        "min_vol_multiple": 0.0,
        "min_gap_open_pct": -10.0,
        "atr_window": 3,
        "atr_method": "wilder",
        "lookback_days": 3,
        "horizon_days": 5,
        "sr_min_ratio": 0.0,
        "sr_lookback": 3,
        "use_precedent": False,
        "use_atr_feasible": False,
        "exit_model": "pct_tp_only",
        "min_rr_required": 0.5,
    }

    storage = DummyStorage()
    D = dates[-1]

    cand_df, out_df, fail_count, stats = sigscan.scan_day(storage, D, params)

    assert fail_count == 0
    assert set(gate_calls) == {"PASS", "FAIL"}
    assert len(option_calls) == 1
    assert list(out_df["ticker"]) == ["PASS"]
    assert out_df.iloc[0]["entry_model"] == "sr_breakout"
    assert stats["candidates"] == len(cand_df) == 2
    assert stats["passed_gate"] == 1
    assert stats["failed_gate"] == 1
    assert stats["skipped_no_entry_model"] == 0
    assert (out_df["rule_fail_reasons"] == "").all()


def test_scan_day_allows_missing_rr_with_atr_mode(monkeypatch):
    dates = pd.bdate_range("2022-01-03", periods=6)
    df_prices = pd.DataFrame(
        {
            "date": dates,
            "open": [100, 101, 102, 103, 104, 105],
            "high": [101, 102, 103, 104, 105, 106],
            "low": [99, 100, 101, 102, 103, 104],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5],
            "volume": [1_000_000] * 6,
        }
    )

    membership = pd.DataFrame(
        {
            "ticker": ["ATR"],
            "start_date": [dates.min()],
            "end_date": [pd.NaT],
        }
    )

    monkeypatch.setattr(
        sigscan,
        "_load_members",
        lambda storage, cache_salt=None: membership.copy(),
    )
    monkeypatch.setattr(sigscan, "_load_prices", lambda _storage, _ticker: df_prices.copy())

    def _fake_compute_metrics(df, D_ts, vol_lookback, atr_window, atr_method, sr_lookback):
        return {
            "close_up_pct": 5.0,
            "vol_multiple": 2.0,
            "gap_open_pct": 0.5,
            "atr21": 2.5,
            "support": 100.0,
            "resistance": 120.0,
            "sr_ratio": 3.0,
            "sr_support": 100.0,
            "sr_resistance": 120.0,
            "sr_window_len": int(sr_lookback),
            "entry_open": 100.0,
            "atr_method": atr_method,
        }

    monkeypatch.setattr(sigscan, "_compute_metrics", _fake_compute_metrics)
    monkeypatch.setattr(sigscan, "atr_feasible", lambda *args, **kwargs: True)

    def _fake_simulate_pct_target_only(prices, entry_ts, entry_price, tp_pct_percent, horizon):
        del prices, horizon
        exit_price = entry_price * (1.0 + tp_pct_percent / 100.0)
        exit_date = entry_ts + pd.Timedelta(days=1)
        return {
            "exit_date": exit_date,
            "exit_price": exit_price,
            "exit_reason": "tp_hit",
            "tp_price_abs_target": exit_price,
        }

    monkeypatch.setattr(sigscan, "simulate_pct_target_only", _fake_simulate_pct_target_only)

    monkeypatch.setattr(
        sigscan,
        "compute_vertical_spread_trade",
        lambda **kwargs: {
            "opt_structure": "CALL_VERTICAL_DEBIT",
            "K1": kwargs.get("entry_price", 100.0) - 1.0,
            "K2": kwargs.get("entry_price", 100.0),
            "width_frac": 0.01,
            "debit_entry": 1.0,
            "contracts": 1,
            "cash_outlay": 100.0,
            "fees_entry": 1.3,
            "chain_tick": 1.0,
            "opt_reason": "",
        },
    )

    params = {
        "min_close_up_pct": 0.0,
        "min_vol_multiple": 0.0,
        "min_gap_open_pct": -10.0,
        "atr_window": 3,
        "atr_method": "wilder",
        "lookback_days": 3,
        "horizon_days": 5,
        "sr_min_ratio": 0.0,
        "sr_lookback": 3,
        "use_precedent": False,
        "use_atr_feasible": False,
        "exit_model": "pct_tp_only",
        "tp_mode": "atr_multiple",
        "tp_atr_multiple": 1.0,
        "entry_model_default": "sr_breakout",
    }

    storage = DummyStorage()
    D = dates[-1]

    cand_df, out_df, fail_count, stats = sigscan.scan_day(storage, D, params)

    assert fail_count == 0
    assert stats["candidates"] > 0
    assert stats["passed_gate"] > 0
    assert stats["failed_rr_missing"] == 0
    assert stats["failed_rr_below"] == 0
    assert (cand_df["rr_ratio"].isna()).all()
    assert "rr_fail" not in cand_df.iloc[0]["rule_fail_reasons"]
    assert out_df.iloc[0]["passes_all_rules"] == 1
