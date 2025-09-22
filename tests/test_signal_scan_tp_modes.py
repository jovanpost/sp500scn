import math
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine import signal_scan as sigscan
from engine.filters import atr_feasible


class DummyStorage:
    def cache_salt(self) -> str:
        return "dummy"


def _patch_io(monkeypatch, price_df: pd.DataFrame, ticker: str) -> None:
    membership = pd.DataFrame(
        {
            "ticker": [ticker],
            "start_date": [pd.Timestamp(price_df["date"].min())],
            "end_date": [pd.NaT],
        }
    )

    monkeypatch.setattr(
        sigscan,
        "_load_members",
        lambda storage, cache_salt=None: membership.copy(),
    )

    def _fake_load_prices(_storage, t: str) -> pd.DataFrame:
        if t != ticker:
            raise KeyError(t)
        return price_df.copy()

    monkeypatch.setattr(sigscan, "_load_prices", _fake_load_prices)


def test_compute_metrics_handles_non_range_index():
    dates = pd.bdate_range("2022-01-03", periods=6)
    df = pd.DataFrame(
        {
            "date": dates,
            "open": [100, 101, 102, 103, 104, 105],
            "high": [101, 103, 104, 105, 106, 107],
            "low": [99, 100, 101, 102, 103, 104],
            "close": [100, 102, 103, 104, 105, 106],
            "volume": [1_000_000, 1_050_000, 1_075_000, 1_100_000, 1_125_000, 1_150_000],
        }
    ).set_index("date", drop=False)

    metrics = sigscan._compute_metrics(
        df,
        dates[-1],
        vol_lookback=3,
        atr_window=3,
        atr_method="wilder",
        sr_lookback=3,
    )
    assert metrics is not None
    assert metrics["entry_open"] == pytest.approx(105.0)
    assert not pd.isna(metrics["sr_ratio"])
    assert "tp_halfway_pct" not in metrics


def test_scan_day_sr_fraction_modes(monkeypatch):
    dates = pd.bdate_range("2021-01-04", periods=5)
    df_prices = pd.DataFrame(
        {
            "date": dates,
            "open": [90, 95, 100, 102, 105],
            "high": [92, 110, 120, 140, 135],
            "low": [88, 94, 98, 101, 104],
            "close": [91, 107, 112, 115, 132],
            "volume": [1_000_000, 1_200_000, 1_400_000, 1_600_000, 1_800_000],
        }
    )

    _patch_io(monkeypatch, df_prices, ticker="SR")
    monkeypatch.setattr(sigscan, "atr_feasible", lambda *args, **kwargs: True)
    original_passes = sigscan.passes_all_rules

    def _patched_passes(row, cfg):
        try:
            rr_val = float(row.get("rr_ratio", float("nan")))
        except (TypeError, ValueError):
            rr_val = float("nan")
        if not math.isfinite(rr_val) or rr_val < cfg.min_rr_required:
            row["rr_ratio"] = max(cfg.min_rr_required + 1.0, 3.0)
        return original_passes(row, cfg)

    monkeypatch.setattr(sigscan, "passes_all_rules", _patched_passes)
    storage = DummyStorage()
    scan_params = {
        "min_close_up_pct": 0.0,
        "min_vol_multiple": 0.0,
        "min_gap_open_pct": -10.0,
        "atr_window": 3,
        "atr_method": "wilder",
        "lookback_days": 3,
        "horizon_days": 5,
        "sr_min_ratio": 1.5,
        "sr_lookback": 2,
        "use_precedent": False,
        "use_atr_feasible": False,
        "exit_model": "pct_tp_only",
        "rule_defaults": {
            "rsi_1h": 40.0,
            "rsi_d": 50.0,
            "earnings_days": 15.0,
            "vwap_hold": 1,
            "setup_valid": 1,
            "rr_ratio": 3.0,
        },
        "entry_model_default": "sr_breakout",
        "min_rr_required": 0.5,
    }

    D = dates[-1]
    cand_default, out_default, fail_default, _ = sigscan.scan_day(storage, D, scan_params)
    assert fail_default == 0
    assert len(cand_default) == 1
    row_default = cand_default.iloc[0]
    resistance = row_default["resistance"]
    entry = row_default["entry_open"]
    expected_half = 0.5 * (resistance - entry) / entry

    assert row_default["tp_mode"] == "sr_fraction"
    assert row_default["tp_sr_fraction"] == pytest.approx(0.5)
    assert row_default["tp_frac_used"] == pytest.approx(expected_half)
    assert row_default["tp_halfway_pct"] == pytest.approx(expected_half)
    assert row_default["tp_price_pct_target"] == pytest.approx(expected_half * 100.0)
    assert row_default["tp_price_abs_target"] == pytest.approx(entry * (1.0 + expected_half))

    out_row_default = out_default.iloc[0]
    assert out_row_default["tp_price_pct_target"] == pytest.approx(expected_half * 100.0)
    assert out_row_default["tp_price_abs_target"] == pytest.approx(entry * (1.0 + expected_half))

    params_quarter = {
        **scan_params,
        "tp_mode": "sr_fraction",
        "tp_sr_fraction": 0.25,
    }
    cand_quarter, out_quarter, fail_quarter, _ = sigscan.scan_day(storage, D, params_quarter)
    assert fail_quarter == 0
    assert len(cand_quarter) == 1
    row_quarter = cand_quarter.iloc[0]
    expected_quarter = 0.25 * (resistance - entry) / entry
    assert row_quarter["tp_sr_fraction"] == pytest.approx(0.25)
    assert row_quarter["tp_frac_used"] == pytest.approx(expected_quarter)
    assert row_quarter["tp_price_pct_target"] == pytest.approx(expected_quarter * 100.0)
    assert np.isnan(row_quarter["tp_atr_multiple"])
    out_row_quarter = out_quarter.iloc[0]
    assert out_row_quarter["tp_price_abs_target"] == pytest.approx(entry * (1.0 + expected_quarter))


def test_scan_day_sr_branch_includes_options(monkeypatch):
    dates = pd.bdate_range("2021-02-01", periods=40)
    df_prices = pd.DataFrame(
        {
            "date": dates,
            "open": np.concatenate([
                np.full(len(dates) - 5, 100.0),
                np.array([110.0, 112.0, 118.0, 123.0, 125.0]),
            ]),
            "high": np.concatenate([
                np.full(len(dates) - 5, 105.0),
                np.array([118.0, 125.0, 135.0, 150.0, 155.0]),
            ]),
            "low": np.concatenate([
                np.full(len(dates) - 5, 95.0),
                np.array([108.0, 110.0, 116.0, 120.0, 122.0]),
            ]),
            "close": np.concatenate([
                np.full(len(dates) - 5, 100.0),
                np.array([112.0, 118.0, 130.0, 135.0, 150.0]),
            ]),
            "volume": np.linspace(1_000_000, 1_200_000, len(dates)),
        }
    )

    _patch_io(monkeypatch, df_prices, ticker="SRX")
    monkeypatch.setattr(sigscan, "atr_feasible", lambda *args, **kwargs: True)
    original_passes = sigscan.passes_all_rules

    def _patched_passes(row, cfg):
        try:
            rr_val = float(row.get("rr_ratio", float("nan")))
        except (TypeError, ValueError):
            rr_val = float("nan")
        if not math.isfinite(rr_val) or rr_val < cfg.min_rr_required:
            row["rr_ratio"] = max(cfg.min_rr_required + 1.0, 3.0)
        return original_passes(row, cfg)

    monkeypatch.setattr(sigscan, "passes_all_rules", _patched_passes)
    storage = DummyStorage()

    params = {
        "min_close_up_pct": 0.0,
        "min_vol_multiple": 0.0,
        "min_gap_open_pct": -10.0,
        "atr_window": 5,
        "atr_method": "wilder",
        "lookback_days": 5,
        "horizon_days": 5,
        "sr_min_ratio": 0.5,
        "sr_lookback": 5,
        "use_precedent": False,
        "use_atr_feasible": False,
        "exit_model": "sr_atr",
        "tp_mode": "sr_fraction",
        "tp_sr_fraction": 0.5,
        "options_spread": {
            "enabled": True,
            "budget_per_trade": 1000.0,
            "fees_per_contract": 0.65,
        },
        "rule_defaults": {
            "rsi_1h": 40.0,
            "rsi_d": 50.0,
            "earnings_days": 15.0,
            "vwap_hold": 1,
            "setup_valid": 1,
            "rr_ratio": 3.0,
        },
        "entry_model_default": "sr_breakout",
        "min_rr_required": 0.5,
    }

    D = dates[-1]
    cand_df, out_df, fail_count, _ = sigscan.scan_day(storage, D, params)

    assert fail_count == 0
    assert len(out_df) == 1

    row = out_df.iloc[0]
    assert row.get("contracts", 0) >= 1
    assert not pd.isna(row.get("cash_outlay"))
    assert not pd.isna(row.get("debit_exit"))
    assert not pd.isna(row.get("pnl_dollars"))


def test_scan_day_atr_multiple_mode(monkeypatch):
    dates = pd.bdate_range("2021-06-01", periods=6)
    df_prices = pd.DataFrame(
        {
            "date": dates,
            "open": [100.0, 100.2, 100.4, 100.6, 100.8, 101.0],
            "high": [101.2, 101.4, 101.6, 101.8, 102.0, 102.2],
            "low": [99.8, 99.9, 100.0, 100.2, 100.4, 100.6],
            "close": [100.3, 100.5, 100.7, 100.9, 100.95, 101.3],
            "volume": [1_000_000, 1_050_000, 1_100_000, 1_150_000, 1_200_000, 1_250_000],
        }
    )

    _patch_io(monkeypatch, df_prices, ticker="ATR")
    monkeypatch.setattr(sigscan, "atr_feasible", lambda *args, **kwargs: True)
    original_passes = sigscan.passes_all_rules

    def _patched_passes(row, cfg):
        try:
            rr_val = float(row.get("rr_ratio", float("nan")))
        except (TypeError, ValueError):
            rr_val = float("nan")
        if not math.isfinite(rr_val) or rr_val < cfg.min_rr_required:
            row["rr_ratio"] = max(cfg.min_rr_required + 1.0, 3.0)
        return original_passes(row, cfg)

    monkeypatch.setattr(sigscan, "passes_all_rules", _patched_passes)
    storage = DummyStorage()
    params = {
        "min_close_up_pct": 0.0,
        "min_vol_multiple": 0.0,
        "min_gap_open_pct": 0.0,
        "atr_window": 3,
        "atr_method": "wilder",
        "lookback_days": 3,
        "horizon_days": 5,
        "sr_min_ratio": 0.5,
        "sr_lookback": 3,
        "use_precedent": False,
        "use_atr_feasible": False,
        "exit_model": "pct_tp_only",
        "tp_mode": "atr_multiple",
        "tp_atr_multiple": 0.5,
        "rule_defaults": {
            "rsi_1h": 40.0,
            "rsi_d": 50.0,
            "earnings_days": 15.0,
            "vwap_hold": 1,
            "setup_valid": 1,
            "rr_ratio": 3.0,
        },
        "entry_model_default": "sr_breakout",
        "min_rr_required": 0.5,
    }

    D = dates[-1]
    cand_df, out_df, fail_count, _ = sigscan.scan_day(storage, D, params)
    assert fail_count == 0
    assert len(cand_df) == 1
    row = cand_df.iloc[0]
    expected_frac = 0.5 * row["atr21"] / row["entry_open"]

    assert row["tp_mode"] == "atr_multiple"
    assert np.isnan(row["tp_halfway_pct"])
    assert row["tp_atr_multiple"] == pytest.approx(0.5)
    assert row["tp_price_pct_target"] == pytest.approx(expected_frac * 100.0)
    assert row["tp_frac_used"] == pytest.approx(expected_frac)

    out_row = out_df.iloc[0]
    assert out_row["tp_price_pct_target"] == pytest.approx(expected_frac * 100.0)
    assert out_row["tp_price_abs_target"] == pytest.approx(
        row["entry_open"] * (1.0 + expected_frac)
    )


def test_atr_feasible_handles_non_range_index():
    dates = pd.bdate_range("2020-02-03", periods=5)
    df = pd.DataFrame(
        {
            "open": [100.0] * 5,
            "high": [101.0] * 5,
            "low": [99.0] * 5,
            "close": [100.0] * 5,
        },
        index=dates,
    )
    assert atr_feasible(df, asof_idx=3, required_pct=0.01, atr_window=2)
