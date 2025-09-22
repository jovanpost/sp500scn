import copy
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine import signal_scan as sigscan  # noqa: E402


class DummyStorage:
    def cache_salt(self) -> str:
        return "dummy"


BASE_PARAMS = {
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
    "tp_mode": "sr_fraction",
    "tp_sr_fraction": 0.5,
    "entry_model_default": "sr_breakout",
    "min_rr_required": 0.5,
    "rule_defaults": {
        "rsi_1h": 40.0,
        "rsi_d": 50.0,
        "earnings_days": 10.0,
        "vwap_hold": 1,
        "setup_valid": 1,
    },
}


def _setup_scan_environment(monkeypatch):
    dates = pd.bdate_range("2024-01-02", periods=12)
    df_prices = pd.DataFrame(
        {
            "date": dates,
            "open": [100 + i * 0.5 for i in range(len(dates))],
            "high": [101 + i * 0.5 for i in range(len(dates))],
            "low": [99 + i * 0.5 for i in range(len(dates))],
            "close": [100.5 + i * 0.5 for i in range(len(dates))],
            "volume": [1_000_000 + 10_000 * i for i in range(len(dates))],
        }
    )

    membership = pd.DataFrame(
        {
            "ticker": ["OPT"],
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

    def _fake_metrics(df, D_ts, vol_lookback, atr_window, atr_method, sr_lookback):
        return {
            "close_up_pct": 5.0,
            "vol_multiple": 2.0,
            "gap_open_pct": 0.5,
            "atr21": 2.5,
            "support": 98.0,
            "resistance": 110.0,
            "sr_ratio": 3.0,
            "sr_support": 98.0,
            "sr_resistance": 110.0,
            "sr_window_len": int(sr_lookback),
            "entry_open": 100.0,
            "atr_method": atr_method,
        }

    monkeypatch.setattr(sigscan, "_compute_metrics", _fake_metrics)
    monkeypatch.setattr(sigscan, "atr_feasible", lambda *args, **kwargs: True)
    monkeypatch.setattr(sigscan, "passes_all_rules", lambda row, cfg: (True, []))

    def _fake_simulate(prices, entry_ts, entry_price, tp_pct_percent, horizon):
        del prices, horizon
        exit_price = float(entry_price) * 1.05
        exit_date = entry_ts + pd.Timedelta(days=1)
        return {
            "exit_date": exit_date,
            "exit_price": exit_price,
            "exit_reason": "tp_hit",
            "tp_price_abs_target": exit_price,
        }

    monkeypatch.setattr(sigscan, "simulate_pct_target_only", _fake_simulate)
    return DummyStorage(), dates[-1]


def test_options_disabled_skips_options(monkeypatch):
    storage, target_day = _setup_scan_environment(monkeypatch)

    option_calls: list[bool] = []

    def _record_options(**kwargs):
        option_calls.append(True)
        return {
            "opt_structure": "CALL_VERTICAL_DEBIT",
            "contracts": 2,
            "opt_reason": "called",
        }

    monkeypatch.setattr(sigscan, "compute_vertical_spread_trade", _record_options)

    params = copy.deepcopy(BASE_PARAMS)
    params["options_spread_enabled"] = False
    params["options_spread"] = {
        "enabled": True,
        "budget_per_trade": 1000.0,
        "fees_per_contract": 0.65,
    }

    cand_df, out_df, fail_count, stats = sigscan.scan_day(storage, target_day, params)

    assert fail_count == 0
    assert option_calls == []
    assert not out_df.empty
    first_row = out_df.iloc[0]
    assert int(first_row.get("contracts", 0)) == 0
    assert first_row.get("opt_reason") == "options_flag_off"
    assert any(
        evt.get("event") == "options_flag" and not evt.get("options_spread_enabled")
        for evt in stats.get("events", [])
    )


def test_options_enabled_invokes_options(monkeypatch):
    storage, target_day = _setup_scan_environment(monkeypatch)

    option_calls: list[bool] = []

    def _record_options(**kwargs):
        option_calls.append(True)
        return {
            "opt_structure": "CALL_VERTICAL_DEBIT",
            "contracts": 3,
            "opt_reason": "options_on",
        }

    monkeypatch.setattr(sigscan, "compute_vertical_spread_trade", _record_options)

    params = copy.deepcopy(BASE_PARAMS)
    params["options_spread_enabled"] = True
    params["options_spread"] = {
        "enabled": False,
        "budget_per_trade": 1000.0,
        "fees_per_contract": 0.65,
    }

    cand_df, out_df, fail_count, stats = sigscan.scan_day(storage, target_day, params)

    assert fail_count == 0
    assert option_calls == [True]
    assert not out_df.empty
    first_row = out_df.iloc[0]
    assert first_row.get("opt_reason") == "options_on"
    assert any(
        evt.get("event") == "options_flag" and evt.get("options_spread_enabled")
        for evt in stats.get("events", [])
    )
