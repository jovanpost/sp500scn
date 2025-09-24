from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import engine.stocks_only_scanner as sos
from engine.scan_shared.precursor_flags import build_precursor_flags
import pandas as pd



def _make_base_params(**overrides) -> sos.StocksOnlyScanParams:
    params: sos.StocksOnlyScanParams = {
        "start": pd.Timestamp("2022-03-01"),
        "end": pd.Timestamp("2022-03-01"),
        "horizon_days": 3,
        "sr_lookback": 2,
        "sr_min_ratio": 1.0,
        "min_yup_pct": 0.0,
        "min_gap_pct": 0.0,
        "min_volume_multiple": 0.0,
        "volume_lookback": 1,
        "exit_model": "atr",
        "atr_window": 3,
        "atr_method": "wilder",
        "tp_atr_multiple": 1.0,
        "sl_atr_multiple": 1.0,
        "use_sp_filter": False,
        "cash_per_trade": sos.DEFAULT_CASH_CAP,
    }
    params.update(overrides)
    return params


def _constant_price_frame(
    *,
    start: str,
    periods: int,
    gap_index: int | None = None,
    gap_pct: float = 0.0,
    vol_spikes: dict[int, float] | None = None,
) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=periods)
    close = pd.Series(100.0, index=dates)
    open_prices = close.copy()
    high = close + 2.0
    low = close - 2.0
    volume = pd.Series(1_000_000.0, index=dates)

    if gap_index is not None:
        idx = dates[gap_index]
        prev_close = close.iloc[gap_index - 1]
        open_prices.loc[idx] = prev_close * (1.0 + gap_pct / 100.0)
        high.loc[idx] = open_prices.loc[idx] + 2.0
        low.loc[idx] = open_prices.loc[idx] - 2.0

    if vol_spikes:
        for offset, mult in vol_spikes.items():
            volume.iloc[offset] = 1_000_000.0 * mult

    frame = pd.DataFrame(
        {
            "date": dates,
            "open": open_prices.values,
            "high": high.values,
            "low": low.values,
            "close": close.values,
            "volume": volume.values,
        }
    )
    return frame


def test_any_logic_includes_on_single_match():
    prices = _constant_price_frame(start="2022-01-03", periods=60, gap_index=55, gap_pct=5.0)
    scan_day = pd.Timestamp(prices.iloc[-1]["date"]).normalize()
    params = _make_base_params(start=scan_day, end=scan_day)
    params["precursors"] = {
        "enabled": True,
        "within_days": 10,
        "logic": "ANY",
        "conditions": [{"flag": "gap_up_ge_gpct_prev", "min_gap_pct": 3.0}],
    }

    ledger, _ = sos.run_scan(params, prices_by_ticker={"AAA": prices})
    assert not ledger.empty
    row = ledger.iloc[0]
    assert row["precursor_score"] == 1
    assert row["precursor_flags_hit"] == ["gap_up_ge_gpct_prev"]
    assert row["precursor_last_seen_days_ago"]["gap_up_ge_gpct_prev"] >= 1.0


def test_all_logic_requires_all_flags():
    prices = _constant_price_frame(start="2022-01-03", periods=60, gap_index=55, gap_pct=5.0)
    scan_day = pd.Timestamp(prices.iloc[-1]["date"]).normalize()
    params = _make_base_params(start=scan_day, end=scan_day)
    params["precursors"] = {
        "enabled": True,
        "within_days": 10,
        "logic": "ALL",
        "conditions": [
            {"flag": "gap_up_ge_gpct_prev", "min_gap_pct": 3.0},
            {"flag": "ema_20_50_cross_up"},
        ],
    }

    ledger, _ = sos.run_scan(params, prices_by_ticker={"AAA": prices})
    assert ledger.empty


def test_threshold_params_respected_bb_atr_gap_vol():
    prices = _constant_price_frame(
        start="2021-01-04",
        periods=130,
        gap_index=120,
        gap_pct=6.0,
        vol_spikes={126: 2.5, 127: 5.0},
    )
    scan_day = pd.Timestamp(prices.iloc[-1]["date"]).normalize()
    params = _make_base_params(start=scan_day, end=scan_day)

    high_thresholds = {
        "enabled": True,
        "within_days": 20,
        "logic": "ALL",
        "conditions": [
            {"flag": "atr_squeeze_pct", "max_percentile": 100.0},
            {"flag": "bb_squeeze_pct", "max_percentile": 100.0},
            {"flag": "gap_up_ge_gpct_prev", "min_gap_pct": 6.0},
            {"flag": "vol_mult_d1_ge_x", "min_mult": 2.0},
            {"flag": "vol_mult_d2_ge_x", "min_mult": 2.0},
        ],
    }
    params_high = {**params, "precursors": high_thresholds}
    ledger_high, _ = sos.run_scan(params_high, prices_by_ticker={"AAA": prices})
    assert not ledger_high.empty

    strict_thresholds = {
        "enabled": True,
        "within_days": 20,
        "logic": "ALL",
        "conditions": [
            {"flag": "atr_squeeze_pct", "max_percentile": 10.0},
            {"flag": "bb_squeeze_pct", "max_percentile": 10.0},
            {"flag": "gap_up_ge_gpct_prev", "min_gap_pct": 7.0},
            {"flag": "vol_mult_d1_ge_x", "min_mult": 3.0},
            {"flag": "vol_mult_d2_ge_x", "min_mult": 3.0},
        ],
    }
    params_strict = {**params, "precursors": strict_thresholds}
    ledger_strict, _ = sos.run_scan(params_strict, prices_by_ticker={"AAA": prices})
    assert ledger_strict.empty


def test_new_high_windows_20_63():
    dates = pd.bdate_range("2021-01-04", periods=130)
    close = pd.Series(50.0 + 0.5 * pd.RangeIndex(len(dates)), index=dates)
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": close.values,
            "high": close.values + 2.0,
            "low": close.values - 2.0,
            "close": close.values,
            "volume": [1_000_000.0] * len(dates),
        }
    )
    scan_day = pd.Timestamp(dates[-5]).normalize()
    params = _make_base_params(start=scan_day, end=scan_day)
    params["sr_min_ratio"] = 0.0
    params["precursors"] = {
        "enabled": True,
        "within_days": 30,
        "logic": "ALL",
        "conditions": [
            {"flag": "new_high_20"},
            {"flag": "new_high_63"},
        ],
    }

    ledger, _ = sos.run_scan(params, prices_by_ticker={"AAA": frame})
    assert not ledger.empty
    hits = ledger.iloc[0]["precursor_flags_hit"]
    assert set(hits) == {"new_high_20", "new_high_63"}


def test_parity_with_lab_sample():
    prices = _constant_price_frame(
        start="2021-06-01",
        periods=90,
        gap_index=70,
        gap_pct=5.0,
        vol_spikes={86: 2.5, 87: 5.0},
    )
    scan_day = pd.Timestamp(prices.iloc[-1]["date"]).normalize()
    params = _make_base_params(start=scan_day, end=scan_day)
    conditions = [
        {"flag": "gap_up_ge_gpct_prev", "min_gap_pct": 4.0},
        {"flag": "vol_mult_d1_ge_x", "min_mult": 2.0},
        {"flag": "vol_mult_d2_ge_x", "min_mult": 2.0},
    ]
    params["precursors"] = {
        "enabled": True,
        "within_days": 15,
        "logic": "ANY",
        "conditions": conditions,
    }

    panel_with_flags, _ = build_precursor_flags(prices, params["precursors"])
    window_start = scan_day - pd.tseries.offsets.BDay(15)
    window = panel_with_flags.loc[(panel_with_flags.index >= window_start) & (panel_with_flags.index < scan_day)]
    expected_flags = set()
    for condition in conditions:
        flag = condition["flag"]
        if flag == "gap_up_ge_gpct_prev":
            hits = window["gap_up_pct_prev"] >= float(condition.get("min_gap_pct", 3.0))
        elif flag == "vol_mult_d1_ge_x":
            hits = window["vol_mult_d1"] >= float(condition.get("min_mult", 1.5))
        elif flag == "vol_mult_d2_ge_x":
            hits = window["vol_mult_d2"] >= float(condition.get("min_mult", 1.5))
        else:
            hits = window.get(flag, pd.Series(False, index=window.index))
        if bool(hits.fillna(False).any()):
            expected_flags.add(flag)

    ledger, _ = sos.run_scan(params, prices_by_ticker={"AAA": prices})
    assert not ledger.empty
    actual_flags = set(ledger.iloc[0]["precursor_flags_hit"])
    assert actual_flags == expected_flags


def test_scanner_unchanged_when_disabled():
    prices = _constant_price_frame(start="2022-01-03", periods=40)
    scan_day = pd.Timestamp(prices.iloc[-1]["date"]).normalize()
    base_params = _make_base_params(start=scan_day, end=scan_day)

    ledger_no_precursors, summary_no_precursors = sos.run_scan(
        base_params, prices_by_ticker={"AAA": prices}
    )

    params_disabled = {**base_params, "precursors": {"enabled": False}}
    ledger_disabled, summary_disabled = sos.run_scan(
        params_disabled, prices_by_ticker={"AAA": prices}
    )

    pd.testing.assert_frame_equal(ledger_no_precursors, ledger_disabled)
    assert summary_no_precursors == summary_disabled