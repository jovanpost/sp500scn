from __future__ import annotations

import pandas as pd
import pytest

import engine.stocks_only_scanner as sos


def _basic_params(**overrides):
    params: sos.StocksOnlyScanParams = {
        "start": pd.Timestamp("2022-01-03"),
        "end": pd.Timestamp("2022-01-10"),
        "horizon_days": 5,
        "sr_lookback": 3,
        "sr_min_ratio": 2.0,
        "min_yup_pct": 0.0,
        "min_gap_pct": 0.0,
        "min_volume_multiple": 0.0,
        "volume_lookback": 3,
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


def test_wilder_atr_matches_manual():
    dates = pd.bdate_range("2022-01-03", periods=5)
    df = pd.DataFrame(
        {
            "date": dates,
            "open": [10, 11, 12, 13, 14],
            "high": [11, 12, 13, 14, 15],
            "low": [9, 10, 11, 12, 13],
            "close": [10, 11, 12, 13, 14],
            "volume": [1_000_000] * 5,
        }
    )

    panel = sos._prepare_panel(df, _basic_params(), ticker="AAA")
    # Wilder ATR with window=3, seed is average of first three true ranges (all 2.0)
    # Values are then shifted by one in the panel.
    assert panel.loc[dates[3], "atr_value"] == pytest.approx(2.0)
    assert panel.loc[dates[4], "atr_value"] == pytest.approx(2.0)


@pytest.mark.parametrize(
    "price, expected",
    [
        (333.0, 3),
        (1001.0, 0),
        (0.0, 0),
    ],
)
def test_compute_shares_obeys_cap(price: float, expected: int):
    assert sos._compute_shares(price, sos.DEFAULT_CASH_CAP) == expected


def test_simulate_exit_prefers_tp(monkeypatch):
    dates = pd.bdate_range("2022-01-03", periods=3)
    bars = pd.DataFrame(
        {
            "date": dates,
            "open": [10.0, 10.5, 11.0],
            "high": [10.6, 11.5, 11.2],
            "low": [9.8, 10.4, 10.9],
            "close": [10.4, 11.4, 11.1],
        }
    )

    info = sos._simulate_exit(bars, dates[0], 10.0, 11.0, 9.5, horizon_days=5)
    assert info is not None
    assert info["exit_reason"] == "tp"


def test_simulate_exit_hits_sl_first():
    dates = pd.bdate_range("2022-01-03", periods=2)
    bars = pd.DataFrame(
        {
            "date": dates,
            "open": [10.0, 9.0],
            "high": [11.5, 9.5],
            "low": [8.5, 8.8],
            "close": [9.0, 9.1],
        }
    )

    info = sos._simulate_exit(bars, dates[0], 10.0, 11.0, 9.0, horizon_days=5)
    assert info is not None
    assert info["exit_reason"] == "sl"


def test_simulate_exit_timeout():
    dates = pd.bdate_range("2022-01-03", periods=3)
    bars = pd.DataFrame(
        {
            "date": dates,
            "open": [10.0, 10.2, 10.4],
            "high": [10.3, 10.4, 10.5],
            "low": [9.9, 10.0, 10.1],
            "close": [10.1, 10.2, 10.3],
        }
    )

    info = sos._simulate_exit(bars, dates[0], 10.0, 11.0, 9.0, horizon_days=2)
    assert info is not None
    assert info["exit_reason"] == "timeout"
    assert info["exit_date"].date() == dates[2].date()


def test_sr_ratio_gate():
    assert sos._sr_ratio_ok(10.0, 8.0, 14.0, 2.0)
    assert not sos._sr_ratio_ok(10.0, 9.25, 11.0, 2.0)
    assert not sos._sr_ratio_ok(10.0, 8.0, 11.0, 2.0)


def test_filter_thresholds():
    params = _basic_params(min_yup_pct=1.0, min_gap_pct=0.5, min_volume_multiple=1.2)
    row = pd.Series(
        {
            "yesterday_up_pct": 1.5,
            "open_gap_pct": 0.6,
            "volume_multiple": 1.25,
        }
    )
    assert sos._passes_filters(row, params)

    failing_row = pd.Series(
        {
            "yesterday_up_pct": 0.9,
            "open_gap_pct": 0.6,
            "volume_multiple": 1.25,
        }
    )
    assert not sos._passes_filters(failing_row, params)
