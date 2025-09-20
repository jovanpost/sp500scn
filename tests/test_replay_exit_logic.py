import pandas as pd
import pytest

from engine.replay import replay_trade, simulate_pct_target_only


def _adi_slice_2020():
    dates = pd.bdate_range("2020-03-02", "2020-05-29")
    dates = dates.difference(pd.DatetimeIndex([pd.Timestamp("2020-04-10")]))
    df = pd.DataFrame(index=dates)
    df["Open"] = 90.0
    df["High"] = 100.0
    df["Low"] = 85.0
    df["Close"] = 90.0

    df.loc[pd.Timestamp("2020-03-20"), ["Open", "High", "Low", "Close"]] = [
        92.41,
        93.37,
        84.88,
        85.08,
    ]
    df.loc[pd.Timestamp("2020-03-23"), "Low"] = 80.0
    df.loc[pd.Timestamp("2020-04-27"), "High"] = 108.56
    df.loc[pd.Timestamp("2020-04-28"), ["High", "Low", "Close"]] = [111.80, 105.0, 111.0]
    df.loc[pd.Timestamp("2020-04-29"), ["Open", "High", "Low", "Close"]] = [
        120.0,
        130.0,
        75.0,
        125.0,
    ]
    df.loc[pd.Timestamp("2020-04-30"), ["Open", "High", "Low", "Close"]] = [
        118.0,
        128.0,
        112.0,
        118.0,
    ]
    df.index.name = "date"
    return df


def test_simulate_pct_target_exit_and_excursions():
    df = _adi_slice_2020()
    entry_date = pd.Timestamp("2020-03-20")
    result = simulate_pct_target_only(
        df,
        entry_date,
        entry_open=92.41,
        tp_pct=18.88,
        horizon_bdays=40,
    )

    assert result is not None
    assert result["hit"] is True
    assert result["exit_reason"] == "tp"
    assert result["exit_date"] == pd.Timestamp("2020-04-28")
    assert result["tp_touch_date"] == pd.Timestamp("2020-04-28")
    assert result["days_to_exit"] == 27
    assert result["exit_bar_high"] == pytest.approx(111.80, rel=1e-6)
    assert result["exit_bar_low"] == pytest.approx(105.0, rel=1e-6)
    assert result["tp_price_abs_target"] == pytest.approx(92.41 * (1.1888), rel=1e-6)
    assert result["mfe_pct"] == pytest.approx((111.80 / 92.41 - 1.0) * 100.0, rel=1e-9)
    assert result["mae_pct"] == pytest.approx((80.0 / 92.41 - 1.0) * 100.0, rel=1e-9)
    assert result["mfe_date"] == pd.Timestamp("2020-04-28")
    assert result["mae_date"] == pd.Timestamp("2020-03-23")
    # High and low spikes after exit must be ignored
    assert result["mfe_pct"] < (130.0 / 92.41 - 1.0) * 100.0
    assert result["mae_pct"] > (75.0 / 92.41 - 1.0) * 100.0


def test_replay_trade_stop_counts_days_and_excursions():
    dates = pd.bdate_range("2023-01-02", periods=4)
    bars = pd.DataFrame(
        {
            "date": dates,
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [103.0, 101.0, 103.0, 104.0],
            "low": [99.0, 89.0, 98.0, 100.0],
            "close": [101.0, 90.0, 101.0, 102.0],
        }
    )

    result = replay_trade(
        bars,
        entry_ts=dates[0],
        entry_price=100.0,
        tp_price=120.0,
        stop_price=90.0,
        horizon_days=5,
    )

    assert result["hit"] is False
    assert result["exit_reason"] == "stop"
    assert result["days_to_exit"] == 2
    assert result["exit_date"] == dates[1]
    assert pd.isna(result["tp_touch_date"])
    assert result["exit_bar_low"] == pytest.approx(89.0)
    assert result["mae_pct"] == pytest.approx((89.0 / 100.0 - 1.0) * 100.0)
    assert result["mfe_pct"] == pytest.approx((103.0 / 100.0 - 1.0) * 100.0)
