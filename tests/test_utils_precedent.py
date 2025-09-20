import pandas as pd
import pytest
from pandas.tseries.offsets import BDay

from engine.utils_precedent import compute_precedent_hit_details, tp_fraction_from_row


def test_tp_fraction_precedence_absolute_target():
    assert tp_fraction_from_row(100.0, 105.0, None, None) == pytest.approx(0.05)


def test_tp_fraction_fallback_halfway_pct_fraction():
    assert tp_fraction_from_row(None, None, 2.0, None) == pytest.approx(2.0)


def test_tp_fraction_fallback_tp_price_pct_target():
    assert tp_fraction_from_row(None, None, None, 12.5) == pytest.approx(0.125)


def test_compute_hits_simple_5pct_no_peek_two_starts():
    idx = pd.bdate_range("2020-01-01", periods=40)
    df = pd.DataFrame({"open": 100.0, "high": 100.0}, index=idx)
    df.loc[idx[10], "high"] = 105.0
    df.loc[idx[15], "high"] = 105.0
    D = idx[30]
    hits, details = compute_precedent_hit_details(
        df, D, tp_frac=0.05, lookback_bdays=252, window_bdays=21
    )
    assert hits == len(details)
    assert hits >= 2
    first_touch_days = set()
    for e in details:
        assert e["target_pct"] == pytest.approx(5.0)
        S = pd.Timestamp(e["date"])
        hit_day = S + BDay(int(e["days_to_hit"]))
        assert hit_day <= D - BDay(1)
        assert pd.Timestamp(e["hit_date"]) == hit_day
        first_touch_days.add(hit_day)
    assert pd.Timestamp("2020-01-15") in first_touch_days
    assert pd.Timestamp("2020-01-22") in first_touch_days


def test_compute_hits_large_target_over_100pct():
    idx = pd.bdate_range("2023-01-02", periods=8)
    df = pd.DataFrame({"open": [10] * 8, "high": [10, 10, 21, 10, 10, 10, 10, 10]}, index=idx)
    D = idx[7]
    hits, details = compute_precedent_hit_details(
        df, D, tp_frac=1.0, lookback_bdays=252, window_bdays=21
    )
    assert hits >= 1
    assert all(item["target_pct"] == pytest.approx(100.0) for item in details)


def test_cross_start_same_future_day_counts_both():
    dates = pd.to_datetime(["2023-03-01", "2023-03-02", "2023-03-03", "2023-03-06"])
    df = pd.DataFrame({"open": [100, 100, 100, 100], "high": [101, 101, 101, 105]}, index=dates)
    D = pd.Timestamp("2023-03-07")
    hits, details = compute_precedent_hit_details(
        df, D, tp_frac=0.05, lookback_bdays=5, window_bdays=5
    )
    assert hits == len(details)
    assert hits >= 2
    start_dates = {d["date"] for d in details}
    assert {"2023-03-02", "2023-03-03"}.issubset(start_dates)
