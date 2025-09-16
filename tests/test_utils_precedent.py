import pandas as pd
import pytest

from engine.utils_precedent import compute_precedent_hit_details


def test_compute_precedent_hits_details_fraction_and_percent():
    idx = pd.bdate_range("2023-01-02", periods=6)
    prices = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            "high": [101.0, 102.0, 107.0, 101.0, 103.0, 106.0],
        },
        index=idx,
    )

    asof = pd.Timestamp("2023-01-10")
    hits, details = compute_precedent_hit_details(
        prices,
        asof_date=asof,
        tp_value=0.05,
        lookback_bdays=3,
        window_bdays=3,
    )

    assert hits == 2
    assert [d["date"] for d in details] == ["2023-01-05", "2023-01-06"]
    assert details[0]["days_to_hit"] == 2
    assert details[1]["days_to_hit"] == 1
    assert details[0]["entry_price"] == pytest.approx(100.0)
    assert details[0]["target_pct"] == pytest.approx(5.0, abs=1e-6)
    assert details[0]["max_gain_pct"] == pytest.approx(6.0, abs=1e-6)

    hits_pct, details_pct = compute_precedent_hit_details(
        prices,
        asof_date=asof,
        tp_value=5.0,
        lookback_bdays=3,
        window_bdays=3,
    )
    assert hits_pct == hits
    assert details_pct == details


def test_compute_precedent_hits_no_peeking():
    idx = pd.bdate_range("2023-01-06", periods=4)
    prices = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [100.0, 100.0, 100.0, 106.0],
        },
        index=idx,
    )

    asof = pd.Timestamp("2023-01-11")
    hits, details = compute_precedent_hit_details(
        prices,
        asof_date=asof,
        tp_value=0.05,
        lookback_bdays=5,
        window_bdays=3,
    )

    assert hits == 0
    assert details == []
