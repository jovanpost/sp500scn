import pandas as pd

from engine.metrics import has_21d_runup_precedent


def test_has_21d_runup_precedent_true():
    dates = pd.date_range("2023-01-01", periods=40)
    close = pd.Series([100.0] * 40)
    high = close.copy()
    # create a 21-day window with 10% run-up
    high.iloc[0:21] = 110.0
    df = pd.DataFrame({"date": dates, "close": close, "high": high})
    assert has_21d_runup_precedent(df, lookback_days=30, horizon_days=21, target_move_pct=0.09)


def test_has_21d_runup_precedent_false():
    dates = pd.date_range("2023-01-01", periods=40)
    close = pd.Series([100.0] * 40)
    high = close.copy()
    df = pd.DataFrame({"date": dates, "close": close, "high": high})
    assert not has_21d_runup_precedent(
        df, lookback_days=30, horizon_days=21, target_move_pct=0.05
    )

