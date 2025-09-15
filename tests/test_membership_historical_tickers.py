import pandas as pd

from data_lake import membership


def test_historical_tickers_union(monkeypatch):
    df = pd.DataFrame(
        [
            {"ticker": "AAA", "start_date": "2000-01-01", "end_date": "2001-01-01"},
            {"ticker": "BBB", "start_date": "2000-01-01", "end_date": None},
            {"ticker": "AAA", "start_date": "2002-01-01", "end_date": None},
        ]
    )

    monkeypatch.setattr(
        membership,
        "load_membership",
        lambda storage=None, cache_salt="": df,
    )

    assert membership.historical_tickers() == ["AAA", "BBB"]

