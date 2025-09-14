import pandas as pd

from data_lake import storage as stg
from engine import universe


def test_tidy_prices_drops_duplicates_and_naive():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime([
                "2020-01-01",
                "2020-01-01",
                "2020-01-02",
            ]).tz_localize("America/New_York"),
            "open": [1, 2, 3],
            "high": [1, 2, 3],
            "low": [1, 2, 3],
            "close": [1, 2, 3],
            "volume": [10, 20, 30],
            "ticker": ["AAA", "AAA", "AAA"],
        }
    )
    out = stg._tidy_prices(df)
    assert out.index.tz is None
    assert out.index.tolist() == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]


def test_members_on_date_tz_handling():
    members = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "start_date": [pd.Timestamp("2020-01-01", tz="America/New_York")],
            "end_date": [pd.Timestamp("2020-12-31", tz="America/New_York")],
        }
    )
    d = pd.Timestamp("2020-06-01")
    out = universe.members_on_date(members, d)
    assert out["ticker"].tolist() == ["AAA"]
    d2 = pd.Timestamp("2021-01-01")
    out2 = universe.members_on_date(members, d2)
    assert out2.empty
