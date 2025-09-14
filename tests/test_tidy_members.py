import pandas as pd
from engine import universe


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
