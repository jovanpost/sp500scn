import pandas as pd

from engine.universe import members_on_date


def test_members_on_date_handles_string_dates():
    m = pd.DataFrame({
        "ticker": ["AAA", "BBB", "CCC"],
        "start_date": ["2010-01-01", "2011-06-15", "2012-01-01"],
        "end_date": ["2010-12-31", None, "2014-05-05"],
    })
    day = "2011-06-15"
    active = members_on_date(m, day)
    assert active["ticker"].tolist() == ["BBB"]
