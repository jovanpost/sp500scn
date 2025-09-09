import pandas as pd

from engine.universe import members_on_date


def test_members_on_date_string_dtype():
    df = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "start_date": pd.Series(["2020-01-01", "2020-07-01"], dtype="string"),
            "end_date": pd.Series(["2020-06-30", pd.NA], dtype="string"),
        }
    )

    res = members_on_date(df, pd.Timestamp("2020-05-01"))
    assert res["ticker"].tolist() == ["AAA"]

    res = members_on_date(df, pd.Timestamp("2020-08-01"))
    assert res["ticker"].tolist() == ["BBB"]
