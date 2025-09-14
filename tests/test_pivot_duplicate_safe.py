import pandas as pd


def test_pivot_duplicate_safe():
    tidy = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "BBB"],
            "date": [
                pd.Timestamp("2020-01-01"),
                pd.Timestamp("2020-01-01"),
                pd.Timestamp("2020-01-01"),
            ],
            "close": [1.0, 2.0, 3.0],
        }
    )
    tidy = tidy.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
    wide = tidy.pivot_table(index="date", columns="ticker", values="close", aggfunc="last")
    wide = wide.loc[:, ~wide.columns.duplicated()].sort_index()
    assert wide.loc[pd.Timestamp("2020-01-01"), "AAA"] == 2.0
    assert wide.shape == (1, 2)
