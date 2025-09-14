import pandas as pd
from data_lake import storage as stg


def test_tidy_prices_drops_duplicates_and_tz_naive():
    df = pd.DataFrame(
        {
            "ticker": ["aaa", "AAA"],
            "date": [
                pd.Timestamp("2020-01-01", tz="UTC"),
                pd.Timestamp("2020-01-01", tz="UTC"),
            ],
            "open": [1, 1],
            "high": [1, 1],
            "low": [1, 1],
            "close": [1, 2],
            "volume": [10, 20],
        }
    )
    out, dropped = stg._tidy_prices(df)
    assert dropped == 1
    assert len(out) == 1
    assert out["ticker"].tolist() == ["AAA"]
    assert out["date"].dt.tz is None
