import os, sys, pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data_lake import storage as stg

def test_tidy_prices_normalizes_schema():
    df = pd.DataFrame({
        "date": [pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2020-01-01", tz="UTC")],
        "open": [1, 1],
        "high": [1, 1],
        "low": [1, 1],
        "close": [1, 2],
        "volume": [10, 20],
    })
    out = stg._tidy_prices(df, ticker="aaa")
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Ticker"]
    assert out.index.tolist() == [pd.Timestamp("2020-01-01")]
    assert out.index.tz is None
    assert out.loc[pd.Timestamp("2020-01-01"), "Close"] == 2
    assert out.loc[pd.Timestamp("2020-01-01"), "Adj Close"] == 2
    assert out.loc[pd.Timestamp("2020-01-01"), "Ticker"] == "AAA"
