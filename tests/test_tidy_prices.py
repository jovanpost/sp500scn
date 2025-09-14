# tests/test_tidy_prices.py

import os
import sys
import pandas as pd

# Make package importable from tests/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_lake import storage as stg  # noqa: E402


def test_tidy_prices_normalizes_schema():
    df = pd.DataFrame(
        {
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
    out = stg._tidy_prices(df, ticker="aaa")

    # Expect a DatetimeIndex (tz-naive) and Yahoo-style columns
    assert isinstance(out.index, pd.DatetimeIndex)
    assert out.index.tz is None
    assert out.index.tolist() == [pd.Timestamp("2020-01-01")]

    assert list(out.columns) == ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Ticker"]
    assert out.loc[pd.Timestamp("2020-01-01"), "Close"] == 2
    assert out.loc[pd.Timestamp("2020-01-01"), "Adj Close"] == 2
    assert out.loc[pd.Timestamp("2020-01-01"), "Ticker"] == "AAA"


def test_tidy_prices_renames_ticker_column():
    df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-02")],
            "open": [1],
            "high": [1],
            "low": [1],
            "close": [1],
            "volume": [10],
            "ticker": ["bbb"],
        }
    )
    out = stg._tidy_prices(df)
    assert "Ticker" in out.columns
    assert "ticker" not in out.columns
    assert out.loc[pd.Timestamp("2020-01-02"), "Ticker"] == "BBB"


def test_tidy_prices_prefers_existing_Ticker_column():
    df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-03")],
            "open": [1],
            "high": [1],
            "low": [1],
            "close": [1],
            "volume": [10],
            "ticker": ["ccc"],
            "Ticker": ["ddd"],
        }
    )
    out = stg._tidy_prices(df)
    # Lowercase 'ticker' should be dropped when 'Ticker' already exists
    assert "ticker" not in out.columns
    assert out.loc[pd.Timestamp("2020-01-03"), "Ticker"] == "DDD"