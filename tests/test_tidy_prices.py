import pandas as pd

from data_lake.storage import _tidy_prices


def test_tidy_prices_normalizes_schema():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=3, tz="America/New_York"),
            "open": [1.0, 2.0, 3.0],
            "high": [2.0, 3.0, 4.0],
            "low": [0.5, 1.5, 2.5],
            "close": [1.5, 2.5, 3.5],
            "volume": [100, 200, 300],
            "ticker": ["aapl", "aapl", "aapl"],
        }
    )

    tidy = _tidy_prices(df, ticker="aapl")

    assert list(tidy.columns) == ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Ticker"]
    assert tidy.index.name == "date"
    assert tidy.index.tz is None
    assert tidy["Ticker"].unique().tolist() == ["AAPL"]
    assert "Adj Close" in tidy.columns


def test_tidy_prices_deduplicates_last_wins():
    df = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-01", "2020-01-02"],
            "Open": [1, 2, 3],
            "High": [2, 3, 4],
            "Low": [0, 1, 2],
            "Close": [1, 2, 3],
            "Volume": [10, 20, 30],
        }
    )

    tidy = _tidy_prices(df, ticker="MSFT")

    assert len(tidy) == 2
    first_day = pd.Timestamp("2020-01-01")
    assert tidy.loc[first_day, "Open"] == 2
    assert tidy.loc[first_day, "Ticker"] == "MSFT"
