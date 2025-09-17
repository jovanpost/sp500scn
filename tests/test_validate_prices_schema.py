import pandas as pd
import pytest

from data_lake.storage import validate_prices_schema


def _frame(**overrides):
    base = {
        "date": [pd.Timestamp("2020-03-20")],
        "Ticker": ["RAW"],
        "Open": [92.41],
        "High": [93.37],
        "Low": [84.88],
        "Close": [85.08],
        "Adj Close": [77.08],
        "Volume": [100],
        "Dividends": [0.52],
        "Stock Splits": [0.0],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_accepts_raw():
    validate_prices_schema(_frame())


def test_flags_back_adjusted():
    with pytest.raises(ValueError, match="Adjusted OHLC detected"):
        validate_prices_schema(_frame(**{"Adj Close": [85.08]}))


def test_allows_equal_when_no_actions():
    df = _frame(Dividends=[0.0])
    df["Adj Close"] = df["Close"]
    validate_prices_schema(df)


def test_warns_non_strict(caplog):
    df = _frame(**{"Adj Close": [85.08]})
    df["Adj Close"] = df["Close"]
    with caplog.at_level("WARNING"):
        validate_prices_schema(df, strict=False)
    assert any("Adjusted OHLC detected" in rec.getMessage() for rec in caplog.records)
