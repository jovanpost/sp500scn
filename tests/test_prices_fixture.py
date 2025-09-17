import io
from pathlib import Path

import pandas as pd
import pytest
import streamlit as st

from data_lake import storage as stg


def _mini_adi_frame() -> pd.DataFrame:
    """Build a deterministic mini price frame including a dividend day."""
    return pd.DataFrame([
        {
            "date": "2020-03-18",
            "Open": 87.00,
            "High": 91.50,
            "Low": 85.10,
            "Close": 90.12,
            "Adj Close": 81.60,
            "Volume": 12_345_678,
            "Dividends": 0.0,
            "Stock Splits": 0.0,
            "Ticker": "ADI",
        },
        {
            "date": "2020-03-19",
            "Open": 88.20,
            "High": 92.00,
            "Low": 86.00,
            "Close": 89.55,
            "Adj Close": 81.06,
            "Volume": 11_000_000,
            "Dividends": 0.0,
            "Stock Splits": 0.0,
            "Ticker": "ADI",
        },
        {
            "date": "2020-03-20",
            "Open": 92.41,
            "High": 93.37,
            "Low": 84.88,
            "Close": 85.08,
            "Adj Close": 77.08,
            "Volume": 20_000_000,
            "Dividends": 0.62,
            "Stock Splits": 0.0,
            "Ticker": "ADI",
        },
        {
            "date": "2020-03-23",
            "Open": 80.00,
            "High": 84.00,
            "Low": 78.00,
            "Close": 82.50,
            "Adj Close": 74.50,
            "Volume": 15_000_000,
            "Dividends": 0.0,
            "Stock Splits": 0.0,
            "Ticker": "ADI",
        },
    ])


def test_fixture_loads_raw_prices(tmp_path: Path):
    storage = stg.Storage()
    storage.local_root = tmp_path
    st.cache_data.clear()

    df = _mini_adi_frame()
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    storage.write_bytes("prices/ADI.parquet", buf.getvalue())

    out = stg.load_prices_cached(storage, cache_salt=storage.cache_salt(), tickers=["ADI"])
    assert not out.empty

    row = out[out["date"] == pd.Timestamp("2020-03-20")].iloc[0]
    assert row["Open"] == pytest.approx(92.41, rel=1e-6)
    assert row["Close"] == pytest.approx(85.08, rel=1e-6)
    assert row["Adj Close"] == pytest.approx(77.08, rel=1e-6)
    assert row["Dividends"] == pytest.approx(0.62, rel=1e-6)
