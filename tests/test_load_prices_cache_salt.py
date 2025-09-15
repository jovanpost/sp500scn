import io
import os
import sys
import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data_lake import storage as stg


def test_load_prices_cached_respects_cache_salt(monkeypatch):
    s = stg.Storage()

    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=2),
            "open": [1, 2],
            "high": [1, 2],
            "low": [1, 2],
            "close": [1, 2],
            "volume": [10, 20],
        }
    )
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)

    calls = {"n": 0}

    def fake_exists(self, path: str) -> bool:
        return True

    def fake_read_parquet_df(self, path: str):
        calls["n"] += 1
        return pd.read_parquet(io.BytesIO(buf.getvalue()))

    monkeypatch.setattr(stg.Storage, "exists", fake_exists)
    monkeypatch.setattr(stg.Storage, "read_parquet_df", fake_read_parquet_df)

    start = pd.Timestamp("2020-01-01")
    end = pd.Timestamp("2020-01-02")
    st.cache_data.clear()
    # Regression test for bug where cache_salt was ignored
    stg.load_prices_cached(s, ["AAA"], start, end, cache_salt="a")
    stg.load_prices_cached(s, ["AAA"], start, end, cache_salt="b")

    assert calls["n"] == 2
