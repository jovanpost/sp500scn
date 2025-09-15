import io
import os
import sys
import io
import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data_lake import storage as stg


def test_cache_reuses_loaded_prices(monkeypatch):
    st.cache_data.clear()
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

    stg.load_prices_cached(s, cache_salt=s.cache_salt(), tickers=["AAA"], start=start, end=end)
    stg.load_prices_cached(s, cache_salt=s.cache_salt(), tickers=["AAA"], start=start, end=end)

    assert calls["n"] == 1


def test_cache_busts_on_cache_salt(monkeypatch):
    st.cache_data.clear()
    s1 = stg.Storage()
    s2 = stg.Storage()

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

    # Ensure storages report different salts
    monkeypatch.setattr(s1, "cache_salt", lambda: "salt1")
    monkeypatch.setattr(s2, "cache_salt", lambda: "salt2")

    stg.load_prices_cached(s1, cache_salt=s1.cache_salt(), tickers=["AAA"], start=start, end=end)
    stg.load_prices_cached(s2, cache_salt=s2.cache_salt(), tickers=["AAA"], start=start, end=end)

    assert calls["n"] == 2
