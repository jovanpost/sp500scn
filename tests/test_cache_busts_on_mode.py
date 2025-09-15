import io
import os
import sys
import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data_lake import storage as stg


def test_cache_busts_on_mode(monkeypatch):
    st.cache_data.clear()
    s1 = stg.Storage()

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

    def fake_read_parquet(self, path: str):
        calls["n"] += 1
        return pd.read_parquet(io.BytesIO(buf.getvalue()))

    monkeypatch.setattr(stg.Storage, "read_parquet", fake_read_parquet)

    start = pd.Timestamp("2020-01-01")
    end = pd.Timestamp("2020-01-02")

    stg.load_prices_cached(s1, ["AAA"], start, end, cache_salt="mode=local")
    stg.load_prices_cached(s1, ["AAA"], start, end, cache_salt="mode=local")
    s2 = stg.Storage()
    stg.load_prices_cached(s2, ["AAA"], start, end, cache_salt="mode=supabase")

    assert calls["n"] == 2
