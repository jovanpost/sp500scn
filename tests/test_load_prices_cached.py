import io
import os, sys, io, pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data_lake import storage as stg


def test_load_prices_cached_concat_and_filter(monkeypatch):
    s = stg.Storage()

    df_a = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=3, tz="America/New_York"),
            "open": [1, 2, 3],
            "high": [1, 2, 3],
            "low": [1, 2, 3],
            "close": [1, 2, 3],
            "volume": [10, 20, 30],
        }
    )
    buf_a = io.BytesIO()
    df_a.to_parquet(buf_a, index=False)

    df_b = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=3, tz="America/New_York"),
            "open": [4, 5, 6],
            "high": [4, 5, 6],
            "low": [4, 5, 6],
            "close": [4, 5, 6],
            "volume": [40, 50, 60],
        }
    )
    buf_b = io.BytesIO()
    df_b.to_parquet(buf_b, index=False)

    def fake_exists(self, path: str) -> bool:
        return path in {"prices/AAA.parquet", "prices/BBB.parquet"}

    def fake_read_parquet_df(self, path: str):
        if path == "prices/AAA.parquet":
            return pd.read_parquet(io.BytesIO(buf_a.getvalue()))
        if path == "prices/BBB.parquet":
            return pd.read_parquet(io.BytesIO(buf_b.getvalue()))
        raise FileNotFoundError(path)

    monkeypatch.setattr(stg.Storage, "exists", fake_exists)
    monkeypatch.setattr(stg.Storage, "read_parquet_df", fake_read_parquet_df)

    st.cache_data.clear()
    out = stg.load_prices_cached(
        s,
        ["AAA", "BBB", "CCC"],
        pd.Timestamp("2020-01-02"),
        pd.Timestamp("2020-01-03"),
    )
    out = out.set_index("date")

    assert sorted(out["Ticker"].unique()) == ["AAA", "BBB"]
    assert out.index.min() == pd.Timestamp("2020-01-02")
    assert out.index.max() == pd.Timestamp("2020-01-03")


def test_load_prices_cached_uses_cache(monkeypatch):
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
    stg.load_prices_cached(s, ["AAA"], start, end)
    stg.load_prices_cached(s, ["AAA"], start, end)

    assert calls["n"] == 1
