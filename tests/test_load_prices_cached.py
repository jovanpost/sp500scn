import pandas as pd
from data_lake import storage as stg

def test_load_prices_cached_reads_parquet(tmp_path, monkeypatch):
    monkeypatch.setattr(stg, "LOCAL_ROOT", tmp_path)
    s = stg.Storage()
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=2),
            "open": [1.0, 2.0],
            "high": [1.0, 2.0],
            "low": [1.0, 2.0],
            "close": [1.0, 2.0],
            "volume": [10, 20],
        }
    )
    p = tmp_path / "prices" / "AAA.parquet"
    p.parent.mkdir(parents=True)
    df.to_parquet(p)
    out = stg.load_prices_cached(s, ["AAA"], pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02"))
    assert not out.empty
    assert out["ticker"].unique().tolist() == ["AAA"]


def test_load_prices_cached_uses_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(stg, "LOCAL_ROOT", tmp_path)
    s = stg.Storage()
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=1), "open": [1], "high": [1], "low": [1], "close": [1], "volume": [1]})
    p = tmp_path / "prices" / "AAA.parquet"
    p.parent.mkdir(parents=True)
    df.to_parquet(p)

    calls = {"n": 0}

    orig = stg.Storage.read_bytes

    def fake_read_bytes(self, path):
        calls["n"] += 1
        return orig(self, path)

    monkeypatch.setattr(stg.Storage, "read_bytes", fake_read_bytes)

    start = pd.Timestamp("2020-01-01")
    end = pd.Timestamp("2020-01-01")
    stg.load_prices_cached(s, ["AAA"], start, end)
    stg.load_prices_cached(s, ["AAA"], start, end)

    assert calls["n"] == 1
