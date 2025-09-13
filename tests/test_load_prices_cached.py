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
