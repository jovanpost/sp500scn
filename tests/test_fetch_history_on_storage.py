import os, sys, pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import prices as up
from data_lake import storage as stg


def test_fetch_history_on_storage(monkeypatch):
    df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-01")],
            "open": [1],
            "high": [1],
            "low": [1],
            "close": [1],
            "volume": [100],
        }
    )

    def fake_read_parquet(self, path: str):
        assert path == "prices/AAA.parquet"
        return df

    monkeypatch.setattr(stg.Storage, "read_parquet", fake_read_parquet)

    out = up.fetch_history("AAA")
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    assert out.iloc[0]["Open"] == 1
