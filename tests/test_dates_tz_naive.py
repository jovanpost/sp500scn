import io
import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data_lake import storage as stg


def test_load_prices_index_tz_naive(monkeypatch):
    s = stg.Storage()
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=4, tz="America/New_York"),
            "open": [1, 2, 3, 4],
            "high": [1, 2, 3, 4],
            "low": [1, 2, 3, 4],
            "close": [1, 2, 3, 4],
            "volume": [10, 20, 30, 40],
        }
    )
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)

    def fake_read_bytes(self, path: str) -> bytes:
        assert path == "prices/AAA.parquet"
        return buf.getvalue()

    monkeypatch.setattr(stg.Storage, "read_bytes", fake_read_bytes)

    start = pd.Timestamp("2020-01-01")
    end = pd.Timestamp("2020-01-04")
    out = stg.load_prices_cached(s, ["AAA"], start, end)

    assert out.index.tz is None
    assert out.index.equals(out.index.normalize())

    start2 = pd.Timestamp("2020-01-02").normalize()
    end2 = pd.Timestamp("2020-01-03").normalize()
    sliced = out.loc[(out.index >= start2) & (out.index <= end2)]
    assert len(sliced) == 2
    assert sliced.index.min() == start2
    assert sliced.index.max() == end2
