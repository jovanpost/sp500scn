from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest

from data_lake import storage
from data_lake.storage import ConfigurationError


def _set_env(monkeypatch, layout: str = "flat", prefix: str = "lake/prices") -> None:
    monkeypatch.setenv("LAKE_LAYOUT", layout)
    monkeypatch.setenv("LAKE_PRICES_PREFIX", prefix)


def _make_storage(tmp_path: Path, monkeypatch, *, layout: str = "flat"):
    local_root = tmp_path / ".lake"
    prices_dir = local_root / "prices"
    prices_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(storage, "LOCAL_ROOT", local_root)
    _set_env(monkeypatch, layout=layout)
    return storage.Storage()


def test_filter_tickers_with_parquet_returns_present_and_missing(tmp_path, monkeypatch):
    st = _make_storage(tmp_path, monkeypatch)
    prices_dir = Path(storage.LOCAL_ROOT) / "prices"
    (prices_dir / "AAPL.parquet").write_text("a")
    (prices_dir / "MSFT.parquet").write_text("m")

    present, missing = storage.filter_tickers_with_parquet(
        st, ["aapl", "MSFT", "GOOG", None, "AAPL"]
    )

    assert present == ["AAPL", "MSFT"]
    assert missing == ["GOOG"]


def test_filter_tickers_with_parquet_handles_empty_and_missing(tmp_path, monkeypatch):
    st = _make_storage(tmp_path, monkeypatch)

    present, missing = storage.filter_tickers_with_parquet(st, [])
    assert present == []
    assert missing == []

    present_missing, missing_only = storage.filter_tickers_with_parquet(st, ["XYZ"])
    assert present_missing == []
    assert missing_only == ["XYZ"]


def test_filter_tickers_with_parquet_supabase_pagination(monkeypatch):
    _set_env(monkeypatch)
    st = storage.Storage()
    st.mode = "supabase"
    st.bucket = "lake"

    class Bucket:
        def __init__(self):
            self.names = [f"SYM{i}.parquet" for i in range(220)]

        def list(self, path="", limit=100, offset=0, **kwargs):
            assert path in {"", "prices"}
            start = offset
            end = min(offset + limit, len(self.names))
            slice_ = self.names[start:end]
            return {"data": [{"name": name} for name in slice_]}

    class Client:
        class StorageAPI:
            def from_(self, bucket):
                assert bucket == "lake"
                return Bucket()

        storage = StorageAPI()

    st.supabase_client = Client()

    present, missing = storage.filter_tickers_with_parquet(
        st, ["sym0", "SYM150", "missing"]
    )

    assert present == ["SYM0", "SYM150"]
    assert missing == ["MISSING"]


def test_filter_tickers_with_parquet_normalizes_ticker_variants(tmp_path, monkeypatch):
    st = _make_storage(tmp_path, monkeypatch)
    prices_dir = Path(storage.LOCAL_ROOT) / "prices"
    (prices_dir / "BRK_B.parquet").write_text("b")
    (prices_dir / "BF-B.parquet").write_text("bf")

    present, missing = storage.filter_tickers_with_parquet(
        st, ["brk.b", "BF.B", "ally"]
    )

    assert present == ["BRK.B", "BF.B"]
    assert missing == ["ALLY"]


def test_filter_tickers_with_parquet_partitioned_layout(tmp_path, monkeypatch):
    st = _make_storage(tmp_path, monkeypatch, layout="partitioned")
    base = Path(storage.LOCAL_ROOT) / "prices" / "AAPL" / "date=2020-01-01"
    base.mkdir(parents=True)
    (base / "part-0.parquet").write_text("data")

    present, missing = storage.filter_tickers_with_parquet(st, ["AAPL", "MSFT"])

    assert present == ["AAPL"]
    assert missing == ["MSFT"]


def test_filter_tickers_with_parquet_invalid_layout(monkeypatch):
    _set_env(monkeypatch, layout="bogus")
    st = storage.Storage()

    with pytest.raises(ConfigurationError):
        storage.filter_tickers_with_parquet(st, ["AAPL"])
