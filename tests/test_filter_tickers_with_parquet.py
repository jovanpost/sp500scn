from __future__ import annotations

from pathlib import Path

from data_lake import storage


def _make_storage(tmp_path: Path, monkeypatch):
    local_root = tmp_path / ".lake"
    prices_dir = local_root / "prices"
    prices_dir.mkdir(parents=True)
    monkeypatch.setattr(storage, "LOCAL_ROOT", local_root)
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


def test_filter_tickers_with_parquet_supabase_pagination():
    st = storage.Storage()
    st.mode = "supabase"

    class Bucket:
        def __init__(self):
            self.names = [f"SYM{i}.parquet" for i in range(220)]

        def list(self, path="", limit=100, offset=0, **kwargs):
            assert path in {"", "prices"}
            start = offset
            end = min(offset + limit, len(self.names))
            slice_ = self.names[start:end]
            return {"data": [{"name": name} for name in slice_]}

    st.bucket = Bucket()

    present, missing = storage.filter_tickers_with_parquet(
        st, ["sym0", "SYM150", "missing"]
    )

    assert present == ["SYM0", "SYM150"]
    assert missing == ["MISSING"]
