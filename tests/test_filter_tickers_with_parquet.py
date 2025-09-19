from __future__ import annotations

from pathlib import Path
import importlib
import sys

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


def test_ui_fallback_filter(monkeypatch, tmp_path):
    from data_lake import storage as dl_storage

    if hasattr(dl_storage, "filter_tickers_with_parquet"):
        monkeypatch.delattr(dl_storage, "filter_tickers_with_parquet", raising=False)

    local_root = tmp_path / ".lake"
    monkeypatch.setattr(dl_storage, "LOCAL_ROOT", local_root)
    st = dl_storage.Storage()

    prices_dir = local_root / "prices"
    prices_dir.mkdir(parents=True, exist_ok=True)
    (prices_dir / "AAPL.parquet").write_text("data")

    module_name = "ui.pages.45_YdayVolSignal_Open"
    sys.modules.pop(module_name, None)
    page_module = importlib.import_module(module_name)

    present, missing = page_module.filter_tickers_with_parquet(st, ["AAPL", "MSFT"])

    assert present == ["AAPL"]
    assert missing == ["MSFT"]
    assert page_module.filter_tickers_with_parquet is page_module._fallback_filter_tickers_with_parquet

    sys.modules.pop(module_name, None)
