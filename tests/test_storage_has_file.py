import importlib
import sys
from pathlib import Path


def test_storage_has_file_without_exists(tmp_path):
    # Ensure repository root is on path so the UI module can be imported
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    mod = importlib.import_module("ui.pages.90_Data_Lake_Phase1")

    class DummyStorage:
        def read_bytes(self, path: str) -> bytes:
            return (tmp_path / path).read_bytes()

    base = tmp_path / "prices"
    base.mkdir(parents=True, exist_ok=True)
    (base / "AAPL.parquet").write_text("x")

    st = DummyStorage()
    assert mod._storage_has_file(st, "prices/AAPL.parquet") is True
    assert mod._storage_has_file(st, "prices/MSFT.parquet") is False
