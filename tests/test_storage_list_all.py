from pathlib import Path
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data_lake import storage


def test_list_all_local_paginates(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "LOCAL_ROOT", tmp_path)
    st = storage.Storage()
    base = tmp_path / "prices"
    base.mkdir(parents=True, exist_ok=True)
    for i in range(150):
        (base / f"{i}.parquet").write_text("x")
    items = st.list_all("prices")
    assert len(items) == 150
    assert Path(items[-1]).suffix == ".parquet"


def test_list_all_handles_apiresponse():
    """Ensure list_all unwraps APIResponse-style results."""

    st = storage.Storage()
    st.mode = "supabase"

    class APIResp:
        def __init__(self, data):
            self.data = data

    class Bucket:
        def list(self, *args, **kwargs):
            return APIResp([
                {"name": "a.parquet"},
                {"name": "b.parquet"},
            ])

    st.bucket = Bucket()
    items = st.list_all("prices")
    assert items == ["prices/a.parquet", "prices/b.parquet"]
