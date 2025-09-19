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


def test_list_all_supabase_paginates():
    st = storage.Storage()
    st.mode = "supabase"

    class Bucket:
        def __init__(self):
            self.calls: list[tuple[str | None, int, int]] = []

        def list(self, path="", limit=100, offset=0, **kwargs):
            self.calls.append((path, limit, offset))
            assert path in {"", "prices"}
            total = 205
            start = offset
            end = min(offset + limit, total)
            data = [{"name": f"{i}.parquet"} for i in range(start, end)]
            return {"data": data}

    bucket = Bucket()
    st.bucket = bucket

    items = st.list_all("prices")

    assert len(items) == 205
    assert set(items) == {f"prices/{i}.parquet" for i in range(205)}
    assert len(bucket.calls) >= 3
