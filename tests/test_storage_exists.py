from pathlib import Path

from data_lake import storage


def test_exists_local(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "LOCAL_ROOT", tmp_path)
    st = storage.Storage()
    base = tmp_path / "prices"
    base.mkdir(parents=True, exist_ok=True)
    (base / "AAPL.parquet").write_text("x")
    assert st.exists("prices/AAPL.parquet")
    assert not st.exists("prices/MSFT.parquet")


def test_exists_handles_apiresponse():
    st = storage.Storage()
    st.mode = "supabase"

    class APIResp:
        def __init__(self, data):
            self.data = data

    class Bucket:
        def list(self, folder, *args, **kwargs):
            if folder == "prices":
                return APIResp([{"name": "AAPL.parquet"}])
            return APIResp([])

    st.bucket = Bucket()
    assert st.exists("prices/AAPL.parquet") is True
    assert st.exists("prices/MSFT.parquet") is False


def test_exists_handles_fileobject():
    st = storage.Storage()
    st.mode = "supabase"

    class FileObject:
        def __init__(self, name: str):
            self.name = name

    class APIResp:
        def __init__(self, data):
            self.data = data

    class Bucket:
        def list(self, folder, *args, **kwargs):
            if folder == "prices":
                return APIResp([FileObject("AAPL.parquet")])
            return APIResp([])

    st.bucket = Bucket()
    assert st.exists("prices/AAPL.parquet") is True
    assert st.exists("prices/MSFT.parquet") is False


def test_exists_handles_string_list():
    st = storage.Storage()
    st.mode = "supabase"

    class Bucket:
        def list(self, folder, *args, **kwargs):
            if folder == "prices":
                return ["AAPL.parquet"]
            return []

    st.bucket = Bucket()
    assert st.exists("prices/AAPL.parquet") is True
    assert st.exists("prices/MSFT.parquet") is False


def test_exists_local_custom_root(tmp_path):
    """exists() and list_prefix should respect ``local_root`` overrides."""
    lake = tmp_path / ".lake" / "prices"
    lake.mkdir(parents=True)
    (lake / "AAPL.parquet").write_text("x")
    st = storage.Storage()
    st.local_root = tmp_path / ".lake"
    assert st.exists("prices/")
    assert st.exists("prices/AAPL.parquet")
    assert not st.exists("prices/MSFT.parquet")


def test_list_prefix_relative_root(tmp_path, monkeypatch):
    """list_prefix and exists work when ``local_root`` is relative."""
    monkeypatch.chdir(tmp_path)
    rel = Path("lake")
    (rel / "history").mkdir(parents=True)
    (rel / "history" / "AAPL.parquet").write_text("x")
    monkeypatch.setattr(storage, "LOCAL_ROOT", rel)
    st = storage.Storage()
    assert st.list_prefix("history/") == ["history/AAPL.parquet"]
    assert st.exists("history/")
    assert st.exists("history/AAPL.parquet")
