from pathlib import Path

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
