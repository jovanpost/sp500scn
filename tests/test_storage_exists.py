from data_lake import storage


def test_list_prefix_local(tmp_path, monkeypatch):
    local_root = tmp_path / ".lake"
    prices_dir = local_root / "prices"
    prices_dir.mkdir(parents=True)
    (prices_dir / "AAPL.parquet").write_text("a")
    (prices_dir / "MSFT.parquet").write_text("m")

    monkeypatch.setattr(storage, "LOCAL_ROOT", local_root)
    st = storage.Storage()

    entries = st.list_prefix("prices")
    assert sorted(entries) == ["prices/AAPL.parquet", "prices/MSFT.parquet"]


def test_exists_local(tmp_path, monkeypatch):
    local_root = tmp_path / ".lake"
    prices_dir = local_root / "prices"
    prices_dir.mkdir(parents=True)
    (prices_dir / "AAPL.parquet").write_text("a")

    monkeypatch.setattr(storage, "LOCAL_ROOT", local_root)
    st = storage.Storage()

    assert st.exists("prices/AAPL.parquet") is True
    assert st.exists("prices/MSFT.parquet") is False
