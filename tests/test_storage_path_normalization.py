import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data_lake.storage import (
    Storage,
    _normalize_storage_key,
    _resolve_prices_prefix,
)


@pytest.mark.parametrize(
    "bucket, raw, expected",
    [
        ("lake", "prices/AAPL.parquet", "prices/AAPL.parquet"),
        ("lake", "/prices/AAPL.parquet", "prices/AAPL.parquet"),
        ("lake", "lake/prices/AAPL.parquet", "prices/AAPL.parquet"),
        ("lake", "/lake/prices/AAPL.parquet", "prices/AAPL.parquet"),
        ("lake", "//lake///prices//AAPL.parquet", "prices/AAPL.parquet"),
        ("lake", "lake", ""),
        ("", "prices/AAPL.parquet", "prices/AAPL.parquet"),
    ],
)
def test_normalize_storage_key(bucket, raw, expected):
    assert _normalize_storage_key(raw, bucket=bucket) == expected


@pytest.mark.parametrize(
    "env_value, expected",
    [
        ("prices", "prices"),
        ("/prices", "prices"),
        ("lake/prices", "prices"),
        ("lake/prices/", "prices"),
        ("//lake///prices", "prices"),
        ("lake", ""),
    ],
)
def test_resolve_prices_prefix_handles_bucket(monkeypatch, env_value, expected):
    monkeypatch.setenv("LAKE_PRICES_PREFIX", env_value)
    storage = Storage()
    storage.bucket = "lake"
    assert _resolve_prices_prefix(storage) == expected


def test_read_bytes_normalizes_supabase_key():
    storage = Storage()
    storage.mode = "supabase"
    storage.bucket = "lake"

    called: dict[str, str] = {}

    class Bucket:
        def download(self, key: str) -> bytes:
            called["key"] = key
            return b"payload"

    class Client:
        class StorageAPI:
            def __init__(self, bucket_obj: Bucket) -> None:
                self._bucket = bucket_obj

            def from_(self, bucket: str) -> Bucket:
                assert bucket == "lake"
                return self._bucket

        def __init__(self) -> None:
            self.storage = Client.StorageAPI(Bucket())

    storage.supabase_client = Client()
    data = storage.read_bytes("lake/prices/AAA.parquet")

    assert data == b"payload"
    assert called["key"] == "prices/AAA.parquet"
