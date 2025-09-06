import pytest
from types import SimpleNamespace

from data_lake.storage import Storage


def make_storage(monkeypatch, buckets):
    """Create a Storage instance with a fake Supabase client."""
    import data_lake.storage as storage

    class FakeAPI:
        def __init__(self, buckets):
            self._buckets = buckets
            self.created = False
        def list_buckets(self):
            return SimpleNamespace(data=self._buckets)
        def create_bucket(self, name, public=True):
            self.created = True
        def from_(self, name):
            return SimpleNamespace(name=name)

    fake_client = SimpleNamespace(storage=FakeAPI(buckets))

    monkeypatch.setattr(storage, "_supabase_creds", lambda: ("url", "key"))
    monkeypatch.setattr(storage, "_classify_key", lambda _: "service_role")
    monkeypatch.setattr(storage, "create_client", lambda url, key: fake_client)

    store = Storage()
    return store, fake_client.storage


def test_creates_bucket_when_missing(monkeypatch):
    store, api = make_storage(monkeypatch, [{"name": "foo"}])
    assert api.created is True
    assert store.bucket_exists is True


def test_detects_existing_bucket(monkeypatch):
    store, api = make_storage(monkeypatch, [{"name": "lake"}])
    assert api.created is False
    assert store.bucket_exists is True
