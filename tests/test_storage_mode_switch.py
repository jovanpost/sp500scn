import types
import os
import sys
import streamlit as st
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data_lake import storage as stg


def test_force_supabase_requires_package(monkeypatch):
    monkeypatch.setenv("FORCE_SUPABASE", "1")
    monkeypatch.setattr(st, "secrets", {"supabase": {"url": "u", "key": "k", "force": True}})
    monkeypatch.setattr(stg, "supabase_available", lambda: (False, "missing"))
    with pytest.raises(RuntimeError):
        stg.Storage()


def test_cache_salt_changes_with_url(monkeypatch):
    monkeypatch.setenv("FORCE_SUPABASE", "1")
    monkeypatch.setattr(stg, "supabase_available", lambda: (True, ""))

    class DummyBucket:
        def list(self, prefix):
            return []

    class DummyClient:
        def __init__(self):
            self.storage = types.SimpleNamespace(
                from_=lambda name: DummyBucket(),
                create_bucket=lambda name: None,
            )

    monkeypatch.setattr(stg, "create_client", lambda url, key: DummyClient())

    monkeypatch.setattr(
        st,
        "secrets",
        {"supabase": {"url": "https://a.supabase.co", "key": "k", "force": True}},
    )
    s1 = stg.Storage()
    salt1 = s1.cache_salt()

    monkeypatch.setattr(
        st,
        "secrets",
        {"supabase": {"url": "https://b.supabase.co", "key": "k", "force": True}},
    )
    s2 = stg.Storage()

    assert s1.mode == s2.mode == "supabase"
    assert salt1 != s2.cache_salt()
