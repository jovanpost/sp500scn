from __future__ import annotations

import os, io, json, pathlib
from typing import Optional, Tuple

try:
    from supabase import create_client  # type: ignore
except Exception:
    create_client = None

import streamlit as st

from .schemas import StorageMode

LOCAL_ROOT = pathlib.Path(".lake").resolve()


def _supabase_creds() -> Optional[Tuple[str, str]]:
    try:
        cfg = st.secrets.get("supabase", {})
        url, key = cfg.get("url"), cfg.get("key")
        if url and key:
            return url, key
    except Exception:
        pass
    return None


class Storage:
    def __init__(self) -> None:
        creds = _supabase_creds()
        self.mode: StorageMode = "supabase" if (create_client and creds) else "local"
        if self.mode == "supabase":
            url, key = creds  # type: ignore
            self.client = create_client(url, key)
            self.bucket = self.client.storage.from_("lake")  # Expect a public bucket named 'lake'
        else:
            LOCAL_ROOT.mkdir(parents=True, exist_ok=True)

    def info(self) -> str:
        return f"storage: {self.mode}"

    def write_bytes(self, path: str, data: bytes) -> str:
        if self.mode == "supabase":
            self.bucket.upload(path, io.BytesIO(data), {"upsert": True})
            return path
        p = (LOCAL_ROOT / path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return str(p)

    def read_bytes(self, path: str) -> bytes:
        if self.mode == "supabase":
            resp = self.bucket.download(path)
            return resp
        p = (LOCAL_ROOT / path)
        return p.read_bytes()

    def exists(self, path: str) -> bool:
        if self.mode == "supabase":
            try:
                self.bucket.download(path)
                return True
            except Exception:
                return False
        return (LOCAL_ROOT / path).exists()
