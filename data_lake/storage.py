from __future__ import annotations

import os, io, json, pathlib, tempfile
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
        self.creds_present = bool(creds)
        self.mode: StorageMode = "supabase" if (create_client and creds) else "local"
        self.bucket_exists = False
        if self.mode == "supabase":
            url, key = creds  # type: ignore
            self.client = create_client(url, key)
            # Ensure a public bucket named 'lake' exists; create if missing.
            try:
                resp = self.client.storage.list_buckets()
                data = getattr(resp, "data", None) or []
                names = {
                    (getattr(b, "name", None) or (b.get("name") if isinstance(b, dict) else None))
                    for b in data
                }
                self.bucket_exists = "lake" in names
                if not self.bucket_exists:
                    self.client.storage.create_bucket("lake", public=True)
                    self.bucket_exists = True
            except Exception:
                # Ignore errors from older SDKs or insufficient permissions
                self.bucket_exists = False
            self.bucket = self.client.storage.from_("lake")
        else:
            LOCAL_ROOT.mkdir(parents=True, exist_ok=True)

    def info(self) -> str:
        bucket_status = "n/a"
        if self.mode == "supabase":
            bucket_status = "ok" if self.bucket_exists else "missing"
        return f"storage: {self.mode} (bucket: {bucket_status})"

    def write_bytes(self, path: str, data: bytes) -> str:
        if self.mode == "supabase":
            # storage3 expects string header values; bools break httpx header building.
            opts = {
                "cache-control": "3600",
                "content-type": "application/octet-stream",
                "upsert": "true",
            }
            # Upload via temp file path (API accepts str path or file-like)
            tmp = tempfile.NamedTemporaryFile(delete=False)
            try:
                tmp.write(data)
                tmp.flush()
                tmp.close()
                self.bucket.upload(path, tmp.name, opts)
            finally:
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass
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
