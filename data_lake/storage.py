from __future__ import annotations

import os, io, json, base64, pathlib, tempfile, logging
from typing import Optional, Tuple

try:
    from supabase import create_client  # type: ignore
except Exception:
    create_client = None

import streamlit as st

from .schemas import StorageMode

LOCAL_ROOT = pathlib.Path(".lake").resolve()

log = logging.getLogger(__name__)


def _supabase_creds() -> Optional[Tuple[str, str]]:
    try:
        cfg = st.secrets.get("supabase", {})
        url, key = cfg.get("url"), cfg.get("key")
        if url and key:
            return url, key
    except Exception:
        pass
    return None


def _classify_key(k: str) -> str:
    if not k:
        return "missing"
    if k.startswith("sb_"):
        return "publishable"
    parts = k.split(".")
    if len(parts) != 3:
        return "not_jwt"
    try:
        payload = json.loads(
            base64.urlsafe_b64decode(parts[1] + "==").decode()
        )
        return str(payload.get("role", "unknown"))
    except Exception:
        return "invalid_jwt"


def _classify_jwt(k: str) -> dict:
    if not k:
        return {"valid": False, "kind": "missing"}
    if k.startswith("sb_"):
        return {"valid": False, "kind": "publishable"}
    parts = k.split(".")
    if len(parts) != 3:
        return {"valid": False, "kind": "not_jwt"}
    try:
        hdr = json.loads(base64.urlsafe_b64decode(parts[0] + "==").decode("utf-8"))
        pl = json.loads(base64.urlsafe_b64decode(parts[1] + "==").decode("utf-8"))
        return {"valid": True, "kind": "jwt", "role": pl.get("role"), "alg": hdr.get("alg")}
    except Exception as e:
        return {"valid": False, "kind": "invalid_jwt", "error": str(e)}


class Storage:
    def __init__(self) -> None:
        creds = _supabase_creds()
        key = creds[1] if creds else ""
        self.key_info = _classify_jwt(key or "")
        self.key_role = _classify_key(key)
        self.creds_present = bool(creds)
        self.error: Optional[str] = None
        self.mode: StorageMode = "supabase" if (create_client and creds) else "local"
        self.bucket_exists = False
        if self.mode == "supabase":
            if self.key_role not in {"service_role", "anon"}:
                self.error = f"invalid key role: {self.key_role}"
                self.mode = "local"
                LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
            else:
                url = creds[0]  # type: ignore
                self.client = create_client(url, key)
                # Ensure a public bucket named 'lake' exists; create if missing.
                try:
                    resp = self.client.storage.list_buckets()
                    data = resp.data or []
                    names = {b.name for b in data}
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
        info = f"storage: {self.mode} (key:{self.key_role})"
        if self.mode == "supabase":
            bucket_status = "ok" if self.bucket_exists else "missing"
            info += f" (bucket:{bucket_status})"
        return info

    def write_bytes(self, path: str, data: bytes) -> str:
        if self.mode == "supabase":
            # storage3 expects string option values; bools break httpx header building.
            file_opts = {"upsert": "true", "contentType": "application/octet-stream"}
            # Upload via temp file path (API accepts str path or file-like)
            tmp = tempfile.NamedTemporaryFile(delete=False)
            try:
                tmp.write(data)
                tmp.flush()
                tmp.close()
                self.bucket.upload(path, tmp.name, file_options=file_opts)
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

    def selftest(self) -> dict:
        info = {"mode": self.mode, "key_info": self.key_info, "bucket": "lake"}
        try:
            if self.mode != "supabase":
                info["note"] = "local mode"
                return info
            resp = self.client.storage.list_buckets()
            names = {getattr(b, "name", None) for b in (resp.data or [])}
            info["buckets"] = sorted([n for n in names if n])
            info["lake_exists"] = "lake" in names
            if not info["lake_exists"]:
                self.client.storage.create_bucket("lake", public=True)
                info["created_lake"] = True
            bkt = self.client.storage.from_("lake")
            file_opts = {"upsert": "true", "contentType": "text/plain"}
            tmp = tempfile.NamedTemporaryFile(delete=False)
            tmp.write(b"ping")
            tmp.flush()
            tmp.close()
            path = "diagnostics/ping.txt"
            bkt.upload(path, tmp.name, file_options=file_opts)
            read = bkt.download(path)
            bkt.remove([path])
            info["write_read_ok"] = read == b"ping"
        except Exception as e:
            info["error"] = f"{type(e).__name__}: {e}"
        return info
