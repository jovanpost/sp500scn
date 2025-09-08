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


def _bucket_names(client) -> set[str]:
    """Return set of bucket names, handling APIResponse vs list."""
    try:
        resp = client.storage.list_buckets()
        buckets = getattr(resp, "data", resp) or []
        names = []
        for b in buckets:
            if isinstance(b, dict):
                names.append(b.get("name"))
            else:
                names.append(getattr(b, "name", None))
        return {n for n in names if n}
    except Exception:
        return set()


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
                # Ensure a public bucket named 'lake'; handle storage3 variants.
                try:
                    names = _bucket_names(self.client)
                    self.bucket_exists = "lake" in names
                    if not self.bucket_exists:
                        # Create as public; re-check to avoid race in concurrent boots.
                        self.client.storage.create_bucket("lake", public=True)
                        names = _bucket_names(self.client)
                        self.bucket_exists = "lake" in names
                    if not self.bucket_exists:
                        raise RuntimeError("Supabase storage bucket 'lake' not available")
                except Exception as e:
                    self.error = f"Failed to ensure Supabase bucket 'lake': {e}"
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
            # storage3 expects header 'x-upsert' as string; it converts file_options internally.
            opts = {"upsert": "true", "contentType": "application/octet-stream"}
            # Upload via temp file path (API accepts str path or file-like)
            tmp = tempfile.NamedTemporaryFile(delete=False)
            try:
                tmp.write(data)
                tmp.flush()
                tmp.close()
                self.bucket.upload(path, tmp.name, file_options=opts)
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

    def list_prefix(self, prefix: str) -> list[str]:
        """Return flat list of object names under a prefix."""
        if self.mode == "supabase":
            try:
                try:
                    resp = self.bucket.list(prefix)
                except TypeError:
                    resp = self.bucket.list(path=prefix)
                items = getattr(resp, "data", resp) or []
                names = []
                for it in items:
                    if isinstance(it, dict):
                        name = it.get("name")
                    else:
                        name = getattr(it, "name", None)
                    if name:
                        names.append(name)
                return names
            except Exception:
                return []
        base = LOCAL_ROOT / prefix
        if not base.exists():
            return []
        return [p.name for p in base.iterdir() if p.is_file()]

    def selftest(self) -> dict:
        info = {"mode": self.mode, "key_info": self.key_info, "bucket": "lake"}
        try:
            if self.mode != "supabase":
                info["note"] = "local mode"
                return info
            names = _bucket_names(self.client)
            info["buckets"] = sorted(names)
            info["lake_exists"] = "lake" in names
            if not info["lake_exists"]:
                self.client.storage.create_bucket("lake", public=True)
                info["created_lake"] = True
            bkt = self.client.storage.from_("lake")
            opts = {"upsert": "true", "contentType": "text/plain"}
            tmp = tempfile.NamedTemporaryFile(delete=False)
            tmp.write(b"ping")
            tmp.flush()
            tmp.close()
            path = "diagnostics/ping.txt"
            bkt.upload(path, tmp.name, file_options=opts)
            read = bkt.download(path)
            bkt.remove([path])
            info["write_read_ok"] = read == b"ping"
            return info
        except Exception as e:
            return {"error": str(e)}
