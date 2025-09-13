from __future__ import annotations

import os, io, json, base64, pathlib, tempfile, logging
from typing import Optional, Tuple, Any

import pandas as pd
import yfinance as yf
from requests.exceptions import RequestException
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

try:
    from supabase import create_client, Client  # type: ignore
except Exception:
    create_client = None
    Client = None

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


def _as_buckets_list(resp: Any):
    # storage3 may return list, dict with 'data', or object with .data
    if resp is None:
        return []
    if isinstance(resp, list):
        return resp
    if isinstance(resp, dict) and "data" in resp:
        return resp["data"]
    if hasattr(resp, "data"):
        return resp.data
    return []


def _bucket_name(b: Any) -> Optional[str]:
    # dict or object
    if isinstance(b, dict):
        return b.get("name") or b.get("id")
    return getattr(b, "name", None) or getattr(b, "id", None)


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
                self.supabase_client = self.client
                # Ensure a bucket named 'lake' exists (no public kw in storage3==0.12.x)
                try:
                    resp = self.client.storage.list_buckets()
                    names = {_bucket_name(b) for b in _as_buckets_list(resp)}
                    self.bucket_exists = "lake" in names
                    if not self.bucket_exists:
                        try:
                            self.client.storage.create_bucket("lake")
                            self.bucket_exists = True
                        except Exception:
                            pass
                except Exception:
                    self.bucket_exists = False
                self.bucket = self.client.storage.from_("lake")
        else:
            LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
            self.supabase_client = None

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

    def read_parquet(self, path: str):
        import pandas as pd
        if self.mode == "supabase":
            buf = self.bucket.download(path)
            return pd.read_parquet(io.BytesIO(buf))
        p = (LOCAL_ROOT / path)
        return pd.read_parquet(p)

    def exists(self, path: str) -> bool:
        if self.mode == "supabase":
            try:
                self.bucket.download(path)
                return True
            except Exception:
                return False
        return (LOCAL_ROOT / path).exists()

    def list_all(self, prefix: str) -> list[str]:
        """Return all object names under prefix handling pagination."""

        if self.mode != "supabase":
            base = (LOCAL_ROOT / prefix)
            return [
                str(p.relative_to(LOCAL_ROOT)).replace("\\", "/")
                for p in base.rglob("*")
                if p.is_file()
            ]

        items: list[str] = []
        offset = 0
        while True:
            try:
                resp = self.bucket.list(
                    prefix,
                    {
                        "limit": 1000,
                        "offset": offset,
                        "sortBy": {"column": "name", "order": "asc"},
                    },
                )
            except TypeError:
                resp = self.bucket.list(
                    path=prefix,
                    limit=1000,
                    offset=offset,
                    sortBy={"column": "name", "order": "asc"},
                )
            batch = getattr(resp, "data", resp) or []
            if not batch:
                break
            items.extend(
                [f"{prefix.rstrip('/')}/{obj['name']}" for obj in batch if "name" in obj]
            )
            if len(batch) < 1000:
                break
            offset += len(batch)
        return items

    def list_prefix(self, prefix: str) -> list[str]:
        """Return list of object names under a prefix."""

        if self.mode == "supabase":
            try:
                bucket = self.client.storage.from_("lake")
                names: list[str] = []
                offset = 0
                page = 200
                while True:
                    # storage3 list() treats the 'path' like a folder; ensure clean prefix
                    path = prefix.strip("/")
                    resp = bucket.list(path=path, limit=page, offset=offset)
                    items = getattr(resp, "data", resp) or []
                    if not items:
                        break
                    for it in items:
                        n = it.get("name") if isinstance(it, dict) else getattr(it, "name", "")
                        if n:
                            names.append(f"{path}/{n}" if not n.startswith(f"{path}/") else n)
                    if len(items) < page:
                        break
                    offset += page
                return names
            except Exception:
                return []
        else:
            base = (LOCAL_ROOT / prefix)
            if not base.exists():
                return []
            return [
                str(p.relative_to(LOCAL_ROOT)).replace("\\", "/")
                for p in base.rglob("*") if p.is_file()
            ]

    def exists_prefix(self, prefix: str) -> bool:
        """Return True if any object exists under the given prefix."""

        prefix = prefix.rstrip("/") + "/"
        if self.mode == "supabase":
            try:
                try:
                    resp = self.bucket.list(prefix, limit=1)
                except TypeError:
                    resp = self.bucket.list(path=prefix, limit=1)
                items = getattr(resp, "data", resp) or []
                return bool(items)
            except Exception:
                return False
        base = LOCAL_ROOT / prefix
        return base.exists() and any(base.iterdir())

    def selftest(self) -> dict:
        info = {"mode": self.mode, "key_info": self.key_info, "bucket": "lake"}
        try:
            if self.mode != "supabase":
                info["note"] = "local mode"
                return info
            resp = self.client.storage.list_buckets()
            names = {_bucket_name(b) for b in _as_buckets_list(resp)}
            info["buckets"] = sorted(n for n in names if n)
            info["lake_exists"] = "lake" in names
            if not info["lake_exists"]:
                try:
                    self.client.storage.create_bucket("lake")
                    info["created_lake"] = True
                except Exception:
                    info["created_lake"] = False
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


@st.cache_data(show_spinner=False)
def load_prices_cached(
    _storage: Storage, tickers: list[str], start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Fetch OHLCV data for tickers using Supabase first, then yfinance."""
    prices: list[pd.DataFrame] = []
    missing: list[str] = []

    supabase: Client | None = getattr(_storage, "supabase_client", None)
    if supabase:
        for t in tickers:
            try:
                query = (
                    supabase.table("sp500_ohlcv")
                    .select("*")
                    .eq("ticker", t)
                    .gte("date", start.strftime("%Y-%m-%d"))
                    .lte("date", end.strftime("%Y-%m-%d"))
                    .execute()
                )
                df = pd.DataFrame(query.data)
                if not df.empty:
                    df["Date"] = pd.to_datetime(df["date"])
                    df = df[
                        ["Date", "Open", "High", "Low", "Close", "Volume", "ticker"]
                    ].set_index("Date")
                    prices.append(df)
                else:
                    missing.append(t)
            except Exception as e:
                st.warning(f"Supabase query failed for {t}: {str(e)[:50]}... Falling back to yfinance.")
                missing.append(t)
    else:
        missing = list(tickers)

    if missing:

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(min=4, max=10),
            retry=retry_if_exception_type(
                (RequestException, json.JSONDecodeError, ConnectionError)
            ),
        )
        def _dl(ticker: str) -> pd.DataFrame:
            try:
                return yf.download(
                    ticker,
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    progress=False,
                )
            except Exception as e:
                st.warning(f"yfinance failed for {ticker}: {str(e)[:50]}... Skipping.")
                return pd.DataFrame()

        for t in missing:
            df = _dl(t)
            if not df.empty:
                df["ticker"] = t
                prices.append(df)

    if prices:
        all_prices = pd.concat(prices, ignore_index=False)
        if not all_prices.columns.is_unique:
            all_prices = all_prices.loc[:, ~all_prices.columns.duplicated()]
        return all_prices
    return pd.DataFrame()
