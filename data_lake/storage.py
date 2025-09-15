"""Storage backend with optional Supabase support."""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import pathlib
from pathlib import Path
from typing import Any, Iterable, List, Tuple

import pandas as pd
import streamlit as st

try:  # pragma: no cover - optional dependency
    from supabase import create_client, Client  # type: ignore
except Exception:  # pragma: no cover - tests may not have package
    create_client = None  # type: ignore
    Client = None  # type: ignore

log = logging.getLogger(__name__)

DEFAULT_LOCAL_ROOT = pathlib.Path("data")
LOCAL_ROOT = DEFAULT_LOCAL_ROOT


def supabase_available() -> Tuple[bool, str]:
    """Best-effort check that Supabase client can be created."""

    try:
        url = st.secrets.get("SUPABASE_URL")
    except Exception:
        url = None
    try:
        key = st.secrets.get("SUPABASE_ANON_KEY")
    except Exception:
        key = None
    if url is None or key is None:
        try:
            cfg = st.secrets.get("supabase", {})  # type: ignore[assignment]
        except Exception:
            cfg = {}
        url = url or cfg.get("url")
        key = key or cfg.get("key")
    url = url or os.getenv("SUPABASE_URL")
    key = key or os.getenv("SUPABASE_ANON_KEY")
    if not url or not key:
        return False, "missing credentials"
    if create_client is None:
        return False, "supabase package not installed"
    return True, ""


def _classify_key(key: str) -> str:
    if key.startswith("sb_"):
        return "publishable"
    parts = key.split(".")
    if len(parts) != 3:
        return "not_jwt"
    try:
        padded = parts[1] + "=" * (-len(parts[1]) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded).decode())
    except Exception:
        return "invalid_jwt"
    return payload.get("role", "invalid_jwt")


def _tidy_prices(df: pd.DataFrame, ticker: str | None = None) -> pd.DataFrame:
    """Normalize OHLCV schema to Yahoo-style columns, add Ticker, and drop dupes (last wins)."""

    df = df.copy()
    # If both exist, prefer "Ticker"
    if "ticker" in df.columns and "Ticker" in df.columns:
        df = df.drop(columns=["ticker"])

    rename_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "adjclose": "Adj Close",
        "adj_close": "Adj Close",
        "adj close": "Adj Close",
        "ticker": "Ticker",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            df[col] = pd.NA

    if "Adj Close" not in df.columns:
        df["Adj Close"] = df.get("Close")

    if "Ticker" in df.columns:
        df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    elif ticker is not None:
        df["Ticker"] = str(ticker).upper()
    else:
        df["Ticker"] = "UNKNOWN"

    # index by date; strip tz and drop duplicates (keep last)
    if "date" in df.columns:
        idx = pd.to_datetime(df["date"], errors="coerce")
        # if tz-aware, convert to naive
        try:
            idx = idx.dt.tz_localize(None)
        except Exception:
            pass
        df = df.assign(date=idx).dropna(subset=["date"]).set_index("date")
    df = df[["Open", "High", "Low", "Close", "Adj Close", "Volume", "Ticker"]]
    df = df[~df.index.duplicated(keep="last")]

    return df


class Storage:
    """Wrapper around local files or Supabase Storage."""

    def __init__(self, local_root: pathlib.Path | None = None, bucket: str | None = None):
        self.local_root = (local_root or LOCAL_ROOT).resolve()

        try:
            supabase_cfg = st.secrets.get("supabase", {})  # type: ignore[assignment]
        except Exception:
            supabase_cfg = {}
        try:
            self.bucket = bucket or st.secrets.get("SUPABASE_BUCKET")
        except Exception:
            self.bucket = None
        self.bucket = self.bucket or supabase_cfg.get("bucket") or bucket or "lake"

        try:
            raw_flag = st.secrets.get("FORCE_SUPABASE", None)
        except Exception:
            raw_flag = None
        if raw_flag is None:
            raw_flag = supabase_cfg.get("force")
        if raw_flag is None:
            raw_flag = os.getenv("FORCE_SUPABASE")
        self.force_supabase: bool = (
            str(raw_flag).lower() in {"1", "true", "yes"} if raw_flag is not None else False
        )

        try:
            self.supabase_url = st.secrets.get("SUPABASE_URL")
        except Exception:
            self.supabase_url = None
        self.supabase_url = self.supabase_url or supabase_cfg.get("url") or os.getenv("SUPABASE_URL")
        try:
            self.supabase_key = st.secrets.get("SUPABASE_ANON_KEY")
        except Exception:
            self.supabase_key = None
        self.supabase_key = self.supabase_key or supabase_cfg.get("key") or os.getenv("SUPABASE_ANON_KEY")

        self.key_info: dict[str, str] = {
            "kind": _classify_key(self.supabase_key) if self.supabase_key else "service_role",
        }

        self.supabase_client: Client | None = None
        self.mode: str = "local"
        self._prices_cache: dict[str, pd.DataFrame] = {}

        self._maybe_init_supabase()

    # ------------------------------------------------------------------
    def _supabase_secrets_present(self) -> bool:
        return bool(self.supabase_url and self.supabase_key and create_client is not None)

    def _init_supabase(self) -> bool:
        try:
            assert self._supabase_secrets_present()
            self.supabase_client = create_client(self.supabase_url, self.supabase_key)  # type: ignore
            return True
        except Exception as e:  # pragma: no cover - defensive
            log.warning("Supabase init failed: %s", e)
            self.supabase_client = None
            return False

    def _maybe_init_supabase(self) -> None:
        if (self.force_supabase or self._supabase_secrets_present()) and self._init_supabase():
            self.mode = "supabase"
        else:
            if self.force_supabase:
                raise RuntimeError(
                    "Supabase required but unavailable. Set SUPABASE_URL and SUPABASE_ANON_KEY or disable FORCE_SUPABASE."
                )
            self.mode = "local"

    def ensure_supabase(self) -> bool:
        if self.mode == "supabase":
            return True
        if self._supabase_secrets_present() and self._init_supabase():
            self.mode = "supabase"
            return True
        return False

    def supabase_available(self) -> bool:
        """Return ``True`` if a Supabase client is configured."""
        return self.supabase_client is not None

    # ------------------------------------------------------------------
    def _norm(self, path: str) -> pathlib.Path:
        return self.local_root / path

    # ------------------------------------------------------------------
    def list_prefix(self, prefix: str) -> list[str]:
        """List files beneath `prefix` for both local and Supabase (SDK v2). Handles pagination and
        SDK variants that either return a raw list or an object without `.data`."""
        if self.mode == "local":
            root = (self.local_root or LOCAL_ROOT).resolve()
            base = (root / prefix).resolve()
            if not base.exists():
                return []
            return [
                str(p.relative_to(root)).replace("\\", "/")
                for p in base.rglob("*") if p.is_file()
            ]
        norm_prefix = prefix.rstrip("/")
        api = self.supabase_client.storage.from_(self.bucket)
        out: list[str] = []
        offset = 0
        while True:
            # Prefer path positional; some builds reject keyword "prefix"
            try:
                resp = api.list(norm_prefix)  # returns up to default page size (often 100)
                data = getattr(resp, "data", resp)
                # Try larger pages if supported
                if isinstance(data, list) and len(data) == 0 and offset == 0:
                    resp = api.list(norm_prefix, limit=1000, offset=0)
                    data = getattr(resp, "data", resp)
            except TypeError:
                # Alternative signature
                try:
                    resp = api.list(norm_prefix, limit=1000, offset=offset)
                except TypeError:
                    resp = api.list(norm_prefix)
                data = getattr(resp, "data", resp)

            if not isinstance(data, list) or not data:
                break
            out.extend([
                f"{norm_prefix}/{(d if isinstance(d, str) else d.get('name') if isinstance(d, dict) else getattr(d, 'name', ''))}"
                for d in data
            ])
            # paginate if page size looks capped
            page_size = len(data)
            if page_size < 1000:
                break
            offset += page_size
        return out

    def exists(self, path: str) -> bool:
        parent = str(Path(path).parent).replace("\\", "/")
        name = Path(path).name
        try:
            children = self.list_prefix(parent)
        except Exception:
            return False
        return any(Path(p).name == name for p in children)

    # ------------------------------------------------------------------
    def read_bytes(self, path: str) -> bytes:
        if self.mode == "supabase" and self.supabase_client is not None:
            res = self.supabase_client.storage.from_(self.bucket).download(path)
            if hasattr(res, "content"):
                return res.content  # supabase-py
            return res  # type: ignore
        return self._norm(path).read_bytes()

    def write_bytes(self, path: str, data: bytes, *, content_type: str | None = None) -> None:
        if self.mode == "supabase" and self.supabase_client is not None:
            opts = {"contentType": content_type} if content_type else None
            self.supabase_client.storage.from_(self.bucket).upload(path, data, file_options=opts)
        else:
            dest = self._norm(path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)

    def read_parquet_df(self, path: str) -> pd.DataFrame:
        b = self.read_bytes(path)
        return pd.read_parquet(io.BytesIO(b))

    def read_parquet(self, path: str) -> pd.DataFrame:
        if self.mode == "supabase":
            return self.read_parquet_df(path)
        return pd.read_parquet(self._norm(path))

    # ------------------------------------------------------------------
    def diagnostics(self) -> dict[str, Any]:
        """Return basic information about the current storage backend."""
        return {
            "mode": self.mode,
            "bucket": self.bucket if self.mode == "supabase" else None,
            "local_root": str(self.local_root) if self.mode == "local" else None,
            "supabase_available": self.supabase_available(),
        }

    def info(self) -> str:
        if self.mode == "local":
            return f"mode={self.mode} root={self.local_root}"
        return f"mode={self.mode} bucket={self.bucket}"

    def cache_salt(self) -> str:
        return f"mode={self.mode}|url={self.supabase_url or ''}"

    def selftest(self) -> dict[str, Any]:
        return {"ok": True, "mode": self.mode}

    def list_all(self, prefix: str) -> List[str]:
        if self.mode == "supabase":
            norm = prefix.rstrip("/")
            items: Iterable | None = None
            if self.supabase_client is not None:
                res = self.supabase_client.storage.from_(self.bucket).list(norm)
                items = getattr(res, "data", res)
            elif hasattr(self.bucket, "list"):
                res = self.bucket.list(norm)
                items = getattr(res, "data", res)
            out: List[str] = []
            for it in items or []:
                name = None
                if isinstance(it, dict):
                    name = it.get("name")
                else:
                    name = getattr(it, "name", None) or str(it)
                if name:
                    out.append(f"{norm}/{name}")
            return sorted(out)
        norm = prefix.rstrip("/")
        base = self._norm(norm)
        if not base.exists():
            return []
        return [f"{norm}/{p.name}" for p in sorted(base.iterdir()) if p.is_file()]

    @classmethod
    def from_env(cls) -> "Storage":
        return cls()


@st.cache_data(hash_funcs={Storage: lambda _: 0}, show_spinner=False)
def load_prices_cached(_storage: "Storage",
                       tickers: list[str],
                       start: pd.Timestamp | None = None,
                       end: pd.Timestamp | None = None,
                       cache_salt: str = "") -> pd.DataFrame:
    """Load OHLCV for `tickers` from object storage Parquet files only.
    Output columns: Open, High, Low, Close, Adj Close, Volume, Ticker; index is naive datetime."""
    frames: list[pd.DataFrame] = []
    for t in tickers:
        path = f"prices/{t}.parquet"
        if not getattr(_storage, "exists", lambda *_: False)(path):
            continue
        try:
            raw = _storage.read_parquet_df(path)
        except Exception:
            # fallback: bytes -> parquet
            raw = pd.read_parquet(io.BytesIO(_storage.read_bytes(path)))
        tidy = _tidy_prices(raw, ticker=t)
        if start is not None or end is not None:
            s = pd.Timestamp(start) if start is not None else None
            e = pd.Timestamp(end) if end is not None else None
            if s is not None:
                tidy = tidy[tidy.index >= s]
            if e is not None:
                tidy = tidy[tidy.index <= e]
        # keep date as a column too for downstream code that expects it
        tidy = tidy.reset_index().rename(columns={"index": "date"})
        frames.append(tidy)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    # ensure expected dtypes
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    try:
        out["date"] = out["date"].dt.tz_localize(None)
    except Exception:
        pass
    out = out.dropna(subset=["date"])
    return out

