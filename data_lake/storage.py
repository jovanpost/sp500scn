"""Storage backend with optional Supabase support."""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import pathlib
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
    df = df.copy()
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
        "Date": "date",
        "Datetime": "date",
        "timestamp": "date",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    if "date" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "date"})
        elif "time" in df.columns:
            df = df.rename(columns={"time": "date"})
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if getattr(df["date"].dtype, "tz", None) is not None:
            df["date"] = df["date"].dt.tz_localize(None)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            df[col] = pd.NA
    if "Adj Close" not in df.columns:
        df["Adj Close"] = df.get("Close")
    if "Ticker" in df.columns:
        df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    elif ticker is not None:
        df["Ticker"] = str(ticker).upper()
    if "date" in df.columns:
        df = df.dropna(subset=["date"])
    if "date" in df.columns and "Ticker" in df.columns:
        df = df.sort_values("date").drop_duplicates(subset=["date", "Ticker"], keep="last")
    cols = ["date", "Open", "High", "Low", "Close", "Adj Close", "Volume", "Ticker"]
    df = df[[c for c in cols if c in df.columns]]
    if "date" in df.columns:
        df = df.set_index("date")
    return df


class Storage:
    """Wrapper around local files or Supabase Storage."""

    def __init__(self, local_root: pathlib.Path | None = None, bucket: str | None = None):
        self.local_root = (local_root or DEFAULT_LOCAL_ROOT).resolve()

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

    # ------------------------------------------------------------------
    def _norm(self, path: str) -> pathlib.Path:
        return self.local_root / path

    # ------------------------------------------------------------------
    def list_prefix(self, prefix: str) -> List[str]:
        if self.mode == "supabase" and self.supabase_client is not None:
            r = self.supabase_client.storage.from_(self.bucket).list(prefix=prefix)
            items = r or []
            names: List[str] = []
            for item in items:
                name = None
                if isinstance(item, dict):
                    name = item.get("name") or item.get("Key")
                else:
                    name = getattr(item, "name", None)
                if name:
                    names.append(f"{prefix.rstrip('/')}/{name}".lstrip("/"))
            return names
        base = self._norm(prefix)
        if not base.exists():
            return []
        out: List[str] = []
        for p in base.rglob("*"):
            if p.is_file():
                out.append(str(p.relative_to(self.local_root)).replace("\\", "/"))
        return out

    def exists(self, path: str) -> bool:
        if self.mode == "supabase" and self.supabase_client is not None:
            parent = str(pathlib.Path(path).parent).rstrip("/") + "/"
            children = set(self.list_prefix(parent))
            return any(c.endswith(path) for c in children)
        return self._norm(path).exists()

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
        sample: dict[str, Any] = {}
        try:
            sample["has_membership"] = self.exists("membership/sp500_members.parquet")
            sample["has_AAPL"] = self.exists("prices/AAPL.parquet")
        except Exception as e:
            sample["err"] = repr(e)
        return {
            "mode": self.mode,
            "bucket": self.bucket if self.mode == "supabase" else None,
            "supabase_ok": self.supabase_client is not None,
            "sample": sample,
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
        if self.mode == "supabase" and self.supabase_client is not None:
            res = self.supabase_client.storage.from_(self.bucket).list(prefix)
            items = res or []
            out: List[str] = []
            for it in items:
                name = None
                if isinstance(it, dict):
                    name = it.get("name")
                else:
                    name = getattr(it, "name", None)
                if name:
                    out.append(f"{prefix}/{name}")
            return sorted(out)
        base = self._norm(prefix)
        if not base.exists():
            return []
        return [f"{prefix}/{p.name}" for p in sorted(base.iterdir()) if p.is_file()]

    @classmethod
    def from_env(cls) -> "Storage":
        return cls()


@st.cache_data(show_spinner=False, hash_funcs={Storage: lambda _: 0})
def load_prices_cached(
    _storage: Storage,
    tickers: Iterable[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_salt: str,
) -> pd.DataFrame:
    supabase: Client | None = getattr(_storage, "supabase_client", None)
    if supabase is not None and _storage.mode == "supabase":
        prices: list[pd.DataFrame] = []
        failed_tickers: set[str] = set()
        PAGE_SIZE = 1000
        for ticker in tickers:
            try:
                offset = 0
                ticker_rows: list[dict] = []
                while True:
                    resp = (
                        supabase.table("sp500_ohlcv")
                        .select("ticker, date, open, high, low, close, volume")
                        .eq("ticker", ticker)
                        .gte("date", start.strftime("%Y-%m-%d"))
                        .lte("date", end.strftime("%Y-%m-%d"))
                        .order("date")
                        .range(offset, offset + PAGE_SIZE - 1)
                        .execute()
                    )
                    if not resp.data:
                        break
                    ticker_rows.extend(resp.data)
                    if len(resp.data) < PAGE_SIZE:
                        break
                    offset += PAGE_SIZE
                if ticker_rows:
                    df = pd.DataFrame(ticker_rows)
                    if not df.empty:
                        df = _tidy_prices(df, ticker)
                        prices.append(df)
                    else:
                        failed_tickers.add(ticker)
                else:
                    failed_tickers.add(ticker)
            except Exception as e:  # pragma: no cover - defensive
                failed_tickers.add(ticker)
                st.error(f"Supabase query failed for {ticker}: {str(e)[:50]}.")
        if failed_tickers:
            st.warning(
                f"Failed to load data for {len(failed_tickers)} tickers: {', '.join(sorted(failed_tickers))}"
            )
        if prices:
            all_prices = pd.concat(prices, axis=0)
            if not all_prices.columns.is_unique:
                all_prices = all_prices.loc[:, ~all_prices.columns.duplicated(keep="first")]
                st.info("Dropped duplicate columns in prices.")
            return all_prices
        st.error("No price data loaded from Supabase. Check table name or data availability.")
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for t in tickers:
        key = str(t).upper()
        if key not in _storage._prices_cache:
            try:
                raw = _storage.read_parquet_df(f"prices/{key}.parquet")
            except FileNotFoundError:
                continue
            tidy = _tidy_prices(raw, key)
            _storage._prices_cache[key] = tidy
        df = _storage._prices_cache[key]
        mask = pd.Series(True, index=df.index)
        if start is not None:
            mask &= df.index >= start
        if end is not None:
            mask &= df.index <= end
        subset = df.loc[mask]
        if not subset.empty:
            frames.append(subset)
    if frames:
        return pd.concat(frames).sort_index()
    return pd.DataFrame(
        columns=["Open", "High", "Low", "Close", "Adj Close", "Volume", "Ticker"]
    )


print("load_prices_cached defined and imported successfully")

