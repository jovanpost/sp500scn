# (full replacement content)
# storage.py
from __future__ import annotations

import io
import os
import base64
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Sequence

import pandas as pd
import streamlit as st
from pandas.api.types import is_numeric_dtype

# HTTP for optional HEAD verification
import requests

log = logging.getLogger(__name__)

try:
    from supabase import Client, create_client  # type: ignore
except Exception as exc:  # pragma: no cover - supabase optional
    Client = Any  # type: ignore[assignment]
    create_client = None  # type: ignore[assignment]
    _SUPABASE_IMPORT_ERROR: Exception | None = exc
else:  # pragma: no cover
    _SUPABASE_IMPORT_ERROR = None


# Back-compat alias to the prices module (kept; we override below anyway)
try:
    from .prices import load_prices_cached as _dl_load_prices_cached  # type: ignore[attr-defined]
except Exception:
    _dl_load_prices_cached = None


def load_prices_cached(*args, **kwargs):
    if _dl_load_prices_cached is None:
        from .prices import load_prices_cached as _impl
        return _impl(*args, **kwargs)
    return _dl_load_prices_cached(*args, **kwargs)


LOCAL_ROOT = Path(".lake")
DEFAULT_BUCKET = "lake"


class ConfigurationError(RuntimeError):
    pass


PRICE_COLUMNS = [
    "date",
    "Ticker",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
    "Dividends",
    "Stock Splits",
]


def _coerce_path(value: Any) -> str:
    if isinstance(value, os.PathLike):  # type: ignore[arg-type]
        return os.fspath(value)  # type: ignore[arg-type]
    return str(value or "")


def _split_clean_parts(raw: str) -> list[str]:
    if not raw:
        return []
    normalized = raw.replace("\\", "/")
    parts: list[str] = []
    for part in normalized.split("/"):
        piece = part.strip()
        if not piece or piece in {".", ".."}:
            continue
        parts.append(piece)
    return parts


def _parts_match(parts: Sequence[str], prefix: Sequence[str]) -> bool:
    if len(parts) < len(prefix):
        return False
    if not prefix:
        return True
    return [p.lower() for p in parts[: len(prefix)]] == [p.lower() for p in prefix]


def _normalize_storage_key(
    raw: Any,
    *,
    bucket: str | None = None,
) -> str:
    """Return a clean storage key with bucket prefixes stripped."""

    text = _coerce_path(raw).strip()
    if not text:
        return ""

    parts = _split_clean_parts(text)
    bucket_parts = _split_clean_parts(str(bucket or ""))
    if bucket_parts and _parts_match(parts, bucket_parts):
        parts = parts[len(bucket_parts) :]

    return "/".join(parts)


def supabase_available() -> tuple[bool, str]:
    if create_client is None:
        reason = (
            "supabase python client not installed"
            if _SUPABASE_IMPORT_ERROR is None
            else str(_SUPABASE_IMPORT_ERROR)
        )
        return False, reason
    return True, ""


def validate_prices_schema(df: pd.DataFrame, *, strict: bool = True) -> None:
    if df is None or df.empty:
        return
    req = set(PRICE_COLUMNS)
    missing = sorted(req - set(df.columns))
    if missing:
        raise ValueError(f"Missing required price columns: {', '.join(missing)}")

    dates = pd.to_datetime(df["date"], errors="coerce")
    if dates.isna().any() and df["date"].notna().any():
        raise ValueError("Non-datetime values in 'date' column")

    tick = df["Ticker"].dropna().astype(str).str.strip()
    if not tick.empty and not (tick == tick.str.upper()).all():
        raise ValueError("Ticker column must be uppercase")

    for col in [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
        "Dividends",
        "Stock Splits",
    ]:
        if col in df.columns:
            series = df[col].dropna()
            if not series.empty and not is_numeric_dtype(series):
                pd.to_numeric(series, errors="raise")

    actions_present = False
    for col in ("Dividends", "Stock Splits"):
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce").fillna(0)
            if values.ne(0).any():
                actions_present = True
                break

    if actions_present:
        close = pd.to_numeric(df["Close"], errors="coerce")
        adj_close = pd.to_numeric(df["Adj Close"], errors="coerce")
        mask = close.notna() & adj_close.notna()
        if mask.any() and (close[mask] == adj_close[mask]).all():
            msg = "Adjusted OHLC detected; store raw OHLC and keep 'Adj Close' separately."
            if strict:
                raise ValueError(msg)
            log.warning(msg)


def _is_truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _coerce_secrets_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    try:
        return dict(obj)
    except Exception:
        return {}


def _classify_key(key: str | None) -> str:
    if not key:
        return "missing"
    if key.startswith("sb_"):
        return "publishable"
    parts = key.split(".")
    if len(parts) != 3:
        return "not_jwt"
    payload_b64 = parts[1]
    padding = "=" * (-len(payload_b64) % 4)
    try:
        decoded = base64.urlsafe_b64decode((payload_b64 + padding).encode("utf-8"))
        payload = json.loads(decoded.decode("utf-8"))
    except Exception:
        return "invalid_jwt"
    role = payload.get("role") if isinstance(payload, dict) else None
    if isinstance(role, str) and role:
        return role
    return "invalid_jwt"


@dataclass(slots=True)
class _SupabaseConfig:
    url: str | None
    key: str | None
    bucket: str
    force: bool


def _load_supabase_config(
    url: str | None = None,
    key: str | None = None,
    bucket: str | None = None,
) -> _SupabaseConfig:
    secrets_cfg: dict[str, Any] = {}
    try:
        secrets_cfg = _coerce_secrets_dict(getattr(st, "secrets", {})).get("supabase", {})
    except Exception:
        secrets_cfg = {}
    env_bucket = os.getenv("SUPABASE_BUCKET")
    env_url = os.getenv("SUPABASE_URL")
    env_key = os.getenv("SUPABASE_KEY")

    cfg_url = url or env_url or secrets_cfg.get("url")
    cfg_key = key or env_key or secrets_cfg.get("key")
    cfg_bucket = bucket or env_bucket or secrets_cfg.get("bucket") or DEFAULT_BUCKET

    force_env = os.getenv("FORCE_SUPABASE")
    force_cfg = secrets_cfg.get("force")
    force = _is_truthy(force_env) or _is_truthy(force_cfg)

    return _SupabaseConfig(cfg_url, cfg_key, cfg_bucket, force)


class Storage:
    """Abstracts access to the Parquet data lake (local or Supabase)."""

    def __init__(
        self,
        mode: str | None = None,
        bucket: str | None = None,
        supabase_url: str | None = None,
        supabase_key: str | None = None,
        supabase_client: Client | None = None,
    ) -> None:
        cfg = _load_supabase_config(supabase_url, supabase_key, bucket)

        self.mode: str = "local"
        self.bucket: str | Any = bucket or cfg.bucket
        self.local_root: Path = Path(LOCAL_ROOT)
        self.supabase_url: str | None = cfg.url
        self.supabase_key: str | None = cfg.key
        self.supabase_client: Client | None = supabase_client
        self.force_supabase: bool = cfg.force
        self.key_info: dict[str, Any] = {"kind": _classify_key(cfg.key), "key": cfg.key}

        requested_mode = mode or ("supabase" if (cfg.url and cfg.key) else None)
        want_supabase = requested_mode == "supabase" or self.force_supabase

        if want_supabase:
            ok, reason = supabase_available()
            if not ok:
                raise RuntimeError(f"Supabase client unavailable: {reason}")
            if not (cfg.url and cfg.key):
                raise RuntimeError("Supabase configuration requires url+key")
            if self.supabase_client is None:
                assert create_client is not None
                self.supabase_client = create_client(cfg.url, cfg.key)  # type: ignore[misc]
            self.mode = "supabase"
            self.bucket = cfg.bucket
        else:
            self.mode = "local"
            self.bucket = cfg.bucket

    # ---------------- Basic helpers ----------------
    def supabase_available(self) -> tuple[bool, str]:
        return supabase_available()

    def diagnostics(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "bucket": self.bucket,
            "local_root": str(self._resolved_root()) if self.mode == "local" else None,
            "supabase_url": self.supabase_url,
            "force_supabase": self.force_supabase,
        }

    def info(self) -> str:
        return f"Storage(mode={self.mode}, bucket={self.bucket})"

    def cache_salt(self) -> str:
        return f"mode={self.mode}|url={self.supabase_url or ''}|bucket={self.bucket}"

    def selftest(self) -> dict[str, Any]:
        return {"ok": True, "mode": self.mode}

    # ---------------- I/O primitives ----------------
    def _resolved_root(self) -> Path:
        root = Path(self.local_root if self.local_root is not None else LOCAL_ROOT)
        if not root.is_absolute():
            root = (Path.cwd() / root).resolve()
        return root

    def _bucket_name(self) -> str:
        return str(self.bucket).strip("/") if isinstance(self.bucket, str) else ""

    def _normalize_key(self, path: Any) -> str:
        return _normalize_storage_key(path, bucket=self._bucket_name())

    def _norm(self, path: str) -> Path:
        rel = self._normalize_key(path)
        base = self._resolved_root()
        return base / rel if rel else base

    def _get_bucket_api(self):  # type: ignore[no-untyped-def]
        if self.supabase_client is None:
            raise RuntimeError("Supabase client not configured")
        return self.supabase_client.storage.from_(self.bucket)

    def read_bytes(self, path: str) -> bytes:
        norm = self._normalize_key(path)
        if self.mode == "local":
            full = self._norm(norm)
            return full.read_bytes()

        api = self._get_bucket_api()
        try:
            result = api.download(norm)
        except Exception as exc:
            log.error(
                "Supabase download failed (bucket=%s, key=%s, raw=%s)",
                self._bucket_name() or self.bucket,
                norm,
                path,
                exc_info=True,
            )
            raise
        data = getattr(result, "data", result)
        if isinstance(data, bytes):
            return data
        if hasattr(data, "read"):
            return data.read()
        if isinstance(data, dict) and "data" in data:
            payload = data["data"]
            if isinstance(payload, (bytes, bytearray)):
                return bytes(payload)
        raise TypeError(f"Unsupported download response: {type(result)!r}")

    def read_parquet_df(self, path: str) -> pd.DataFrame:
        norm = self._normalize_key(path)
        if self.mode == "local":
            full = self._norm(norm)
            return pd.read_parquet(full)
        data = self.read_bytes(path)
        return pd.read_parquet(io.BytesIO(data))

    def write_bytes(self, path: str, payload: bytes, *, content_type: str | None = None) -> None:
        norm = self._normalize_key(path)
        if self.mode == "local":
            dest = self._norm(norm)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(payload)
            return

        api = self._get_bucket_api()
        options = {}
        if content_type:
            options = {"content_type": content_type}
        try:
            api.upload(norm, payload, file_options=options or None)
        except Exception as exc:
            log.error(
                "Supabase upload failed (bucket=%s, key=%s, raw=%s)",
                self._bucket_name() or self.bucket,
                norm,
                path,
                exc_info=True,
            )
            raise

    # ---------------- Listing helpers ----------------
    def list_prefix(self, prefix: str) -> List[str]:
        norm = self._normalize_key(prefix)
        if self.mode == "supabase":
            if self.supabase_client is not None:
                api = self._get_bucket_api()
            elif hasattr(self.bucket, "list"):
                api = self.bucket  # type: ignore[assignment]
            else:
                raise RuntimeError("Supabase client not configured")

            limit = 1000
            offset = 0
            out: list[str] = []
            seen: set[str] = set()
            supports_offset = True

            while True:
                try:
                    response = api.list(
                        path=norm,
                        limit=limit,
                        offset=offset,
                        sort_by={"column": "name", "order": "asc"},
                    )
                except TypeError:
                    # Older clients (or Streamlit’s proxy) don’t accept kwargs → no pagination.
                    supports_offset = False
                    try:
                        response = api.list(norm) if norm else api.list()
                    except TypeError:
                        response = api.list(path=norm) if norm else api.list()
                except Exception as exc:
                    log.error(
                        "Supabase list failed (bucket=%s, prefix=%s, raw=%s)",
                        self._bucket_name() or self.bucket,
                        norm,
                        prefix,
                        exc_info=True,
                    )
                    raise

                items = getattr(response, "data", response)
                if isinstance(items, dict):
                    items = items.get("data")
                raw_items = list(items or [])

                new_items = 0
                for item in raw_items:
                    if isinstance(item, dict):
                        name = item.get("name")
                    else:
                        name = getattr(item, "name", None) or str(item)
                    if not name:
                        continue
                    name = str(name).lstrip("/")
                    full = f"{norm}/{name}" if norm else name
                    if full in seen:
                        continue
                    seen.add(full)
                    out.append(full)
                    new_items += 1

                if not supports_offset:
                    break
                if len(raw_items) < limit or new_items == 0:
                    break
                offset += limit

            return sorted(out)

        # Local FS
        base = self._norm(norm)
        if not base.exists() or not base.is_dir():
            return []
        entries: list[str] = []
        for child in sorted(base.iterdir()):
            rel = f"{norm}/{child.name}" if norm else child.name
            if child.is_dir():
                entries.append(f"{rel}/")
            elif child.is_file():
                entries.append(rel)
        return entries

    def list_all(self, prefix: str) -> List[str]:
        return self.list_prefix(prefix)

    def exists(self, path: str) -> bool:
        if not path or path.endswith("/"):
            return False
        norm = self._normalize_key(path)
        if not norm:
            return False
        if "/" in norm:
            parent, name = norm.rsplit("/", 1)
        else:
            parent, name = "", norm
        items = self.list_prefix(parent)
        target = f"{parent}/{name}" if parent else name
        return target in set(items)

    # ------------- Optional HTTP HEAD presence -------------
    def _public_object_url(self, path: str) -> str:
        base = (self.supabase_url or "").rstrip("/")
        bkt = str(self.bucket).strip("/")
        p = self._normalize_key(path)
        return f"{base}/storage/v1/object/public/{bkt}/{p}"

    def http_head_exists(self, path: str) -> bool:
        norm = self._normalize_key(path)
        if self.mode == "local":
            return (self._norm(norm)).exists()
        # Public URL HEAD
        try:
            url = self._public_object_url(norm)
            r = requests.head(url, allow_redirects=True, timeout=8)
            if r.status_code == 200:
                return True
            if r.status_code in (301, 302, 307, 308):
                r2 = requests.get(url, headers={"Range": "bytes=0-0"}, timeout=12)
                if r2.status_code in (200, 206):
                    return True
            if r.status_code not in (401, 403):
                return False
        except Exception:
            pass
        # Signed URL fallback (private buckets)
        try:
            if create_client is None or not (self.supabase_url and self.supabase_key):
                return False
            api = self._get_bucket_api()
            signed = api.create_signed_url(norm, 60)
            signed_url = None
            if isinstance(signed, dict):
                signed_url = (
                    signed.get("signedURL")
                    or signed.get("signed_url")
                    or (signed.get("data") or {}).get("signedURL")
                )
            if not signed_url:
                return False
            r3 = requests.head(signed_url, allow_redirects=True, timeout=8)
            if r3.status_code == 200:
                return True
            if r3.status_code in (301, 302, 307, 308):
                r4 = requests.get(signed_url, headers={"Range": "bytes=0-0"}, timeout=12)
                return r4.status_code in (200, 206)
            return False
        except Exception:
            return False

    # ---------------- Factories ----------------
    @classmethod
    def from_env(cls) -> "Storage":
        return cls()


# ====== Price helpers =======================================================

def _normalize_key(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def _normalize_ticker_symbol(raw: str) -> str:
    return str(raw).strip().upper()


_TICKER_CANON_TRANSLATION = str.maketrans({".": "_", "-": "_", " ": "_"})

def _canonicalize_ticker_symbol(raw: str) -> str:
    return _normalize_ticker_symbol(raw).translate(_TICKER_CANON_TRANSLATION)


def _candidate_price_stems(ticker: str) -> list[str]:
    normalized = _normalize_ticker_symbol(ticker)
    canonical = _canonicalize_ticker_symbol(normalized)
    variants: set[str] = {normalized, canonical}
    swaps = {
        normalized.replace(".", "_"),
        normalized.replace(".", "-"),
        normalized.replace("-", "_"),
        normalized.replace("-", "."),
        normalized.replace("_", "."),
        normalized.replace("_", "-"),
        canonical.replace("_", "."),
        canonical.replace("_", "-"),
    }
    variants.update(filter(None, swaps))
    return sorted({v.strip().upper() for v in variants if v})


def _tidy_prices(df: pd.DataFrame, ticker: str | None = None) -> pd.DataFrame:
    if df is None or df.empty:
        empty = pd.DataFrame(columns=[c for c in PRICE_COLUMNS if c != "date"])
        empty.index = pd.DatetimeIndex([], name="date")
        log.debug("_tidy_prices: empty frame for %s", ticker)
        return empty

    working = df.copy()
    if not isinstance(working.index, pd.DatetimeIndex):
        date_col = next(
            (c for c in working.columns if _normalize_key(c) in {"date", "timestamp"}),
            None,
        )
        if date_col:
            working["date"] = working[date_col]
        else:
            working = working.reset_index().rename(
                columns={working.index.name or "index": "date"}
            )
    else:
        working = working.reset_index().rename(columns={working.index.name or "index": "date"})

    rename_map: dict[str, str] = {}
    for col in list(working.columns):
        key = _normalize_key(col)
        if key in {"date", "timestamp"}:
            rename_map[col] = "date"
        elif key == "open":
            rename_map[col] = "Open"
        elif key == "high":
            rename_map[col] = "High"
        elif key == "low":
            rename_map[col] = "Low"
        elif key in {"close", "close_price"}:
            rename_map.setdefault(col, "Close")
        elif key in {"adj_close", "adjusted_close", "adjclose", "close_adj"}:
            rename_map[col] = "Adj Close"
        elif key == "volume":
            rename_map[col] = "Volume"
        elif key in {"ticker", "symbol"}:
            rename_map[col] = "Ticker"
        elif key in {"dividends", "dividend"}:
            rename_map[col] = "Dividends"
        elif key in {"stock_splits", "stocksplits", "split"}:
            rename_map[col] = "Stock Splits"

    working = working.rename(columns=rename_map)
    working["date"] = pd.to_datetime(working["date"], errors="coerce").dt.tz_localize(None)
    working = working.dropna(subset=["date"])

    if "Ticker" not in working.columns:
        working["Ticker"] = ticker or pd.NA
    working["Ticker"] = working["Ticker"].astype("string").str.upper().str.strip()

    keep = PRICE_COLUMNS.copy()
    for col in keep:
        if col not in working.columns:
            if col == "Adj Close":
                working[col] = pd.NA
            elif col in {"Dividends", "Stock Splits"}:
                working[col] = 0.0
            elif col == "date":
                continue
            else:
                working[col] = pd.NA

    for col in ["Open", "High", "Low", "Close", "Adj Close", "Dividends", "Stock Splits"]:
        if col in working.columns:
            working[col] = pd.to_numeric(working[col], errors="coerce")

    if "Volume" in working.columns:
        vol = pd.to_numeric(working["Volume"], errors="coerce")
        try:
            working["Volume"] = vol.astype("Int64")
        except Exception:
            working["Volume"] = vol

    ordered = ["date"] + [c for c in keep if c != "date"]
    working = working[ordered]
    working = working.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    working = working.set_index("date").rename_axis("date")
    return working[[c for c in keep if c != "date"]]


@st.cache_data(hash_funcs={Storage: lambda _: 0}, show_spinner=False)
def load_prices_cached(
    _storage: Storage,
    *args,
    cache_salt: str | list[str] = "",
    tickers: list[str] | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    legacy_args = list(args)
    if legacy_args and tickers is None:
        tickers = legacy_args.pop(0)
    if legacy_args and start is None:
        start = legacy_args.pop(0)
    if legacy_args and end is None:
        end = legacy_args.pop(0)

    if tickers is None and isinstance(cache_salt, (list, tuple)):
        tickers = list(cache_salt)
        cache_salt = ""

    raw_tickers: list[Any]
    if tickers is None:
        raw_tickers = []
    elif isinstance(tickers, (str, bytes)):
        raw_tickers = [tickers]
    else:
        try:
            raw_tickers = list(tickers)
        except TypeError:
            raw_tickers = [tickers]

    tickers = [str(t).upper() for t in raw_tickers if t]
    if not tickers:
        return pd.DataFrame(columns=PRICE_COLUMNS)

    def _naive(ts: pd.Timestamp | None) -> pd.Timestamp | None:
        if ts is None:
            return None
        stamp = pd.Timestamp(ts)
        try:
            stamp = stamp.tz_localize(None)
        except TypeError:
            if stamp.tzinfo is not None:
                stamp = stamp.tz_convert(None)
        except Exception:
            pass
        return stamp

    start_ts = _naive(start)
    end_ts = _naive(end)

    prefix = _resolve_prices_prefix(_storage)
    layout = _resolve_layout(_storage, prefix)
    display_prefix = prefix or "<root>"

    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        layouts_to_try = [layout]
        if layout == "partitioned":
            layouts_to_try = ["partitioned", "flat"]

        df_raw: pd.DataFrame | None = None
        errors: list[tuple[str, str, Exception | None]] = []

        for layout_attempt in layouts_to_try:
            key = _build_storage_key(prefix, ticker, layout_attempt)
            normalized_attempt = _storage._normalize_key(key)
            try:
                if layout_attempt == "flat":
                    candidate = _storage.read_parquet_df(key)
                else:
                    candidate = _load_partitioned_price(_storage, key)
            except FileNotFoundError:
                errors.append((layout_attempt, normalized_attempt, None))
                continue
            except Exception as exc:
                log.error(
                    "load_prices_cached: failed to load (bucket=%s, prefix=%s, key=%s, layout_attempt=%s, detected_layout=%s): %s",
                    _storage._bucket_name() or _storage.bucket,
                    display_prefix,
                    normalized_attempt,
                    layout_attempt,
                    layout,
                    exc,
                    exc_info=True,
                )
                if layout_attempt == "flat":
                    try:
                        candidate = pd.read_parquet(io.BytesIO(_storage.read_bytes(key)))
                    except FileNotFoundError:
                        errors.append((layout_attempt, normalized_attempt, None))
                        continue
                    except Exception as exc2:
                        errors.append((layout_attempt, normalized_attempt, exc2))
                        continue
                else:
                    errors.append((layout_attempt, normalized_attempt, exc))
                    continue
            df_raw = candidate
            break

        if df_raw is None:
            if errors:
                last_layout, last_key, _ = errors[-1]
            else:
                last_layout = layouts_to_try[-1]
                last_key = _storage._normalize_key(
                    _build_storage_key(prefix, ticker, last_layout)
                )
            log.warning(
                "load_prices_cached: missing price data (bucket=%s, prefix=%s, key=%s, layout_attempt=%s, detected_layout=%s)",
                _storage._bucket_name() or _storage.bucket,
                display_prefix,
                last_key,
                last_layout,
                layout,
            )
            continue

        tidy = _tidy_prices(df_raw, ticker=ticker)
        tidy.index = pd.to_datetime(tidy.index, errors="coerce").tz_localize(None)
        if tidy.index.isna().any():
            bad = int(tidy.index.isna().sum())
            log.debug("load_prices_cached: dropping %s rows with bad dates for %s", bad, ticker)
            tidy = tidy[tidy.index.notna()]
        if start_ts is not None:
            tidy = tidy[tidy.index >= start_ts]
        if end_ts is not None:
            tidy = tidy[tidy.index <= end_ts]
        frames.append(tidy)

    if not frames:
        return pd.DataFrame(columns=PRICE_COLUMNS)

    out = pd.concat(frames, axis=0, ignore_index=False)
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out.index.name = "date"
    out = out.sort_index()
    out = out.reset_index()
    out = out.dropna(subset=["date"])
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    out = out.sort_values(["Ticker", "date"])
    out = out.drop_duplicates(subset=["Ticker", "date"], keep="last")

    for col in PRICE_COLUMNS:
        if col not in out.columns:
            if col in {"Dividends", "Stock Splits"}:
                out[col] = 0.0
            else:
                out[col] = pd.NA

    out = out[PRICE_COLUMNS]

    out["Ticker"] = out["Ticker"].astype("string").str.upper().str.strip()
    for col in ["Open", "High", "Low", "Close", "Adj Close", "Dividends", "Stock Splits"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if "Volume" in out.columns:
        vol = pd.to_numeric(out["Volume"], errors="coerce")
        try:
            out["Volume"] = vol.astype("Int64")
        except Exception:
            out["Volume"] = vol

    validate_prices_schema(out)
    return out.reset_index(drop=True)


# -------- Manifest support (FAST PATH) --------------------------------

@st.cache_data(hash_funcs={Storage: lambda _: 0}, show_spinner=False)
def _load_price_manifest(storage: Storage) -> set[str] | None:
    """
    Try to load a cached list of available price files.
    Supported:
      - prices/_manifest.parquet  (with column 'Ticker')
      - prices/_manifest.csv      (column 'Ticker' or first column)
      - prices/_manifest.txt      (one ticker per line)
    Returns uppercased ticker set, or None if not found.
    """
    candidates = [
        "prices/_manifest.parquet",
        "prices/_manifest.csv",
        "prices/_manifest.txt",
    ]
    for path in candidates:
        try:
            if path.endswith(".parquet"):
                df = storage.read_parquet_df(path)
                col = "Ticker" if "Ticker" in df.columns else df.columns[0]
                vals = df[col].dropna().astype(str).str.upper().str.strip()
                tickers = set(v for v in vals.tolist() if v)
                if tickers:
                    log.info("price-filter: using manifest %s (%d tickers)", path, len(tickers))
                    return tickers
            elif path.endswith(".csv"):
                raw = storage.read_bytes(path).decode("utf-8", errors="ignore")
                df = pd.read_csv(io.StringIO(raw))
                col = "Ticker" if "Ticker" in df.columns else df.columns[0]
                vals = df[col].dropna().astype(str).str.upper().str.strip()
                tickers = set(v for v in vals.tolist() if v)
                if tickers:
                    log.info("price-filter: using manifest %s (%d tickers)", path, len(tickers))
                    return tickers
            else:
                raw = storage.read_bytes(path).decode("utf-8", errors="ignore")
                vals = [line.strip().upper() for line in raw.splitlines()]
                tickers = set(v for v in vals if v and not v.startswith("#"))
                if tickers:
                    log.info("price-filter: using manifest %s (%d tickers)", path, len(tickers))
                    return tickers
        except FileNotFoundError:
            continue
        except Exception as exc:
            log.warning("price-filter: manifest %s load failed: %s", path, exc)
            continue
    return None


def _resolve_prices_prefix(storage: Storage) -> str:
    raw_prefix = os.getenv("LAKE_PRICES_PREFIX", "lake/prices")
    bucket = storage._bucket_name() if hasattr(storage, "_bucket_name") else ""
    prefix = _normalize_storage_key(raw_prefix, bucket=bucket)
    return prefix.strip("/")


def _resolve_layout(storage: Storage, prefix: str) -> str:
    val = (os.getenv("LAKE_LAYOUT", "auto") or "auto").strip().lower()
    if val in {"flat", "partitioned"}:
        return val
    if val not in {"auto"}:
        raise ConfigurationError(
            "Invalid LAKE_LAYOUT value. Expected 'flat', 'partitioned', or 'auto'."
        )
    try:
        entries = storage.list_prefix(prefix)
    except Exception:
        return "flat"
    for e in entries[:200]:
        name = Path(str(e)).name
        if os.path.splitext(name)[1].lower() == ".parquet":
            return "flat"
    return "partitioned"


def _build_storage_key(prefix: str, stem: str, layout: str) -> str:
    base = f"{prefix}/{stem}" if prefix else stem
    if layout == "flat":
        return f"{base}.parquet"
    return base


def _display_key(path: str, layout: str) -> str:
    return f"{path}/" if layout == "partitioned" else path


def _gather_partition_objects(storage: Storage, base_key: str) -> list[str]:
    queue = [base_key.rstrip("/")]
    seen: set[str] = set()
    parquet_paths: set[str] = set()

    while queue:
        current = queue.pop()
        if not current or current in seen:
            continue
        seen.add(current)
        try:
            entries = storage.list_prefix(current)
        except Exception as exc:  # pragma: no cover - handled by caller
            raise
        for entry in entries:
            normalized = entry.rstrip("/")
            if not normalized:
                continue
            if normalized.lower().endswith(".parquet"):
                parquet_paths.add(normalized)
            elif normalized not in seen and normalized != current:
                queue.append(normalized)

    return sorted(parquet_paths)


def _load_partitioned_price(storage: Storage, base_key: str) -> pd.DataFrame:
    parts = _gather_partition_objects(storage, base_key)
    if not parts:
        raise FileNotFoundError(base_key)
    frames: list[pd.DataFrame] = []
    for path in parts:
        df = storage.read_parquet_df(path)
        if not isinstance(df, pd.DataFrame):  # pragma: no cover - safety
            continue
        frames.append(df)
    if not frames:
        raise FileNotFoundError(base_key)
    return pd.concat(frames, axis=0, ignore_index=False)


def _classify_probe_exception(exc: Exception) -> str:
    text = str(exc).lower()
    if any(t in text for t in ["401", "403", "forbidden", "permission", "denied", "rls"]):
        return "401"
    return "404"


def _probe_key(storage: Storage, key: str, layout: str) -> str:
    try:
        if layout == "flat":
            return "200" if storage.http_head_exists(key) else "404"
        entries = storage.list_prefix(key)
        return "200" if entries else "404"
    except Exception as exc:  # pragma: no cover
        return _classify_probe_exception(exc)


def filter_tickers_with_parquet(
    storage: Storage, tickers: Iterable[str]
) -> tuple[list[str], list[str]]:
    """Split ``tickers`` into those with and without price parquet data."""
    requested: list[str] = []
    canonical_by_ticker: dict[str, str] = {}
    seen_canonical: set[str] = set()
    for raw in tickers:
        if not raw:
            continue
        ticker = _normalize_ticker_symbol(raw)
        if not ticker:
            continue
        canonical = _canonicalize_ticker_symbol(ticker)
        if canonical in seen_canonical:
            continue
        requested.append(ticker)
        canonical_by_ticker[ticker] = canonical
        seen_canonical.add(canonical)

    if not requested:
        return [], []

    # ---- 0) Manifest fast path -----------------------------------------
    manifest = _load_price_manifest(storage)
    if manifest is not None:
        present = [t for t in requested if _canonicalize_ticker_symbol(t) in manifest]
        missing = [t for t in requested if _canonicalize_ticker_symbol(t) not in manifest]
        log.info("price-filter: manifest resolved %d/%d tickers", len(present), len(requested))
        return present, missing

    # ---- 1) Listing-based path -----------------------------------------
    prefix = _resolve_prices_prefix(storage)
    layout = _resolve_layout(storage, prefix)
    display_prefix = prefix or "<root>"
    bucket_name = storage._bucket_name() if hasattr(storage, "_bucket_name") else storage.bucket

    available_canonical: set[str] = set()

    if layout == "flat":
        # 1a) Try directory list
        try:
            entries = storage.list_prefix(prefix)
        except Exception as exc:
            raise ConfigurationError(
                f"Unable to list price objects under prefix '{display_prefix}': {exc}"
            ) from exc

        parquet_stems: set[str] = set()
        for entry in entries:
            name = Path(str(entry)).name
            stem, ext = os.path.splitext(name)
            if not stem:
                continue
            if ext and ext.lower() != ".parquet":
                continue
            parquet_stems.add(stem.upper())

        for stem in parquet_stems:
            canonical = _canonicalize_ticker_symbol(stem)
            if canonical:
                available_canonical.add(canonical)

        # 1b) If the list looks short, do a LIMITED HEAD probe sweep
        MAX_HEAD_PROBES = int(os.getenv("MAX_HEAD_PROBES", "60"))
        suspicious = len(requested) >= 100 and len(available_canonical) < max(1, len(requested) // 2)
        if suspicious and MAX_HEAD_PROBES > 0:
            log.warning(
                "price-filter: suspicious listing (bucket=%s prefix=%s layout=%s %s present/%s requested) — "
                "probing up to %s tickers via HEAD",
                bucket_name,
                display_prefix,
                layout,
                len(available_canonical),
                len(requested),
                MAX_HEAD_PROBES,
            )
            probes = 0
            for t in requested:
                if probes >= MAX_HEAD_PROBES:
                    break
                canon = canonical_by_ticker[t]
                if canon in available_canonical:
                    continue
                for stem in _candidate_price_stems(t):
                    key = _build_storage_key(prefix, stem, layout="flat")
                    try:
                        if storage.http_head_exists(key):
                            available_canonical.add(_canonicalize_ticker_symbol(stem))
                            break
                    except Exception:
                        pass
                probes += 1

    else:
        # Partitioned layout → verify existence by listing each stem path (cheap)
        probed_paths: set[str] = set()
        for ticker in requested:
            for stem in _candidate_price_stems(ticker):
                key = _build_storage_key(prefix, stem, layout)
                if key in probed_paths:
                    continue
                probed_paths.add(key)
                try:
                    entries = storage.list_prefix(key)
                except Exception as exc:
                    raise ConfigurationError(
                        f"Unable to list partitioned price objects under '{key}': {exc}"
                    ) from exc
                if entries:
                    canonical = _canonicalize_ticker_symbol(stem)
                    if canonical:
                        available_canonical.add(canonical)
                    break

    present = [t for t in requested if canonical_by_ticker.get(t) in available_canonical]
    missing = [t for t in requested if canonical_by_ticker.get(t) not in available_canonical]

    # Breadcrumbs when coverage is poor (cheap probes for 3 keys)
    if len(requested) >= 100 and len(present) < max(1, len(requested) // 2):
        probes = []
        for t in missing[:3]:
            stem = _candidate_price_stems(t)[0]
            key = _build_storage_key(prefix, stem, layout)
            status = _probe_key(storage, key, layout)
            probes.append((_display_key(key, layout), status))
        for k, s in probes:
            log.warning(
                "price-filter: probe %s -> %s (bucket=%s prefix=%s layout=%s)",
                k,
                s,
                bucket_name,
                display_prefix,
                layout,
            )

    log.debug(
        "filter_tickers_with_parquet: bucket=%s prefix=%s layout=%s requested=%s present=%s missing=%s",
        bucket_name,
        display_prefix,
        layout,
        len(requested),
        len(present),
        len(missing),
    )

    return present, missing


__all__ = [
    "Storage",
    "filter_tickers_with_parquet",
    "ConfigurationError",
    "load_prices_cached",
]
