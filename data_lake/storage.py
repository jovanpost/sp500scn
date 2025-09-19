from __future__ import annotations

import io
import os
import base64
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List

import pandas as pd
import streamlit as st
from pandas.api.types import is_numeric_dtype

log = logging.getLogger(__name__)

try:
    from supabase import Client, create_client  # type: ignore
except Exception as exc:  # pragma: no cover - supabase optional
    Client = Any  # type: ignore[assignment]
    create_client = None  # type: ignore[assignment]
    _SUPABASE_IMPORT_ERROR: Exception | None = exc
else:  # pragma: no cover - exercised when package installed
    _SUPABASE_IMPORT_ERROR = None


LOCAL_ROOT = Path(".lake")
DEFAULT_BUCKET = "lake"

# Stable price schema used across ingest + read paths.
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


def supabase_available() -> tuple[bool, str]:
    """Return ``(available, reason)`` for Supabase client support."""

    if create_client is None:
        reason = (
            "supabase python client not installed"
            if _SUPABASE_IMPORT_ERROR is None
            else str(_SUPABASE_IMPORT_ERROR)
        )
        return False, reason
    return True, ""


def validate_prices_schema(df: pd.DataFrame, *, strict: bool = True) -> None:
    """Ensure frames store raw OHLC alongside provider Adj Close."""

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
    try:  # pragma: no cover - defensive; depends on secrets object implementation
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
    """Read Supabase configuration from environment + Streamlit secrets."""

    secrets_cfg: dict[str, Any] = {}
    try:  # pragma: no branch - streamlit secrets may be unavailable
        secrets_cfg = _coerce_secrets_dict(getattr(st, "secrets", {})).get("supabase", {})
    except Exception:  # pragma: no cover - secrets access can raise
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
                assert create_client is not None  # for type checkers
                self.supabase_client = create_client(cfg.url, cfg.key)  # type: ignore[misc]
            self.mode = "supabase"
            self.bucket = cfg.bucket
        else:
            self.mode = "local"
            self.bucket = cfg.bucket

    # ------------------------------------------------------------------
    # Basic helpers
    def supabase_available(self) -> tuple[bool, str]:  # pragma: no cover - thin wrapper
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

    # ------------------------------------------------------------------
    # I/O primitives
    def _resolved_root(self) -> Path:
        root = Path(self.local_root if self.local_root is not None else LOCAL_ROOT)
        if not root.is_absolute():
            root = (Path.cwd() / root).resolve()
        return root

    def _norm(self, path: str) -> Path:
        rel = path.strip("/")
        return self._resolved_root() / rel

    def _get_bucket_api(self):  # type: ignore[no-untyped-def]
        if self.supabase_client is None:
            raise RuntimeError("Supabase client not configured")
        return self.supabase_client.storage.from_(self.bucket)

    def read_bytes(self, path: str) -> bytes:
        norm = path.strip("/")
        if self.mode == "local":
            full = self._norm(norm)
            return full.read_bytes()

        api = self._get_bucket_api()
        result = api.download(norm)
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
        norm = path.strip("/")
        if self.mode == "local":
            full = self._norm(norm)
            return pd.read_parquet(full)
        data = self.read_bytes(norm)
        return pd.read_parquet(io.BytesIO(data))

    def write_bytes(self, path: str, payload: bytes, *, content_type: str | None = None) -> None:
        norm = path.strip("/")
        if self.mode == "local":
            dest = self._norm(norm)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(payload)
            return

        api = self._get_bucket_api()
        options = {}
        if content_type:
            options = {"content_type": content_type}
        api.upload(norm, payload, file_options=options or None)

    # ------------------------------------------------------------------
    # Listing helpers
    def list_prefix(self, prefix: str) -> List[str]:
        norm = prefix.strip("/")
        if self.mode == "supabase":
            if self.supabase_client is not None:
                api = self._get_bucket_api()
            elif hasattr(self.bucket, "list"):
                api = self.bucket  # type: ignore[assignment]
            else:  # pragma: no cover - defensive
                raise RuntimeError("Supabase client not configured")

            limit = 100
            offset = 0
            out: list[str] = []
            seen: set[str] = set()

            while True:
                used_pagination = True
                try:
                    if norm != "":
                        response = api.list(norm, limit=limit, offset=offset)
                    else:
                        response = api.list(limit=limit, offset=offset)
                except TypeError:
                    used_pagination = False
                    if offset > 0:
                        break
                    if norm:
                        response = api.list(prefix=norm)
                    else:
                        response = api.list()

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
                    if norm and name.startswith(f"{norm}/"):
                        full = name
                    else:
                        full = f"{norm}/{name}" if norm else name
                    if full in seen:
                        continue
                    seen.add(full)
                    out.append(full)
                    new_items += 1

                if not used_pagination:
                    break
                if len(raw_items) < limit:
                    break
                if new_items == 0:
                    break
                offset += limit

            return sorted(out)

        base = self._norm(norm)
        if not base.exists() or not base.is_dir():
            return []
        return [
            f"{norm}/{child.name}" if norm else child.name
            for child in sorted(base.iterdir())
            if child.is_file()
        ]

    def list_all(self, prefix: str) -> List[str]:
        return self.list_prefix(prefix)

    def exists(self, path: str) -> bool:
        if not path:
            return False
        if path.endswith("/"):
            return False
        norm = path.strip("/")
        if not norm:
            return False
        if "/" in norm:
            parent, name = norm.rsplit("/", 1)
        else:
            parent, name = "", norm
        items = self.list_prefix(parent)
        target = f"{parent}/{name}" if parent else name
        return target in set(items)

    # ------------------------------------------------------------------
    # Factories
    @classmethod
    def from_env(cls) -> "Storage":
        return cls()


# ====== Price helpers =======================================================


def _normalize_key(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def _normalize_ticker_symbol(raw: str) -> str:
    """Normalize a ticker symbol to uppercase without leading/trailing whitespace."""

    return str(raw).strip().upper()


_TICKER_CANON_TRANSLATION = str.maketrans({".": "_", "-": "_", " ": "_"})


def _canonicalize_ticker_symbol(raw: str) -> str:
    """Return a canonical representation resilient to punctuation differences."""

    return _normalize_ticker_symbol(raw).translate(_TICKER_CANON_TRANSLATION)


def _candidate_price_stems(ticker: str) -> list[str]:
    """Possible filename stems for ``ticker`` in the ``prices`` folder."""

    normalized = _normalize_ticker_symbol(ticker)
    canonical = _canonicalize_ticker_symbol(normalized)

    variants: set[str] = {normalized, canonical}

    # Generate variants that commonly appear in vendor exports ("BRK.B" vs "BRK_B").
    punctuation_swaps = {
        normalized.replace(".", "_"),
        normalized.replace(".", "-"),
        normalized.replace("-", "_"),
        normalized.replace("-", "."),
        normalized.replace("_", "."),
        normalized.replace("_", "-"),
        canonical.replace("_", "."),
        canonical.replace("_", "-"),
    }

    variants.update(filter(None, punctuation_swaps))

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
                # Preserve provider adjustment data without manufacturing values.
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

    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        path = f"prices/{ticker}.parquet"
        try:
            df_raw = _storage.read_parquet_df(path)
        except FileNotFoundError:
            continue
        except Exception:
            try:
                df_raw = pd.read_parquet(io.BytesIO(_storage.read_bytes(path)))
            except FileNotFoundError:
                continue
        tidy = _tidy_prices(df_raw, ticker=ticker)
        # Ensure index is datetime to avoid comparison failures
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

    # Enforce raw OHLC semantics before handing data to callers.
    validate_prices_schema(out)

    return out.reset_index(drop=True)


def filter_tickers_with_parquet(
    storage: Storage, tickers: Iterable[str]
) -> tuple[list[str], list[str]]:
    """Split ``tickers`` into those with and without ``prices/*.parquet`` data."""

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

    available_canonical: set[str] = set()
    entries: list[str]
    try:
        entries = storage.list_prefix("prices")
    except Exception as exc:  # pragma: no cover - defensive guardrail
        log.warning("filter_tickers_with_parquet: list_prefix failed: %s", exc)
        entries = []

    for entry in entries:
        name = Path(str(entry)).name
        stem, ext = os.path.splitext(name)
        if not stem:
            continue
        if ext and ext.lower() != ".parquet":
            continue
        canonical = _canonicalize_ticker_symbol(stem)
        if canonical:
            available_canonical.add(canonical)

    if not available_canonical:
        for ticker in requested:
            for stem in _candidate_price_stems(ticker):
                path = f"prices/{stem}.parquet"
                try:
                    if storage.exists(path):
                        available_canonical.add(_canonicalize_ticker_symbol(stem))
                        break
                except Exception:  # pragma: no cover - defensive
                    continue

    present = [
        t
        for t in requested
        if canonical_by_ticker.get(t) in available_canonical
    ]
    missing = [
        t
        for t in requested
        if canonical_by_ticker.get(t) not in available_canonical
    ]

    log.debug(
        "filter_tickers_with_parquet: requested=%s present=%s missing=%s",
        len(requested),
        len(present),
        len(missing),
    )

    return present, missing
