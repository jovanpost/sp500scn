import base64
import json
import logging
import os
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
import streamlit as st

try:  # pragma: no cover - optional dependency
    from supabase import Client, create_client
except Exception:  # pragma: no cover - tests may not have package
    Client = object  # type: ignore
    create_client = None  # type: ignore


# Default root for local storage operations
DEFAULT_LOCAL_ROOT = Path("data")
# Backwards compatibility for tests that patch ``LOCAL_ROOT``
LOCAL_ROOT = DEFAULT_LOCAL_ROOT


def _supabase_creds() -> Tuple[str, str] | None:
    """Return Supabase (url, key) if configured."""
    try:
        cfg = st.secrets.get("supabase", {})  # type: ignore[attr-defined]
    except Exception:
        cfg = {}
    url = cfg.get("url") or os.getenv("SUPABASE_URL")
    key = cfg.get("key") or os.getenv("SUPABASE_KEY")
    if url and key:
        return str(url), str(key)
    return None


def supabase_available() -> Tuple[bool, str]:
    """Best-effort check that Supabase client can be created."""
    creds = _supabase_creds()
    if not creds:
        return False, "missing credentials"
    if create_client is None:
        return False, "supabase package not installed"
    return True, ""


def _tidy_prices(df: pd.DataFrame, ticker: str | None = None) -> pd.DataFrame:
    """Normalize OHLCV schema to Yahoo-style columns and drop duplicates.

    Output columns (when available):
      date, Open, High, Low, Close, Adj Close, Volume, Ticker
    """
    df = df.copy()

    # If both variants exist, drop the lowercase one to avoid a clash after rename
    if "ticker" in df.columns and "Ticker" in df.columns:
        df = df.drop(columns=["ticker"])

    # Standardize column names
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

    # Ensure we have a 'date' column (pull from index if needed)
    if "date" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "date"})
        elif "time" in df.columns:
            df = df.rename(columns={"time": "date"})

    # Coerce to datetime and strip timezone to keep joins simple
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        # Remove timezone if present
        if getattr(df["date"].dtype, "tz", None) is not None:
            df["date"] = df["date"].dt.tz_localize(None)

    # Ensure required OHLCV columns exist
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            df[col] = pd.NA

    # Ensure Adj Close exists
    if "Adj Close" not in df.columns:
        df["Adj Close"] = df.get("Close")

    # Normalize Ticker
    if "Ticker" in df.columns:
        df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    elif ticker is not None:
        df["Ticker"] = str(ticker).upper()

    # Drop rows without a valid date
    if "date" in df.columns:
        df = df.dropna(subset=["date"])

    # Drop duplicate rows on (date, Ticker); keep the last (often adjusted values)
    if "date" in df.columns and "Ticker" in df.columns:
        df = df.sort_values("date").drop_duplicates(subset=["date", "Ticker"], keep="last")

    # Order columns (keep only those that exist)
    cols = ["date", "Open", "High", "Low", "Close", "Adj Close", "Volume", "Ticker"]
    df = df[[c for c in cols if c in df.columns]]
    if "date" in df.columns:
        df = df.set_index("date")

    return df


def _classify_key(key: str) -> str:
    """Rudimentary classification of Supabase keys.

    Returns one of ``service_role``, ``publishable``, ``not_jwt`` or ``invalid_jwt``.
    """

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


class Storage:
    """Simple wrapper around local file operations for tests.

    The real application supports multiple backends (e.g. Supabase storage).
    The simplified test version only uses the local filesystem but mirrors the
    interface expected by the Streamlit UI.  ``local_root`` defaults to the
    ``data`` directory but can be overridden (the tests use this to point at a
    temporary directory).
    """

    def __init__(self) -> None:
        self.log = logging.getLogger(__name__)
        self.mode = "local"
        self.bucket = None
        self.supabase_client: Client | None = None
        self.supabase_url: str | None = None
        self.bucket_name = "lake"
        try:
            cfg = st.secrets.get("supabase", {})  # type: ignore[attr-defined]
        except Exception:
            cfg = {}
        self.force_supabase = bool(cfg.get("force")) or os.getenv("FORCE_SUPABASE") == "1"
        self.auto_create_bucket = bool(cfg.get("auto_create_bucket", False))
        creds = _supabase_creds()
        if creds:
            self.supabase_url, key = creds
        else:
            key = None
        self.bucket_name = cfg.get("bucket", "lake")
        self.local_root = LOCAL_ROOT
        self._prices_cache: dict[str, pd.DataFrame] = {}
        self.key_info: dict[str, str] = {"kind": _classify_key(key) if key else "service_role"}

        avail, reason = supabase_available()
        if self.force_supabase and not avail:
            raise RuntimeError(f"Supabase required but unavailable: {reason}")

        if avail and create_client is not None:
            try:
                self.supabase_client = create_client(self.supabase_url, key)  # type: ignore[arg-type]
                self.bucket = self.supabase_client.storage.from_(self.bucket_name)
                try:
                    self.bucket.list("")
                except Exception:
                    if self.force_supabase and not self.auto_create_bucket:
                        raise RuntimeError(
                            f"Supabase bucket '{self.bucket_name}' missing."
                        )
                    if self.auto_create_bucket:
                        try:
                            self.supabase_client.storage.create_bucket(self.bucket_name)
                        except Exception:
                            pass
                        self.bucket = self.supabase_client.storage.from_(self.bucket_name)
                    else:
                        self.bucket = None
                if self.bucket is not None:
                    self.mode = "supabase"
            except Exception as e:
                if self.force_supabase:
                    raise RuntimeError(f"Supabase init failed: {e}")
                self.mode = "local"

        # Where local files are stored; tests may monkeypatch ``LOCAL_ROOT``
        self.local_root = LOCAL_ROOT

    # The tests monkeypatch this method, so the implementation can be minimal.
    def read_parquet(self, path: str) -> pd.DataFrame:
        return pd.read_parquet(self.local_root / path)

    # --- simple byte-level helpers used by the UI ---
    def read_bytes(self, path: str) -> bytes:
        """Return raw bytes from ``path``.

        The method tries Supabase first when in ``supabase`` mode."""

        if self.mode == "supabase" and self.bucket is not None:
            try:
                res = self.bucket.download(path)
                data = getattr(res, "data", res)
                if isinstance(data, bytes):
                    return data
                return bytes(data)  # type: ignore[arg-type]
            except Exception as e:  # pragma: no cover - defensive
                self.log.warning("download failed for '%s': %s", path, e)
        return (self.local_root / path).read_bytes()

    def write_bytes(self, path: str, data: bytes) -> None:
        """Write ``data`` to ``path`` under the local ``data`` directory."""

        if self.mode == "supabase" and self.bucket is not None:
            try:
                self.bucket.upload(
                    path,
                    data,
                    file_options={"upsert": True, "contentType": "application/octet-stream"},
                )
                return
            except Exception as e:  # pragma: no cover - defensive
                self.log.warning("upload failed for '%s': %s", path, e)

        dest = self.local_root / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)

    # ------------------------------------------------------------------
    # helpers
    def list_prefix(self, prefix: str) -> list[str]:
        """Return a list of file paths under ``prefix``.

        The return value is a list of paths relative to ``local_root`` (using
        forward slashes) so the caller sees uniform results across storage
        backends.  The method is defensive and tolerates missing directories or
        Supabase errors.
        """

        pfx = prefix.lstrip("/")
        if self.mode == "local":
            root = self.local_root.resolve()
            base = (
                (root / pfx).parent if pfx.endswith(".parquet") else (root / pfx)
            ).resolve()
            results: list[str] = []
            if base.is_file():
                results.append(str(base.relative_to(root)).replace("\\", "/"))
            elif base.exists():
                for path in base.rglob("*"):
                    if path.is_file():
                        rel = str(path.relative_to(root)).replace("\\", "/")
                        results.append(rel)
            return sorted(results)

        if self.bucket:
            path_arg = pfx[:-1] if pfx.endswith("/") and pfx != "/" else pfx
            try:
                res = self.bucket.list(path_arg)
            except Exception as e:  # pragma: no cover - defensive
                self.log.warning("list_prefix failed for '%s': %s", pfx, e)
                return []
            if isinstance(res, dict) and "data" in res:
                items = res["data"]
            elif hasattr(res, "data"):
                items = res.data
            else:
                items = res
            results: list[str] = []
            for it in items or []:
                name = getattr(it, "name", None)
                if name is None and isinstance(it, dict):
                    name = it.get("name")
                if name is None and isinstance(it, str):
                    name = it
                if not name:
                    continue
                base = path_arg.rstrip("/")
                full = f"{base}/{name}" if base else name
                results.append(full)
            return sorted(results)

        return []

    def exists(self, path: str) -> bool:
        """Check if ``path`` exists by delegating to :func:`list_prefix`."""

        norm = path.lstrip("/").rstrip("/")
        if norm == "":
            return False
        parent = norm.rsplit("/", 1)[0] + "/" if "/" in norm else ""
        items = [p.rstrip("/") for p in self.list_prefix(parent)]
        if norm in items:
            return True
        if path.endswith("/"):
            # directory check
            return any(it.startswith(norm + "/") for it in items)
        return False

    def info(self) -> str:
        """Return a short description of the storage configuration."""
        return f"mode={self.mode} root={self.local_root}"

    def cache_salt(self) -> str:
        """Return a string that changes when mode or URL changes."""
        return f"mode={self.mode}|url={self.supabase_url or ''}"

    def selftest(self) -> dict[str, str | bool]:
        """Basic self-test used by the Streamlit diagnostics pane."""
        return {"ok": True, "mode": self.mode}

    def list_all(self, prefix: str) -> list[str]:
        if self.mode == "local":
            base = self.local_root / prefix
            if not base.exists():
                return []
            return [str(Path(prefix) / p.name) for p in sorted(base.iterdir()) if p.is_file()]

        if self.bucket:
            resp = self.bucket.list(prefix)
            data = getattr(resp, "data", resp)
            return [f"{prefix}/{item['name']}" for item in data]

        return []

    @classmethod
    def from_env(cls) -> "Storage":
        """Return a ``Storage`` instance using environment configuration.

        The simplified test implementation always returns a local ``Storage``.
        """

        return cls()


@st.cache_data(show_spinner=False, hash_funcs={Storage: lambda _: 0})
def load_prices_cached(
    _storage: Storage,
    tickers: Iterable[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_salt: str,
) -> pd.DataFrame:
    """Load OHLCV history for ``tickers``.

    Uses Supabase table ``sp500_ohlcv`` when a client is available, otherwise
    falls back to the local parquet cache used in tests.  Results are cached
    via :func:`st.cache_data` to avoid repeated queries.
    """

    supabase: Client | None = getattr(_storage, "supabase_client", None)
    if supabase is not None and _storage.local_root is None:
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
            except Exception as e:
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
                raw = _storage.read_parquet(f"prices/{key}.parquet")
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

