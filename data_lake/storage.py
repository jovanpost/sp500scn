import base64
import json
import logging
from pathlib import Path
from typing import Iterable

import pandas as pd


# Default root for local storage operations
DEFAULT_LOCAL_ROOT = Path("data")
# Backwards compatibility for tests that patch ``LOCAL_ROOT``
LOCAL_ROOT = DEFAULT_LOCAL_ROOT


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
        self.mode = "local"
        self.bucket = None
        # Where local files are stored; tests may monkeypatch ``LOCAL_ROOT``
        self.local_root = LOCAL_ROOT
        self._prices_cache: dict[str, pd.DataFrame] = {}
        # mimic interface of production storage for UI diagnostics
        self.key_info: dict[str, str] = {"kind": "service_role"}
        self.log = logging.getLogger(__name__)

    # The tests monkeypatch this method, so the implementation can be minimal.
    def read_parquet(self, path: str) -> pd.DataFrame:
        return pd.read_parquet(self.local_root / path)

    # --- simple byte-level helpers used by the UI ---
    def read_bytes(self, path: str) -> bytes:
        """Return raw bytes from ``path``.

        The test implementation only supports the local ``data`` directory,
        but the method mirrors the interface of the production storage layer
        used by the Streamlit app.
        """

        return (self.local_root / path).read_bytes()

    def write_bytes(self, path: str, data: bytes) -> None:
        """Write ``data`` to ``path`` under the local ``data`` directory."""

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
            base = (
                (self.local_root / pfx).parent
                if pfx.endswith(".parquet")
                else (self.local_root / pfx)
            ).resolve()
            results: list[str] = []
            if base.is_file():
                results.append(str(base.relative_to(self.local_root)).replace("\\", "/"))
            elif base.exists():
                for path in base.rglob("*"):
                    if path.is_file():
                        rel = str(path.relative_to(self.local_root)).replace("\\", "/")
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
        """Return ``True`` if ``path`` exists in storage."""

        norm = path.lstrip("/")

        # Directory-style check: any file under this prefix?
        if norm.endswith("/"):
            return len(self.list_prefix(norm)) > 0

        parent = Path(norm).parent.as_posix()
        base = Path(norm).name
        for full in self.list_prefix(parent + "/"):
            if full.replace("\\", "/").endswith("/" + base) or full == norm:
                return True
        return False

    def info(self) -> str:
        """Return a short description of the storage configuration."""
        return f"mode={self.mode} root={self.local_root}"

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


def load_prices_cached(
    storage: Storage,
    tickers: Iterable[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Load price history for ``tickers`` from a parquet cache.

    Results are cached on the ``storage`` instance to avoid repeated I/O.
    """

    frames: list[pd.DataFrame] = []
    for t in tickers:
        key = str(t).upper()
        if key not in storage._prices_cache:
            try:
                raw = storage.read_parquet(f"prices/{key}.parquet")
            except FileNotFoundError:
                continue
            tidy = _tidy_prices(raw, key)
            storage._prices_cache[key] = tidy

        df = storage._prices_cache[key]
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

