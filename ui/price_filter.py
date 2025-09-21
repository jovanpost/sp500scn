from __future__ import annotations

import logging
import os
from typing import Callable, Iterable, Tuple

from data_lake.storage import ConfigurationError, Storage

PriceFilterFunc = Callable[[Storage, Iterable[str]], Tuple[list[str], list[str]]]

log = logging.getLogger(__name__)

STRUCTURED_ERROR_MESSAGE = (
    "Price availability check is unavailable. Run canceled. Likely import or storage config issue."
)
CALLOUT_MESSAGE = (
    "Filtering unavailable—import/storage misconfigured. Backtest canceled. "
    "Check LAKE_LAYOUT/LAKE_PRICES_PREFIX and helper import."
)

# ---- Behavior switches -------------------------------------------------------
# Default to the robust per-ticker probe (fixes 'missing parquet' false negatives).
# Set PRICE_FILTER_STRATEGY=helper to use the package helper again.
_PRICE_FILTER_STRATEGY = (os.getenv("PRICE_FILTER_STRATEGY") or "probe").strip().lower()
# Allow falling back to probe if helper import fails.
_ALLOW_FALLBACK = (os.getenv("ALLOW_FALLBACK") or "false").strip().lower() == "true"
# ------------------------------------------------------------------------------


_PRICE_FILTER_INITIALIZED = False
_PRICE_FILTER_FUNC: PriceFilterFunc | None = None

PRICE_FILTER_READY = False
PRICE_FILTER_ERROR: str | None = None
PRICE_FILTER_SOURCE = "uninitialized"


class PriceFilterUnavailableError(RuntimeError):
    """Raised when price availability filtering cannot run safely."""

    code = "PRICE_FILTER_UNAVAILABLE"
    structured_message = STRUCTURED_ERROR_MESSAGE
    user_message = CALLOUT_MESSAGE

    def __init__(self, reason: str | None = None) -> None:
        super().__init__(self.structured_message)
        self.reason = reason


def _resolve_prefix_for_fallback(storage: Storage) -> str:
    """
    Returns path prefix relative to the bucket root that contains the prices.
    Works for both supabase and local backends.
    """
    raw_prefix = os.getenv("LAKE_PRICES_PREFIX", "lake/prices")
    prefix = str(raw_prefix or "").strip().strip("/")
    bucket = str(getattr(storage, "bucket", "") or "").strip().strip("/")
    # If prefix was given as "<bucket>" or "<bucket>/...": normalize to folder-only
    if bucket and prefix == bucket:
        return ""
    if bucket and prefix.startswith(f"{bucket}/"):
        prefix = prefix[len(bucket) + 1 :]
    return prefix.strip("/")


def _fallback_filter_tickers_with_parquet(
    storage: Storage, tickers: Iterable[str]
) -> tuple[list[str], list[str]]:
    """
    Robust per-ticker existence probe:
    - Supports flat files:  prices/TICKER.parquet
    - Supports partitioned: prices/TICKER/...
    - Immune to folder listing pagination/limits.
    """
    seen: set[str] = set()
    present: list[str] = []
    missing: list[str] = []

    prefix = _resolve_prefix_for_fallback(storage)
    layout = (os.getenv("LAKE_LAYOUT") or "flat").strip().lower()

    for raw in tickers or []:
        if not raw:
            continue
        ticker = str(raw).strip().upper()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)

        try:
            if layout == "partitioned":
                # in partitioned layout we expect a folder per ticker
                folder = f"{prefix}/{ticker}" if prefix else ticker
                exists = bool(storage.list_prefix(folder))
            else:
                # flat layout: one parquet per ticker
                key = f"{prefix}/{ticker}.parquet" if prefix else f"{ticker}.parquet"
                exists = bool(storage.exists(key))
        except Exception as e:
            log.warning("price probe failed for %s: %s", ticker, e)
            exists = False

        (present if exists else missing).append(ticker)

    return present, missing


def initialize_price_filter() -> None:
    """Pick the active price filter implementation once."""

    global _PRICE_FILTER_INITIALIZED, _PRICE_FILTER_FUNC
    global PRICE_FILTER_READY, PRICE_FILTER_ERROR, PRICE_FILTER_SOURCE

    if _PRICE_FILTER_INITIALIZED:
        return
    _PRICE_FILTER_INITIALIZED = True

    # Force the robust strategy by default.
    if _PRICE_FILTER_STRATEGY == "probe":
        _PRICE_FILTER_FUNC = _fallback_filter_tickers_with_parquet
        PRICE_FILTER_READY = True
        PRICE_FILTER_ERROR = None
        PRICE_FILTER_SOURCE = "probe"
        log.info("Price filter: using per-ticker probe strategy (recommended).")
        return

    # Optional: try the package helper (faster when listing works correctly)
    try:
        from data_lake.storage import filter_tickers_with_parquet as helper
        _PRICE_FILTER_FUNC = helper
        PRICE_FILTER_READY = True
        PRICE_FILTER_ERROR = None
        PRICE_FILTER_SOURCE = "package"
        log.info("Price filter: using package helper strategy.")
    except Exception as exc:  # pragma: no cover - defensive
        PRICE_FILTER_READY = False
        PRICE_FILTER_ERROR = str(exc)
        if _ALLOW_FALLBACK:
            log.warning(
                "Package helper import failed; falling back to probe strategy: %s",
                exc,
            )
            _PRICE_FILTER_FUNC = _fallback_filter_tickers_with_parquet
            PRICE_FILTER_SOURCE = "fallback"
            PRICE_FILTER_READY = True
            PRICE_FILTER_ERROR = None
        else:
            log.error("Price availability helper import failed: %s", exc)
            _PRICE_FILTER_FUNC = None
            PRICE_FILTER_SOURCE = "unavailable"


def get_price_filter() -> tuple[PriceFilterFunc, str]:
    """Return the active price filter callable and its source."""
    initialize_price_filter()
    if _PRICE_FILTER_FUNC is None:
        raise PriceFilterUnavailableError(PRICE_FILTER_ERROR)
    return _PRICE_FILTER_FUNC, PRICE_FILTER_SOURCE


def raise_unavailable(reason: str | Exception | None = None) -> PriceFilterUnavailableError:
    """Convert a configuration failure into a user-facing error."""
    text = str(reason) if isinstance(reason, Exception) else reason
    if text:
        log.error("Price availability helper unavailable: %s", text)
    else:
        log.error("Price availability helper unavailable for unknown reason.")
    return PriceFilterUnavailableError(text)


def handle_filter_exception(exc: Exception) -> PriceFilterUnavailableError:
    """Normalize storage helper errors into unavailable errors."""
    if isinstance(exc, PriceFilterUnavailableError):
        return exc
    if isinstance(exc, ConfigurationError):
        return raise_unavailable(exc)
    return raise_unavailable(str(exc))

