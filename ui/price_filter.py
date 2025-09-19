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

_ALLOW_FALLBACK = os.getenv("ALLOW_FALLBACK", "false").strip().lower() == "true"

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
    raw_prefix = os.getenv("LAKE_PRICES_PREFIX", "lake/prices")
    prefix = str(raw_prefix or "").strip().strip("/")
    bucket = str(getattr(storage, "bucket", "") or "").strip().strip("/")
    if bucket and prefix == bucket:
        return ""
    if bucket and prefix.startswith(f"{bucket}/"):
        prefix = prefix[len(bucket) + 1 :]
    return prefix.strip("/")


def _fallback_filter_tickers_with_parquet(
    storage: Storage, tickers: Iterable[str]
) -> tuple[list[str], list[str]]:
    seen: set[str] = set()
    present: list[str] = []
    missing: list[str] = []

    prefix = _resolve_prefix_for_fallback(storage)
    layout = os.getenv("LAKE_LAYOUT", "flat").strip().lower() or "flat"

    for raw in tickers or []:
        if not raw:
            continue
        ticker = str(raw).strip().upper()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)

        if layout == "partitioned":
            key = f"{prefix}/{ticker}" if prefix else ticker
            try:
                exists = bool(storage.list_prefix(key))
            except Exception:
                exists = False
        else:
            key = f"{prefix}/{ticker}.parquet" if prefix else f"{ticker}.parquet"
            try:
                exists = storage.exists(key)
            except Exception:
                exists = False

        if exists:
            present.append(ticker)
        else:
            missing.append(ticker)

    return present, missing


def initialize_price_filter() -> None:
    """Perform a one-time import smoke of the price filter helper."""

    global _PRICE_FILTER_INITIALIZED, _PRICE_FILTER_FUNC
    global PRICE_FILTER_READY, PRICE_FILTER_ERROR, PRICE_FILTER_SOURCE

    if _PRICE_FILTER_INITIALIZED:
        return

    _PRICE_FILTER_INITIALIZED = True

    try:
        from data_lake.storage import filter_tickers_with_parquet as helper
    except Exception as exc:  # pragma: no cover - defensive
        PRICE_FILTER_READY = False
        PRICE_FILTER_ERROR = str(exc)
        if _ALLOW_FALLBACK:
            log.warning(
                "Price availability helper import failed; ALLOW_FALLBACK=true so using direct probes: %s",
                exc,
            )
            _PRICE_FILTER_FUNC = _fallback_filter_tickers_with_parquet
            PRICE_FILTER_SOURCE = "fallback"
        else:
            log.error("Price availability helper import failed: %s", exc)
            _PRICE_FILTER_FUNC = None
            PRICE_FILTER_SOURCE = "unavailable"
    else:
        _PRICE_FILTER_FUNC = helper
        PRICE_FILTER_READY = True
        PRICE_FILTER_ERROR = None
        PRICE_FILTER_SOURCE = "package"
        log.info("Price availability helper import succeeded.")


def get_price_filter() -> tuple[PriceFilterFunc, str]:
    """Return the active price filter callable and its source."""

    initialize_price_filter()
    if _PRICE_FILTER_FUNC is None:
        raise PriceFilterUnavailableError(PRICE_FILTER_ERROR)
    return _PRICE_FILTER_FUNC, PRICE_FILTER_SOURCE


def raise_unavailable(reason: str | Exception | None = None) -> PriceFilterUnavailableError:
    """Convert a configuration failure into a user-facing error."""

    text: str | None
    if isinstance(reason, Exception):
        text = str(reason)
    else:
        text = reason

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
