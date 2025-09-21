from __future__ import annotations

import logging
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

# -----------------------------------------------------------------------------
# TRUST-ALL STRATEGY (HOTFIX)
# Always include every requested ticker; we will discover true availability
# when we actually try to read the parquet during preload. This avoids
# false negatives from prefix listing/pagination/path issues.
# -----------------------------------------------------------------------------

def _trust_all_filter(_storage: Storage, tickers: Iterable[str]) -> tuple[list[str], list[str]]:
    seen: set[str] = set()
    present: list[str] = []
    for raw in tickers or []:
        t = str(raw).strip().upper()
        if t and t not in seen:
            seen.add(t)
            present.append(t)
    # We don’t pre-declare anything missing; loading will reveal real misses.
    return present, []

_PRICE_FILTER_INITIALIZED = True
_PRICE_FILTER_FUNC: PriceFilterFunc | None = _trust_all_filter

PRICE_FILTER_READY = True
PRICE_FILTER_ERROR: str | None = None
PRICE_FILTER_SOURCE = "trust"

class PriceFilterUnavailableError(RuntimeError):
    code = "PRICE_FILTER_UNAVAILABLE"
    structured_message = STRUCTURED_ERROR_MESSAGE
    user_message = CALLOUT_MESSAGE
    def __init__(self, reason: str | None = None) -> None:
        super().__init__(self.structured_message)
        self.reason = reason

def initialize_price_filter() -> None:
    # no-op; we’re forcing trust-all
    pass

def get_price_filter() -> tuple[PriceFilterFunc, str]:
    return _PRICE_FILTER_FUNC, PRICE_FILTER_SOURCE

def raise_unavailable(reason: str | Exception | None = None) -> PriceFilterUnavailableError:
    text = str(reason) if isinstance(reason, Exception) else reason
    if text:
        log.error("Price availability helper unavailable: %s", text)
    else:
        log.error("Price availability helper unavailable for unknown reason.")
    return PriceFilterUnavailableError(text)

def handle_filter_exception(exc: Exception) -> PriceFilterUnavailableError:
    if isinstance(exc, PriceFilterUnavailableError):
        return exc
    if isinstance(exc, ConfigurationError):
        return raise_unavailable(exc)
    return raise_unavailable(str(exc))
