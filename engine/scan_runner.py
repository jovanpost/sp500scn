from __future__ import annotations

import dataclasses
import logging
import random
from dataclasses import dataclass, asdict
from typing import Any, Callable

import numpy as np
import pandas as pd

from engine.stocks_only_scanner import (
    DEFAULT_CASH_CAP as _DEFAULT_CASH_CAP,
    ScanSummary as _LegacyScanSummary,
    run_scan as _legacy_run_scan,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class StocksOnlyScanParams:
    start: pd.Timestamp
    end: pd.Timestamp
    horizon_days: int
    sr_lookback: int
    sr_min_ratio: float
    min_yup_pct: float
    min_gap_pct: float
    min_volume_multiple: float
    volume_lookback: int
    exit_model: str
    atr_window: int
    atr_method: str
    tp_atr_multiple: float
    sl_atr_multiple: float
    use_sp_filter: bool
    cash_per_trade: float = _DEFAULT_CASH_CAP
    precursors: dict[str, Any] | None = None
    seed: int | None = 42


def _coerce_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def _prepare_payload(params: StocksOnlyScanParams) -> dict[str, Any]:
    payload = asdict(params)
    payload["start"] = _coerce_timestamp(payload["start"])
    payload["end"] = _coerce_timestamp(payload["end"])
    payload.pop("seed", None)
    return payload


def _ensure_precursor_payload(raw: dict[str, Any] | None) -> dict[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        return None
    cleaned: dict[str, Any] = {str(k): v for k, v in raw.items()}
    if not cleaned.get("enabled"):
        return None
    conditions = []
    for condition in cleaned.get("conditions", []) or []:
        if not isinstance(condition, dict):
            continue
        flag = str(condition.get("flag", "")).strip()
        if not flag:
            continue
        entry = {"flag": flag}
        if condition.get("max_percentile") is not None:
            entry["max_percentile"] = float(condition.get("max_percentile"))
        if condition.get("min_gap_pct") is not None:
            entry["min_gap_pct"] = float(condition.get("min_gap_pct"))
        if condition.get("min_mult") is not None:
            entry["min_mult"] = float(condition.get("min_mult"))
        conditions.append(entry)
    if not conditions:
        return None
    cleaned["conditions"] = conditions
    return cleaned


def run_scan(
    params: StocksOnlyScanParams,
    *,
    storage: Any | None = None,
    prices_by_ticker: dict[str, pd.DataFrame] | None = None,
    membership: pd.DataFrame | None = None,
    progress: Callable[[int, int, str], None] | None = None,
    debug: Any | None = None,
) -> dict[str, Any]:
    """Execute the unified shares-only backtest.

    The implementation delegates to :mod:`engine.stocks_only_scanner` and wraps the
    result into a deterministic, serialisable payload so the UI layers only need to
    deal with a single structure.
    """

    if params.seed is not None:
        np.random.seed(int(params.seed))
        random.seed(int(params.seed))

    payload = _prepare_payload(params)
    payload["precursors"] = _ensure_precursor_payload(params.precursors)

    legacy_kwargs: dict[str, Any] = {}
    if storage is not None:
        legacy_kwargs["storage"] = storage
    if prices_by_ticker is not None:
        legacy_kwargs["prices_by_ticker"] = prices_by_ticker
    if membership is not None:
        legacy_kwargs["membership"] = membership
    if debug is not None:
        legacy_kwargs["debug"] = debug
    if progress is not None:
        legacy_kwargs["progress"] = progress  # type: ignore[assignment]

    log.debug("scan_runner payload=%s", payload)

    trades_df, summary = _legacy_run_scan(payload, **legacy_kwargs)
    if isinstance(summary, _LegacyScanSummary):
        summary_dict = dataclasses.asdict(summary)
    elif isinstance(summary, dict):
        summary_dict = summary
    else:
        summary_dict = {
            "start": payload["start"],
            "end": payload["end"],
        }

    result = {
        "summary": summary_dict,
        "trades": trades_df.copy(),
        "debug": {"seed": params.seed},
    }
    return result


__all__ = ["StocksOnlyScanParams", "run_scan"]
