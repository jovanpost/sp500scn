from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Tuple, List, Any


@dataclass
class RuleConfig:
    min_rr_required: float = 2.0
    allow_missing_rr: bool = True

    def __post_init__(self) -> None:
        try:
            value = float(self.min_rr_required)
        except (TypeError, ValueError):
            value = 2.0
        if not math.isfinite(value) or value < 2.0:
            value = 2.0
        self.min_rr_required = value

        self.allow_missing_rr = bool(self.allow_missing_rr)


def _to_float(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _to_int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _infer_rr_ratio(row: Mapping[str, Any]) -> float | None:
    rr = _to_float(row.get("rr_ratio"))
    if rr is not None:
        return rr

    entry = _to_float(row.get("entry_open"))
    support = _to_float(row.get("support"))
    tp_abs = _to_float(row.get("tp_price_abs_target"))

    if entry is None or support is None or tp_abs is None:
        return None

    risk = entry - support
    if risk <= 0:
        return None

    reward = tp_abs - entry if tp_abs >= entry else entry - tp_abs
    if reward <= 0:
        return None

    return reward / risk


def passes_all_rules(row: Mapping[str, Any], cfg: RuleConfig) -> Tuple[bool, List[str]]:
    reasons: List[str] = []

    if _to_int(row.get("atr_ok")) != 1:
        reasons.append("atr_fail")
    if _to_int(row.get("sr_ok")) != 1:
        reasons.append("sr_fail")
    if _to_int(row.get("precedent_ok")) != 1:
        reasons.append("precedent_fail")

    rr_value = _infer_rr_ratio(row)
    if rr_value is None:
        if not cfg.allow_missing_rr:
            reasons.append("rr_fail")
    elif rr_value < cfg.min_rr_required:
        reasons.append("rr_fail")

    rsi_1h = _to_float(row.get("rsi_1h"))
    if rsi_1h is None or rsi_1h > 65.0:
        reasons.append("rsi_1h_high")

    rsi_d = _to_float(row.get("rsi_d"))
    if rsi_d is None or rsi_d > 75.0:
        reasons.append("rsi_d_high")

    earnings_days = _to_float(row.get("earnings_days"))
    if earnings_days is None or earnings_days < 5:
        reasons.append("earnings_window_fail")

    if _to_int(row.get("vwap_hold")) != 1:
        reasons.append("vwap_hold_fail")

    if _to_int(row.get("setup_valid")) != 1:
        reasons.append("setup_invalid")

    return (len(reasons) == 0), reasons


__all__ = ["RuleConfig", "passes_all_rules"]
