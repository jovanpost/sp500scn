from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Mapping, Tuple, List, Any


@dataclass
class RuleConfig:
    min_rr_required: float = 2.0
    # If True, a missing R:R does NOT fail the gate (we infer when possible).
    allow_missing_rr: bool = True

    def __post_init__(self) -> None:
        try:
            v = float(self.min_rr_required)
        except (TypeError, ValueError):
            v = 2.0
        if not math.isfinite(v) or v < 2.0:
            v = 2.0
        self.min_rr_required = v
        self.allow_missing_rr = bool(self.allow_missing_rr)


def _to_float(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _to_int(v: Any) -> int | None:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _infer_rr_ratio(row: Mapping[str, Any]) -> float | None:
    # Prefer precomputed rr_ratio
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

    # Reward handles up/down targets
    reward = tp_abs - entry if tp_abs >= entry else entry - tp_abs
    if reward <= 0:
        return None

    return reward / risk


def _env_on(name: str, default: bool = False) -> bool:
    val = os.getenv(name, "1" if default else "0").strip().lower()
    return val in {"1", "true", "yes", "on"}


def passes_all_rules(row: Mapping[str, Any], cfg: RuleConfig) -> Tuple[bool, List[str]]:
    """
    Lenient-by-default rule gate:
      - Missing optional signals (RSI, earnings window, VWAP, setup flag) DO NOT fail.
      - ATR / S:R / Precedent flags respected if present (these are also pre-gates upstream).
      - R:R enforced when known; otherwise honor cfg.allow_missing_rr.
    Env toggles:
      RULES_BYPASS_ALL=1           -> always pass (emergency kill-switch)
      RULES_BYPASS_SOFT=1          -> ignore RSI/earnings/VWAP/setup checks
      RULES_IGNORE_ATR_SR_PREC=1   -> ignore atr_ok/sr_ok/precedent_ok here (still run pre-gates upstream)
    """
    # Emergency bypass
    if _env_on("RULES_BYPASS_ALL", False):
        return True, []

    reasons: List[str] = []

    # Hard-ish structural flags (can be ignored via env)
    if not _env_on("RULES_IGNORE_ATR_SR_PREC", False):
        atr_ok = _to_int(row.get("atr_ok"))
        if atr_ok is not None and atr_ok != 1:
            reasons.append("atr_fail")

        sr_ok = _to_int(row.get("sr_ok"))
        if sr_ok is not None and sr_ok != 1:
            reasons.append("sr_fail")

        prec_ok = _to_int(row.get("precedent_ok"))
        if prec_ok is not None and prec_ok != 1:
            reasons.append("precedent_fail")

    # R:R requirement
    rr_value = _infer_rr_ratio(row)
    if rr_value is None:
        if not cfg.allow_missing_rr:
            reasons.append("rr_fail")
    elif rr_value < cfg.min_rr_required:
        reasons.append("rr_fail")

    # Soft signals — lenient: ONLY fail when present and violating.
    if not _env_on("RULES_BYPASS_SOFT", False):
        rsi_1h = _to_float(row.get("rsi_1h"))
        if rsi_1h is not None and rsi_1h > 65.0:
            reasons.append("rsi_1h_high")

        rsi_d = _to_float(row.get("rsi_d"))
        if rsi_d is not None and rsi_d > 75.0:
            reasons.append("rsi_d_high")

        earnings_days = _to_float(row.get("earnings_days"))
        # Only enforce if present; defaulting to fail on missing was too strict
        if earnings_days is not None and earnings_days < 5:
            reasons.append("earnings_window_fail")

        vwap_hold = _to_int(row.get("vwap_hold"))
        if vwap_hold is not None and vwap_hold != 1:
            reasons.append("vwap_hold_fail")

        setup_valid = _to_int(row.get("setup_valid"))
        if setup_valid is not None and setup_valid != 1:
            reasons.append("setup_invalid")

    return (len(reasons) == 0), reasons


__all__ = ["RuleConfig", "passes_all_rules", "_infer_rr_ratio"]

