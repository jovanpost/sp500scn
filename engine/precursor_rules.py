from __future__ import annotations

from typing import Any

from engine.scan_shared.precursor_flags import DEFAULT_PARAMS

PRECURSOR_FLAG_KEYS = {
    "scanner_precursor_ema": "ema_20_50_cross_up",
    "scanner_precursor_rsi50": "rsi_cross_50",
    "scanner_precursor_rsi60": "rsi_cross_60",
    "scanner_precursor_atr": "atr_squeeze_pct",
    "scanner_precursor_bb": "bb_squeeze_pct",
    "scanner_precursor_nr7": "nr7",
    "scanner_precursor_gap": "gap_up_ge_gpct_prev",
    "scanner_precursor_vol_d1": "vol_mult_d1_ge_x",
    "scanner_precursor_vol_d2": "vol_mult_d2_ge_x",
    "scanner_precursor_sr": "sr_ratio_ge_2",
    "scanner_precursor_high20": "new_high_20",
    "scanner_precursor_high63": "new_high_63",
}

THRESHOLD_KEYS = {
    "atr_squeeze_pct": ("scanner_precursor_atr_threshold", "max_percentile", 1.0, 100.0, DEFAULT_PARAMS["atr_pct_threshold"]),
    "bb_squeeze_pct": ("scanner_precursor_bb_threshold", "max_percentile", 1.0, 100.0, DEFAULT_PARAMS["bb_pct_threshold"]),
    "gap_up_ge_gpct_prev": ("scanner_precursor_gap_threshold", "min_gap_pct", 0.0, None, DEFAULT_PARAMS["gap_min_pct"]),
    "vol_mult_d1_ge_x": ("scanner_precursor_vol_threshold", "min_mult", 0.1, None, DEFAULT_PARAMS["vol_min_mult"]),
    "vol_mult_d2_ge_x": ("scanner_precursor_vol_threshold", "min_mult", 0.1, None, DEFAULT_PARAMS["vol_min_mult"]),
}

ALIASES = {
    "ema20_50_cross_up": "ema_20_50_cross_up",
    "atr_squeeze_q": "atr_squeeze_pct",
    "bb_squeeze_q": "bb_squeeze_pct",
    "gap_pct": "gap_up_ge_gpct_prev",
    "gap_prior_ge_pct": "gap_up_ge_gpct_prev",
    "vol_d1": "vol_mult_d1_ge_x",
    "vol_d2": "vol_mult_d2_ge_x",
    "sr_ratio_gte": "sr_ratio_ge_2",
}


def _coerce_float(value: Any, *, default: float, minimum: float | None = None, maximum: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = float(default)
    if minimum is not None:
        result = max(minimum, result)
    if maximum is not None:
        result = min(maximum, result)
    return result


def _coerce_int(value: Any, *, default: int, minimum: int = 1) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        result = int(default)
    if result < minimum:
        result = minimum
    return result


def _ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [v for v in value if v is not None]
    if isinstance(value, str):
        return [value]
    return []


def build_conditions_from_session(ss: Any) -> dict[str, Any] | None:
    if not getattr(ss, "get", None):
        raise TypeError("session state-like object required")

    if not ss.get("scanner_precursor_enabled", False):
        return None

    within_days = _coerce_int(
        ss.get("scanner_precursor_within", DEFAULT_PARAMS["lookback_days"]),
        default=DEFAULT_PARAMS["lookback_days"],
        minimum=1,
    )
    logic = str(ss.get("scanner_precursor_logic", "ANY") or "ANY").upper()
    if logic not in {"ANY", "ALL"}:
        logic = "ANY"

    conditions: list[dict[str, Any]] = []

    for session_key, flag in PRECURSOR_FLAG_KEYS.items():
        if not ss.get(session_key, False):
            continue
        payload: dict[str, Any] = {"flag": flag}
        if flag in THRESHOLD_KEYS:
            thresh_key, field, minimum, maximum, default = THRESHOLD_KEYS[flag]
            value = _coerce_float(
                ss.get(thresh_key, default),
                default=default,
                minimum=minimum,
                maximum=maximum,
            )
            payload[field] = value
        conditions.append(payload)

    if not conditions:
        return None

    base_thresholds = {
        "atr_pct_threshold": _coerce_float(
            ss.get("scanner_precursor_atr_threshold", DEFAULT_PARAMS["atr_pct_threshold"]),
            default=DEFAULT_PARAMS["atr_pct_threshold"],
            minimum=1.0,
            maximum=100.0,
        ),
        "bb_pct_threshold": _coerce_float(
            ss.get("scanner_precursor_bb_threshold", DEFAULT_PARAMS["bb_pct_threshold"]),
            default=DEFAULT_PARAMS["bb_pct_threshold"],
            minimum=1.0,
            maximum=100.0,
        ),
        "gap_min_pct": _coerce_float(
            ss.get("scanner_precursor_gap_threshold", DEFAULT_PARAMS["gap_min_pct"]),
            default=DEFAULT_PARAMS["gap_min_pct"],
            minimum=0.0,
        ),
        "vol_min_mult": _coerce_float(
            ss.get("scanner_precursor_vol_threshold", DEFAULT_PARAMS["vol_min_mult"]),
            default=DEFAULT_PARAMS["vol_min_mult"],
            minimum=0.1,
        ),
    }

    return {
        "enabled": True,
        "within_days": within_days,
        "logic": logic,
        "conditions": conditions,
        **base_thresholds,
    }


def apply_preset_to_session(preset_json: dict[str, Any], ss: Any) -> list[str]:
    if not isinstance(preset_json, dict):
        raise TypeError("Preset payload must be a dict")
    if not getattr(ss, "__setitem__", None):
        raise TypeError("Session state-like object required")

    ss["scanner_precursor_enabled"] = True

    logic = str(preset_json.get("logic", "ANY") or "ANY").upper()
    if logic not in {"ANY", "ALL"}:
        logic = "ANY"
    ss["scanner_precursor_logic"] = logic

    within_candidates: list[int] = []
    applied_flags: list[str] = []

    for flag_key in PRECURSOR_FLAG_KEYS:
        ss[flag_key] = False

    for condition in _ensure_list(preset_json.get("conditions")):
        if isinstance(condition, dict):
            raw_flag = str(condition.get("flag", "")).strip()
            if not raw_flag:
                continue
            flag = ALIASES.get(raw_flag, raw_flag)
            session_flag_key = None
            for key, candidate in PRECURSOR_FLAG_KEYS.items():
                if candidate == flag:
                    session_flag_key = key
                    break
            if session_flag_key is None:
                continue
            ss[session_flag_key] = True
            applied_flags.append(flag)
            within_val = condition.get("within_days")
            if within_val is not None:
                within_candidates.append(_coerce_int(within_val, default=DEFAULT_PARAMS["lookback_days"]))
            if flag in THRESHOLD_KEYS:
                thresh_key, field, minimum, maximum, default = THRESHOLD_KEYS[flag]
                raw_value = condition.get(field)
                if raw_value is None:
                    raw_value = condition.get("threshold")
                ss[thresh_key] = _coerce_float(
                    raw_value,
                    default=default,
                    minimum=minimum,
                    maximum=maximum,
                )
        else:
            flag = ALIASES.get(str(condition), str(condition))
            for key, candidate in PRECURSOR_FLAG_KEYS.items():
                if candidate == flag:
                    ss[key] = True
                    applied_flags.append(flag)
                    break

    if preset_json.get("within_days") is not None:
        ss["scanner_precursor_within"] = _coerce_int(
            preset_json.get("within_days"),
            default=DEFAULT_PARAMS["lookback_days"],
        )
    elif within_candidates:
        ss["scanner_precursor_within"] = max(within_candidates)
    else:
        ss.setdefault("scanner_precursor_within", DEFAULT_PARAMS["lookback_days"])

    return applied_flags


__all__ = [
    "build_conditions_from_session",
    "apply_preset_to_session",
    "PRECURSOR_FLAG_KEYS",
    "ALIASES",
]
