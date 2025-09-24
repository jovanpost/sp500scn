from __future__ import annotations

from typing import Any

import pandas as pd

from .indicators import IndicatorConfig, compute_common_indicators

DEFAULT_PARAMS = {
    "atr_pct_threshold": 25.0,
    "bb_pct_threshold": 20.0,
    "gap_min_pct": 3.0,
    "vol_min_mult": 1.5,
    "lookback_days": 20,
}

FLAG_COLUMNS = [
    "ema_20_50_cross_up",
    "rsi_cross_50",
    "rsi_cross_60",
    "atr_squeeze_pct",
    "bb_squeeze_pct",
    "nr7",
    "gap_up_ge_gpct_prev",
    "vol_mult_d1_ge_x",
    "vol_mult_d2_ge_x",
    "sr_ratio_ge_2",
    "new_high_20",
    "new_high_63",
]

METRIC_COLUMNS = [
    "atr_pctile",
    "bb_width_pctile",
    "gap_up_pct_prev",
    "vol_mult_d1",
    "vol_mult_d2",
    "sr_ratio",
]


def build_precursor_flags(
    panel: pd.DataFrame,
    params: dict | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach spike precursor flag columns to the panel."""

    config = IndicatorConfig()
    working = compute_common_indicators(panel, config=config)

    raw_params = params or {}
    atr_pct_threshold = float(
        raw_params.get("atr_pct_threshold", DEFAULT_PARAMS["atr_pct_threshold"])
    )
    bb_pct_threshold = float(
        raw_params.get("bb_pct_threshold", DEFAULT_PARAMS["bb_pct_threshold"])
    )
    gap_min_pct = float(raw_params.get("gap_min_pct", DEFAULT_PARAMS["gap_min_pct"]))
    vol_min_mult = float(raw_params.get("vol_min_mult", DEFAULT_PARAMS["vol_min_mult"]))
    lookback_days = int(raw_params.get("lookback_days", DEFAULT_PARAMS["lookback_days"]))

    working["atr_squeeze_pct"] = working["atr_pctile"] <= atr_pct_threshold
    working["bb_squeeze_pct"] = working["bb_width_pctile"] <= bb_pct_threshold
    working["gap_up_ge_gpct_prev"] = working["gap_up_pct_prev"] >= gap_min_pct
    working["vol_mult_d1"] = working["vol_mult_raw"].shift(1)
    working["vol_mult_d2"] = working["vol_mult_raw"].shift(2)
    working["vol_mult_d1_ge_x"] = working["vol_mult_d1"] >= vol_min_mult
    working["vol_mult_d2_ge_x"] = working["vol_mult_d2"] >= vol_min_mult
    working["sr_ratio_ge_2"] = working["sr_ratio"] >= 2.0

    flag_metadata: dict[str, Any] = {
        "ema_20_50_cross_up": {"type": "trend", "fast": 20, "slow": 50},
        "rsi_cross_50": {"type": "momentum", "level": 50, "period": 14},
        "rsi_cross_60": {"type": "momentum", "level": 60, "period": 14},
        "atr_squeeze_pct": {
            "type": "volatility",
            "measure": "atr_percentile",
            "threshold": atr_pct_threshold,
            "window": config.atr_percentile_window,
        },
        "bb_squeeze_pct": {
            "type": "volatility",
            "measure": "bb_width_percentile",
            "threshold": bb_pct_threshold,
            "window": config.bb_percentile_window,
        },
        "nr7": {"type": "range", "window": config.nr7_window},
        "gap_up_ge_gpct_prev": {"type": "gap", "threshold_pct": gap_min_pct},
        "vol_mult_d1_ge_x": {
            "type": "volume",
            "days_ago": 1,
            "threshold": vol_min_mult,
            "lookback": config.volume_lookback,
        },
        "vol_mult_d2_ge_x": {
            "type": "volume",
            "days_ago": 2,
            "threshold": vol_min_mult,
            "lookback": config.volume_lookback,
        },
        "sr_ratio_ge_2": {"type": "sr_ratio", "threshold": 2.0, "lookback": config.sr_lookback},
        "new_high_20": {"type": "new_high", "window": 20},
        "new_high_63": {"type": "new_high", "window": 63},
    }

    flag_metadata.update(
        {
            "atr_pctile": {"type": "volatility", "window": config.atr_percentile_window},
            "bb_width_pctile": {"type": "volatility", "window": config.bb_percentile_window},
            "gap_up_pct_prev": {"type": "gap"},
            "vol_mult_d1": {"type": "volume", "days_ago": 1},
            "vol_mult_d2": {"type": "volume", "days_ago": 2},
        }
    )

    flag_metadata["lookback_days"] = lookback_days
    flag_metadata["indicator_config"] = config
    flag_metadata["flags"] = FLAG_COLUMNS
    flag_metadata["metrics"] = METRIC_COLUMNS

    return working, flag_metadata


__all__ = ["build_precursor_flags", "DEFAULT_PARAMS", "FLAG_COLUMNS", "METRIC_COLUMNS"]
