from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from engine.features import atr as compute_atr


@dataclass(frozen=True)
class IndicatorConfig:
    atr_window: int = 14
    atr_method: str = "wilder"
    atr_percentile_window: int = 63
    bb_period: int = 20
    bb_percentile_window: int = 126
    nr7_window: int = 7
    volume_lookback: int = 63
    sr_lookback: int = 63
    new_high_windows: tuple[int, ...] = (20, 63)

    @property
    def max_lookback(self) -> int:
        """Maximum trailing window required to compute all indicators."""
        extras: Iterable[int] = (
            self.atr_window,
            self.atr_percentile_window,
            self.bb_period,
            self.bb_percentile_window,
            self.nr7_window,
            self.volume_lookback,
            self.sr_lookback,
            max(self.new_high_windows) if self.new_high_windows else 0,
        )
        return int(max(int(x) for x in extras))


def ensure_datetime_index(panel: pd.DataFrame) -> pd.DataFrame:
    """Return panel with a normalized datetime index and sorted rows."""
    working = panel.copy()
    if "date" in working.columns:
        working["date"] = pd.to_datetime(working["date"], errors="coerce").dt.tz_localize(None)
        working = working.dropna(subset=["date"])
        working = working.sort_values("date").drop_duplicates(subset=["date"], keep="last")
        working = working.set_index("date", drop=False)
    else:
        working = working.copy()
        working.index = pd.to_datetime(working.index, errors="coerce").tz_localize(None)
        working = working.dropna(axis=0, subset=working.index.names)
        if "date" not in working.columns:
            working["date"] = working.index
    working.index = working.index.tz_localize(None)
    working.index = working.index.normalize()
    working = working[~working.index.duplicated(keep="last")]
    return working.sort_index()


def rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return pd.Series(np.nan, index=series.index)

    min_periods = min(window, max(3, window // 2))

    def _pct(arr: np.ndarray) -> float:
        if len(arr) == 0:
            return float("nan")
        current = arr[-1]
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            return float("nan")
        rank = (valid <= current).sum() / len(valid)
        return float(rank * 100.0)

    return series.rolling(window, min_periods=min_periods).apply(_pct, raw=True)


def compute_common_indicators(
    panel: pd.DataFrame,
    *,
    config: IndicatorConfig = IndicatorConfig(),
) -> pd.DataFrame:
    """Add common indicator columns used by spike precursor filters."""
    working = ensure_datetime_index(panel)
    close = working["close"].astype(float)
    high = working["high"].astype(float)
    low = working["low"].astype(float)
    open_ = working["open"].astype(float)
    volume = working.get("volume", pd.Series(np.nan, index=working.index)).astype(float)

    # Trend: EMA cross
    ema_fast = close.ewm(span=20, adjust=False, min_periods=20).mean()
    ema_slow = close.ewm(span=50, adjust=False, min_periods=50).mean()
    working["ema_fast_20"] = ema_fast
    working["ema_slow_50"] = ema_slow
    working["ema_20_50_cross_up"] = (ema_fast > ema_slow) & (
        ema_fast.shift(1) <= ema_slow.shift(1)
    )

    # RSI crosses
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    roll = 14
    avg_gain = gain.ewm(alpha=1 / roll, adjust=False, min_periods=roll).mean()
    avg_loss = loss.ewm(alpha=1 / roll, adjust=False, min_periods=roll).mean()
    rs = avg_gain / avg_loss.replace({0: np.nan})
    rsi = 100 - (100 / (1 + rs))
    working["rsi"] = rsi
    working["rsi_cross_50"] = (rsi >= 50.0) & (rsi.shift(1) < 50.0)
    working["rsi_cross_60"] = (rsi >= 60.0) & (rsi.shift(1) < 60.0)

    # ATR squeeze percentile
    atr = compute_atr(
        working[["high", "low", "close"]],
        window=config.atr_window,
        method=config.atr_method,
    )
    working["atr_value"] = atr
    working["atr_pctile"] = rolling_percentile(atr, config.atr_percentile_window)

    # Bollinger Bandwidth squeeze
    mid = close.rolling(config.bb_period, min_periods=config.bb_period).mean()
    std = close.rolling(config.bb_period, min_periods=config.bb_period).std(ddof=0)
    upper = mid + 2 * std
    lower = mid - 2 * std
    bandwidth = (upper - lower) / mid.replace({0: np.nan})
    working["bb_width"] = bandwidth
    working["bb_width_pctile"] = rolling_percentile(bandwidth, config.bb_percentile_window)

    # NR7
    true_range = (high - low).astype(float)
    working["nr7"] = true_range <= true_range.rolling(config.nr7_window).min()

    # Gaps
    prev_close = close.shift(1)
    working["gap_up_pct_prev"] = (open_ / prev_close - 1.0) * 100.0

    # Volume multiples
    vol_avg = volume.rolling(
        config.volume_lookback, min_periods=min(10, config.volume_lookback)
    ).mean()
    volume_multiple = np.where(vol_avg > 0, volume / vol_avg, np.nan)
    working["vol_mult_raw"] = volume_multiple
    working["vol_mult_d1"] = pd.Series(volume_multiple, index=working.index)
    working["vol_mult_d2"] = working["vol_mult_d1"].shift(1)

    # Support / resistance ratio
    support = low.rolling(config.sr_lookback, min_periods=config.sr_lookback).min().shift(1)
    resistance = high.rolling(config.sr_lookback, min_periods=config.sr_lookback).max().shift(1)
    working["support"] = support
    working["resistance"] = resistance
    denom = open_ - support
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (resistance - open_) / denom
    ratio = ratio.where((support < open_) & (resistance > open_))
    working["sr_ratio"] = ratio

    # New highs
    for window in config.new_high_windows:
        if window <= 0:
            continue
        rolling_max = close.shift(1).rolling(window, min_periods=window).max()
        key = f"new_high_{int(window)}"
        working[key] = close >= rolling_max

    return working


__all__ = [
    "IndicatorConfig",
    "compute_common_indicators",
    "ensure_datetime_index",
    "rolling_percentile",
]
