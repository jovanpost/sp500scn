from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from data_lake.storage import Storage, load_prices_cached

from .features import atr as compute_atr


def _normalize_prices_df(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Return a tidy OHLCV frame with normalized column names and unique dates."""

    working = df.copy()
    if "date" not in working.columns:
        index_name = working.index.name
        working = working.reset_index()
        if "date" not in working.columns:
            rename_candidates = []
            if "Date" in working.columns:
                rename_candidates.append("Date")
            if index_name and index_name in working.columns:
                rename_candidates.append(index_name)
            if "index" in working.columns:
                rename_candidates.append("index")
            if "level_0" in working.columns:
                rename_candidates.append("level_0")
            target = next(iter(rename_candidates), working.columns[0])
            working = working.rename(columns={target: "date"})
    if "date" not in working.columns and "Date" in working.columns:
        working = working.rename(columns={"Date": "date"})

    working["date"] = (
        pd.to_datetime(working["date"], errors="coerce").dt.tz_localize(None)
    )
    working = working.dropna(subset=["date"])

    rename_map = {c: c.lower() for c in working.columns}
    working = working.rename(columns=rename_map)

    for col in ["open", "high", "low", "close"]:
        if col not in working.columns:
            raise KeyError(f"Missing required price column '{col}' for {ticker}")
        working[col] = pd.to_numeric(working[col], errors="coerce")

    if "volume" in working.columns:
        working["volume"] = pd.to_numeric(working["volume"], errors="coerce")
    else:
        working["volume"] = float("nan")

    working = (
        working.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    )
    return working.reset_index(drop=True)


def _business_start(ts: pd.Timestamp, lookback: int) -> pd.Timestamp:
    if lookback <= 0:
        return ts
    start = (ts - BDay(int(lookback))).normalize()
    return pd.Timestamp(start).tz_localize(None)


def _rolling_percentile(series: pd.Series, window: int) -> pd.Series:
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


def _discover_all_tickers(storage: Storage) -> list[str]:
    try:
        entries = storage.list_prefix("prices/")
    except Exception:
        return []

    tickers: set[str] = set()
    for entry in entries:
        name = str(entry).replace("\\", "/")
        parts = [p for p in name.split("/") if p]
        if not parts:
            continue
        last = parts[-1]
        if last.startswith("_"):
            continue
        if last.lower().endswith(".parquet"):
            stem = last.rsplit(".parquet", 1)[0]
        else:
            stem = parts[-1]
        if stem:
            tickers.add(stem.upper())
    return sorted(tickers)


def analyze_spike_precursors(
    storage: Storage,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    universe: list[str] | None,
    spike_mode: Literal["pct", "atr"],
    pct_threshold: float | None,
    atr_multiple: float | None,
    atr_window: int,
    atr_method: str,
    lookback_days: int,
    indicator_config: dict[str, Any],
    debug: object | None = None,
) -> tuple[pd.DataFrame, dict]:
    start_ts = pd.Timestamp(start).tz_localize(None)
    end_ts = pd.Timestamp(end).tz_localize(None)
    if start_ts > end_ts:
        start_ts, end_ts = end_ts, start_ts

    universe_list: list[str]
    if universe:
        universe_list = sorted({str(t).upper() for t in universe if t})
    else:
        universe_list = _discover_all_tickers(storage)

    if debug and hasattr(debug, "log_event"):
        try:
            debug.log_event(
                "universe_resolved",
                requested=len(universe or []),
                resolved=len(universe_list),
            )
        except Exception:
            pass

    if not universe_list:
        empty = pd.DataFrame(columns=["ticker", "spike_date"])
        summary = {
            "counts": {
                "spikes": 0,
                "tickers": 0,
                "start": start_ts,
                "end": end_ts,
            },
            "frequency_table": pd.DataFrame(
                columns=["flag", "hit_count", "hit_rate", "median_lead_days"]
            ),
            "combos": pd.DataFrame(columns=["combo", "count", "lift"]),
            "flag_metadata": {},
            "lookback_days": lookback_days,
        }
        return empty, summary

    atr_method = (atr_method or "wilder").strip().lower()

    raw_config = indicator_config if isinstance(indicator_config, dict) else {}
    config = {
        str(key): value
        for key, value in raw_config.items()
        if isinstance(value, dict)
    }

    def _section_state(name: str) -> tuple[dict[str, Any], bool]:
        section = config.get(name)
        if not isinstance(section, dict):
            return {}, False
        enabled_val = section.get("enabled")
        if enabled_val is None:
            enabled_flag = True
        else:
            enabled_flag = bool(enabled_val)
        return section, enabled_flag

    volume_filter_cfg, volume_filter_enabled = _section_state("volume_filter")
    volume_filter_lookback = int(volume_filter_cfg.get("lookback", 63) or 63)
    volume_filter_threshold = float(volume_filter_cfg.get("threshold", 1.5) or 1.5)

    trend_cfg, trend_enabled = _section_state("trend")
    trend_fast = int(trend_cfg.get("fast", 20) or 20)
    trend_slow = int(trend_cfg.get("slow", 50) or 50)

    rsi_cfg, rsi_enabled = _section_state("rsi")
    rsi_period = int(rsi_cfg.get("period", 14) or 14)
    rsi_levels = rsi_cfg.get("levels") or [50.0, 60.0]
    rsi_levels = [float(x) for x in rsi_levels if x is not None]
    if not rsi_levels:
        rsi_levels = [50.0, 60.0]

    atr_cfg, atr_squeeze_enabled = _section_state("atr_squeeze")
    atr_pct_window = int(atr_cfg.get("window", 63) or 63)
    atr_pct_threshold = float(atr_cfg.get("percentile", 25.0) or 25.0)

    bb_cfg, bb_enabled = _section_state("bb")
    bb_period = int(bb_cfg.get("period", 20) or 20)
    bb_pct_window = int(bb_cfg.get("pct_window", 126) or 126)
    bb_percentile = float(bb_cfg.get("percentile", 20.0) or 20.0)

    nr7_cfg, nr7_enabled = _section_state("nr7")
    nr7_window = int(nr7_cfg.get("window", 7) or 7)

    gap_cfg, gap_enabled = _section_state("gap")
    gap_threshold = float(gap_cfg.get("threshold", 3.0) or 3.0)

    volume_cfg, volume_enabled = _section_state("volume")
    volume_threshold = float(volume_cfg.get("threshold", 1.5) or 1.5)
    volume_lookback = int(volume_cfg.get("lookback", 63) or 63)

    sr_cfg, sr_enabled = _section_state("sr")
    sr_threshold = float(sr_cfg.get("threshold", 2.0) or 2.0)
    sr_lookback = int(sr_cfg.get("lookback", 63) or 63)

    new_high_cfg, new_high_enabled = _section_state("new_high")
    new_high_windows = new_high_cfg.get("windows") or [20, 63]
    new_high_windows = [int(w) for w in new_high_windows if w]
    if not new_high_windows:
        new_high_windows = [20, 63]

    pct_threshold_val = float(pct_threshold or 0.0) if spike_mode == "pct" else 0.0
    atr_multiple_val = float(atr_multiple or 0.0) if spike_mode == "atr" else 0.0

    enabled_families = [
        name
        for name, enabled in [
            ("trend", trend_enabled),
            ("rsi", rsi_enabled),
            ("atr_squeeze", atr_squeeze_enabled),
            ("bb", bb_enabled),
            ("nr7", nr7_enabled),
            ("gap", gap_enabled),
            ("volume", volume_enabled),
            ("sr", sr_enabled),
            ("new_high", new_high_enabled),
        ]
        if enabled
    ]

    pad_days = lookback_days + max(
        atr_window,
        volume_filter_lookback if volume_filter_enabled else 0,
        sr_lookback if sr_enabled else 0,
        volume_lookback if volume_enabled else 0,
        bb_pct_window if bb_enabled else 0,
        atr_pct_window if atr_squeeze_enabled else 0,
        max(new_high_windows) if new_high_enabled and new_high_windows else 0,
    )
    pad_days = max(pad_days, lookback_days + 5)
    fetch_start = (start_ts - BDay(pad_days)).tz_localize(None)

    if debug and hasattr(debug, "log_event"):
        try:
            debug.log_event(
                "preload_prices:start",
                tickers=len(universe_list),
                start=str(fetch_start.date()),
                end=str(end_ts.date()),
            )
        except Exception:
            pass

    prices_df = load_prices_cached(
        storage,
        cache_salt=storage.cache_salt(),
        tickers=list(universe_list),
        start=fetch_start,
        end=end_ts,
    )

    if debug and hasattr(debug, "log_event"):
        try:
            debug.log_event(
                "preload_prices:done",
                rows=int(len(prices_df)),
                tickers=len(universe_list),
            )
        except Exception:
            pass

    if prices_df.empty:
        empty = pd.DataFrame(columns=["ticker", "spike_date"])
        summary = {
            "counts": {
                "spikes": 0,
                "tickers": 0,
                "start": start_ts,
                "end": end_ts,
            },
            "frequency_table": pd.DataFrame(
                columns=["flag", "hit_count", "hit_rate", "median_lead_days"]
            ),
            "combos": pd.DataFrame(columns=["combo", "count", "lift"]),
            "flag_metadata": {},
            "lookback_days": lookback_days,
        }
        return empty, summary

    prices_df["date"] = pd.to_datetime(prices_df["date"], errors="coerce").dt.tz_localize(
        None
    )
    prices_df = prices_df.dropna(subset=["date"])

    available_start = prices_df["date"].min()
    available_end = prices_df["date"].max()
    if debug and hasattr(debug, "log_event"):
        try:
            debug.log_event(
                "coverage",
                available_start=str(pd.Timestamp(available_start).date())
                if pd.notna(available_start)
                else None,
                available_end=str(pd.Timestamp(available_end).date())
                if pd.notna(available_end)
                else None,
                requested_start=str(start_ts.date()),
                requested_end=str(end_ts.date()),
            )
        except Exception:
            pass

    spike_rows: list[dict[str, Any]] = []
    bool_columns: set[str] = set()
    lead_columns: dict[str, str] = {}
    flag_metadata: dict[str, dict[str, Any]] = {}

    if debug and hasattr(debug, "log_event"):
        try:
            debug.log_event(
                "config_gates",
                pct_mode=spike_mode == "pct",
                atr_mode=spike_mode == "atr",
                indicators=enabled_families,
                volume_filter=bool(volume_filter_enabled),
            )
        except Exception:
            pass

    selected_flags: dict[str, Any] = {}

    for ticker, frame in prices_df.groupby("Ticker"):
        ticker = str(ticker).upper()
        try:
            panel = _normalize_prices_df(frame, ticker)
        except Exception:
            if debug and hasattr(debug, "log_event"):
                try:
                    debug.log_event("panel_error", ticker=ticker)
                except Exception:
                    pass
            continue

        panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.tz_localize(None)
        panel = panel.dropna(subset=["date"])
        panel = panel.sort_values("date")
        panel = panel.drop_duplicates(subset=["date"], keep="last")
        panel = panel.set_index("date", drop=False)
        panel.index = panel.index.tz_localize(None)
        panel.index = panel.index.normalize()
        panel = panel[~panel.index.duplicated(keep="last")]

        panel = panel.loc[(panel.index >= fetch_start) & (panel.index <= end_ts)].copy()
        if panel.empty:
            continue

        close = panel["close"].astype(float)
        prev_close = close.shift(1)
        panel["pct_change"] = (close / prev_close - 1.0) * 100.0

        atr_series = compute_atr(panel[["high", "low", "close"]], window=atr_window, method=atr_method)
        panel["atr"] = atr_series
        panel["atr_prev"] = panel["atr"].shift(1)
        panel["atr_multiple"] = np.where(
            panel["atr_prev"].abs() > 1e-9,
            (close - prev_close) / panel["atr_prev"],
            np.nan,
        )

        if volume_filter_enabled:
            trailing = panel["volume"].shift(1).rolling(
                volume_filter_lookback, min_periods=min(10, volume_filter_lookback)
            ).mean()
            panel["volume_trailing_avg"] = trailing
            panel["volume_multiple_spike"] = np.where(
                trailing > 0, panel["volume"] / trailing, np.nan
            )
        else:
            panel["volume_trailing_avg"] = np.nan
            panel["volume_multiple_spike"] = np.nan

        if trend_enabled:
            ema_fast = close.ewm(span=trend_fast, adjust=False, min_periods=trend_fast).mean()
            ema_slow = close.ewm(span=trend_slow, adjust=False, min_periods=trend_slow).mean()
            panel["ema_fast"] = ema_fast
            panel["ema_slow"] = ema_slow
            panel["ema_cross_up"] = (
                (ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))
            )

        if rsi_enabled:
            delta = close.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            roll = rsi_period
            avg_gain = gain.ewm(alpha=1 / roll, adjust=False, min_periods=roll).mean()
            avg_loss = loss.ewm(alpha=1 / roll, adjust=False, min_periods=roll).mean()
            rs = avg_gain / avg_loss.replace({0: np.nan})
            rsi = 100 - (100 / (1 + rs))
            panel["rsi"] = rsi
            for level in rsi_levels:
                key = f"rsi_cross_{int(level)}"
                panel[key] = (rsi >= level) & (rsi.shift(1) < level)

        if atr_squeeze_enabled:
            panel["atr_percentile_rank"] = _rolling_percentile(panel["atr"], atr_pct_window)
            panel["atr_squeeze_flag"] = panel["atr_percentile_rank"] <= atr_pct_threshold

        if bb_enabled:
            mid = close.rolling(bb_period, min_periods=bb_period).mean()
            std = close.rolling(bb_period, min_periods=bb_period).std(ddof=0)
            upper = mid + 2 * std
            lower = mid - 2 * std
            width = (upper - lower) / mid.replace({0: np.nan})
            panel["bb_width"] = width
            panel["bb_width_percentile"] = _rolling_percentile(width, bb_pct_window)
            panel["bb_squeeze_flag"] = panel["bb_width_percentile"] <= bb_percentile

        if nr7_enabled:
            true_range = (panel["high"] - panel["low"]).astype(float)
            panel["nr7_flag"] = true_range <= true_range.rolling(nr7_window).min()

        if gap_enabled:
            panel["gap_pct"] = (panel["open"] / prev_close - 1.0) * 100.0

        if volume_enabled:
            vol_avg = panel["volume"].rolling(
                volume_lookback, min_periods=min(10, volume_lookback)
            ).mean()
            panel["volume_multiple"] = np.where(vol_avg > 0, panel["volume"] / vol_avg, np.nan)

        if sr_enabled:
            support = panel["low"].rolling(sr_lookback, min_periods=sr_lookback).min()
            resistance = panel["high"].rolling(sr_lookback, min_periods=sr_lookback).max()
            price = close
            denom = price - support
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = (resistance - price) / denom
            ratio = ratio.where((support < price) & (resistance > price))
            panel["sr_ratio"] = ratio

        if new_high_enabled:
            for window in new_high_windows:
                rolling_max = close.shift(1).rolling(window, min_periods=window).max()
                key = f"new_high_{window}"
                panel[key] = close >= rolling_max

        if panel.empty:
            continue

        spike_mask = pd.Series(False, index=panel.index)
        if spike_mode == "pct":
            spike_mask = panel["pct_change"] >= pct_threshold_val
        else:
            diff = close - prev_close
            spike_mask = diff >= (atr_multiple_val * panel["atr_prev"])

        spike_mask &= panel.index.to_series().between(start_ts, end_ts)
        spike_mask &= prev_close.notna()

        if volume_filter_enabled:
            spike_mask &= (
                panel["volume_multiple_spike"] >= volume_filter_threshold
            )

        spike_dates = panel.index[spike_mask]
        if not len(spike_dates):
            continue

        for spike_date in spike_dates:
            spike_row: dict[str, Any] = {
                "ticker": ticker,
                "spike_date": spike_date,
                "close": float(panel.loc[spike_date, "close"]),
                "pct_change_t": float(panel.loc[spike_date, "pct_change"]),
                "atr_mult_t": float(panel.loc[spike_date, "atr_multiple"]),
                "volume_multiple_t": float(
                    panel.loc[spike_date, "volume_multiple_spike"]
                )
                if volume_filter_enabled
                else np.nan,
            }

            window_start = _business_start(spike_date, lookback_days)
            window_df = panel.loc[(panel.index >= window_start) & (panel.index < spike_date)]

            lead_candidates: list[float] = []
            active_flags: list[str] = []

            if trend_enabled and "ema_cross_up" in panel:
                flag_col = f"ema_cross_{trend_fast}_{trend_slow}_any"
                lead_col = f"ema_cross_{trend_fast}_{trend_slow}_lead_days"
                crosses = window_df.index[window_df.get("ema_cross_up", False)]
                hit = len(crosses) > 0
                spike_row[flag_col] = bool(hit)
                if hit:
                    leads = [int((spike_date - d).days) for d in crosses]
                    lead = float(min(leads)) if leads else float("nan")
                    spike_row[lead_col] = lead
                    lead_candidates.append(lead)
                    active_flags.append(flag_col)
                else:
                    spike_row[lead_col] = float("nan")
                bool_columns.add(flag_col)
                lead_columns[flag_col] = lead_col
                flag_metadata.setdefault(
                    flag_col,
                    {
                        "type": "ema_cross",
                        "fast": trend_fast,
                        "slow": trend_slow,
                    },
                )

            if rsi_enabled and "rsi" in panel:
                spike_row["rsi14_max"] = float(window_df["rsi"].max()) if not window_df.empty else np.nan
                for level in rsi_levels:
                    flag_col = f"rsi_{rsi_period}_cross_{int(level)}"
                    lead_col = f"{flag_col}_lead_days"
                    cross_col = f"rsi_cross_{int(level)}"
                    if cross_col not in panel:
                        continue
                    crosses = window_df.index[window_df.get(cross_col, False)]
                    hit = len(crosses) > 0
                    spike_row[flag_col] = bool(hit)
                    if hit:
                        leads = [int((spike_date - d).days) for d in crosses]
                        lead = float(min(leads)) if leads else float("nan")
                        spike_row[lead_col] = lead
                        lead_candidates.append(lead)
                        active_flags.append(flag_col)
                    else:
                        spike_row[lead_col] = float("nan")
                    bool_columns.add(flag_col)
                    lead_columns[flag_col] = lead_col
                    flag_metadata.setdefault(
                        flag_col,
                        {
                            "type": "rsi_above",
                            "period": rsi_period,
                            "level": level,
                        },
                    )

            if atr_squeeze_enabled and "atr_squeeze_flag" in panel:
                flag_col = f"atr_squeeze_le_{atr_pct_threshold:g}p"
                lead_col = f"{flag_col}_lead_days"
                hits = window_df.index[window_df.get("atr_squeeze_flag", False)]
                hit = len(hits) > 0
                spike_row[flag_col] = bool(hit)
                spike_row["atr_pctl_min"] = (
                    float(window_df["atr_percentile_rank"].min())
                    if not window_df.empty
                    else np.nan
                )
                if hit:
                    leads = [int((spike_date - d).days) for d in hits]
                    lead = float(min(leads)) if leads else float("nan")
                    spike_row[lead_col] = lead
                    lead_candidates.append(lead)
                    active_flags.append(flag_col)
                else:
                    spike_row[lead_col] = float("nan")
                bool_columns.add(flag_col)
                lead_columns[flag_col] = lead_col
                flag_metadata.setdefault(
                    flag_col,
                    {
                        "type": "squeeze",
                        "measure": "atr_percentile",
                        "threshold": atr_pct_threshold,
                    },
                )

            if bb_enabled and "bb_squeeze_flag" in panel:
                flag_col = f"bb_squeeze_le_{bb_percentile:g}p"
                lead_col = f"{flag_col}_lead_days"
                hits = window_df.index[window_df.get("bb_squeeze_flag", False)]
                hit = len(hits) > 0
                spike_row[flag_col] = bool(hit)
                spike_row["bb_width_pctl_min"] = (
                    float(window_df["bb_width_percentile"].min())
                    if not window_df.empty
                    else np.nan
                )
                if hit:
                    leads = [int((spike_date - d).days) for d in hits]
                    lead = float(min(leads)) if leads else float("nan")
                    spike_row[lead_col] = lead
                    lead_candidates.append(lead)
                    active_flags.append(flag_col)
                else:
                    spike_row[lead_col] = float("nan")
                bool_columns.add(flag_col)
                lead_columns[flag_col] = lead_col
                flag_metadata.setdefault(
                    flag_col,
                    {
                        "type": "squeeze",
                        "measure": "bb_width_percentile",
                        "threshold": bb_percentile,
                    },
                )

            if nr7_enabled and "nr7_flag" in panel:
                flag_col = "nr7_any"
                lead_col = "nr7_lead_days"
                hits = window_df.index[window_df.get("nr7_flag", False)]
                hit = len(hits) > 0
                spike_row[flag_col] = bool(hit)
                spike_row["nr7_count"] = int(window_df.get("nr7_flag", False).sum())
                if hit:
                    leads = [int((spike_date - d).days) for d in hits]
                    lead = float(min(leads)) if leads else float("nan")
                    spike_row[lead_col] = lead
                    lead_candidates.append(lead)
                    active_flags.append(flag_col)
                else:
                    spike_row[lead_col] = float("nan")
                bool_columns.add(flag_col)
                lead_columns[flag_col] = lead_col
                flag_metadata.setdefault(flag_col, {"type": "pattern", "name": "nr7"})

            if gap_enabled and "gap_pct" in panel:
                flag_col = f"gap_prior_ge_{gap_threshold:g}pct"
                lead_col = f"{flag_col}_lead_days"
                hits = window_df.index[window_df.get("gap_pct", 0.0) >= gap_threshold]
                hit = len(hits) > 0
                spike_row[flag_col] = bool(hit)
                if hit:
                    leads = [int((spike_date - d).days) for d in hits]
                    lead = float(min(leads)) if leads else float("nan")
                    spike_row[lead_col] = lead
                    lead_candidates.append(lead)
                    active_flags.append(flag_col)
                else:
                    spike_row[lead_col] = float("nan")
                bool_columns.add(flag_col)
                lead_columns[flag_col] = lead_col
                flag_metadata.setdefault(
                    flag_col,
                    {"type": "gap", "threshold_pct": gap_threshold},
                )

            if volume_enabled and "volume_multiple" in panel:
                day1 = panel.index.get_loc(spike_date) - 1
                day2 = panel.index.get_loc(spike_date) - 2
                if day1 >= 0:
                    prev_day = panel.index[day1]
                    mult = panel.iloc[day1]["volume_multiple"]
                    flag_col = f"vol_day1_ge_{volume_threshold:g}x"
                    lead_col = f"{flag_col}_lead_days"
                    hit = float(mult) >= volume_threshold if pd.notna(mult) else False
                    spike_row[flag_col] = bool(hit)
                    spike_row[lead_col] = 1.0 if hit else float("nan")
                    if hit:
                        lead_candidates.append(1.0)
                        active_flags.append(flag_col)
                    bool_columns.add(flag_col)
                    lead_columns[flag_col] = lead_col
                    flag_metadata.setdefault(
                        flag_col,
                        {
                            "type": "volume_multiple",
                            "days_ago": 1,
                            "threshold": volume_threshold,
                        },
                    )
                if day2 >= 0:
                    prev2 = panel.index[day2]
                    mult2 = panel.iloc[day2]["volume_multiple"]
                    flag_col = f"vol_day2_ge_{volume_threshold:g}x"
                    lead_col = f"{flag_col}_lead_days"
                    hit2 = float(mult2) >= volume_threshold if pd.notna(mult2) else False
                    spike_row[flag_col] = bool(hit2)
                    spike_row[lead_col] = 2.0 if hit2 else float("nan")
                    if hit2:
                        lead_candidates.append(2.0)
                        active_flags.append(flag_col)
                    bool_columns.add(flag_col)
                    lead_columns[flag_col] = lead_col
                    flag_metadata.setdefault(
                        flag_col,
                        {
                            "type": "volume_multiple",
                            "days_ago": 2,
                            "threshold": volume_threshold,
                        },
                    )

            if sr_enabled and "sr_ratio" in panel:
                flag_col = f"sr_ratio_ge_{sr_threshold:g}"
                lead_col = f"{flag_col}_lead_days"
                hits = window_df.index[window_df.get("sr_ratio", 0.0) >= sr_threshold]
                hit = len(hits) > 0
                spike_row[flag_col] = bool(hit)
                spike_row["sr_ratio_max"] = (
                    float(window_df["sr_ratio"].max()) if not window_df.empty else np.nan
                )
                if hit:
                    leads = [int((spike_date - d).days) for d in hits]
                    lead = float(min(leads)) if leads else float("nan")
                    spike_row[lead_col] = lead
                    lead_candidates.append(lead)
                    active_flags.append(flag_col)
                else:
                    spike_row[lead_col] = float("nan")
                bool_columns.add(flag_col)
                lead_columns[flag_col] = lead_col
                flag_metadata.setdefault(
                    flag_col,
                    {"type": "sr_ratio", "threshold": sr_threshold},
                )

            if new_high_enabled:
                for window in new_high_windows:
                    key = f"new_high_{window}"
                    if key not in panel:
                        continue
                    flag_col = f"{key}_any"
                    lead_col = f"{flag_col}_lead_days"
                    hits = window_df.index[window_df.get(key, False)]
                    hit = len(hits) > 0
                    spike_row[flag_col] = bool(hit)
                    if hit:
                        leads = [int((spike_date - d).days) for d in hits]
                        lead = float(min(leads)) if leads else float("nan")
                        spike_row[lead_col] = lead
                        lead_candidates.append(lead)
                        active_flags.append(flag_col)
                    else:
                        spike_row[lead_col] = float("nan")
                    bool_columns.add(flag_col)
                    lead_columns[flag_col] = lead_col
                    flag_metadata.setdefault(
                        flag_col,
                        {"type": "new_high", "window": window},
                    )

            spike_row["lead_time_first_precursor"] = (
                float(min(lead_candidates)) if lead_candidates else float("nan")
            )

            spike_rows.append(spike_row)

            if debug and hasattr(debug, "log_event"):
                try:
                    debug.log_event(
                        "spike_detected",
                        ticker=ticker,
                        date=str(spike_date.date()),
                        pct=float(spike_row["pct_change_t"]),
                        atr_mult=float(spike_row["atr_mult_t"]),
                    )
                    if active_flags:
                        debug.log_event(
                            "precursor_flags",
                            ticker=ticker,
                            date=str(spike_date.date()),
                            flags=active_flags,
                        )
                except Exception:
                    pass

    if not spike_rows:
        empty_df = pd.DataFrame(columns=["ticker", "spike_date"])
        summary = {
            "counts": {
                "spikes": 0,
                "tickers": len(universe_list),
                "start": start_ts,
                "end": end_ts,
            },
            "frequency_table": pd.DataFrame(
                columns=["flag", "hit_count", "hit_rate", "median_lead_days"]
            ),
            "combos": pd.DataFrame(columns=["combo", "count", "lift"]),
            "flag_metadata": flag_metadata,
            "lookback_days": lookback_days,
        }
        return empty_df, summary

    spikes_df = pd.DataFrame(spike_rows)
    spikes_df = spikes_df.sort_values(["spike_date", "ticker"]).reset_index(drop=True)

    for col in bool_columns:
        if col in spikes_df.columns:
            spikes_df[col] = spikes_df[col].fillna(False).astype(bool)

    frequency_rows = []
    total_spikes = len(spikes_df)

    for flag in sorted(bool_columns):
        if flag not in spikes_df.columns:
            continue
        hits_df = spikes_df[spikes_df[flag]]
        hit_count = int(hits_df.shape[0])
        hit_rate = float(hit_count / total_spikes) if total_spikes else 0.0
        lead_col = lead_columns.get(flag)
        median_lead = (
            float(hits_df[lead_col].median())
            if lead_col and lead_col in hits_df.columns and not hits_df.empty
            else float("nan")
        )
        frequency_rows.append(
            {
                "flag": flag,
                "hit_count": hit_count,
                "hit_rate": hit_rate,
                "median_lead_days": median_lead,
            }
        )

    frequency_table = pd.DataFrame(frequency_rows).sort_values(
        ["hit_rate", "hit_count"], ascending=[False, False]
    )

    lead_stats = {}
    for flag, lead_col in lead_columns.items():
        if lead_col in spikes_df.columns:
            valid = spikes_df.loc[spikes_df[flag], lead_col].dropna()
            if not valid.empty:
                lead_stats[flag] = {
                    "min": float(valid.min()),
                    "median": float(valid.median()),
                    "mean": float(valid.mean()),
                    "max": float(valid.max()),
                }

    combos_df = pd.DataFrame(columns=["combo", "count", "lift"])
    if total_spikes <= 5000 and len(bool_columns) >= 2:
        records = []
        flags_sorted = sorted([f for f in bool_columns if f in spikes_df.columns])
        base_rates = {f: float(spikes_df[f].mean()) for f in flags_sorted}
        for i, flag_a in enumerate(flags_sorted):
            for flag_b in flags_sorted[i + 1 :]:
                pair_mask = spikes_df[flag_a] & spikes_df[flag_b]
                count = int(pair_mask.sum())
                if count == 0:
                    continue
                rate = count / total_spikes if total_spikes else 0.0
                expected = base_rates.get(flag_a, 0.0) * base_rates.get(flag_b, 0.0)
                lift = rate / expected if expected > 0 else np.nan
                records.append(
                    {
                        "combo": f"{flag_a} + {flag_b}",
                        "count": count,
                        "lift": float(lift) if np.isfinite(lift) else np.nan,
                    }
                )
        if records:
            combos_df = pd.DataFrame(records).sort_values(
                "count", ascending=False
            ).head(10)

    summary = {
        "counts": {
            "spikes": total_spikes,
            "tickers": int(spikes_df["ticker"].nunique()),
            "start": start_ts,
            "end": end_ts,
        },
        "frequency_table": frequency_table.reset_index(drop=True),
        "combos": combos_df.reset_index(drop=True),
        "lead_stats": lead_stats,
        "flag_metadata": flag_metadata,
        "lookback_days": lookback_days,
    }

    if debug and hasattr(debug, "log_event"):
        try:
            debug.log_event(
                "summary:done",
                spikes=total_spikes,
                bool_flags=len(bool_columns),
            )
        except Exception:
            pass

    return spikes_df, summary

