from __future__ import annotations

import datetime as dt
import json
import logging
import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Literal, TypedDict

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from data_lake.membership import load_membership
from data_lake.storage import Storage, load_prices_cached
from engine.features import atr as compute_atr
from engine.replay import replay_trade
from engine.scan_shared.indicators import IndicatorConfig
from engine.scan_shared.precursor_flags import (
    DEFAULT_PARAMS as PRECURSOR_DEFAULTS,
    FLAG_COLUMNS as PRECURSOR_FLAG_COLUMNS,
    METRIC_COLUMNS as PRECURSOR_METRIC_COLUMNS,
    build_precursor_flags,
)

log = logging.getLogger(__name__)

DEFAULT_CASH_CAP = 1_000.0
_PRECURSOR_CONFIG = IndicatorConfig()
_PRECURSOR_MAX_LOOKBACK = _PRECURSOR_CONFIG.max_lookback
_PRECURSOR_EVENT_OFFSETS = {
    "vol_mult_d1": 1,
    "vol_mult_d1_ge_x": 1,
    "vol_mult_d2": 2,
    "vol_mult_d2_ge_x": 2,
}


class PrecursorCondition(TypedDict, total=False):
    flag: str
    max_percentile: float
    min_gap_pct: float
    min_mult: float


class PrecursorsParams(TypedDict, total=False):
    enabled: bool
    within_days: int
    logic: Literal["ANY", "ALL"]
    conditions: list[PrecursorCondition]
    atr_pct_threshold: float
    bb_pct_threshold: float
    gap_min_pct: float
    vol_min_mult: float


class StocksOnlyScanParams(TypedDict, total=False):
    start: pd.Timestamp | dt.date | str
    end: pd.Timestamp | dt.date | str
    horizon_days: int
    sr_lookback: int
    sr_min_ratio: float
    min_yup_pct: float
    min_gap_pct: float
    min_volume_multiple: float
    volume_lookback: int
    exit_model: Literal["atr", "sr"]
    atr_window: int
    atr_method: str
    tp_atr_multiple: float
    sl_atr_multiple: float
    use_sp_filter: bool
    cash_per_trade: float
    precursors: PrecursorsParams


@dataclass
class ScanSummary:
    start: pd.Timestamp
    end: pd.Timestamp
    tickers_scanned: int
    candidates: int
    trades: int
    wins: int
    total_capital: float
    total_pnl: float
    win_rate: float


def _log_event(event: str, **fields: object) -> None:
    payload = {"event": event, **fields}
    try:
        log.info("stocks_only_scanner %s", json.dumps(payload, default=str))
    except TypeError:
        log.info("stocks_only_scanner %s", payload)


def _normalize_timestamp(value: pd.Timestamp | dt.date | str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def _compute_padding(params: StocksOnlyScanParams) -> int:
    horizon = int(params.get("horizon_days", 30) or 30)
    padding = max(
        int(params.get("sr_lookback", 21) or 21),
        int(params.get("volume_lookback", 20) or 20),
        int(params.get("atr_window", 14) or 14),
        3,
    )
    precursors = params.get("precursors")
    precursor_padding = 0
    if isinstance(precursors, dict) and precursors.get("enabled"):
        within_days = int(
            precursors.get("within_days", PRECURSOR_DEFAULTS["lookback_days"]) or PRECURSOR_DEFAULTS["lookback_days"]
        )
        precursor_padding = within_days + _PRECURSOR_MAX_LOOKBACK
    return max(padding, precursor_padding) + horizon + 5


def _build_membership_index(
    members: pd.DataFrame | None,
) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp | None]]]:
    if members is None or members.empty:
        return {}
    working = members.copy()
    working["ticker"] = working["ticker"].astype(str).str.upper().str.strip()
    working["start_date"] = pd.to_datetime(working["start_date"], errors="coerce").dt.tz_localize(None)
    if "end_date" in working.columns:
        working["end_date"] = pd.to_datetime(working["end_date"], errors="coerce").dt.tz_localize(None)
    else:
        working["end_date"] = pd.NaT

    index: dict[str, list[tuple[pd.Timestamp, pd.Timestamp | None]]] = {}
    for row in working.itertuples(index=False):
        start = getattr(row, "start_date")
        end = getattr(row, "end_date", pd.NaT)
        end_ts: pd.Timestamp | None
        if pd.isna(end):
            end_ts = None
        else:
            end_ts = pd.Timestamp(end).tz_localize(None)
        ticker = getattr(row, "ticker")
        if pd.isna(start) or not ticker:
            continue
        start_ts = pd.Timestamp(start).tz_localize(None)
        index.setdefault(ticker, []).append((start_ts, end_ts))
    return index


def _is_member(
    index: dict[str, list[tuple[pd.Timestamp, pd.Timestamp | None]]],
    ticker: str,
    day: pd.Timestamp,
) -> bool:
    intervals = index.get(ticker)
    if not intervals:
        return False
    for start, end in intervals:
        if start is None:
            continue
        if start <= day and (end is None or day <= end):
            return True
    return False


def _normalize_prices_df(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
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
    working["date"] = pd.to_datetime(working["date"], errors="coerce").dt.tz_localize(None)
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
    working = working.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return working


def _prepare_panel(df: pd.DataFrame, params: StocksOnlyScanParams, *, ticker: str) -> pd.DataFrame:
    sr_lookback = int(params.get("sr_lookback", 21) or 21)
    volume_lookback = int(params.get("volume_lookback", 20) or 20)
    atr_window = int(params.get("atr_window", 14) or 14)
    atr_method = str(params.get("atr_method", "wilder") or "wilder").lower()

    panel = _normalize_prices_df(df, ticker)
    panel = panel.assign(ticker=ticker)

    panel = panel.sort_values("date").reset_index(drop=True)

    close = panel["close"].astype(float)
    panel["yesterday_up_pct"] = (close.shift(1) / close.shift(2) - 1.0) * 100.0
    panel["open_gap_pct"] = (panel["open"] / close.shift(1) - 1.0) * 100.0

    vol = panel["volume"].astype(float)
    vol_sma = vol.rolling(volume_lookback, min_periods=volume_lookback).mean().shift(1)
    panel["volume_multiple"] = vol / vol_sma

    atr_series = compute_atr(panel[["high", "low", "close"]], window=atr_window, method=atr_method)
    panel["atr_value"] = atr_series.shift(1)

    panel["support"] = (
        panel["low"].rolling(sr_lookback, min_periods=sr_lookback).min().shift(1)
    )
    panel["resistance"] = (
        panel["high"].rolling(sr_lookback, min_periods=sr_lookback).max().shift(1)
    )

    denom = panel["open"] - panel["support"]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (panel["resistance"] - panel["open"]) / denom
    ratio = ratio.where((panel["support"] < panel["open"]) & (panel["resistance"] > panel["open"]))
    panel["sr_ratio"] = ratio

    panel = panel.set_index("date", drop=False)
    return panel


def _compute_shares(entry_price: float, cash_cap: float) -> int:
    if entry_price <= 0 or cash_cap <= 0:
        return 0
    if entry_price > cash_cap:
        return 0
    shares = int(math.floor(cash_cap / entry_price))
    return shares if shares >= 1 else 0


def _sr_ratio_ok(entry_price: float, support: float, resistance: float, min_ratio: float) -> bool:
    if any(math.isnan(x) for x in [entry_price, support, resistance, min_ratio]):
        return False
    if support >= entry_price or resistance <= entry_price:
        return False
    down = entry_price - support
    up = resistance - entry_price
    if down <= 0:
        return False
    ratio = up / down
    return ratio >= min_ratio


def _filter_rejection_reasons(
    row: pd.Series, params: StocksOnlyScanParams
) -> list[str]:
    reasons: list[str] = []

    min_yup = float(params.get("min_yup_pct", 0.0) or 0.0)
    min_gap = float(params.get("min_gap_pct", 0.0) or 0.0)
    min_vol = float(params.get("min_volume_multiple", 1.0) or 0.0)

    yup = float(row.get("yesterday_up_pct", float("nan")))
    gap = float(row.get("open_gap_pct", float("nan")))
    vol_mult = float(row.get("volume_multiple", float("nan")))

    if not math.isfinite(yup):
        reasons.append("yup_missing")
    elif yup < min_yup:
        reasons.append("yup_below_min")

    if not math.isfinite(gap):
        reasons.append("gap_missing")
    elif gap < min_gap:
        reasons.append("gap_below_min")

    if not math.isfinite(vol_mult):
        reasons.append("volume_missing")
    elif vol_mult < min_vol:
        reasons.append("volume_below_min")

    return reasons


def _passes_filters(row: pd.Series, params: StocksOnlyScanParams) -> bool:
    return not _filter_rejection_reasons(row, params)


def _simulate_exit(
    bars: pd.DataFrame,
    entry_date: pd.Timestamp,
    entry_price: float,
    tp_price: float,
    sl_price: float,
    horizon_days: int,
) -> dict | None:
    result = replay_trade(
        bars[["date", "open", "high", "low", "close"]].copy(),
        entry_date,
        entry_price,
        tp_price,
        sl_price,
        horizon_days=horizon_days,
    )
    reason = result.get("exit_reason")
    if reason == "stop":
        reason = "sl"
    if reason not in {"tp", "sl", "timeout"}:
        return None
    exit_date = result.get("exit_date")
    if pd.isna(exit_date):
        return None
    return {
        "exit_reason": reason,
        "exit_price": float(result.get("exit_price", float("nan"))),
        "exit_date": pd.Timestamp(exit_date).tz_localize(None),
    }


def run_scan(
    params: StocksOnlyScanParams,
    *,
    storage: Storage | None = None,
    prices_by_ticker: dict[str, pd.DataFrame] | None = None,
    membership: pd.DataFrame | None = None,
    progress: Callable[[int, int, str], None] | None = None,
    debug: object | None = None,
) -> tuple[pd.DataFrame, ScanSummary]:
    start_ts = _normalize_timestamp(params.get("start"))
    end_ts = _normalize_timestamp(params.get("end"))
    if start_ts is None or end_ts is None:
        raise ValueError("Both start and end dates are required")
    if end_ts < start_ts:
        raise ValueError("End date must be on or after start date")

    horizon = int(params.get("horizon_days", 30) or 30)
    cash_cap = float(params.get("cash_per_trade", DEFAULT_CASH_CAP) or DEFAULT_CASH_CAP)
    sr_min_ratio_raw = params.get("sr_min_ratio", 2.0)
    try:
        sr_min_ratio = float(2.0 if sr_min_ratio_raw is None else sr_min_ratio_raw)
    except (TypeError, ValueError):
        sr_min_ratio = 2.0
    exit_model = str(params.get("exit_model", "atr") or "atr").lower()
    tp_mult = float(params.get("tp_atr_multiple", 1.0) or 1.0)
    sl_mult = float(params.get("sl_atr_multiple", 1.0) or 1.0)
    use_sp_filter = bool(params.get("use_sp_filter", True))

    raw_precursors = params.get("precursors") or {}
    precursors_enabled = bool(raw_precursors.get("enabled"))
    precursor_within_days = int(
        raw_precursors.get("within_days", PRECURSOR_DEFAULTS["lookback_days"]) or PRECURSOR_DEFAULTS["lookback_days"]
    )
    precursor_logic = str(raw_precursors.get("logic", "ANY") or "ANY").upper()
    if precursor_logic not in {"ANY", "ALL"}:
        precursor_logic = "ANY"
    precursor_conditions: list[PrecursorCondition] = []
    for condition in raw_precursors.get("conditions", []) or []:
        if not isinstance(condition, dict):
            continue
        flag = str(condition.get("flag", "")).strip()
        if not flag:
            continue
        payload: PrecursorCondition = PrecursorCondition(flag=flag)
        if condition.get("max_percentile") is not None:
            payload["max_percentile"] = float(condition.get("max_percentile"))
        if condition.get("min_gap_pct") is not None:
            payload["min_gap_pct"] = float(condition.get("min_gap_pct"))
        if condition.get("min_mult") is not None:
            payload["min_mult"] = float(condition.get("min_mult"))
        precursor_conditions.append(payload)
    if precursors_enabled and not precursor_conditions:
        precursors_enabled = False

    precursor_base_params = {
        key: float(raw_precursors.get(key, PRECURSOR_DEFAULTS[key]))
        for key in PRECURSOR_DEFAULTS
    }

    def _eval_precursor_condition(
        window: pd.DataFrame, flag: str, payload: PrecursorCondition
    ) -> tuple[bool, list[pd.Timestamp]]:
        if window.empty:
            return False, []

        def _series(name: str) -> pd.Series | None:
            if name not in window.columns:
                return None
            return window[name]

        mask: pd.Series | None = None
        if flag == "atr_squeeze_pct":
            series = _series("atr_pctile")
            threshold = float(payload.get("max_percentile", precursor_base_params["atr_pct_threshold"]))
            if series is not None:
                mask = series <= threshold
        elif flag == "bb_squeeze_pct":
            series = _series("bb_width_pctile")
            threshold = float(payload.get("max_percentile", precursor_base_params["bb_pct_threshold"]))
            if series is not None:
                mask = series <= threshold
        elif flag == "gap_up_ge_gpct_prev":
            series = _series("gap_up_pct_prev")
            threshold = float(payload.get("min_gap_pct", precursor_base_params["gap_min_pct"]))
            if series is not None:
                mask = series >= threshold
        elif flag == "vol_mult_d1_ge_x":
            series = _series("vol_mult_d1")
            threshold = float(payload.get("min_mult", precursor_base_params["vol_min_mult"]))
            if series is not None:
                mask = series >= threshold
        elif flag == "vol_mult_d2_ge_x":
            series = _series("vol_mult_d2")
            threshold = float(payload.get("min_mult", precursor_base_params["vol_min_mult"]))
            if series is not None:
                mask = series >= threshold
        elif flag == "sr_ratio_ge_2":
            series = _series("sr_ratio")
            if series is not None:
                mask = series >= 2.0
        else:
            series = _series(flag)
            if series is not None:
                mask = series.astype(bool)

        if mask is None:
            return False, []
        hits = mask.fillna(False)
        if not bool(hits.any()):
            return False, []
        hit_dates = list(window.index[hits])
        return True, hit_dates

    def _dbg_call(method: str, *args, **kwargs):
        if debug is None:
            return None
        fn = getattr(debug, method, None)
        if callable(fn):
            return fn(*args, **kwargs)
        return None

    storage = storage or Storage()
    members = membership
    if members is None:
        members = load_membership(storage, cache_salt=storage.cache_salt())

    membership_index = _build_membership_index(members)

    padding_days = _compute_padding(params)
    fetch_start = start_ts - BDay(padding_days)
    fetch_end = end_ts + BDay(padding_days)

    price_map: dict[str, pd.DataFrame] = {}
    tickers_to_load: Iterable[str]

    if prices_by_ticker is not None:
        tickers_to_load = list(prices_by_ticker.keys())
        for t, frame in prices_by_ticker.items():
            price_map[str(t).upper()] = _prepare_panel(frame, params, ticker=str(t).upper())
        rows_loaded = sum(len(df) for df in price_map.values()) if price_map else 0
        _dbg_call(
            "log_event",
            "prices_loaded",
            requested=len(tickers_to_load),
            loaded=len(price_map),
            rows=rows_loaded,
        )
    else:
        members_for_range = members.copy() if members is not None else pd.DataFrame()
        if not members_for_range.empty:
            members_for_range["start_date"] = pd.to_datetime(
                members_for_range["start_date"], errors="coerce"
            ).dt.tz_localize(None)
            members_for_range["end_date"] = pd.to_datetime(
                members_for_range.get("end_date"), errors="coerce"
            ).dt.tz_localize(None)
            mask = (members_for_range["start_date"] <= end_ts) & (
                members_for_range["end_date"].isna() | (start_ts <= members_for_range["end_date"])
            )
            filtered = members_for_range.loc[mask]
            tickers_to_load = sorted(filtered["ticker"].astype(str).str.upper().unique())
        else:
            tickers_to_load = []

        requested_unique = (
            len(sorted(members_for_range["ticker"].astype(str).str.upper().unique()))
            if not members_for_range.empty
            else 0
        )
        available_count = len(list(tickers_to_load))
        _dbg_call(
            "log_event",
            "ticker_filter",
            requested=requested_unique,
            available=available_count,
            missing=max(0, requested_unique - available_count),
        )

        if not tickers_to_load:
            empty = pd.DataFrame(
                columns=[
                    "entry_date",
                    "exit_date",
                    "ticker",
                    "entry_price",
                    "tp_price",
                    "sl_price",
                    "exit_price",
                    "exit_reason",
                    "shares",
                    "cost",
                    "proceeds",
                    "pnl",
                ]
            )
            summary = ScanSummary(
                start=start_ts,
                end=end_ts,
                tickers_scanned=0,
                candidates=0,
                trades=0,
                wins=0,
                total_capital=0.0,
                total_pnl=0.0,
                win_rate=0.0,
            )
            return empty, summary

        cache_salt = storage.cache_salt()
        _log_event(
            "preload_prices:start",
            tickers=len(list(tickers_to_load)),
            start=str(fetch_start.date()),
            end=str(fetch_end.date()),
        )
        _dbg_call(
            "log_event",
            "preload_prices:start",
            tickers=len(list(tickers_to_load)),
            start=str(fetch_start.date()),
            end=str(fetch_end.date()),
        )
        prices_df = load_prices_cached(
            storage,
            cache_salt=cache_salt,
            tickers=list(tickers_to_load),
            start=fetch_start,
            end=fetch_end,
        )
        _log_event(
            "preload_prices:done",
            rows=int(len(prices_df)),
            tickers=len(list(tickers_to_load)),
        )
        _dbg_call(
            "log_event",
            "preload_prices:done",
            rows=int(len(prices_df)),
            tickers=len(list(tickers_to_load)),
        )
        if not prices_df.empty:
            available_start = pd.to_datetime(prices_df["date"], errors="coerce").min()
            available_end = pd.to_datetime(prices_df["date"], errors="coerce").max()
            if pd.notna(available_start) and pd.notna(available_end):
                _log_event(
                    "coverage",
                    available_start=str(pd.Timestamp(available_start).date()),
                    available_end=str(pd.Timestamp(available_end).date()),
                    requested_start=str(start_ts.date()),
                    requested_end=str(end_ts.date()),
                )
                _dbg_call(
                    "log_event",
                    "coverage",
                    available_start=str(pd.Timestamp(available_start).date()),
                    available_end=str(pd.Timestamp(available_end).date()),
                    requested_start=str(start_ts.date()),
                    requested_end=str(end_ts.date()),
                )
        for ticker, frame in prices_df.groupby("Ticker"):
            price_map[str(ticker).upper()] = _prepare_panel(frame, params, ticker=str(ticker).upper())

        _dbg_call(
            "log_event",
            "prices_loaded",
            requested=len(list(tickers_to_load)),
            loaded=len(price_map),
            rows=int(len(prices_df)),
        )

    tickers_sorted = sorted(price_map.keys())
    _dbg_call("set_tickers", tickers_sorted)
    _log_event(
        "scan:start",
        start=str(start_ts.date()),
        end=str(end_ts.date()),
        horizon=horizon,
        tickers=len(tickers_sorted),
    )
    _dbg_call(
        "log_event",
        "scan:start",
        start=str(start_ts.date()),
        end=str(end_ts.date()),
        horizon=horizon,
        tickers=len(tickers_sorted),
    )

    ledger_rows: list[dict[str, object]] = []
    candidate_count = 0

    bdays = pd.bdate_range(start_ts, end_ts)
    panel_by_day_cache: dict[str, pd.DataFrame] = {}
    precursor_panel_cache: dict[str, pd.DataFrame] = {}
    scan_timer_start = time.perf_counter()
    _dbg_call(
        "log_event",
        "run_backtest:start",
        tickers=len(tickers_sorted),
        days=len(bdays),
    )
    for idx, ticker in enumerate(tickers_sorted, 1):
        panel = price_map[ticker]
        if ticker in panel_by_day_cache:
            panel_by_day = panel_by_day_cache[ticker]
        else:
            by_day = panel.copy()

            idx = (
                pd.to_datetime(by_day.index, errors="coerce")
                .tz_localize(None)
                .normalize()
            )
            by_day.index = idx

            by_day = by_day[~by_day.index.duplicated(keep="last")]

            panel_by_day_cache[ticker] = by_day
            panel_by_day = by_day

        if panel_by_day.empty:
            continue

        # Ensure the index is not named "date" to avoid ambiguity when creating the bars
        # dataframe that feeds into replay_trade.
        panel_by_day = panel_by_day.rename_axis(None)
        if "date" in panel_by_day.columns:
            panel_by_day = panel_by_day.drop(columns=["date"])

        precursor_panel: pd.DataFrame | None = None
        if precursors_enabled:
            precursor_panel = precursor_panel_cache.get(ticker)
            if precursor_panel is None:
                base_frame = panel.reset_index(drop=True).copy()
                flags_panel, _ = build_precursor_flags(base_frame, params=precursor_base_params)
                keep_cols = [
                    col
                    for col in {"date", *PRECURSOR_FLAG_COLUMNS, *PRECURSOR_METRIC_COLUMNS}
                    if col in flags_panel.columns
                ]
                precursor_panel = flags_panel[keep_cols].copy()
                if "date" not in precursor_panel.columns:
                    precursor_panel["date"] = flags_panel.index
                precursor_panel.index = (
                    pd.to_datetime(precursor_panel.index, errors="coerce")
                    .tz_localize(None)
                    .normalize()
                )
                precursor_panel = precursor_panel[~precursor_panel.index.duplicated(keep="last")]
                precursor_panel_cache[ticker] = precursor_panel

        bars = (
            panel_by_day
            .reset_index(names="date")
            [["date", "open", "high", "low", "close"]]
            .copy()
        )
        if progress is not None:
            progress(idx, len(tickers_sorted), ticker)

        for day in bdays:
            normalized_day = pd.Timestamp(day).tz_localize(None).normalize()
            if use_sp_filter and not _is_member(membership_index, ticker, day):
                continue
            if normalized_day not in panel_by_day.index:
                _dbg_call(
                    "record_rejection",
                    ticker=ticker,
                    date=str(day.date()),
                    reasons=["missing_price"],
                )
                continue
            row = panel_by_day.loc[normalized_day]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            entry_price = float(row.get("open", float("nan")))
            if not math.isfinite(entry_price) or entry_price <= 0:
                _dbg_call(
                    "record_rejection",
                    ticker=ticker,
                    date=str(day.date()),
                    reasons=["invalid_entry_price"],
                )
                continue
            support = float(row.get("support", float("nan")))
            resistance = float(row.get("resistance", float("nan")))
            if not _sr_ratio_ok(entry_price, support, resistance, sr_min_ratio):
                _dbg_call(
                    "record_rejection",
                    ticker=ticker,
                    date=str(day.date()),
                    reasons=["sr_ratio"],
                )
                continue

            filter_reasons = _filter_rejection_reasons(row, params)
            if filter_reasons:
                _dbg_call(
                    "record_rejection",
                    ticker=ticker,
                    date=str(day.date()),
                    reasons=filter_reasons,
                )
                continue

            precursor_hits: list[str] = []
            precursor_last_seen: dict[str, float] = {}
            if precursors_enabled:
                if precursor_panel is None or precursor_panel.empty:
                    _dbg_call(
                        "record_rejection",
                        ticker=ticker,
                        date=str(day.date()),
                        reasons=["precursor_data_missing"],
                    )
                    continue
                window_start = (normalized_day - BDay(precursor_within_days)).normalize()
                window_mask = (precursor_panel.index >= window_start) & (
                    precursor_panel.index <= normalized_day
                )
                window_df = precursor_panel.loc[window_mask]

                condition_results: list[tuple[str, bool, list[pd.Timestamp]]] = []
                for condition in precursor_conditions:
                    flag = condition.get("flag", "")
                    passed, hits = _eval_precursor_condition(window_df, flag, condition)
                    condition_results.append((flag, passed, hits))

                if condition_results:
                    if precursor_logic == "ALL":
                        precursors_ok = all(item[1] for item in condition_results)
                    else:
                        precursors_ok = any(item[1] for item in condition_results)
                else:
                    precursors_ok = True

                if not precursors_ok:
                    _dbg_call(
                        "record_rejection",
                        ticker=ticker,
                        date=str(day.date()),
                        reasons=["precursor_filters"],
                    )
                    continue

                for flag, passed, hits in condition_results:
                    if not passed or not hits:
                        continue
                    precursor_hits.append(flag)
                    leads: list[int] = []
                    offset = _PRECURSOR_EVENT_OFFSETS.get(flag, 0)
                    for hit in hits:
                        if pd.isna(hit):
                            continue
                        hit_ts = pd.Timestamp(hit).tz_localize(None).normalize()
                        event_ts = hit_ts
                        if offset:
                            event_ts = (event_ts - BDay(offset)).normalize()
                            delta = int(
                                np.busday_count(
                                    event_ts.date(), normalized_day.date()
                                )
                            )
                        else:
                            delta = int((normalized_day - event_ts).days)
                        if delta < 0:
                            continue
                        leads.append(delta)
                    if leads:
                        precursor_last_seen[flag] = float(min(leads))

            candidate_count += 1

            unique_precursor_hits = sorted({flag for flag in precursor_hits})
            precursor_score = len(unique_precursor_hits)
            precursor_last_seen_clean = {
                flag: precursor_last_seen[flag]
                for flag in unique_precursor_hits
                if flag in precursor_last_seen
            }

            shares = _compute_shares(entry_price, cash_cap)
            _dbg_call(
                "record_candidate",
                ticker=ticker,
                date=str(day.date()),
                entry_price=float(entry_price),
                shares=int(shares),
                exit_model=exit_model,
            )
            if shares < 1:
                _dbg_call(
                    "record_rejection",
                    ticker=ticker,
                    date=str(day.date()),
                    reasons=["shares_under_cap"],
                )
                continue

            if exit_model == "sr":
                tp_price = resistance
                sl_price = support
            else:
                atr_val = float(row.get("atr_value", float("nan")))
                if not math.isfinite(atr_val) or atr_val <= 0:
                    _dbg_call(
                        "record_rejection",
                        ticker=ticker,
                        date=str(day.date()),
                        reasons=["atr_unavailable"],
                    )
                    continue
                tp_price = entry_price + tp_mult * atr_val
                sl_price = entry_price - sl_mult * atr_val

            if not math.isfinite(tp_price) or not math.isfinite(sl_price):
                _dbg_call(
                    "record_rejection",
                    ticker=ticker,
                    date=str(day.date()),
                    reasons=["invalid_tp_sl"],
                )
                continue
            if sl_price >= entry_price or tp_price <= entry_price:
                _dbg_call(
                    "record_rejection",
                    ticker=ticker,
                    date=str(day.date()),
                    reasons=["invalid_tp_sl"],
                )
                continue

            _log_event(
                "trade_open",
                ticker=ticker,
                entry=str(day.date()),
                entry_price=float(entry_price),
                tp=float(tp_price),
                sl=float(sl_price),
                shares=int(shares),
            )
            _dbg_call(
                "record_trade_open",
                ticker=ticker,
                date=str(day.date()),
                entry_price=float(entry_price),
                tp=float(tp_price),
                sl=float(sl_price),
                shares=int(shares),
            )

            exit_info = _simulate_exit(bars, day, entry_price, tp_price, sl_price, horizon)
            if exit_info is None:
                _dbg_call(
                    "record_rejection",
                    ticker=ticker,
                    date=str(day.date()),
                    reasons=["backtest_failed"],
                )
                continue

            exit_reason = str(exit_info["exit_reason"])
            exit_price = float(exit_info["exit_price"])
            exit_date = pd.Timestamp(exit_info["exit_date"]).tz_localize(None)

            cost = shares * entry_price
            proceeds = shares * exit_price
            pnl = proceeds - cost

            ledger_rows.append(
                {
                    "entry_date": day,
                    "exit_date": exit_date,
                    "ticker": ticker,
                    "entry_price": float(entry_price),
                    "tp_price": float(tp_price),
                    "sl_price": float(sl_price),
                    "exit_price": float(exit_price),
                    "exit_reason": exit_reason,
                    "shares": int(shares),
                    "cost": float(cost),
                    "proceeds": float(proceeds),
                    "pnl": float(pnl),
                    "precursor_flags_hit": unique_precursor_hits if precursors_enabled else [],
                    "precursor_last_seen_days_ago": precursor_last_seen_clean
                    if precursors_enabled
                    else {},
                    "precursor_score": precursor_score if precursors_enabled else 0,
                }
            )

            _log_event(
                "trade_close",
                ticker=ticker,
                exit=str(exit_date.date()),
                exit_reason=exit_reason,
                exit_price=float(exit_price),
                pnl=float(pnl),
            )
            _dbg_call(
                "record_trade_exit",
                ticker=ticker,
                exit_date=str(exit_date.date()),
                exit_reason=exit_reason,
                exit_price=float(exit_price),
                pnl=float(pnl),
            )

    ledger = pd.DataFrame(ledger_rows)
    if not ledger.empty:
        ledger = ledger.sort_values(["entry_date", "ticker"]).reset_index(drop=True)

    trades = int(len(ledger))
    wins = int((ledger["exit_reason"].str.lower() == "tp").sum()) if trades else 0
    total_capital = float(ledger["cost"].sum()) if trades else 0.0
    total_pnl = float(ledger["pnl"].sum()) if trades else 0.0
    win_rate = (wins / trades) if trades else 0.0

    summary = ScanSummary(
        start=start_ts,
        end=end_ts,
        tickers_scanned=len(tickers_sorted),
        candidates=candidate_count,
        trades=trades,
        wins=wins,
        total_capital=total_capital,
        total_pnl=total_pnl,
        win_rate=win_rate,
    )

    _dbg_call(
        "log_event",
        "run_backtest:done",
        ms=int((time.perf_counter() - scan_timer_start) * 1000),
        trades=trades,
        candidates=candidate_count,
    )
    _log_event(
        "bt_stats",
        trades=trades,
        wins=wins,
        total_capital=total_capital,
        total_pnl=total_pnl,
        win_rate=win_rate,
    )
    _dbg_call(
        "log_event",
        "bt_stats",
        trades=trades,
        wins=wins,
        total_capital=total_capital,
        total_pnl=total_pnl,
        win_rate=win_rate,
    )
    _log_event("scan:done", trades=trades, candidates=candidate_count)
    _dbg_call(
        "log_event",
        "scan:done",
        trades=trades,
        candidates=candidate_count,
    )
    _dbg_call(
        "set_counts",
        tickers=len(tickers_sorted),
        candidates=candidate_count,
        trades=trades,
    )

    return ledger, summary


__all__ = [
    "StocksOnlyScanParams",
    "ScanSummary",
    "run_scan",
    "_compute_shares",
    "_sr_ratio_ok",
    "_passes_filters",
    "_simulate_exit",
]
