from __future__ import annotations

import json
import logging
import datetime as dt
from typing import TypedDict, Tuple, List, Dict, Any, Callable

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from data_lake.storage import Storage
import streamlit as st
from .replay import replay_trade, simulate_pct_target_only
from .filters import atr_feasible
from .utils_precedent import compute_precedent_hit_details, tp_fraction_from_row

log = logging.getLogger(__name__)


class ScanParams(TypedDict, total=False):
    min_close_up_pct: float
    min_vol_multiple: float
    min_gap_open_pct: float
    atr_window: int
    lookback_days: int
    horizon_days: int
    sr_min_ratio: float
    use_precedent: bool
    use_atr_feasible: bool
    precedent_lookback: int
    precedent_window: int
    min_precedent_hits: int
    exit_model: str


@st.cache_data(show_spinner=False, hash_funcs={Storage: lambda _: 0})
def _load_members(storage: Storage, cache_salt: str) -> pd.DataFrame:
    m = storage.read_parquet_df("membership/sp500_members.parquet")
    m["start_date"] = pd.to_datetime(m["start_date"], errors="coerce", utc=True).dt.tz_localize(None)
    m["end_date"] = pd.to_datetime(m["end_date"], errors="coerce", utc=True).dt.tz_localize(None)
    return m


def members_on_date(m: pd.DataFrame, date: dt.date) -> pd.DataFrame:
    D = pd.to_datetime(date)
    start = pd.to_datetime(m["start_date"])
    end = pd.to_datetime(m["end_date"])
    mask = (start <= D) & (end.isna() | (D <= end))
    return m.loc[mask]


def _load_prices(storage: Storage, ticker: str) -> pd.DataFrame | None:
    try:
        df = storage.read_parquet_df(f"prices/{ticker}.parquet")
        if "date" not in df.columns:
            df["date"] = pd.to_datetime(df.get("index") or df.get("Date"))
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        return df[["date", "open", "high", "low", "close", "volume"]].dropna().sort_values("date")
    except Exception as e:
        log.warning("price load failed for %s: %s", ticker, e)
        return None


def _compute_metrics(df: pd.DataFrame, D: dt.date, vol_lookback: int, atr_window: int) -> Dict[str, Any] | None:
    D = pd.to_datetime(D)
    if D not in df["date"].values:
        idx = df["date"].searchsorted(D)
        if idx == 0 or idx >= len(df):
            return None
        D = df["date"].iloc[idx]

    idx = df.index[df["date"] == D]
    if len(idx) == 0:
        return None
    i = idx[0]
    if i == 0:
        return None
    dm1 = i - 1

    close_up_pct = (
        (df.loc[dm1, "close"] / df.loc[dm1 - 1, "close"] - 1.0) * 100
        if dm1 > 0
        else np.nan
    )

    w0 = max(0, dm1 - vol_lookback + 1)
    lookback_vol = df.loc[w0:dm1, "volume"].mean()
    vol_multiple = (
        df.loc[dm1, "volume"] / lookback_vol
        if lookback_vol and not np.isnan(lookback_vol)
        else np.nan
    )

    gap_open_pct = (df.loc[i, "open"] / df.loc[dm1, "close"] - 1.0) * 100

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(atr_window, min_periods=1).mean()
    atr_val = float(atr.loc[dm1]) if dm1 in atr.index else np.nan
    atr_pct = atr_val / df.loc[dm1, "close"] * 100 if df.loc[dm1, "close"] else np.nan

    if dm1 >= atr_window:
        ret = (df.loc[dm1, "close"] / df.loc[dm1 - atr_window, "close"] - 1.0) * 100
    else:
        ret = np.nan

    lo_win = df.loc[max(0, dm1 - atr_window) : dm1, "low"].min()
    hi_win = df.loc[max(0, dm1 - atr_window) : dm1, "high"].max()

    entry = float(df.loc[i, "open"])
    support = float(lo_win)
    resistance = float(hi_win)
    sr_ratio = np.nan
    tp_halfway_pct = np.nan
    if support > 0 and resistance > entry:
        up = resistance - entry
        down = entry - support
        sr_ratio = up / down if down > 0 else np.nan
        tp_halfway_pct = (entry + up / 2) / entry - 1.0

    return {
        "close_up_pct": float(close_up_pct) if not np.isnan(close_up_pct) else np.nan,
        "vol_multiple": float(vol_multiple) if not np.isnan(vol_multiple) else np.nan,
        "gap_open_pct": float(gap_open_pct),
        "atr21": float(atr_val) if not np.isnan(atr_val) else np.nan,
        "atr21_pct": float(atr_pct) if not np.isnan(atr_pct) else np.nan,
        "ret21_pct": float(ret) if not np.isnan(ret) else np.nan,
        "support": support,
        "resistance": resistance,
        "sr_ratio": sr_ratio,
        "tp_halfway_pct": float(tp_halfway_pct) if not np.isnan(tp_halfway_pct) else np.nan,
        "entry_open": entry,
    }


def scan_day(
    storage: Storage,
    D: pd.Timestamp,
    params: ScanParams,
    on_step: Callable[[int, int, str], None] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, int, dict]:
    """
    Returns (candidates_df, outcomes_df, fail_count, debug_info).
    Must produce same columns as the UI page.
    """

    members = _load_members(storage, cache_salt=storage.cache_salt())
    active = members_on_date(members, D.date())["ticker"].dropna().unique().tolist()
    total = len(active)

    vol_lookback = int(params.get("lookback_days", 63))
    atr_window = int(params.get("atr_window", 14))
    min_close = float(params.get("min_close_up_pct", 0.0))
    min_vol = float(params.get("min_vol_multiple", 0.0))
    min_gap = float(params.get("min_gap_open_pct", 0.0))
    horizon = int(params.get("horizon_days", 30))
    sr_min = float(params.get("sr_min_ratio", 2.0))
    use_precedent = bool(params.get("use_precedent", True))
    use_atr_feasible = bool(params.get("use_atr_feasible", True))
    prec_lookback = int(params.get("precedent_lookback", 252))
    prec_window = int(params.get("precedent_window", 21))
    min_prec_hits = int(params.get("min_precedent_hits", 1))
    exit_model = str(params.get("exit_model", "pct_tp_only") or "pct_tp_only")
    exit_model = exit_model.strip().lower()

    cand_rows: List[Dict[str, Any]] = []
    out_rows: List[Dict[str, Any]] = []
    fail_count = 0

    stats = {"universe": len(active), "loaded": 0, "candidates": 0}
    prec_hit_values: List[int] = []
    prec_fail_count = 0

    for idx, t in enumerate(active, 1):
        try:
            df = _load_prices(storage, t)
            if df is None or len(df) < max(vol_lookback, atr_window) + 2:
                fail_count += 1
                continue
            stats["loaded"] += 1

            D_ts = pd.to_datetime(D)
            if D_ts not in df["date"].values:
                idx_loc = df["date"].searchsorted(D_ts)
                if idx_loc == 0 or idx_loc >= len(df):
                    fail_count += 1
                    continue
                D_ts = df["date"].iloc[idx_loc]
            idx_found = df.index[df["date"] == D_ts]
            if len(idx_found) == 0 or idx_found[0] == 0:
                fail_count += 1
                continue
            i = idx_found[0]
            dm1 = i - 1

            m = _compute_metrics(df, D_ts, vol_lookback, atr_window)
            if not m:
                fail_count += 1
                continue

            df_idx = df.set_index("date").sort_index()
            df_idx.index = pd.to_datetime(df_idx.index).tz_localize(None)
            df_idx = df_idx[~df_idx.index.duplicated(keep="last")]

            if (
                (not np.isnan(m["close_up_pct"]) and m["close_up_pct"] >= min_close)
                and (not np.isnan(m["vol_multiple"]) and m["vol_multiple"] >= min_vol)
                and m["gap_open_pct"] >= min_gap
                and (not np.isnan(m["sr_ratio"]) and m["sr_ratio"] >= sr_min)
            ):
                row: Dict[str, Any] = {"ticker": t, **m}
                row["date"] = pd.Timestamp(D_ts).tz_localize(None)
                row["sr_window_used"] = int(atr_window)

                tp_frac = tp_fraction_from_row(
                    row.get("entry_open"),
                    row.get("tp_price_abs_target"),
                    row.get("tp_halfway_pct"),
                    row.get("tp_price_pct_target"),
                )
                tp_frac_valid = (
                    tp_frac is not None and not pd.isna(tp_frac) and float(tp_frac) > 0
                )

                row["tp_frac_used"] = (
                    float(tp_frac) if tp_frac_valid else float("nan")
                )
                row["tp_pct_used"] = (
                    float(tp_frac) * 100.0 if tp_frac_valid else float("nan")
                )

                hits_count = 0
                hits_details: List[Dict[str, object]] = []
                if use_precedent and tp_frac_valid:
                    hits_count, hits_details = compute_precedent_hit_details(
                        prices_one_ticker=df_idx,
                        asof_date=pd.Timestamp(D_ts),
                        tp_frac=float(tp_frac),
                        lookback_bdays=prec_lookback,
                        window_bdays=prec_window,
                    )
                    prec_hit_values.append(int(hits_count))

                row["precedent_hits"] = int(hits_count)
                row["precedent_details_hits"] = json.dumps(
                    hits_details, separators=(",", ":"), ensure_ascii=False
                )
                row["precedent_hit_start_dates"] = ",".join(
                    e.get("date", "") for e in hits_details
                )

                prec_ok_bool = True
                if use_precedent and tp_frac_valid:
                    prec_ok_bool = hits_count >= min_prec_hits
                    if not prec_ok_bool:
                        prec_fail_count += 1
                elif use_precedent and not tp_frac_valid:
                    prec_ok_bool = True

                row["precedent_ok"] = int(prec_ok_bool)

                atr_ok_bool = (
                    bool(atr_feasible(df, dm1, float(tp_frac), atr_window))
                    if tp_frac_valid
                    else False
                )
                row["atr_ok"] = int(atr_ok_bool)

                # ---- Persist ATR numbers for transparency (no logic change) ----
                h = df["high"]
                l = df["low"]
                c = df["close"]

                tr = pd.concat(
                    [
                        (h - l).abs(),
                        (h - c.shift(1)).abs(),
                        (l - c.shift(1)).abs(),
                    ],
                    axis=1,
                ).max(axis=1)
                atr_series = tr.rolling(int(atr_window), min_periods=int(atr_window)).mean()
                atr_value_dm1 = (
                    float(atr_series.iloc[dm1])
                    if dm1 < len(atr_series) and not pd.isna(atr_series.iloc[dm1])
                    else float("nan")
                )

                tp_required_dollars = (
                    float(row["entry_open"]) * float(tp_frac)
                    if tp_frac_valid
                    else float("nan")
                )
                atr_budget_dollars = (
                    atr_value_dm1 * int(atr_window)
                    if not pd.isna(atr_value_dm1)
                    else float("nan")
                )

                row["atr_window"] = int(atr_window)
                row["atr_value_dm1"] = (
                    None if pd.isna(atr_value_dm1) else round(atr_value_dm1, 6)
                )
                row["atr_budget_dollars"] = (
                    None if pd.isna(atr_budget_dollars) else round(atr_budget_dollars, 6)
                )
                row["tp_required_dollars"] = (
                    None if pd.isna(tp_required_dollars) else round(tp_required_dollars, 6)
                )
                # ---------------------------------------------------------------

                reasons: List[str] = []
                if use_precedent and tp_frac_valid and not prec_ok_bool:
                    reasons.append("precedent_fail")
                if not atr_ok_bool:
                    reasons.append("atr_insufficient")
                row["reasons"] = ",".join(reasons)

                include = ((not use_precedent) or prec_ok_bool) and (
                    (not use_atr_feasible) or atr_ok_bool
                )
                if include:
                    cand_rows.append(row)

                    if exit_model == "pct_tp_only":
                        if not tp_frac_valid:
                            continue
                        entry_open = float(row["entry_open"])
                        price_cols = [
                            col for col in ("open", "high", "low", "close") if col in df_idx.columns
                        ]
                        if not price_cols:
                            continue
                        tp_pct_percent = float(tp_frac) * 100.0
                        res = simulate_pct_target_only(
                            df_idx[price_cols],
                            pd.Timestamp(D_ts),
                            entry_open,
                            tp_pct_percent,
                            horizon,
                        )
                        if res is None:
                            continue
                        out_row = {
                            **row,
                            "exit_model": exit_model,
                            "tp_price": res.get("tp_price_abs_target"),
                            **res,
                        }
                        out_rows.append(out_row)
                    else:
                        tp_price = row["entry_open"] * (1 + row["tp_halfway_pct"])
                        stop_price = row["support"]
                        out = replay_trade(
                            df[["date", "open", "high", "low", "close"]],
                            pd.to_datetime(D),
                            row["entry_open"],
                            tp_price,
                            stop_price,
                            horizon_days=horizon,
                        )
                        out_row = {**row, "exit_model": exit_model, "tp_price": tp_price, **out}
                        out_rows.append(out_row)
        finally:
            if on_step:
                try:
                    on_step(idx, total, t)
                except Exception:
                    pass

    cand_df = pd.DataFrame(cand_rows)
    out_df = pd.DataFrame(out_rows)

    checked_rows = out_rows[:50]
    json_match = 0
    peek_violations = 0
    target_pct_mismatch = 0

    for row in checked_rows:
        try:
            details = json.loads(row.get("precedent_details_hits", ""))
            if not isinstance(details, list):
                details = []
        except Exception:
            details = []

        hits_count = int(row.get("precedent_hits", 0) or 0)
        if hits_count == len(details):
            json_match += 1

        D_row = pd.Timestamp(row.get("date")).tz_localize(None) if row.get("date") else None
        tp_frac_row = tp_fraction_from_row(
            row.get("entry_open"),
            row.get("tp_price_abs_target"),
            row.get("tp_halfway_pct"),
            row.get("tp_price_pct_target"),
        )
        tp_frac_row_valid = (
            tp_frac_row is not None and not pd.isna(tp_frac_row) and float(tp_frac_row) > 0
        )
        target_pct_row = float(tp_frac_row) * 100.0 if tp_frac_row_valid else None

        for event in details:
            try:
                start_date = pd.Timestamp(event.get("date"))
            except Exception:
                start_date = None
            days_to_hit = event.get("days_to_hit")
            try:
                days_to_hit_int = int(days_to_hit)
            except Exception:
                days_to_hit_int = None
            if start_date is not None and days_to_hit_int is not None and D_row is not None:
                start_date = start_date.tz_localize(None)
                hit_date = start_date + BDay(days_to_hit_int)
                if hit_date > D_row - BDay(1):
                    peek_violations += 1
            if target_pct_row is not None and event.get("target_pct") is not None:
                try:
                    target_pct_event = float(event.get("target_pct"))
                except Exception:
                    continue
                if abs(target_pct_event - target_pct_row) > 1e-6:
                    target_pct_mismatch += 1

    stats.setdefault("events", [])
    stats["events"].append(
        {
            "event": "precedent_sanity",
            "date": str(pd.Timestamp(D).tz_localize(None).date()),
            "checked": int(len(checked_rows)),
            "json_match": int(json_match),
            "no_peek_violations": int(peek_violations),
            "target_pct_mismatch": int(target_pct_mismatch),
        }
    )

    if out_rows:
        checked_atr_rows = out_rows[:50]
        expected_window = int(params.get("atr_window", 14))
        atr_present = sum(
            1
            for r in checked_atr_rows
            if r.get("atr_value_dm1") is not None
            and r.get("atr_window") == expected_window
        )
        stats["events"].append(
            {
                "event": "atr_sanity",
                "checked": int(len(checked_atr_rows)),
                "present": int(atr_present),
                "window_default": 14,
            }
        )

    if prec_hit_values:
        stats["precedent_hits_min"] = int(min(prec_hit_values))
        stats["precedent_hits_median"] = float(np.median(prec_hit_values))
        stats["precedent_hits_max"] = int(max(prec_hit_values))
    else:
        stats["precedent_hits_min"] = None
        stats["precedent_hits_median"] = None
        stats["precedent_hits_max"] = None
    stats["precedent_fail_count"] = int(prec_fail_count)
    stats["candidates"] = len(cand_df)

    return cand_df, out_df, fail_count, stats
