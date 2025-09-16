from __future__ import annotations

import io
import logging
import datetime as dt
from typing import TypedDict, Tuple, List, Dict, Any, Callable

import numpy as np
import pandas as pd

from data_lake.storage import Storage
import streamlit as st
from .replay import replay_trade, simulate_pct_target_only
from .filters import atr_feasible, precedent_hits_pct_target

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
    atr_window = int(params.get("atr_window", 21))
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
                tp_halfway_pct = m.get("tp_halfway_pct")
                required_pct = tp_halfway_pct
                tp_pct_percent = (
                    float(required_pct) * 100.0
                    if required_pct is not None and not np.isnan(required_pct)
                    else float("nan")
                )
                hits = 0
                prec_ok = True
                tp_value = (
                    float(required_pct)
                    if required_pct is not None and not np.isnan(required_pct)
                    else float("nan")
                )
                if use_precedent:
                    if pd.isna(tp_value):
                        prec_ok = False
                        prec_fail_count += 1
                    else:
                        hits = precedent_hits_pct_target(
                            df_idx,
                            pd.Timestamp(D_ts),
                            tp_value,
                            lookback_bdays=prec_lookback,
                            window_bdays=prec_window,
                        )
                        prec_hit_values.append(int(hits))
                        prec_ok = hits >= min_prec_hits
                        if not prec_ok:
                            prec_fail_count += 1
                atr_ok = atr_feasible(
                    df, dm1, required_pct, atr_window
                ) if (required_pct is not None and not np.isnan(required_pct)) else False
                reasons: List[str] = []
                if use_precedent and not prec_ok:
                    reasons.append("precedent_fail")
                if not atr_ok:
                    reasons.append("atr_insufficient")

                row = {
                    "ticker": t,
                    **m,
                    "precedent_hits": int(hits),
                    "precedent_ok": int(prec_ok),
                    "atr_ok": int(atr_ok),
                    "reasons": ",".join(reasons),
                }
                include = ((not use_precedent) or prec_ok) and (
                    (not use_atr_feasible) or atr_ok
                )
                if include:
                    cand_rows.append(row)

                    if exit_model == "pct_tp_only":
                        if required_pct is None or np.isnan(required_pct):
                            continue
                        entry_open = float(row["entry_open"])
                        price_cols = [
                            col for col in ("open", "high", "low", "close") if col in df_idx.columns
                        ]
                        if not price_cols:
                            continue
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
