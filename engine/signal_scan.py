from __future__ import annotations

import json
import logging
import datetime as dt
from typing import TypedDict, Tuple, List, Dict, Any, Callable, Literal

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from data_lake.storage import Storage
import streamlit as st
from .features import atr as compute_atr
from .replay import replay_trade, simulate_pct_target_only
from .filters import atr_feasible
from .utils_precedent import compute_precedent_hit_details, tp_fraction_from_row

log = logging.getLogger(__name__)


class ScanParams(TypedDict, total=False):
    min_close_up_pct: float
    min_vol_multiple: float
    min_gap_open_pct: float
    atr_window: int
    atr_method: str
    lookback_days: int
    horizon_days: int
    sr_min_ratio: float
    sr_lookback: int
    use_precedent: bool
    use_atr_feasible: bool
    precedent_lookback: int
    precedent_window: int
    min_precedent_hits: int
    exit_model: str
    tp_mode: Literal["sr_fraction", "atr_multiple"]
    tp_sr_fraction: float
    tp_atr_multiple: float


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
    """Load one ticker and enforce a clean schema with RangeIndex and unique dates."""
    try:
        df = storage.read_parquet_df(f"prices/{ticker}.parquet")
        if "date" not in df.columns:
            df["date"] = pd.to_datetime(df.get("index") or df.get("Date"))
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

        df = (
            df[["date", "open", "high", "low", "close", "volume"]]
            .dropna()
            .sort_values("date")
        )
        # Deduplicate same-day bars (keep last) then force positional index
        df = df[~df["date"].duplicated(keep="last")].reset_index(drop=True)
        return df
    except Exception as e:
        log.warning("price load failed for %s: %s", ticker, e)
        return None


def _align_date_pos(dates_np: np.ndarray, D: pd.Timestamp) -> int | None:
    """Return the *position* of the first bar at/after D using numpy-only ops."""
    if dates_np.dtype != "datetime64[ns]":
        dates_np = dates_np.astype("datetime64[ns]")
    target = np.datetime64(pd.Timestamp(D).to_pydatetime(), "ns")
    pos = int(np.searchsorted(dates_np, target, side="left"))
    if pos < 0 or pos >= len(dates_np):
        return None
    return pos


def _compute_metrics(
    df: pd.DataFrame,
    D: dt.date,
    vol_lookback: int,
    atr_window: int,
    atr_method: str,
    sr_lookback: int,
) -> Dict[str, Any] | None:
    """Compute all per-ticker metrics at date D using *positional* indexing only."""
    # Safety: positional index and unique dates
    if not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index(drop=True)
    if df["date"].duplicated().any():
        df = df[~df["date"].duplicated(keep="last")].reset_index(drop=True)

    dates_np = pd.to_datetime(df["date"]).values.astype("datetime64[ns]")
    pos = _align_date_pos(dates_np, pd.to_datetime(D))
    if pos is None or pos == 0:
        return None  # no bar at/after D or not enough history for D-1
    i = pos
    dm1 = i - 1

    # Day-over-day close change (D-1 vs D-2)
    close_up_pct = (
        (df["close"].iloc[dm1] / df["close"].iloc[dm1 - 1] - 1.0) * 100.0
        if dm1 > 0 else np.nan
    )

    # Volume multiple over lookback window up to D-1
    w0 = max(0, dm1 - vol_lookback + 1)
    lookback_vol = float(df["volume"].iloc[w0:dm1 + 1].mean())
    vol_multiple = (
        float(df["volume"].iloc[dm1]) / lookback_vol
        if (lookback_vol and not np.isnan(lookback_vol)) else np.nan
    )

    # Gap open (D open vs D-1 close)
    gap_open_pct = (df["open"].iloc[i] / df["close"].iloc[dm1] - 1.0) * 100.0

    # ATR (positional). compute_atr returns a Series aligned to df order
    atr_series = compute_atr(df[["high", "low", "close"]], window=atr_window, method=atr_method)
    atr_val = float(atr_series.iloc[dm1]) if dm1 < len(atr_series) else np.nan
    atr_pct = (atr_val / df["close"].iloc[dm1] * 100.0) if df["close"].iloc[dm1] else np.nan

    # Rolling return over ATR window ending at D-1
    ret = (
        (df["close"].iloc[dm1] / df["close"].iloc[dm1 - atr_window] - 1.0) * 100.0
        if dm1 >= atr_window else np.nan
    )

    # Support/Resistance over sr_lookback ending at D-1
    sr_start = max(0, dm1 - sr_lookback + 1) if sr_lookback and sr_lookback > 0 else 0
    sr_slice = df.iloc[sr_start:dm1 + 1]
    support = float(sr_slice["low"].min()) if not sr_slice.empty else float("nan")
    resistance = float(sr_slice["high"].max()) if not sr_slice.empty else float("nan")

    entry = float(df["open"].iloc[i])
    sr_ratio = np.nan
    if support > 0 and resistance > entry:
        up = resistance - entry
        down = entry - support
        sr_ratio = (up / down) if down > 0 else np.nan

    return {
        "close_up_pct": float(close_up_pct) if not np.isnan(close_up_pct) else np.nan,
        "vol_multiple": float(vol_multiple) if not np.isnan(vol_multiple) else np.nan,
        "gap_open_pct": float(gap_open_pct),
        "atr21": float(atr_val) if not np.isnan(atr_val) else np.nan,
        "atr21_pct": float(atr_pct) if not np.isnan(atr_pct) else np.nan,
        "ret21_pct": float(ret) if not np.isnan(ret) else np.nan,
        "support": support,
        "resistance": resistance,
        "sr_support": support,
        "sr_resistance": resistance,
        "sr_window_len": int(sr_slice.shape[0]) if not sr_slice.empty else 0,
        "sr_ratio": sr_ratio,
        "entry_open": entry,
        "atr_method": atr_method,
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
    atr_method = str(params.get("atr_method", "wilder") or "wilder").strip().lower()
    min_close = float(params.get("min_close_up_pct", 0.0))
    min_vol = float(params.get("min_vol_multiple", params.get("min_volume_multiple", 0.0)))
    min_gap = float(params.get("min_gap_open_pct", 0.0))
    horizon = int(params.get("horizon_days", 30))
    sr_min = float(params.get("sr_min_ratio", 2.0))
    sr_lookback = int(params.get("sr_lookback", 21))
    use_precedent = bool(params.get("use_precedent", True))
    use_atr_feasible = bool(params.get("use_atr_feasible", True))
    prec_lookback = int(params.get("precedent_lookback", 252))
    prec_window = int(params.get("precedent_window", 21))
    min_prec_hits = int(params.get("min_precedent_hits", 1))
    exit_model = str(params.get("exit_model", "pct_tp_only") or "pct_tp_only").strip().lower()
    tp_mode = str(params.get("tp_mode", "sr_fraction") or "sr_fraction").strip().lower()
    if tp_mode not in ("sr_fraction", "atr_multiple"):
        tp_mode = "sr_fraction"

    def _coerce_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    tp_sr_fraction = _coerce_float(params.get("tp_sr_fraction", 0.50), 0.50)
    tp_atr_multiple = _coerce_float(params.get("tp_atr_multiple", 0.50), 0.50)

    cand_rows: List[Dict[str, Any]] = []
    out_rows: List[Dict[str, Any]] = []
    fail_count = 0

    stats = {"universe": len(active), "loaded": 0, "candidates": 0}
    stats.setdefault("events", [])
    prec_hit_values: List[int] = []
    prec_fail_count = 0

    for idx, t in enumerate(active, 1):
        try:
            df = _load_prices(storage, t)
            if df is None or len(df) < max(vol_lookback, atr_window) + 2:
                fail_count += 1
                continue
            stats["loaded"] += 1

            # Align D to first bar at/after D using numpy positions
            dates_np = pd.to_datetime(df["date"]).values.astype("datetime64[ns]")
            pos = _align_date_pos(dates_np, pd.to_datetime(D))
            if pos is None or pos == 0:
                fail_count += 1
                continue
            D_ts = df["date"].iloc[pos]
            i = int(pos)
            dm1 = i - 1

            m = _compute_metrics(
                df, D_ts, vol_lookback, atr_window, atr_method, sr_lookback
            )
            if not m:
                fail_count += 1
                continue

            # Build a time-indexed view for replay/precedent (dedup again to be safe)
            df_idx = (
                df.set_index("date")
                .sort_index()
                [~df.set_index("date").index.duplicated(keep="last")]
            )
            df_idx.index = pd.to_datetime(df_idx.index).tz_localize(None)

            # Filters
            close_ok = (not np.isnan(m["close_up_pct"]) and m["close_up_pct"] >= min_close)
            vol_ok = (not np.isnan(m["vol_multiple"]) and m["vol_multiple"] >= min_vol)
            gap_ok = m["gap_open_pct"] >= min_gap
            try:
                sr_ratio_float = float(m.get("sr_ratio"))
            except (TypeError, ValueError):
                sr_ratio_float = float("nan")
            sr_ok = not np.isnan(sr_ratio_float) and sr_ratio_float >= sr_min

            if not sr_ok:
                stats["events"].append(
                    {
                        "event": "sr_reject",
                        "ticker": t,
                        "date": str(pd.Timestamp(D_ts).date()),
                        "sr_ratio": None if np.isnan(sr_ratio_float) else float(sr_ratio_float),
                        "sr_min": float(sr_min),
                    }
                )
                continue

            if not (close_ok and vol_ok and gap_ok):
                continue

            row: Dict[str, Any] = {"ticker": t, **m}
            row["date"] = pd.Timestamp(D_ts).tz_localize(None)
            row["sr_window_used"] = int(m.get("sr_window_len", 0) or 0)
            row["sr_support"] = m.get("sr_support")
            row["sr_resistance"] = m.get("sr_resistance")
            row["sr_ok"] = int(sr_ok)

            sr_fraction_effective = float("nan")
            if not np.isnan(tp_sr_fraction) and tp_sr_fraction > 0:
                sr_fraction_effective = min(float(tp_sr_fraction), 1.0)
            atr_multiple_effective = float("nan")
            if not np.isnan(tp_atr_multiple) and tp_atr_multiple > 0:
                atr_multiple_effective = float(tp_atr_multiple)

            row["tp_mode"] = tp_mode
            row["tp_sr_fraction"] = (
                float(sr_fraction_effective)
                if tp_mode == "sr_fraction" and not np.isnan(sr_fraction_effective)
                else float("nan")
            )
            row["tp_atr_multiple"] = (
                float(atr_multiple_effective)
                if tp_mode == "atr_multiple" and not np.isnan(atr_multiple_effective)
                else float("nan")
            )

            entry_open_val = row.get("entry_open")
            resistance_val = row.get("resistance")
            atr_val_dm1 = row.get("atr21")

            tp_price_pct_target = float("nan")
            tp_price_abs_target = float("nan")
            tp_halfway_pct = float("nan")

            entry_valid = (
                entry_open_val is not None
                and not pd.isna(entry_open_val)
                and float(entry_open_val) > 0
            )

            if entry_valid:
                entry_float = float(entry_open_val)
                if tp_mode == "sr_fraction":
                    if not np.isnan(sr_fraction_effective) and sr_fraction_effective > 0:
                        try:
                            resistance_float = float(resistance_val)
                        except (TypeError, ValueError):
                            resistance_float = float("nan")
                        if not np.isnan(resistance_float) and resistance_float > entry_float:
                            up = resistance_float - entry_float
                            tp_candidate = (sr_fraction_effective * up) / entry_float
                            if tp_candidate > 0:
                                tp_halfway_pct = tp_candidate
                                tp_price_pct_target = tp_candidate * 100.0
                                tp_price_abs_target = entry_float * (1.0 + tp_candidate)
                else:  # ATR multiple mode
                    if not np.isnan(atr_multiple_effective) and atr_multiple_effective > 0:
                        try:
                            atr_float = float(atr_val_dm1)
                        except (TypeError, ValueError):
                            atr_float = float("nan")
                        if not np.isnan(atr_float):
                            tp_candidate = (atr_multiple_effective * atr_float) / entry_float
                            if tp_candidate > 0:
                                tp_price_pct_target = tp_candidate * 100.0
                                tp_price_abs_target = entry_float * (1.0 + tp_candidate)

            row["tp_halfway_pct"] = (
                float(tp_halfway_pct) if not np.isnan(tp_halfway_pct) else float("nan")
            )
            row["tp_price_pct_target"] = (
                float(tp_price_pct_target)
                if not np.isnan(tp_price_pct_target)
                else float("nan")
            )
            row["tp_price_abs_target"] = (
                float(tp_price_abs_target)
                if not np.isnan(tp_price_abs_target)
                else float("nan")
            )

            tp_frac = float("nan")
            tp_pct_val = row.get("tp_price_pct_target")
            if tp_pct_val is not None and not pd.isna(tp_pct_val):
                tp_frac = float(tp_pct_val) / 100.0
            else:
                tp_frac = tp_fraction_from_row(
                    row.get("entry_open"),
                    row.get("tp_price_abs_target"),
                    row.get("tp_halfway_pct"),
                    row.get("tp_price_pct_target"),
                )

            tp_frac_valid = (
                tp_frac is not None
                and not pd.isna(tp_frac)
                and float(tp_frac) > 0
            )
            row["tp_frac_used"] = float(tp_frac) if tp_frac_valid else float("nan")
            row["tp_pct_used"] = float(tp_frac) * 100.0 if tp_frac_valid else float("nan")

            # Precedent hits
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
            row["precedent_hit_start_dates"] = ",".join(e.get("date", "") for e in hits_details)

            hit_dates = []
            for e in hits_details:
                hit_date_val = e.get("hit_date")
                if hit_date_val:
                    try:
                        hit_dates.append(pd.Timestamp(hit_date_val).tz_localize(None))
                    except Exception:
                        continue
            row["precedent_max_hit_date"] = max(hit_dates).date().isoformat() if hit_dates else ""

            # Precedent gate
            prec_ok_bool = True
            if use_precedent and tp_frac_valid:
                prec_ok_bool = hits_count >= min_prec_hits
                if not prec_ok_bool:
                    prec_fail_count += 1
                    stats["events"].append(
                        {
                            "event": "precedent_reject",
                            "ticker": t,
                            "date": str(pd.Timestamp(D_ts).date()),
                            "hits": int(hits_count),
                            "required": int(min_prec_hits),
                        }
                    )
            elif use_precedent and not tp_frac_valid:
                prec_ok_bool = True
            row["precedent_ok"] = int(prec_ok_bool)

            # ATR feasible check (positional)
            df_pos = df.reset_index(drop=True)
            atr_ok_bool = (
                bool(
                    atr_feasible(
                        df_pos,
                        int(dm1),
                        float(tp_frac),
                        atr_window,
                        atr_method=atr_method,
                    )
                )
                if tp_frac_valid
                else False
            )
            row["atr_ok"] = int(atr_ok_bool)

            # ---- Persist ATR numbers for transparency ----
            atr_value_dm1 = float(m.get("atr21")) if m.get("atr21") is not None else float("nan")
            entry_for_target = row.get("entry_open")
            tp_required_dollars = (
                float(entry_for_target) * float(tp_frac)
                if tp_frac_valid
                and entry_for_target is not None
                and not pd.isna(entry_for_target)
                else float("nan")
            )
            atr_budget_dollars = atr_value_dm1 * int(atr_window) if not pd.isna(atr_value_dm1) else float("nan")

            row["atr_window"] = int(atr_window)
            row["atr_method"] = atr_method
            atr_value_raw = None if pd.isna(atr_value_dm1) else float(atr_value_dm1)
            row["atr_value_dm1"] = None if atr_value_raw is None else round(atr_value_raw, 6)
            row["atr_dminus1"] = atr_value_raw
            row["atr_budget_dollars"] = None if pd.isna(atr_budget_dollars) else round(atr_budget_dollars, 6)
            row["tp_required_dollars"] = None if pd.isna(tp_required_dollars) else round(tp_required_dollars, 6)
            # ------------------------------------------------

            reasons: List[str] = []
            if use_precedent and tp_frac_valid and not prec_ok_bool:
                reasons.append("precedent_fail")
            if not atr_ok_bool:
                reasons.append("atr_insufficient")
            row["reasons"] = ",".join(reasons)

            include = ((not use_precedent) or prec_ok_bool) and ((not use_atr_feasible) or atr_ok_bool)
            if include:
                cand_rows.append(row)

                if exit_model == "pct_tp_only":
                    if not tp_frac_valid:
                        continue
                    entry_open_val = row.get("entry_open")
                    if entry_open_val is None or pd.isna(entry_open_val):
                        continue
                    entry_open = float(entry_open_val)
                    price_cols = [c for c in ("open", "high", "low", "close") if c in df_idx.columns]
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
                    if not tp_frac_valid:
                        continue
                    entry_val = row.get("entry_open")
                    if entry_val is None or pd.isna(entry_val):
                        continue
                    tp_price = float(entry_val) * (1.0 + float(tp_frac))
                    stop_price = row.get("support")
                    out = replay_trade(
                        df[["date", "open", "high", "low", "close"]],
                        pd.to_datetime(D),
                        float(entry_val),
                        tp_price,
                        stop_price,
                        horizon_days=horizon,
                    )
                    out_row = {**row, "exit_model": exit_model, "tp_price": tp_price, **out}
                    out_rows.append(out_row)
        except Exception as e:
            # Hard-guard: skip any pathological ticker instead of aborting whole run
            fail_count += 1
            log.warning("scan_day: ticker %s skipped due to error: %s", t, e)
        finally:
            if on_step:
                try:
                    on_step(idx, total, t)
                except Exception:
                    pass

    cand_df = pd.DataFrame(cand_rows)
    out_df = pd.DataFrame(out_rows)

    # Light sanity checks
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
                "window_default": int(expected_window),
                "method": atr_method,
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



