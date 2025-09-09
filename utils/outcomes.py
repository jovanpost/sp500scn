#!/usr/bin/env python3
"""Shared helpers for reading/writing and evaluating data/history/outcomes.csv."""

from __future__ import annotations

import math
from datetime import datetime, date
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .numeric import safe_float
from .prices import fetch_history
from .io import OUTCOMES_CSV, read_csv, write_csv

DEFAULT_OUT_PATH = OUTCOMES_CSV

OUTCOLS = [
    "Ticker",
    "EvalDate",
    "Price",
    "Change%",
    "RelVol(TimeAdj63d)",
    "LastPrice",
    "LastPriceAt",
    "PctToTarget",
    "EntryTimeET",
    "Status",
    "result_status",
    "HitDateET",
    "Expiry",
    "BuyK",
    "SellK",
    "TP",
    "Notes",
    "run_date",
]


# ----- basic IO -----
def read_outcomes(path: str | Path = DEFAULT_OUT_PATH) -> pd.DataFrame:
    """Read outcomes.csv if it exists, else empty DataFrame."""
    df = read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=OUTCOLS)
    return df


def write_outcomes(df: pd.DataFrame, path: str | Path = DEFAULT_OUT_PATH) -> None:
    """Write outcomes DataFrame to CSV ensuring expected columns."""
    for c in OUTCOLS:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[OUTCOLS]
    write_csv(path, df)


# ----- helpers used across operations -----

def _first_nonempty(*vals):
    for v in vals:
        if v is None:
            continue
        if isinstance(v, float) and math.isnan(v):
            continue
        s = str(v).strip()
        if s and s.lower() != "nan":
            return v
    return None


def _to_date_str(x):
    if x is None:
        return None
    try:
        if isinstance(x, pd.Timestamp):
            return x.date().isoformat()
        s = str(x)
        if len(s) >= 10:
            return s[:10]
    except Exception:
        pass
    return None


def parse_date(s: Any) -> date | None:
    """Best effort conversion of ``s`` to a :class:`datetime.date`.

    Accepts ``date``/``datetime`` objects, ``pandas`` timestamps or
    strings. Returns ``None`` if parsing fails or ``s`` is null/NaN.
    """
    if s is None:
        return None
    if isinstance(s, datetime):
        return s.date()
    if isinstance(s, date):
        return s
    try:
        dt = pd.to_datetime(s, errors="coerce")
    except Exception:
        return None
    if pd.isna(dt):
        return None
    try:
        return dt.date()
    except Exception:
        return None


def _parse_expiry_from_passrow(row: pd.Series) -> str | None:
    raw = _first_nonempty(row.get("OptExpiry"), row.get("Expiry"))
    if raw:
        return _to_date_str(raw)
    ev = _to_date_str(row.get("EvalDate"))
    if ev:
        try:
            d = pd.Timestamp(ev) + pd.Timedelta(days=30)
            return d.date().isoformat()
        except Exception:
            return ev
    return None


# ----- upsert/backfill (from run_and_log) -----
def upsert_and_backfill_outcomes(
    df_pass: pd.DataFrame, outcomes_path: str | Path = DEFAULT_OUT_PATH
) -> pd.DataFrame:
    """Insert new outcomes rows and backfill details for pending ones."""
    out = read_outcomes(outcomes_path).copy()
    if df_pass is None or df_pass.empty:
        write_outcomes(out, outcomes_path)
        return out

    df_pass = df_pass.copy()
    for c in [
        "Ticker",
        "EvalDate",
        "Price",
        "Change%",
        "RelVol(TimeAdj63d)",
        "BuyK",
        "SellK",
        "TP",
        "EntryTimeET",
        "HitDateET",
        "Notes",
        "run_date",
    ]:
        if c not in df_pass.columns:
            df_pass[c] = pd.NA

    df_pass["Ticker"] = df_pass["Ticker"].astype(str).str.upper()
    df_pass["EvalDate"] = df_pass["EvalDate"].apply(_to_date_str)
    df_pass["Price"] = df_pass["Price"].map(safe_float)
    df_pass["Change%"] = df_pass["Change%"].map(safe_float)
    df_pass["RelVol(TimeAdj63d)"] = df_pass["RelVol(TimeAdj63d)"].map(safe_float)
    df_pass["BuyK"] = df_pass["BuyK"].map(safe_float)
    df_pass["SellK"] = df_pass["SellK"].map(safe_float)
    df_pass["TP"] = df_pass["TP"].map(safe_float)
    today_str = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
    df_pass["run_date"] = df_pass["run_date"].apply(_to_date_str).fillna(today_str)

    tgt = df_pass["SellK"].combine_first(df_pass["TP"])
    df_pass["PctToTarget"] = np.where(
        (df_pass["Price"].notna()) & (tgt.notna()) & (~pd.isna(tgt)),
        (df_pass["Price"] - tgt) / tgt,
        pd.NA,
    )
    df_pass["LastPrice"] = df_pass["Price"]
    df_pass["LastPriceAt"] = df_pass["EntryTimeET"]
    df_pass["Status"] = "PENDING"
    df_pass["result_status"] = "PENDING"
    df_pass["Expiry"] = df_pass.apply(_parse_expiry_from_passrow, axis=1)

    new_rows = df_pass[OUTCOLS].copy()

    if out.empty:
        out = new_rows
    else:
        key_cols = ["Ticker", "EvalDate"]
        merged = pd.merge(out, new_rows, on=key_cols, how="outer", suffixes=("", "_new"))
        for col in OUTCOLS:
            if col in key_cols:
                continue
            new_col = f"{col}_new"
            if new_col in merged.columns:
                merged[col] = merged[col].combine_first(merged[new_col])
                merged.drop(columns=new_col, inplace=True)
        out = merged[OUTCOLS]
        out["run_date"] = out["run_date"].fillna("")
        out = (
            out.sort_values(["Ticker", "run_date", "EvalDate"])
            .drop_duplicates(subset=["Ticker", "run_date"], keep="first")
            .reset_index(drop=True)
        )

    write_outcomes(out, outcomes_path)
    return out


# ----- evaluation -----
def evaluate_outcomes(df: pd.DataFrame, mode: str = "pending") -> pd.DataFrame:
    """Evaluate outcome rows in ``df``.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to evaluate.
    mode : {"pending", "historical"}, optional
        Select evaluation behavior. ``"pending"`` (default) checks for TP hits
        or expiry on rows where ``result_status`` is ``PENDING``. ``"historical"``
        applies window-based scoring used for historical backtesting.
    """
    if df.empty:
        return df

    # Ensure blank result_status values are treated as pending so they
    # participate in evaluation and get written back correctly.
    if "result_status" in df.columns:
        df["result_status"] = df["result_status"].apply(
            lambda x: "PENDING" if pd.isna(x) or str(x).strip() == "" else x
        )

    mode = str(mode).lower()
    if mode not in {"pending", "historical"}:
        raise ValueError("mode must be 'pending' or 'historical'")

    if mode == "pending":
        today = pd.Timestamp.utcnow().date()

        # Determine earliest EvalDate per ticker so we can fetch once.
        ticker_starts: dict[str, pd.Timestamp] = {}
        for _, r in df.iterrows():
            status = str(r.get("result_status") or r.get("Status") or "").upper()
            if status == "SETTLED":
                continue
            tkr = str(r.get("Ticker", "")).upper()
            ev = _to_date_str(r.get("EvalDate"))
            if tkr and ev:
                try:
                    ev_ts = pd.Timestamp(ev)
                except Exception:
                    continue
                prev = ticker_starts.get(tkr)
                if prev is None or ev_ts < prev:
                    ticker_starts[tkr] = ev_ts

        # Cache history DataFrames per ticker.
        histories: dict[str, pd.DataFrame | None] = {}
        for tkr, start_dt in ticker_starts.items():
            histories[tkr] = fetch_history(tkr, start=start_dt)

        for i, r in df.iterrows():
            tkr = str(r.get("Ticker", "")).upper()
            ev = _to_date_str(r.get("EvalDate"))
            expiry = _to_date_str(r.get("Expiry"))
            target = _first_nonempty(safe_float(r.get("SellK")), safe_float(r.get("TP")))

            hist = histories.get(tkr)
            last_price = None
            last_price_at = None
            if hist is not None and not hist.empty and "Close" in hist.columns:
                try:
                    closes = hist["Close"].dropna().astype(float)
                    if not closes.empty:
                        last_price = float(closes.iloc[-1])
                        ts = closes.index[-1]
                        if isinstance(ts, pd.Timestamp):
                            if ts.tzinfo is None:
                                ts = ts.tz_localize("UTC")
                            ts = ts.tz_convert("America/New_York")
                            last_price_at = ts.strftime("%Y-%m-%d %H:%M:%S ET")
                        else:
                            last_price_at = datetime.now(ZoneInfo("America/New_York")).strftime(
                                "%Y-%m-%d %H:%M:%S ET"
                            )
                except Exception:
                    last_price = None
                    last_price_at = None

            if last_price is not None:
                df.at[i, "LastPrice"] = last_price
                df.at[i, "LastPriceAt"] = last_price_at
                if target is not None and not math.isnan(float(target)):
                    try:
                        df.at[i, "PctToTarget"] = (last_price - float(target)) / float(target)
                    except Exception:
                        pass

            status = str(r.get("result_status") or r.get("Status") or "").upper()
            if status == "SETTLED":
                continue

            if not expiry and ev:
                try:
                    expiry = (pd.Timestamp(ev) + pd.Timedelta(days=30)).date().isoformat()
                except Exception:
                    expiry = ev

            hit_date = None
            if tkr and ev and (target is not None):
                try:
                    if hist is not None and not hist.empty and "High" in hist.columns:
                        try:
                            hist_slice = hist.loc[pd.Timestamp(ev) :]
                        except Exception:
                            hist_slice = hist
                        highs = hist_slice["High"].astype(float)
                        meet = highs >= float(target)
                        if meet.any():
                            first_idx = meet.idxmax()
                            if isinstance(first_idx, pd.Timestamp):
                                hit_date = first_idx.date().isoformat()
                            else:
                                hit_date = _to_date_str(first_idx)
                except Exception:
                    hit_date = None

            if hit_date:
                df.at[i, "HitDateET"] = f"{hit_date} 16:00:00 ET"
                df.at[i, "Notes"] = (
                    "HIT_BY_SELLK" if pd.notna(r.get("SellK")) else "HIT_BY_TP"
                )
                df.at[i, "Status"] = "SETTLED"
                df.at[i, "result_status"] = "SETTLED"
                continue

            if expiry:
                try:
                    exp_d = pd.Timestamp(expiry).date()
                    if today > exp_d:
                        df.at[i, "Status"] = "SETTLED"
                        df.at[i, "result_status"] = "SETTLED"
                        df.at[i, "Notes"] = "EXPIRED_NO_HIT"
                except Exception:
                    pass

        return df

    # Historical mode
    rows = df.to_dict(orient="records")

    # Determine history range per ticker so we can fetch once.
    today = date.today()
    ticker_ranges: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for row in rows:
        tkr = str(row.get("Ticker", "")).strip().upper()
        if not tkr:
            continue
        eval_date = parse_date(row.get("EvalDate"))
        window_end = parse_date(row.get("WindowEnd"))
        target = row.get("TargetLevel", np.nan)
        try:
            target = float(target)
        except Exception:
            target = np.nan
        if eval_date is None or window_end is None or not np.isfinite(target):
            continue
        start = pd.Timestamp(eval_date)
        stop = pd.Timestamp(min(window_end, today))
        if start > stop:
            continue
        prev = ticker_ranges.get(tkr)
        if prev is None:
            ticker_ranges[tkr] = (start, stop)
        else:
            prev_start, prev_stop = prev
            if start < prev_start:
                prev_start = start
            if stop > prev_stop:
                prev_stop = stop
            ticker_ranges[tkr] = (prev_start, prev_stop)

    # Fetch histories once per ticker.
    histories: dict[str, pd.DataFrame | None] = {}
    for tkr, (start, stop) in ticker_ranges.items():
        histories[tkr] = fetch_history(
            tkr,
            start=start,
            end=stop + pd.Timedelta(days=1),
            )

    updated: list[dict[str, Any]] = []
    for row in rows:
        outcome = str(row.get("Outcome", "PENDING")).upper()
        if outcome not in ("PENDING", "YES", "NO"):
            outcome = "PENDING"

        if outcome != "PENDING":
            row["CheckedAtUTC"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            updated.append(row)
            continue

        tkr = str(row.get("Ticker", "")).strip().upper()
        if not tkr:
            row["CheckedAtUTC"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            updated.append(row)
            continue

        eval_date = parse_date(row.get("EvalDate"))
        window_end = parse_date(row.get("WindowEnd"))
        target = row.get("TargetLevel", np.nan)
        try:
            target = float(target)
        except Exception:
            target = np.nan

        if eval_date is None or window_end is None or not np.isfinite(target):
            row["CheckedAtUTC"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            updated.append(row)
            continue

        start = pd.Timestamp(eval_date)
        stop = pd.Timestamp(min(window_end, today))
        if start > stop:
            row["CheckedAtUTC"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            updated.append(row)
            continue

        hist = histories.get(tkr)
        if hist is None or hist.empty:
            row["CheckedAtUTC"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            updated.append(row)
            continue
        try:
            hist = hist.loc[start:stop]
        except Exception:
            pass

        highs = hist["High"].dropna()
        if highs.empty:
            row["CheckedAtUTC"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            updated.append(row)
            continue

        hit_mask = highs >= target
        if hit_mask.any():
            first_idx = hit_mask[hit_mask].index[0]
            row["Outcome"] = "YES"
            row["HitDate"] = first_idx.date().isoformat()
            row["MaxHigh"] = float(highs.max())
        else:
            row["MaxHigh"] = float(highs.max())
            if today > window_end:
                row["Outcome"] = "NO"

        row["CheckedAtUTC"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        updated.append(row)

    return pd.DataFrame(updated)

