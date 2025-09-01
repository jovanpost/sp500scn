#!/usr/bin/env python3
"""Shared helpers for reading/writing and evaluating data/history/outcomes.csv."""

from __future__ import annotations

import math
from datetime import datetime, date
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
    """Insert new outcome rows and backfill pending ones using vector ops."""

    out = read_outcomes(outcomes_path).copy()
    df_pass = df_pass.copy() if df_pass is not None else pd.DataFrame()

    # Normalize keys
    for df in (out, df_pass):
        if not df.empty:
            df["Ticker"] = df["Ticker"].astype(str).str.upper()
            df["EvalDate"] = df["EvalDate"].map(_to_date_str)

    pass_cols = ["Ticker", "EvalDate", "Price", "EntryTimeET", "BuyK", "SellK", "TP", "Expiry"]
    for c in pass_cols:
        if c not in df_pass.columns:
            df_pass[c] = pd.NA

    if not df_pass.empty:
        expiry = df_pass.get("OptExpiry")
        if expiry is None:
            expiry = pd.Series(index=df_pass.index, dtype="object")
        expiry = expiry.combine_first(df_pass.get("Expiry"))
        ev = pd.to_datetime(df_pass["EvalDate"], errors="coerce")
        expiry = pd.to_datetime(expiry, errors="coerce")
        expiry = expiry.fillna(ev + pd.Timedelta(days=30))
        df_pass["Expiry"] = expiry.dt.date.astype(str)

    merged = pd.merge(
        out,
        df_pass[pass_cols],
        on=["Ticker", "EvalDate"],
        how="outer",
        suffixes=("", "_pass"),
    )

    fill_cols = ["BuyK", "SellK", "TP", "Price", "Expiry"]
    merged[fill_cols] = merged[fill_cols].combine_first(
        merged[[f"{c}_pass" for c in fill_cols]].rename(
            columns=lambda x: x.replace("_pass", "")
        )
    )

    new_mask = merged["Status"].isna()
    merged.loc[new_mask, [
        "Status",
        "result_status",
        "HitDateET",
        "Notes",
        "run_date",
    ]] = [
        "PENDING",
        "PENDING",
        "",
        "",
        pd.Timestamp.utcnow().strftime("%Y-%m-%d"),
    ]

    missing_exp = merged["Expiry"].isna() | (merged["Expiry"] == "")
    ev_dt = pd.to_datetime(merged["EvalDate"], errors="coerce")
    merged.loc[missing_exp & ev_dt.notna(), "Expiry"] = (
        ev_dt + pd.Timedelta(days=30)
    ).dt.date.astype(str)

    merged["run_date"] = merged["run_date"].fillna("")

    merged = merged.drop(columns=[f"{c}_pass" for c in fill_cols])
    merged = (
        merged.sort_values(["Ticker", "EvalDate", "run_date"])
        .drop_duplicates(subset=["Ticker", "EvalDate"], keep="last")
        .reset_index(drop=True)
    )

    write_outcomes(merged, outcomes_path)
    return merged


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
            histories[tkr] = fetch_history(tkr, start=start_dt, auto_adjust=False)

        for i, r in df.iterrows():
            status = str(r.get("result_status") or r.get("Status") or "").upper()
            if status == "SETTLED":
                continue

            tkr = str(r.get("Ticker", "")).upper()
            ev = _to_date_str(r.get("EvalDate"))
            expiry = _to_date_str(r.get("Expiry"))
            target = _first_nonempty(safe_float(r.get("SellK")), safe_float(r.get("TP")))

            if not expiry and ev:
                try:
                    expiry = (pd.Timestamp(ev) + pd.Timedelta(days=30)).date().isoformat()
                except Exception:
                    expiry = ev

            hit_date = None
            if tkr and ev and (target is not None):
                hist = histories.get(tkr)
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
    updated = []
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

        today = date.today()

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

        hist = fetch_history(
            tkr,
            start=start,
            end=stop + pd.Timedelta(days=1),
            auto_adjust=False,
        )
        if hist is None or hist.empty:
            row["CheckedAtUTC"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            updated.append(row)
            continue

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

