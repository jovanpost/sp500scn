#!/usr/bin/env python3
"""Shared helpers for reading/writing and evaluating data/history/outcomes.csv."""

from __future__ import annotations

import csv
import math
import os
from datetime import datetime, date, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

REPO_ROOT = Path(__file__).resolve().parents[1]
HIST_DIR = REPO_ROOT / "data" / "history"
DEFAULT_OUT_PATH = HIST_DIR / "outcomes.csv"

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
def read_outcomes(path: str = str(DEFAULT_OUT_PATH)) -> pd.DataFrame:
    """Read outcomes.csv if it exists, else empty DataFrame."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=OUTCOLS)
    try:
        df = pd.read_csv(path)
        if df.empty:
            return pd.DataFrame(columns=OUTCOLS)
        return df
    except Exception:
        return pd.DataFrame(columns=OUTCOLS)


def write_outcomes(df: pd.DataFrame, path: str = str(DEFAULT_OUT_PATH)) -> None:
    """Write outcomes DataFrame to CSV ensuring expected columns."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for c in OUTCOLS:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[OUTCOLS]
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


# ----- helpers used across operations -----
def _safe_float(x):
    try:
        if x is None:
            return None
        if isinstance(x, float) and math.isnan(x):
            return None
        return float(x)
    except Exception:
        return None


def _first_nonempty(*vals):
    for v in vals:
        if v is not None and str(v).strip() != "":
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
    df_pass: pd.DataFrame, outcomes_path: str = str(DEFAULT_OUT_PATH)
) -> pd.DataFrame:
    """Insert new outcomes rows and backfill details for pending ones."""
    out = read_outcomes(outcomes_path).copy()

    if df_pass is None:
        df_pass = pd.DataFrame()
    else:
        df_pass = df_pass.copy()

    key = lambda t, ev: (str(t).upper(), _to_date_str(ev) or "")
    pass_map = {key(r.get("Ticker"), r.get("EvalDate")): r for _, r in df_pass.iterrows()}

    existing_keys = set()
    if not out.empty:
        for _, r in out.iterrows():
            existing_keys.add(key(r.get("Ticker"), r.get("EvalDate")))

    rows_to_append = []
    for k, r in pass_map.items():
        if k in existing_keys:
            continue
        tkr, ev = k
        rows_to_append.append(
            {
                "Ticker": tkr,
                "EvalDate": ev,
                "Price": _safe_float(r.get("Price")),
                "EntryTimeET": r.get("EntryTimeET"),
                "Status": "PENDING",
                "result_status": "PENDING",
                "HitDateET": "",
                "Expiry": _parse_expiry_from_passrow(r),
                "BuyK": _safe_float(r.get("BuyK")),
                "SellK": _safe_float(r.get("SellK")),
                "TP": _safe_float(r.get("TP")),
                "Notes": "",
                "run_date": pd.Timestamp.utcnow().strftime("%Y-%m-%d"),
            }
        )

    if rows_to_append:
        out = pd.concat([out, pd.DataFrame(rows_to_append)], ignore_index=True)

    if not out.empty:
        for i, r in out.iterrows():
            if str(r.get("result_status") or r.get("Status") or "").upper() == "SETTLED":
                continue
            k = key(r.get("Ticker"), r.get("EvalDate"))
            pr = pass_map.get(k)
            if pr is None:
                if not _to_date_str(r.get("Expiry")) and _to_date_str(r.get("EvalDate")):
                    try:
                        exp = (
                            pd.Timestamp(_to_date_str(r.get("EvalDate")))
                            + pd.Timedelta(days=30)
                        ).date().isoformat()
                        out.at[i, "Expiry"] = exp
                    except Exception:
                        pass
                continue
            if not _to_date_str(r.get("Expiry")):
                out.at[i, "Expiry"] = _parse_expiry_from_passrow(pr)
            if pd.isna(r.get("BuyK")) or r.get("BuyK") == "":
                out.at[i, "BuyK"] = _safe_float(pr.get("BuyK"))
            if pd.isna(r.get("SellK")) or r.get("SellK") == "":
                out.at[i, "SellK"] = _safe_float(pr.get("SellK"))
            if pd.isna(r.get("TP")) or r.get("TP") == "":
                out.at[i, "TP"] = _safe_float(pr.get("TP"))
            if pd.isna(r.get("Price")) or r.get("Price") == "":
                out.at[i, "Price"] = _safe_float(pr.get("Price"))

    if not out.empty:
        out["run_date"] = out["run_date"].fillna("")
        out = (
            out.sort_values(["Ticker", "EvalDate", "run_date"])
            .drop_duplicates(subset=["Ticker", "EvalDate"], keep="last")
            .reset_index(drop=True)
        )

    write_outcomes(out, outcomes_path)
    return out


# ----- settlement (hits/expiry) -----
def _first_hit_date_since(ticker: str, start_date: str, threshold: float) -> str | None:
    try:
        if not ticker or threshold is None:
            return None
        df = yf.download(
            ticker,
            start=start_date,
            progress=False,
            auto_adjust=False,
            interval="1d",
        )
        if df is None or df.empty or "High" not in df.columns:
            return None
        meet = df["High"] >= float(threshold)
        if meet.any():
            first_idx = meet.idxmax()
            if isinstance(first_idx, pd.Timestamp):
                return first_idx.date().isoformat()
            return _to_date_str(first_idx)
    except Exception:
        return None
    return None


def settle_pending_outcomes(outcomes_path: str = str(DEFAULT_OUT_PATH)) -> pd.DataFrame:
    """Mark pending outcomes as settled if hit or expired."""
    out = read_outcomes(outcomes_path).copy()
    if out.empty:
        return out

    today = pd.Timestamp.utcnow().date()
    changed = False

    for i, r in out.iterrows():
        status = str(r.get("result_status") or r.get("Status") or "").upper()
        if status == "SETTLED":
            continue

        tkr = str(r.get("Ticker", "")).upper()
        ev = _to_date_str(r.get("EvalDate"))
        expiry = _to_date_str(r.get("Expiry"))
        target = _first_nonempty(_safe_float(r.get("SellK")), _safe_float(r.get("TP")))

        if not expiry and ev:
            try:
                expiry = (pd.Timestamp(ev) + pd.Timedelta(days=30)).date().isoformat()
            except Exception:
                expiry = ev

        hit_date = None
        if tkr and ev and (target is not None):
            hit_date = _first_hit_date_since(tkr, ev, target)

        if hit_date:
            out.at[i, "HitDateET"] = f"{hit_date} 16:00:00 ET"
            out.at[i, "Notes"] = (
                "HIT_BY_SELLK" if pd.notna(r.get("SellK")) else "HIT_BY_TP"
            )
            out.at[i, "Status"] = "SETTLED"
            out.at[i, "result_status"] = "SETTLED"
            changed = True
            continue

        if expiry:
            try:
                exp_d = pd.Timestamp(expiry).date()
                if today > exp_d:
                    out.at[i, "Status"] = "SETTLED"
                    out.at[i, "result_status"] = "SETTLED"
                    out.at[i, "Notes"] = "EXPIRED_NO_HIT"
                    changed = True
            except Exception:
                pass

    if changed:
        write_outcomes(out, outcomes_path)
    return out


# ----- check_hits logic -----
def _parse_date(s):
    try:
        return datetime.strptime(str(s), "%Y-%m-%d").date()
    except Exception:
        return None


def check_pending_hits(df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate pending rows for TP hits or expiry."""
    if df.empty:
        return df

    today = datetime.now(timezone.utc).date()

    pend_mask = df.get("result_status", "") == "PENDING"
    df_p = df[pend_mask].copy()
    for idx, r in df_p.iterrows():
        tkr = str(r["Ticker"])
        tp = r.get("TP", np.nan)
        tp = float(tp) if pd.notna(tp) else np.nan

        d0 = _parse_date(r.get("EvalDate", ""))
        exp = _parse_date(r.get("OptExpiry", ""))

        if pd.isna(tp) or d0 is None:
            continue

        try:
            hist = yf.Ticker(tkr).history(start=d0, end=today, auto_adjust=False)
        except Exception:
            hist = None

        hit_time = ""
        hit_price = ""
        hit = False
        if hist is not None and not hist.empty and "High" in hist.columns:
            highs = hist["High"].astype(float)
            hit_idx = highs[highs >= tp]
            if not hit_idx.empty:
                hit = True
                first = hit_idx.index[0]
                hit_time = first.strftime("%Y-%m-%d")
                hit_price = float(highs.loc[first])

        if hit:
            df.loc[df.index == idx, "result_status"] = "HIT"
            df.loc[df.index == idx, "result_note"] = "TP reached by daily high"
            df.loc[df.index == idx, "hit_time"] = hit_time
            df.loc[df.index == idx, "hit_price"] = hit_price
        else:
            if exp is not None and today > exp:
                df.loc[df.index == idx, "result_status"] = "MISS"
                df.loc[df.index == idx, "result_note"] = "Expired without TP"
                df.loc[df.index == idx, "hit_time"] = ""
                df.loc[df.index == idx, "hit_price"] = ""
    return df


# ----- score_history logic -----
def _coerce_date(s):
    try:
        return pd.to_datetime(s).date()
    except Exception:
        return None


def _check_row(row: dict) -> dict:
    outcome = str(row.get("Outcome", "PENDING")).upper()
    if outcome not in ("PENDING", "YES", "NO"):
        outcome = "PENDING"

    if outcome != "PENDING":
        return row

    tkr = str(row.get("Ticker", "")).strip().upper()
    if not tkr:
        return row

    eval_date = _coerce_date(row.get("EvalDate"))
    window_end = _coerce_date(row.get("WindowEnd"))
    target = row.get("TargetLevel", np.nan)
    try:
        target = float(target)
    except Exception:
        target = np.nan

    today = date.today()

    if eval_date is None or window_end is None or not np.isfinite(target):
        row["CheckedAtUTC"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        return row

    start = pd.Timestamp(eval_date)
    stop = pd.Timestamp(min(window_end, today))

    if start > stop:
        row["CheckedAtUTC"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        return row

    try:
        df = yf.Ticker(tkr).history(
            start=start,
            end=stop + pd.Timedelta(days=1),
            auto_adjust=False,
            actions=False,
        )
        if df is None or df.empty:
            row["CheckedAtUTC"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            return row

        highs = df["High"].dropna()
        if highs.empty:
            row["CheckedAtUTC"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            return row

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
        return row

    except Exception:
        row["CheckedAtUTC"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        return row


def score_history(df: pd.DataFrame) -> pd.DataFrame:
    """Apply hit/miss scoring logic to historical rows."""
    if df.empty:
        return df
    rows = df.to_dict(orient="records")
    updated = [_check_row(r) for r in rows]
    return pd.DataFrame(updated)

