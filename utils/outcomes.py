#!/usr/bin/env python3
"""Shared helpers for reading/writing and evaluating data/history/outcomes.csv."""

from __future__ import annotations

import math
from datetime import datetime, date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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

        for i, r in df.iterrows():
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
                try:
                    hist = fetch_history(tkr, start=ev, auto_adjust=False)
                    if hist is not None and not hist.empty and "High" in hist.columns:
                        highs = hist["High"].astype(float)
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

