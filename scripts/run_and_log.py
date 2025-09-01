#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=======================================================================
Bibliography (Section Index)
=======================================================================
1. Imports & CLI
2. Paths, Constants, Helpers
3. Outcomes Upsert (non-destructive append/update)
4. Screener Runner (invoke library, gather DataFrames)
5. Main (glue: run, save pass file, write logs, upsert outcomes)
=======================================================================
"""

# --------------------------------------------------------------------
# 1. Imports & CLI
# --------------------------------------------------------------------
import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

# Ensure repo root on path for utility imports and local modules
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.io import HISTORY_DIR, OUTCOMES_CSV, read_csv, write_csv

# Screener module (must be importable from repo root)
try:
    import swing_options_screener as sos
except Exception as e:
    print(f"[FATAL] Could not import swing_options_screener: {e}", file=sys.stderr)
    sys.exit(1)

# Optional universe helper (only used if present)
try:
    import sp_universe as spuni  # optional
except Exception:
    spuni = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run swing scan and write history/outcomes."
    )
    p.add_argument(
        "--universe",
        default="sp500",
        help="Universe key (kept for logs/metadata; not passed to screener unless tickers param is supported).",
    )
    p.add_argument(
        "--with-options",
        action="store_true",
        help="Ask the screener to include options fields.",
    )
    p.add_argument(
        "--also-save-scan",
        action="store_true",
        help="If available, save raw scan DF alongside pass file.",
    )
    return p.parse_args()


# --------------------------------------------------------------------
# 2. Paths, Constants, Helpers
# --------------------------------------------------------------------
LOGS_DIR = Path("data/logs")

UTC_NOW = datetime.now(timezone.utc)
STAMP = UTC_NOW.strftime("%Y%m%d-%H%M")


def ensure_dirs() -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def pick(df: pd.DataFrame, col: str, default=None):
    """Safe column getter for a homogeneous value; returns default if missing."""
    try:
        if df is None or df.empty or col not in df.columns:
            return default
        vals = df[col].dropna().unique()
        if len(vals) == 0:
            return default
        return vals[0]
    except Exception:
        return default


def safe_str(x):
    return "" if x is None else str(x)


 
# --------------------------------------------------------------------
# 3. Outcomes Upsert (non-destructive append/update + backfill)
# --------------------------------------------------------------------
import math
from datetime import timezone
import yfinance as yf
import os

OUTCOLS = [
    "Ticker","EvalDate","Price","EntryTimeET",
    "Status","result_status","HitDateET",
    "Expiry","BuyK","SellK","TP","Notes",
    "run_date"
]

# ---------- small utils ----------

def _read_csv(path: str) -> pd.DataFrame:
    df = read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=OUTCOLS)
    for c in OUTCOLS:
        if c not in df.columns:
            df[c] = pd.NA
    return df


def _write_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for c in OUTCOLS:
        if c not in df.columns:
            df[c] = pd.NA
    write_csv(path, df[OUTCOLS])

def _safe_float(x):
    try:
        if x is None:
            return None
        # handle pandas NA/NaN
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
    """Return YYYY-MM-DD or None."""
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
    """
    Prefer OptExpiry/Expiry in the pass row; else fall back to EvalDate + 30d.
    """
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

# ---------- upsert/backfill ----------

def upsert_and_backfill_outcomes(df_pass: pd.DataFrame, outcomes_path: str) -> pd.DataFrame:
    """
    - Insert new PENDING rows for today's passes (Ticker+EvalDate key).
    - Backfill missing Expiry/BuyK/SellK/TP/Price on existing *PENDING* rows.
    """
    out = _read_csv(outcomes_path).copy()

    if df_pass is None:
        df_pass = pd.DataFrame()
    else:
        df_pass = df_pass.copy()

    # Build quick lookup keyed by (Ticker, EvalDate)
    key = lambda t, ev: (str(t).upper(), _to_date_str(ev) or "")
    pass_map = {}
    for _, r in df_pass.iterrows():
        pass_map[key(r.get("Ticker"), r.get("EvalDate"))] = r

    existing_keys = set()
    if not out.empty:
        for _, r in out.iterrows():
            existing_keys.add(key(r.get("Ticker"), r.get("EvalDate")))

    rows_to_append = []

    # 1) Insert new (only if not already present)
    for k, r in pass_map.items():
        if k in existing_keys:
            continue
        tkr, ev = k
        rows_to_append.append({
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
        })

    if rows_to_append:
        out = pd.concat([out, pd.DataFrame(rows_to_append)], ignore_index=True)

    # 2) Backfill existing PENDING with missing Expiry/strikes/TP/Price
    if not out.empty:
        for i, r in out.iterrows():
            if str(r.get("result_status") or r.get("Status") or "").upper() == "SETTLED":
                continue
            k = key(r.get("Ticker"), r.get("EvalDate"))
            pr = pass_map.get(k)
            if pr is None:
                # fallback: if expiry missing, set to EvalDate+30
                if not _to_date_str(r.get("Expiry")) and _to_date_str(r.get("EvalDate")):
                    try:
                        exp = (pd.Timestamp(_to_date_str(r.get("EvalDate"))) + pd.Timedelta(days=30)).date().isoformat()
                        out.at[i, "Expiry"] = exp
                    except Exception:
                        pass
                continue
            # backfill only if missing
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

    # De-dupe defensively (keep latest run_date if duplicates slipped in)
    if not out.empty:
        out["run_date"] = out["run_date"].fillna("")
        out = (
            out.sort_values(["Ticker","EvalDate","run_date"])
               .drop_duplicates(subset=["Ticker","EvalDate"], keep="last")
               .reset_index(drop=True)
        )

    _write_csv(out, outcomes_path)
    return out

# ---------- settlement (hits/expiry) ----------

def _first_hit_date_since(ticker: str, start_date: str, threshold: float) -> str | None:
    try:
        if not ticker or threshold is None:
            return None
        df = yf.download(ticker, start=start_date, progress=False, auto_adjust=False, interval="1d")
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

def settle_pending_outcomes(outcomes_path: str) -> pd.DataFrame:
    out = _read_csv(outcomes_path).copy()
    if out.empty:
        return out

    today = pd.Timestamp.utcnow().date()
    changed = False

    for i, r in out.iterrows():
        status = str(r.get("result_status") or r.get("Status") or "").upper()
        if status == "SETTLED":
            continue

        tkr = str(r.get("Ticker","")).upper()
        ev  = _to_date_str(r.get("EvalDate"))
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
            out.at[i, "Notes"] = "HIT_BY_SELLK" if pd.notna(r.get("SellK")) else "HIT_BY_TP"
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
        _write_csv(out, outcomes_path)
    return out
    

# --------------------------------------------------------------------
# 4. Screener Runner (invoke library, gather DataFrames)
# --------------------------------------------------------------------
from typing import Tuple, Optional
import swing_options_screener as sos

def _safe_engine_run_scan() -> dict:
    """
    Call sos.run_scan() across historical signature variants and normalize
    the result to a dict with keys: {'pass': DataFrame|None, 'scan': DataFrame|None}
    """
    import pandas as _pd

    # Try different parameter names used across versions of your engine
    try:
        out = sos.run_scan(market="sp500", with_options=True)
    except TypeError:
        try:
            out = sos.run_scan(universe="sp500", with_options=True)
        except TypeError:
            out = sos.run_scan(with_options=True)

    df_pass, df_scan = None, None

    if isinstance(out, dict):
        cand = out.get("pass", None)
        if isinstance(cand, _pd.DataFrame):
            df_pass = cand
        cand = out.get("pass_df", None)
        if df_pass is None and isinstance(cand, _pd.DataFrame):
            df_pass = cand
        cand = out.get("pass_df_unadjusted", None)
        if df_pass is None and isinstance(cand, _pd.DataFrame):
            df_pass = cand

        cand = out.get("scan", None)
        if isinstance(cand, _pd.DataFrame):
            df_scan = cand
        cand = out.get("scan_df", None)
        if df_scan is None and isinstance(cand, _pd.DataFrame):
            df_scan = cand

    elif isinstance(out, (list, tuple)):
        if len(out) >= 1 and isinstance(out[0], _pd.DataFrame):
            df_pass = out[0]
        if len(out) >= 2 and isinstance(out[1], _pd.DataFrame):
            df_scan = out[1]

    elif isinstance(out, _pd.DataFrame):
        # Some versions just return the passing table
        df_pass = out

    return {"pass": df_pass, "scan": df_scan}

# --------------------------------------------------------------------
# 5. Main (glue: run, save pass file, write logs, upsert outcomes)
# --------------------------------------------------------------------

from datetime import datetime, timezone

def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

def main() -> int:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        res = _safe_engine_run_scan()
        df_pass: Optional[pd.DataFrame] = res.get("pass")
        # df_scan = res.get("scan")  # currently unused here

        wrote_pass = False
        if isinstance(df_pass, pd.DataFrame) and not df_pass.empty:
            # filename with UTC timestamp
            pass_name = f"pass_{_utc_ts()}.csv"
            pass_path = HISTORY_DIR / pass_name
            write_csv(pass_path, df_pass)
            print(f"[run_and_log] wrote {pass_path}")
            wrote_pass = True

            # Update outcomes.csv (insert new, backfill, settle)
            try:
                # Import the upsert/settle helpers from this same file
                from run_and_log import upsert_and_backfill_outcomes, settle_pending_outcomes  # type: ignore  # noqa: E402
            except Exception:
                # When running as a script, the functions are already in this module
                pass

            upsert_and_backfill_outcomes(df_pass, str(OUTCOMES_CSV))
            settle_pending_outcomes(str(OUTCOMES_CSV))
        else:
            print("[run_and_log] scan returned no passing tickers.")

        # Always succeed if we reached here without exceptions
        # (having 0 passes is not a failure)
        return 0

    except Exception as e:
        # Make failures visible to GitHub Actions
        print(f"[run_and_log] FATAL: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())




