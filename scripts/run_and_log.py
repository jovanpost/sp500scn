#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=======================================================================
Bibliography (Section Index)
=======================================================================
1. Imports & CLI
2. Paths, Constants, Helpers
3. Robust IO (read/write CSV)
4. Outcomes Upsert (non-destructive append/update)
5. Screener Runner (invoke library, gather DataFrames)
6. Main (glue: run, save pass file, write logs, upsert outcomes)
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
import inspect
import pandas as pd

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
HISTORY_DIR = Path("data/history")
LOGS_DIR = Path("data/logs")
OUTCOMES_FILE = HISTORY_DIR / "outcomes.csv"

UTC_NOW = datetime.now(timezone.utc)
STAMP = UTC_NOW.strftime("%Y%m%d-%H%M")

PASS_PATH = HISTORY_DIR / f"pass_{STAMP}.csv"
SCAN_PATH = HISTORY_DIR / f"scan_{STAMP}.csv"
LOG_PATH = LOGS_DIR / f"scan_{STAMP}.txt"


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
# 3. Robust IO (read/write CSV)
# --------------------------------------------------------------------
def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as e:
            print(f"[WARN] Failed reading {path}: {e}", file=sys.stderr)
    return pd.DataFrame()


def write_csv(path: Path, df: pd.DataFrame) -> None:
    try:
        df.to_csv(path, index=False)
    except Exception as e:
        print(f"[ERROR] Failed writing {path}: {e}", file=sys.stderr)

# --------------------------------------------------------------------
# 4. Outcomes Upsert (non-destructive append/update + backfill)
# --------------------------------------------------------------------
import csv
import math
from datetime import timezone
import pandas as pd
import yfinance as yf
import os

OUTCOLS = [
    "Ticker","EvalDate","Price","EntryTimeET",
    "Status","result_status","HitDateET",
    "Expiry","BuyK","SellK","TP","Notes",
    "run_date"
]

def _read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=OUTCOLS)
    try:
        df = pd.read_csv(path)
        if df.empty:
            return pd.DataFrame(columns=OUTCOLS)
        return df
    except Exception:
        return pd.DataFrame(columns=OUTCOLS)

def _write_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for c in OUTCOLS:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[OUTCOLS]
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)

def _safe_float(x):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
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

def upsert_and_backfill_outcomes(df_pass: pd.DataFrame, outcomes_path: str) -> pd.DataFrame:
    """
    - Insert new PENDING rows for today's passes (Ticker+EvalDate key).
    - Backfill missing Expiry/BuyK/SellK/TP/Price on existing *PENDING* rows.
    """
    out = _read_csv(outcomes_path)
    if df_pass is None:
        df_pass = pd.DataFrame()

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

    # 1) Insert new
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
            "run_date": pd.Timestamp.utcnow().strftime("%Y-%m-%d")
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
            # backfill fields only if missing
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

    _write_csv(out, outcomes_path)
    return out

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
    out = _read_csv(outcomes_path)
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
# 5. Screener Runner (invoke library, gather DataFrames)
# --------------------------------------------------------------------
def _candidate_tickers_from_universe(universe_key: str):
    """Try to turn a universe key into a list of tickers (if helpers exist)."""
    if spuni is None:
        return None
    try:
        if hasattr(spuni, "get_universe"):
            return spuni.get_universe(universe_key)
        if hasattr(spuni, "UNIVERSES"):
            return spuni.UNIVERSES.get(universe_key)
    except Exception:
        pass
    return None


def _extract_df(out, *keys) -> pd.DataFrame:
    """If screener returns dict, try multiple keys; if DF, return it; else empty DF."""
    if isinstance(out, pd.DataFrame):
        return out
    if isinstance(out, dict):
        for k in keys:
            v = out.get(k)
            if isinstance(v, pd.DataFrame):
                return v
    return pd.DataFrame()


def run_screener(universe: str, with_options: bool, save_scan: bool):
    """
    Calls sos.run_scan defensively:
      - only passes kwargs that the function actually supports (introspection)
      - if it returns dict, tries common keys to pull pass/scan dataframes
    """
    fn = getattr(sos, "run_scan", None)
    if fn is None:
        raise RuntimeError("swing_options_screener.run_scan not found")

    sig = inspect.signature(fn)
    params = sig.parameters.keys()

    kwargs = {}
    if "with_options" in params:
        kwargs["with_options"] = bool(with_options)

    # only pass tickers if the function supports it
    if "tickers" in params:
        tickers = _candidate_tickers_from_universe(universe)
        if tickers:
            kwargs["tickers"] = tickers

    # Optional "return_scan_df" style flags (support several common names)
    for flag in ("return_scan_df", "return_full_scan", "return_scan"):
        if flag in params:
            kwargs[flag] = bool(save_scan)

    out = fn(**kwargs)

    # Try to extract dataframes
    df_pass = _extract_df(out, "pass", "passes", "df_pass", "pass_df", "passes_df")
    df_scan = _extract_df(out, "scan", "df_scan", "scan_df", "full_scan")

    return df_pass, df_scan

# --------------------------------------------------------------------
# 6. Main (glue: run, save pass file, write logs, upsert+settle outcomes)
# --------------------------------------------------------------------
def main():
    args = parse_args()
    ensure_dirs()

    # 1) run the screener
    try:
        res = sos.run_scan(universe=args.universe, with_options=args.with_options)
    except TypeError:
        res = sos.run_scan(market=args.universe, with_options=args.with_options)

    # Normalize outputs
    df_pass = None
    if isinstance(res, dict):
        df_pass = res.get("pass") or res.get("pass_df") or res.get("pass_df_unadjusted")
    elif isinstance(res, (list, tuple)) and len(res) > 0 and isinstance(res[0], pd.DataFrame):
        df_pass = res[0]
    elif isinstance(res, pd.DataFrame):
        df_pass = res

    # 2) save pass file
    if isinstance(df_pass, pd.DataFrame) and not df_pass.empty:
        df_pass.to_csv(PASS_PATH, index=False)
        print(f"[OK] wrote pass file: {PASS_PATH}")
    else:
        print("[INFO] no pass tickers this run (nothing written)")

    # 3) upsert + backfill outcomes
    upsert_and_backfill_outcomes(df_pass if isinstance(df_pass, pd.DataFrame) else pd.DataFrame(), OUTCOMES_FILE)

    # 4) settle any pending rows
    settle_pending_outcomes(OUTCOMES_FILE)

    # 5) log
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        ts = UTC_NOW.strftime("%Y-%m-%d %H:%M:%SZ")
        n = 0 if df_pass is None else len(df_pass)
        f.write(f"[{ts}] universe={args.universe} passes={n}\n")
        if n:
            for t in df_pass.get("Ticker", []):
                f.write(f" - {t}\n")
    print(f"[OK] log: {LOG_PATH}")

if __name__ == "__main__":
    main()

