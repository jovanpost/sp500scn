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
import pandas as pd

# Your screener module
try:
    import swing_options_screener as sos  # must be importable from repo root
except Exception as e:
    print(f"[FATAL] Could not import swing_options_screener: {e}", file=sys.stderr)
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run swing scan and write history/outcomes.")
    p.add_argument("--universe", default="sp500", help="Universe key understood by the screener (default: sp500).")
    p.add_argument("--with-options", action="store_true", help="Ask the screener to include options fields.")
    p.add_argument("--also-save-scan", action="store_true", help="If available, save raw scan DF alongside pass file.")
    return p.parse_args()


# --------------------------------------------------------------------
# 2. Paths, Constants, Helpers
# --------------------------------------------------------------------
DATA_DIR = os.path.join("data", "history")
LOG_DIR = os.path.join("data", "logs")
OUTCOMES_FILE = os.path.join(DATA_DIR, "outcomes.csv")

UTC_NOW = datetime.now(timezone.utc)
STAMP = UTC_NOW.strftime("%Y%m%d-%H%M")

PASS_BASENAME = f"pass_{STAMP}.csv"
SCAN_BASENAME = f"scan_{STAMP}.csv"
LOG_BASENAME = f"scan_{STAMP}.txt"

PASS_PATH = os.path.join(DATA_DIR, PASS_BASENAME)
SCAN_PATH = os.path.join(DATA_DIR, SCAN_BASENAME)
LOG_PATH = os.path.join(LOG_DIR, LOG_BASENAME)


def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)


def pick(df: pd.DataFrame, col: str, default=None):
    """Safe column getter for a homogeneous value, returns default if column missing."""
    if df is None or df.empty:
        return default
    if col not in df.columns:
        return default
    # If the column is constant across rows, return the first value; otherwise return default
    try:
        v = df[col].iloc[0]
        return v
    except Exception:
        return default


# --------------------------------------------------------------------
# 3. Robust IO (read/write CSV)
# --------------------------------------------------------------------
def read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] Could not read CSV {path}: {e}", file=sys.stderr)
        return pd.DataFrame()


def write_csv(df: pd.DataFrame, path: str):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        print(f"[OK] Wrote {path} ({len(df)} rows)")
    except Exception as e:
        print(f"[ERROR] Could not write CSV {path}: {e}", file=sys.stderr)


def append_text(path: str, txt: str):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(txt)
    except Exception as e:
        print(f"[ERROR] Could not write log {path}: {e}", file=sys.stderr)


# --------------------------------------------------------------------
# 4. Outcomes Upsert (non-destructive append/update)
# --------------------------------------------------------------------
# Columns we own inside outcomes.csv (we DO NOT overwrite settlement fields)
CORE_OUTCOME_COLS = [
    "Ticker", "EvalDate", "Price", "EntryTimeET"
]

# Settlement & evaluation columns that might be added later by nightly checks
PROTECTED_COLS = [
    "Outcome", "OutcomeDate", "HitPrice", "MissPrice",
    "Expiry", "BuyK", "SellK", "Width", "TP", "Resistance"
]


def normalize_outcome_row(row: dict) -> dict:
    """
    Construct a minimal outcomes row from a pass row without clobbering any
    potential settlement fields.
    """
    out = {
        "Ticker":       row.get("Ticker"),
        "EvalDate":     row.get("EvalDate"),
        "Price":        row.get("Price"),
        "EntryTimeET":  row.get("EntryTimeET"),
    }
    return out


def upsert_outcomes(outcomes_path: str, pass_df: pd.DataFrame) -> pd.DataFrame:
    """
    Upsert new core rows from pass_df into outcomes.csv without overwriting
    any of the protected settlement fields.
    """
    existing = read_csv(outcomes_path)

    # Prepare incoming minimal core rows
    incoming_core = pass_df.apply(lambda r: pd.Series(normalize_outcome_row(r)), axis=1)
    # Drop duplicates in the incoming set by (Ticker, EvalDate, EntryTimeET)
    incoming_core = incoming_core.drop_duplicates(subset=["Ticker", "EvalDate", "EntryTimeET"])

    if existing.empty:
        merged = incoming_core.copy()
        write_csv(merged, outcomes_path)
        return merged

    # Ensure existing has all CORE columns
    for c in CORE_OUTCOME_COLS:
        if c not in existing.columns:
            existing[c] = None

    # Build a key for matching
    existing["_key"] = existing["Ticker"].astype(str) + "|" + existing["EvalDate"].astype(str) + "|" + existing["EntryTimeET"].astype(str)
    incoming_core["_key"] = incoming_core["Ticker"].astype(str) + "|" + incoming_core["EvalDate"].astype(str) + "|" + incoming_core["EntryTimeET"].astype(str)

    # Find which incoming keys are new
    existing_keys = set(existing["_key"].tolist())
    new_rows = incoming_core[~incoming_core["_key"].isin(existing_keys)].copy()

    if new_rows.empty:
        print("[INFO] No new outcomes to append.")
        existing.drop(columns=["_key"], errors="ignore", inplace=True)
        return existing

    # Append new rows (protected fields remain whatever they were; here new rows have none)
    combined = pd.concat([existing.drop(columns=["_key"], errors="ignore"), new_rows.drop(columns=["_key"], errors="ignore")], ignore_index=True)
    write_csv(combined, outcomes_path)
    return combined


# --------------------------------------------------------------------
# 5. Screener Runner (invoke library, gather DataFrames)
# --------------------------------------------------------------------
def run_screener(universe: str, with_options: bool) -> dict:
    """
    Call your library function. We expect a dictionary or namespace with:
      - 'passes' (DataFrame)   : required
      - 'scan'   (DataFrame)   : optional
      - 'log'    (str)         : optional
    We keep this wrapper resilient regardless of the exact return shape.
    """
    print(f"[INFO] Running scan: universe={universe} with_options={with_options}")
    out = {"passes": pd.DataFrame(), "scan": pd.DataFrame(), "log": ""}

    try:
        # Your module may expose a single function, adjust as needed:
        #   results = sos.run_scan(universe=universe, with_options=with_options)
        # For safety, try a few patterns:
        if hasattr(sos, "run_scan"):
            results = sos.run_scan(universe=universe, with_options=with_options)
        elif hasattr(sos, "main") and callable(sos.main):
            results = sos.main(universe=universe, with_options=with_options)
        else:
            raise RuntimeError("No suitable entry point in swing_options_screener.")

        # Normalize outputs
        if isinstance(results, dict):
            out["passes"] = results.get("passes", pd.DataFrame())
            out["scan"] = results.get("scan", pd.DataFrame())
            out["log"] = results.get("log", "")
        elif isinstance(results, pd.DataFrame):
            out["passes"] = results
        else:
            # Try attributes
            out["passes"] = getattr(results, "passes", pd.DataFrame())
            out["scan"] = getattr(results, "scan", pd.DataFrame())
            out["log"] = getattr(results, "log", "")

        # Enforce required columns minimality on passes
        for c in ["Ticker", "EvalDate", "Price", "EntryTimeET"]:
            if c not in out["passes"].columns:
                out["passes"][c] = None

        print(f"[INFO] Scan done. Passes: {len(out['passes'])}, Scan rows: {len(out['scan'])}")
        return out

    except Exception as e:
        msg = f"[ERROR] Screener failed: {e}\n"
        print(msg, file=sys.stderr)
        out["log"] += msg
        return out


# --------------------------------------------------------------------
# 6. Main (glue: run, save pass file, write logs, upsert outcomes)
# --------------------------------------------------------------------
def main():
    args = parse_args()
    ensure_dirs()

    results = run_screener(args.universe, args.with_options)

    # Write logs if present
    if results.get("log"):
        append_text(LOG_PATH, results["log"])

    # Save pass file (always attempt)
    passes = results.get("passes", pd.DataFrame())
    if passes is None:
        passes = pd.DataFrame()

    # If EvalDate is missing, fill with today's date (YYYY-MM-DD)
    if "EvalDate" not in passes.columns or passes["EvalDate"].isna().all():
        passes["EvalDate"] = UTC_NOW.strftime("%Y-%m-%d")

    write_csv(passes, PASS_PATH)

    # Optional raw scan
    if args.also_save_scan:
        scan_df = results.get("scan", pd.DataFrame())
        if scan_df is None:
            scan_df = pd.DataFrame()
        write_csv(scan_df, SCAN_PATH)

    # Upsert outcomes with the non-destructive rule
    upsert_outcomes(OUTCOMES_FILE, passes)

    # Human-readable console summary
    print("------------------------------------------------------------")
    print(f"[SUMMARY] Finished at {UTC_NOW.isoformat()}")
    print(f"[SUMMARY] pass file : {PASS_PATH}")
    if args.also_save_scan:
        print(f"[SUMMARY] scan file : {SCAN_PATH}")
    print(f"[SUMMARY] outcomes  : {OUTCOMES_FILE}")
    print("------------------------------------------------------------")


if __name__ == "__main__":
    main()
