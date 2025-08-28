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
from pathlib import Path

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
# ============================================================
# 4) Outcomes Upsert (non-destructive append/update)
# ------------------------------------------------------------
# Purpose:
#   - Create "PENDING" rows from the latest pass DataFrame
#   - Upsert them into data/history/outcomes.csv
#     * Never clobber a resolved row (HIT/MISS/EXPIRED/CANCELLED)
#     * Ensure the file and header exist
#     * Deduplicate by a stable key: (Ticker, EvalDate, EntryTimeET)
#
# Public entry point for section:
#   upsert_outcomes_from_pass(df_pass: pd.DataFrame) -> Path
# ============================================================

from typing import List

_RESOLVED_STATES = {"HIT", "MISS", "EXPIRED", "CANCELLED"}

def _ensure_outcomes_file() -> pd.DataFrame:
    """Load outcomes.csv or initialize an empty one with the canonical header."""
    out_file = HIST_DIR / "outcomes.csv"
    if not out_file.exists():
        pd.DataFrame(columns=OUTCOME_COLS).to_csv(out_file, index=False)
        return pd.DataFrame(columns=OUTCOME_COLS)
    try:
        df = pd.read_csv(out_file)
    except Exception:
        # If the file is corrupt, reinitialize with header (last resort).
        df = pd.DataFrame(columns=OUTCOME_COLS)
        df.to_csv(out_file, index=False)
    # Guarantee all columns exist and in the right order
    for c in OUTCOME_COLS:
        if c not in df.columns:
            df[c] = None
    return df[OUTCOME_COLS]

def _make_pending_rows(df_pass: pd.DataFrame) -> pd.DataFrame:
    """
    Map pass rows to canonical OUTCOME_COLS with Status=PENDING.
    Missing source columns are filled with None.
    """
    if df_pass is None or df_pass.empty:
        return pd.DataFrame(columns=OUTCOME_COLS)

    base = df_pass.copy()

    # Ensure required source fields exist (fill missing)
    src_needed = ["Ticker", "EvalDate", "Price", "EntryTimeET", "TP", "OptExpiry", "BuyK", "SellK"]
    for c in src_needed:
        if c not in base.columns:
            base[c] = None

    pending = pd.DataFrame({
        "Ticker":      base["Ticker"],
        "EvalDate":    base["EvalDate"],
        "Price":       base["Price"],
        "EntryTimeET": base["EntryTimeET"],
        "Status":      "PENDING",
        "HitDateET":   None,
        "Expiry":      base["OptExpiry"],
        "BuyK":        base["BuyK"],
        "SellK":       base["SellK"],
        "TP":          base["TP"],
        "Notes":       None,
    }, columns=OUTCOME_COLS)

    # Normalize types (especially dates as strings) to avoid merge weirdness
    for col in ["EvalDate", "EntryTimeET", "HitDateET", "Expiry"]:
        if col in pending.columns:
            pending[col] = pending[col].astype(str).replace({"nan": None, "NaT": None})
    return pending

def _coalesce_cols(df_target: pd.DataFrame, df_source: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    For each column in cols: if target value is null and source has a value, fill it.
    This is used to non-destructively add details to existing rows.
    """
    for c in cols:
        if c in df_target.columns and c in df_source.columns:
            df_target[c] = df_target[c].where(df_target[c].notna(), df_source[c])
    return df_target

def upsert_outcomes_from_pass(df_pass: pd.DataFrame) -> Path:
    """
    Non-destructive upsert of PENDING rows derived from df_pass into outcomes.csv.
    - Keeps resolved rows intact (HIT/MISS/EXPIRED/CANCELLED)
    - Adds any missing PENDING keys
    - Coalesces empty fields (e.g., TP/Expiry) without overwriting existing values
    """
    out_file = HIST_DIR / "outcomes.csv"
    df_out = _ensure_outcomes_file()

    # Build incoming PENDING rows
    df_new = _make_pending_rows(df_pass)
    if df_new.empty:
        # Nothing to upsert; still ensure file exists with proper header
        df_out.to_csv(out_file, index=False)
        return out_file

    # Stable key used for idempotent upsert
    key = ["Ticker", "EvalDate", "EntryTimeET"]

    # Split existing into resolved vs open
    if not df_out.empty:
        resolved_mask = df_out["Status"].isin(_RESOLVED_STATES)
        df_resolved = df_out[resolved_mask].copy()
        df_open     = df_out[~resolved_mask].copy()
    else:
        df_resolved = pd.DataFrame(columns=OUTCOME_COLS)
        df_open     = pd.DataFrame(columns=OUTCOME_COLS)

    # Left-merge new PENDING against existing OPEN to detect matches
    if not df_open.empty:
        merged = df_new.merge(df_open, on=key, how="left", suffixes=("", "_old"), indicator=True)
        # Rows already present in OPEN: keep existing row but coalesce missing fields
        in_open = merged["_merge"].eq("both")
        to_update_keys = merged.loc[in_open, key]

        if not to_update_keys.empty:
            # Extract the current open rows and new rows aligned by key
            open_on_key = df_open.merge(to_update_keys, on=key, how="inner")
            new_on_key  = df_new.merge(to_update_keys, on=key, how="inner")

            # Coalesce into the open rows (do NOT overwrite non-null with null)
            open_on_key = _coalesce_cols(
                open_on_key, new_on_key,
                cols=["Price", "Status", "HitDateET", "Expiry", "BuyK", "SellK", "TP", "Notes"]
            )

            # Rebuild df_open: replace the affected keys with the coalesced version
            keep_open = df_open.merge(to_update_keys, on=key, how="left", indicator=True)
            keep_open = keep_open[keep_open["_merge"] == "left_only"].drop(columns=["_merge"])
            df_open = pd.concat([keep_open, open_on_key], ignore_index=True)

        # Rows not present in OPEN become brand-new OPEN rows
        to_insert = merged["_merge"].eq("left_only")
        insert_rows = merged.loc[to_insert, OUTCOME_COLS]
        df_open = pd.concat([df_open, insert_rows], ignore_index=True)
    else:
        # No open rows exist; all new rows are inserted
        df_open = df_new.copy()

    # Reassemble: resolved rows first preserved, then (updated/new) open rows
    df_final = pd.concat([df_resolved[OUTCOME_COLS], df_open[OUTCOME_COLS]], ignore_index=True)

    # Final de-duplication by key, keeping the last occurrence (most recent write)
    if not df_final.empty:
        df_final = df_final.sort_values(key).drop_duplicates(subset=key, keep="last")

    # Enforce column order and write
    for c in OUTCOME_COLS:
        if c not in df_final.columns:
            df_final[c] = None
    df_final = df_final[OUTCOME_COLS]
    df_final.to_csv(out_file, index=False)
    return out_file

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
