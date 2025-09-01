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

# Shared outcome helpers
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.outcomes import upsert_and_backfill_outcomes, settle_pending_outcomes

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
# 5. Screener Runner (invoke library, gather DataFrames)
# --------------------------------------------------------------------
from typing import Tuple, Optional
import pandas as pd
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
# 6. Main (glue: run, save pass file, write logs, upsert outcomes)
# --------------------------------------------------------------------
def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

def main() -> int:
    ensure_dirs()

    try:
        res = _safe_engine_run_scan()
        df_pass: Optional[pd.DataFrame] = res.get("pass")
        # df_scan = res.get("scan")  # currently unused here

        wrote_pass = False
        if isinstance(df_pass, pd.DataFrame) and not df_pass.empty:
            # filename with UTC timestamp
            pass_name = f"pass_{_utc_ts()}.csv"
            pass_path = HISTORY_DIR / pass_name
            df_pass.to_csv(pass_path, index=False)
            print(f"[run_and_log] wrote {pass_path}")
            wrote_pass = True

            # Update outcomes.csv (insert new, backfill, settle)
            upsert_and_backfill_outcomes(df_pass, str(OUTCOMES_FILE))
            settle_pending_outcomes(str(OUTCOMES_FILE))
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




