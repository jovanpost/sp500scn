#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=======================================================================
Bibliography (Section Index)
=======================================================================
1. Imports & CLI
2. Paths, Constants, Helpers
3. Screener Runner (invoke library, gather DataFrames)
4. Main (glue: run, save pass file, write logs, upsert outcomes)
=======================================================================
"""

# --------------------------------------------------------------------
# 1. Imports & CLI
# --------------------------------------------------------------------
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

# Optional universe helper (only used if present)
try:
    import sp_universe as spuni  # optional
except Exception:
    spuni = None

# Shared outcome helpers
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.io import DATA_DIR, HISTORY_DIR, OUTCOMES_CSV, write_csv
from utils.outcomes import (
    upsert_and_backfill_outcomes,
    evaluate_outcomes,
    write_outcomes,
)

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
LOGS_DIR = DATA_DIR / "logs"


def ensure_dirs() -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------
# 3. Screener Runner (invoke library, gather DataFrames)
# --------------------------------------------------------------------
from typing import Optional
from utils.scan import safe_run_scan

# --------------------------------------------------------------------
# 4. Main (glue: run, save pass file, write logs, upsert outcomes)
# --------------------------------------------------------------------
def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

def main() -> int:
    ensure_dirs()

    try:
        res = safe_run_scan()
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

            # Update outcomes.csv (insert new, backfill, evaluate)
            out_df = upsert_and_backfill_outcomes(df_pass, OUTCOMES_CSV)
            out_df = evaluate_outcomes(out_df, mode="pending")
            write_outcomes(out_df, OUTCOMES_CSV)
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




