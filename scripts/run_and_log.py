# scripts/run_and_log.py
"""
Run the swing screener, save a dated PASS CSV under data/history/,
and append PENDING rows to data/history/outcomes.csv for later settlement.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import pandas as pd

# --- Local imports (repo files) ---
try:
    import swing_options_screener as sos
except Exception as e:
    print(f"[FATAL] Could not import swing_options_screener: {e}", file=sys.stderr)
    raise

try:
    from sp_universe import get_sp500_tickers
except Exception:
    # Fallback: return empty; sos.run_scan will use defaults
    def get_sp500_tickers() -> list[str]:
        return []

# ---------- IO helpers ----------
ROOT = Path(".")
HIST_DIR = ROOT / "data" / "history"
LOG_DIR = ROOT / "data" / "logs"
OUTCOMES_PATH = HIST_DIR / "outcomes.csv"

def _utc_stamp() -> str:
    # e.g., 20250828-1351 (UTC)
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")

def _ensure_dirs():
    HIST_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---------- main ----------
def main():
    _ensure_dirs()

    ap = argparse.ArgumentParser(description="Run screener and write history.")
    ap.add_argument("--universe", choices=["sp500", "custom"], default="sp500")
    ap.add_argument("--tickers", type=str, default="", help="comma/space/newline separated")
    ap.add_argument("--with-options", action="store_true", help="include options columns")
    # pass through some knobs (keep defaults same as app)
    ap.add_argument("--res-days", type=int, default=sos.RES_LOOKBACK_DEFAULT)
    ap.add_argument("--relvol-min", type=float, default=sos.REL_VOL_MIN_DEFAULT)
    ap.add_argument("--rr-min", type=float, default=sos.RR_MIN_DEFAULT)
    ap.add_argument("--stop-mode", choices=["safest","structure"], default="safest")
    ap.add_argument("--opt-days", type=int, default=sos.TARGET_OPT_DAYS_DEFAULT)
    args = ap.parse_args()

    # --- Build universe ---
    if args.universe == "sp500":
        tickers = get_sp500_tickers() or None  # None => sos uses defaults safely
        if not tickers:
            print("[WARN] Could not fetch S&P 500 tickers; using module defaults.")
    else:
        # custom list from string, use module helper if present
        if hasattr(sos, "parse_ticker_text"):
            tickers = sos.parse_ticker_text(args.tickers)
        else:
            raw = args.tickers.replace("\n", ",").replace(" ", ",")
            tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
        if not tickers:
            tickers = None  # let sos pick defaults

    # --- Run scan (returns dict with 'pass_df') ---
    print("[INFO] Running screener…")
    out = sos.run_scan(
        tickers=tickers,
        res_days=args.res_days,
        rel_vol_min=args.relvol_min,
        relvol_median=False,
        rr_min=args.rr_min,
        stop_mode=args.stop_mode,
        with_options=True if args.with_options else True,  # keep options by default
        opt_days=args.opt_days,
    )
    df_pass: pd.DataFrame = out.get("pass_df", pd.DataFrame())

    ts = _utc_stamp()
    pass_csv = HIST_DIR / f"pass_{ts}.csv"

    if df_pass is None or df_pass.empty:
        # still write an empty file so the run is recorded
        print("[INFO] No PASS results. Writing empty pass file.")
        df_pass = pd.DataFrame(columns=[
            "Ticker","EvalDate","Price","EntryTimeET","Change%","RelVol(TimeAdj63d)",
            "Resistance","TP","RR_to_Res","RR_to_TP"
        ])
        df_pass.to_csv(pass_csv, index=False)
        print(f"[DONE] Wrote {pass_csv}")
        # nothing to append to outcomes
        return

    # Save full pass table for the run
    df_pass.to_csv(pass_csv, index=False)
    print(f"[DONE] Wrote PASS table: {pass_csv}  (rows={len(df_pass)})")

    # ---- Append PENDING rows to outcomes.csv ----
    # minimal columns needed by app History tab
    cols_needed = ["Ticker", "EvalDate", "Price", "EntryTimeET"]
    for c in cols_needed:
        if c not in df_pass.columns:
            df_pass[c] = ""  # ensure columns exist

    df_out = df_pass[cols_needed].copy()
    df_out["result_status"] = "PENDING"  # <- key for Pending count

    # Append (create header only if file doesn't exist yet)
    header = not OUTCOMES_PATH.exists()
    df_out.to_csv(OUTCOMES_PATH, mode="a", header=header, index=False)
    print(f"[DONE] Appended {len(df_out)} PENDING rows -> {OUTCOMES_PATH}")

if __name__ == "__main__":
    main()
