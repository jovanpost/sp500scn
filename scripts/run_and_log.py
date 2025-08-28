#!/usr/bin/env python3
# Minimal runner that always writes outputs & a log so Actions has something to save.

import os
import sys
from datetime import datetime
import pandas as pd

# Make repo root importable
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import swing_options_screener as sos  # your module

def ensure_dirs():
    os.makedirs("data/history", exist_ok=True)
    os.makedirs("data/logs", exist_ok=True)

def stamp_et() -> str:
    # just for filenames; UTC is fine too — using ET keeps consistency with UI text
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", choices=["sp500", "custom"], default="sp500")
    ap.add_argument("--tickers", type=str, default="")
    ap.add_argument("--with-options", action="store_true")
    args = ap.parse_args()

    ensure_dirs()
    ts = stamp_et()

    # Run screener (this returns {'pass_df': df})
    result = sos.run_scan(
        tickers=None if args.universe == "sp500" else sos.parse_ticker_text(args.tickers),
        with_options=args.with_options,
    )

    df = result.get("pass_df", pd.DataFrame())

    # Always write an execution log (so artifacts never come up empty)
    log_path = f"data/logs/scan_{ts}.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Run at {datetime.utcnow().isoformat()}Z\n")
        if df is None or df.empty:
            f.write("No PASS tickers found.\n")
        else:
            f.write(f"PASS count: {len(df)}\n")
            f.write("Columns: " + ", ".join(df.columns) + "\n")
            try:
                f.write(df.head(20).to_string(index=False) + "\n")
            except Exception:
                pass

    # Write today’s outputs at the repo root (for easy artifact pickup)
    if df is None or df.empty:
        # Still create empty files so the artifact step finds something
        pd.DataFrame().to_csv("pass_tickers.csv", index=False)
        pd.DataFrame().to_csv("pass_tickers_unadjusted.psv", index=False, sep="|")
    else:
        # Sorted by Price ascending as per your UI preference
        df = df.sort_values(["Price", "Ticker"])
        df.to_csv("pass_tickers.csv", index=False)
        df.to_csv("pass_tickers_unadjusted.psv", index=False, sep="|")

    # Also drop a timestamped copy into history
    hist_csv = f"data/history/pass_{ts}.csv"
    hist_psv = f"data/history/pass_{ts}.psv"
    try:
        pd.read_csv("pass_tickers.csv").to_csv(hist_csv, index=False)
    except Exception:
        pd.DataFrame().to_csv(hist_csv, index=False)
    try:
        pd.read_csv("pass_tickers_unadjusted.psv", sep="|").to_csv(hist_psv, index=False)
    except Exception:
        pd.DataFrame().to_csv(hist_psv, index=False)

    print("Scan complete.")
    print(f"Wrote: pass_tickers.csv, pass_tickers_unadjusted.psv, {hist_csv}, {hist_psv}, {log_path}")

if __name__ == "__main__":
    main()
