#!/usr/bin/env python3
# scripts/score_history.py
# - Reads data/history/outcomes.csv
# - For Outcome == PENDING, checks if TargetLevel was hit (High >= level)
#   between EvalDate (inclusive) and min(WindowEnd, today) (inclusive).
# - If hit: Outcome=YES, HitDate=first hit date, MaxHigh=max high in window
# - Else if today > WindowEnd and not hit: Outcome=NO, MaxHigh=max high
# - Else: still PENDING, but refreshes CheckedAtUTC
import os
import sys
from datetime import datetime, date
import pandas as pd
import numpy as np
import yfinance as yf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

HIST_DIR = os.path.join(ROOT, "data", "history")
OUTCOMES_CSV = os.path.join(HIST_DIR, "outcomes.csv")


def _coerce_date(s):
    try:
        return pd.to_datetime(s).date()
    except Exception:
        return None


def _check_row(row):
    """
    Returns updated row dict with Outcome/HItDate/MaxHigh potentially updated.
    """
    outcome = str(row.get("Outcome", "PENDING")).upper()
    if outcome not in ("PENDING", "YES", "NO"):
        outcome = "PENDING"

    if outcome != "PENDING":
        return row  # already decided

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

    # Safety checks
    if eval_date is None or window_end is None or not np.isfinite(target):
        row["CheckedAtUTC"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        return row

    start = pd.Timestamp(eval_date)
    stop  = pd.Timestamp(min(window_end, today))

    # If start later than stop (clock skew?), do nothing
    if start > stop:
        row["CheckedAtUTC"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        return row

    try:
        # UNADJUSTED daily, like the screener
        df = yf.Ticker(tkr).history(start=start, end=stop + pd.Timedelta(days=1),
                                    auto_adjust=False, actions=False)  # end is exclusive, hence +1 day
        if df is None or df.empty:
            row["CheckedAtUTC"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            return row

        highs = df["High"].dropna()
        if highs.empty:
            row["CheckedAtUTC"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            return row

        # First hit date
        hit_mask = highs >= target
        if hit_mask.any():
            first_idx = hit_mask[hit_mask].index[0]
            row["Outcome"] = "YES"
            row["HitDate"] = first_idx.date().isoformat()
            row["MaxHigh"] = float(highs.max())
        else:
            # Not hit yet
            row["MaxHigh"] = float(highs.max())
            if today > window_end:
                row["Outcome"] = "NO"  # expired without hit

        row["CheckedAtUTC"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        return row

    except Exception:
        row["CheckedAtUTC"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        return row


def main():
    if not os.path.exists(OUTCOMES_CSV):
        print("No outcomes.csv yet; nothing to score.")
        return

    df = pd.read_csv(OUTCOMES_CSV)
    if df.empty:
        print("outcomes.csv empty; nothing to score.")
        return

    rows = df.to_dict(orient="records")
    updated = [_check_row(r) for r in rows]
    new_df = pd.DataFrame(updated)
    new_df.to_csv(OUTCOMES_CSV, index=False)
    print(f"Scored {len(new_df)} rows → wrote outcomes.csv")


if __name__ == "__main__":
    main()

