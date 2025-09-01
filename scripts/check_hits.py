#!/usr/bin/env python3
"""
Update data/history/outcomes.csv:
- For each PENDING row, fetch daily history since EvalDate.
- If any day's HIGH >= TP -> mark HIT, record hit_time + hit_price.
- If past OptExpiry and never hit -> mark MISS.
"""

import os, sys, csv
from datetime import datetime, date, timezone
import pandas as pd
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from utils.prices import fetch_history

HIST_DIR = os.path.join(REPO_ROOT, "data", "history")
OUT_PATH = os.path.join(HIST_DIR, "outcomes.csv")

def _parse_date(s):
    try:
        return datetime.strptime(str(s), "%Y-%m-%d").date()
    except Exception:
        return None

def main():
    if not os.path.exists(OUT_PATH):
        print("No outcomes.csv yet; nothing to check.")
        return

    df = pd.read_csv(OUT_PATH)
    if df.empty:
        print("outcomes.csv empty; nothing to check.")
        return

    today = datetime.now(timezone.utc).date()

    # Work only on PENDING rows
    pend_mask = (df.get("result_status","") == "PENDING")
    df_p = df[pend_mask].copy()
    if df_p.empty:
        print("No pending rows; done.")
        return

    updates = []
    for idx, r in df_p.iterrows():
        tkr = str(r["Ticker"])
        tp  = r.get("TP", np.nan)
        tp  = float(tp) if pd.notna(tp) else np.nan

        d0  = _parse_date(r.get("EvalDate",""))
        exp = _parse_date(r.get("OptExpiry",""))

        if pd.isna(tp) or d0 is None:
            continue

        hist = fetch_history(tkr, start=d0, end=today, auto_adjust=False)

        hit_time = ""
        hit_price = ""

        hit = False
        if hist is not None and not hist.empty and "High" in hist.columns:
            # if any High >= TP
            highs = hist["High"].astype(float)
            hit_idx = highs[highs >= tp]
            if not hit_idx.empty:
                hit = True
                first = hit_idx.index[0]
                hit_time = first.strftime("%Y-%m-%d")
                hit_price = float(highs.loc[first])

        if hit:
            df.loc[df.index == idx, "result_status"] = "HIT"
            df.loc[df.index == idx, "result_note"]   = "TP reached by daily high"
            df.loc[df.index == idx, "hit_time"]      = hit_time
            df.loc[df.index == idx, "hit_price"]     = hit_price
        else:
            # not hit; if past expiry -> MISS
            if exp is not None and today > exp:
                df.loc[df.index == idx, "result_status"] = "MISS"
                df.loc[df.index == idx, "result_note"]   = "Expired without TP"
                df.loc[df.index == idx, "hit_time"]      = ""
                df.loc[df.index == idx, "hit_price"]     = ""

    df.to_csv(OUT_PATH, index=False, quoting=csv.QUOTE_MINIMAL)
    print("Updated outcomes.csv")

if __name__ == "__main__":
    main()

