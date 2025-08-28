# scripts/check_hits.py
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

# --- bootstrap: add repo root so we can import swing_options_screener.py ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# --------------------------------------------------------------------------

import swing_options_screener as sos


HISTORY_DIR = os.path.join("data", "history")
HITS_DIR = os.path.join("data", "hits")

os.makedirs(HITS_DIR, exist_ok=True)


def check_day(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty or "Ticker" not in df.columns:
        return pd.DataFrame()

    out_rows = []
    for _, r in df.iterrows():
        tkr = str(r.get("Ticker", "")).strip().upper()
        tp = r.get("TP", None)
        eval_date = r.get("EvalDate", "")
        if not tkr or pd.isna(tp) or not eval_date:
            continue

        try:
            # fetch from eval date to today, unadjusted daily
            start = pd.to_datetime(eval_date) - timedelta(days=1)
            hist = yf.Ticker(tkr).history(start=start.date(), auto_adjust=False, actions=False)
            if hist is None or hist.empty:
                hit = False
            else:
                # consider Highs on/after eval date
                h = hist[hist.index.date >= pd.to_datetime(eval_date).date()]["High"]
                hit = bool((h >= float(tp)).any())
        except Exception:
            hit = False

        out_rows.append({
            "Ticker": tkr,
            "EvalDate": eval_date,
            "TP": tp,
            "Hit": bool(hit),
        })

    return pd.DataFrame(out_rows)


def main():
    idx_path = os.path.join(HISTORY_DIR, "index.csv")
    if not os.path.exists(idx_path):
        print("No history index yet.")
        return

    idx = pd.read_csv(idx_path)
    all_rows = []
    for _, row in idx.iterrows():
        csv_rel = row.get("csv", "")
        if not csv_rel:
            continue
        csv_path = os.path.join(os.getcwd(), csv_rel)
        if os.path.exists(csv_path):
            day_df = check_day(csv_path)
            if not day_df.empty:
                day_df["Day"] = row["date"]
                all_rows.append(day_df)

    if all_rows:
        out = pd.concat(all_rows, ignore_index=True)
        out.to_csv(os.path.join(HITS_DIR, "summary.csv"), index=False)

        # Commit updates
        os.system('git config user.name "github-actions"')
        os.system('git config user.email "actions@github.com"')
        os.system('git add -A')
        os.system('git commit -m "[skip ci] Update TP hit summary" || echo "Nothing to commit"')
        os.system('git push || echo "Nothing to push"')
    else:
        print("No hits evaluated (no rows).")


if __name__ == "__main__":
    main()
