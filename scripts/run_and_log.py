#!/usr/bin/env python3
"""
Run the screener, save the PASS file into data/history/,
and immediately append/merge a PENDING row for each PASS into
data/history/outcomes.csv (created if missing).

This guarantees the Streamlit UI shows today's items as Pending right away.
"""

import os
import sys
import csv
from datetime import datetime, timedelta, timezone
import pandas as pd

# Make repo root importable
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import swing_options_screener as sos  # your module

HIST_DIR = os.path.join(REPO_ROOT, "data", "history")
LOG_DIR  = os.path.join(REPO_ROOT, "data", "logs")
os.makedirs(HIST_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Columns we’ll ensure exist for outcomes
OPT_COLS = [
    "OptExpiry","BuyK","SellK","Width","DebitMid","DebitCons",
    "MaxProfitMid","MaxProfitCons","RR_Spread_Mid","RR_Spread_Cons",
    "BreakevenMid","PricingNote"
]

OUTCOME_COLS = [
    "Ticker","EvalDate","EntryTimeET","Price","TP","Resistance",
    "SupportType","SupportPrice",
    "OptExpiry","BuyK","SellK","Width",
    "result_status","result_note","hit_time","hit_price"
]


def _utcnow_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _ensure_opt_cols(df: pd.DataFrame) -> pd.DataFrame:
    for c in OPT_COLS:
        if c not in df.columns:
            df[c] = ""
    return df


def _fallback_expiry(eval_date_str: str) -> str:
    """If real option expiry not available, assume ~30 calendar days."""
    try:
        d = datetime.strptime(eval_date_str, "%Y-%m-%d")
        return (d + timedelta(days=30)).date().isoformat()
    except Exception:
        # fallback to 30d from today
        return (datetime.utcnow().date() + timedelta(days=30)).isoformat()


def _make_pending_rows(df_pass: pd.DataFrame) -> pd.DataFrame:
    """
    For each PASS, create a Pending outcome row immediately. If OptExpiry is
    blank, we synthesize one ≈ EvalDate + 30d so the UI can track it.
    """
    rows = []
    for _, r in df_pass.iterrows():
        eval_date = str(r.get("EvalDate",""))
        rows.append({
            "Ticker":       str(r.get("Ticker","")),
            "EvalDate":     eval_date,
            "EntryTimeET":  str(r.get("EntryTimeET","")),
            "Price":        r.get("Price",""),
            "TP":           r.get("TP",""),
            "Resistance":   r.get("Resistance",""),
            "SupportType":  r.get("SupportType",""),
            "SupportPrice": r.get("SupportPrice",""),
            "OptExpiry":    (str(r.get("OptExpiry","")) or _fallback_expiry(eval_date)),
            "BuyK":         r.get("BuyK",""),
            "SellK":        r.get("SellK",""),
            "Width":        r.get("Width",""),
            "result_status":"PENDING",
            "result_note":  "Awaiting TP hit or expiry",
            "hit_time":     "",
            "hit_price":    ""
        })
    return pd.DataFrame(rows, columns=OUTCOME_COLS)


def main():
    # ---------------- Run screener ----------------
    res = sos.run_scan(
        tickers=None,
        with_options=True,            # ensure options columns are requested
        res_days=sos.RES_LOOKBACK_DEFAULT,
        rel_vol_min=sos.REL_VOL_MIN_DEFAULT,
        relvol_median=False,
        rr_min=sos.RR_MIN_DEFAULT,
        stop_mode="safest",
        opt_days=sos.TARGET_OPT_DAYS_DEFAULT,
    )
    df_pass = res.get("pass_df", pd.DataFrame())

    # If no PASS, still write a short log and exit cleanly
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M")
    pass_name = f"pass_{ts}.csv"
    pass_path = os.path.join(HIST_DIR, pass_name)
    if df_pass is None or df_pass.empty:
        # Write an empty-but-valid file so we can see the run happened
        pd.DataFrame(columns=["Ticker","EvalDate","Price","EntryTimeET"]).to_csv(pass_path, index=False)
        print(f"[{_utcnow_iso()}] No passes; wrote empty {pass_path}")
        return

    # Guarantee option columns exist (in case of rate-limit/etc.)
    df_pass = _ensure_opt_cols(df_pass)

    # Save full PASS file
    df_pass.to_csv(pass_path, index=False)
    print(f"[{_utcnow_iso()}] Wrote PASS file: {pass_path} ({len(df_pass)} rows)")

    # ---------------- Seed outcomes as Pending ----------------
    outcomes_path = os.path.join(HIST_DIR, "outcomes.csv")
    pending = _make_pending_rows(df_pass)

    if os.path.exists(outcomes_path):
        cur = pd.read_csv(outcomes_path)
        # Merge by (Ticker, EvalDate); keep existing rows (in case they were updated later)
        key = ["Ticker","EvalDate"]
        merged = pd.concat([cur, pending], ignore_index=True)
        merged = merged.sort_values(by=["EvalDate","Ticker"]).drop_duplicates(subset=key, keep="first")
        merged.to_csv(outcomes_path, index=False, quoting=csv.QUOTE_MINIMAL)
    else:
        pending.to_csv(outcomes_path, index=False, quoting=csv.QUOTE_MINIMAL)

    print(f"[{_utcnow_iso()}] Outcomes updated: {outcomes_path}")


if __name__ == "__main__":
    main()


