#!/usr/bin/env python3
# scripts/run_and_log.py
# - Runs the screener
# - Saves today's PASS list to data/history/pass_YYYYMMDD-HHMM.csv
# - ALSO writes/updates data/history/outcomes.csv with new rows initialized as PENDING

import os
import sys
from datetime import datetime, timedelta
import pandas as pd

# allow importing from repo root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import argparse
import swing_options_screener as sos  # your module

HIST_DIR = os.path.join(ROOT, "data", "history")
os.makedirs(HIST_DIR, exist_ok=True)
OUTCOMES_CSV = os.path.join(HIST_DIR, "outcomes.csv")


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--universe", choices=["custom", "sp500"], default="sp500")
    p.add_argument("--tickers", type=str, default="")
    p.add_argument("--res-days", type=int, default=sos.RES_LOOKBACK_DEFAULT)
    p.add_argument("--relvol-min", type=float, default=sos.REL_VOL_MIN_DEFAULT)
    p.add_argument("--relvol-median", action="store_true")
    p.add_argument("--rr-min", type=float, default=sos.RR_MIN_DEFAULT)
    p.add_argument("--stop-mode", choices=["safest","structure"], default="safest")
    p.add_argument("--with-options", action="store_true", help="append option suggestion")
    p.add_argument("--opt-days", type=int, default=sos.TARGET_OPT_DAYS_DEFAULT)
    return p.parse_args()


def _today_timestamp():
    now = datetime.utcnow()
    return now.strftime("%Y%m%d-%H%M")


def _coerce_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def _evaldate_to_dt(row):
    # EvalDate is ISO date string (yyyy-mm-dd)
    try:
        return datetime.strptime(str(row["EvalDate"]), "%Y-%m-%d")
    except Exception:
        # fallback: try to coerce from other formats
        try:
            return pd.to_datetime(row["EvalDate"]).to_pydatetime()
        except Exception:
            return None


def _initial_outcome_row(row):
    """
    Build the initial outcome fields for a PASS row:
    - TargetType: OPTION_SELLK if we have a SellK, else TP_PRICE
    - TargetLevel: float(SellK) or float(TP)
    - WindowEnd: OptExpiry (if present) else EvalDate + 30 calendar days
    - Outcome=PENDING, HitDate="", MaxHigh=""
    """
    sellk = row.get("SellK", "")
    tp    = row.get("TP", "")
    expiry = row.get("OptExpiry", "")

    if str(sellk).strip() != "":
        target_type = "OPTION_SELLK"
        target_level = _coerce_float(sellk)
    else:
        target_type = "TP_PRICE"
        target_level = _coerce_float(tp)

    # Window end
    if str(expiry).strip() != "":
        window_end = str(expiry)  # already ISO date
    else:
        evdt = _evaldate_to_dt(row)
        if evdt is None:
            # fallback: 30 days from now
            window_end = (datetime.utcnow() + timedelta(days=30)).date().isoformat()
        else:
            window_end = (evdt + timedelta(days=30)).date().isoformat()

    out = dict(
        Outcome="PENDING",
        TargetType=target_type,
        TargetLevel=target_level,
        WindowEnd=window_end,
        HitDate="",
        MaxHigh="",
        CheckedAtUTC=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    )
    return out


def _unique_key(df):
    """
    Build a de-dup key: (Ticker, EvalDate, EntryTimeET).
    If EntryTimeET missing, fall back to (Ticker, EvalDate).
    """
    key = df.get("EntryTimeET")
    if key is None or key.isna().all():
        return df["Ticker"].astype(str) + "|" + df["EvalDate"].astype(str)
    return df["Ticker"].astype(str) + "|" + df["EvalDate"].astype(str) + "|" + df["EntryTimeET"].astype(str)


def main():
    args = _parse_args()

    # Run the screener
    if args.universe == "sp500":
        tickers = None  # sos will pull live S&P 500 when None
    else:
        tickers = sos.parse_ticker_text(args.tickers) if args.tickers else None

    out = sos.run_scan(
        tickers=tickers,
        res_days=args.res_days,
        rel_vol_min=args.relvol_min,
        relvol_median=args.relvol_median,
        rr_min=args.rr_min,
        stop_mode=args.stop_mode,
        with_options=args.with_options,
        opt_days=args.opt_days,
    )

    df = out["pass_df"]
    ts = _today_timestamp()
    pass_path = os.path.join(HIST_DIR, f"pass_{ts}.csv")
    df.to_csv(pass_path, index=False)
    print(f"Wrote {pass_path} with {len(df)} rows")

    # Prepare/merge outcomes with PENDING rows immediately
    cols_keep = list(df.columns)  # keep original columns
    # Add initial outcome fields
    enriched = []
    for r in df.to_dict(orient="records"):
        r2 = dict(r)
        r2.update(_initial_outcome_row(r))
        enriched.append(r2)

    df_new = pd.DataFrame(enriched)

    # Merge into outcomes.csv (append, then drop duplicates by key)
    if os.path.exists(OUTCOMES_CSV):
        old = pd.read_csv(OUTCOMES_CSV)
        merged = pd.concat([old, df_new], ignore_index=True)
    else:
        merged = df_new

    merged["_key"] = _unique_key(merged)
    merged = merged.sort_values(["EvalDate", "Ticker"]).drop_duplicates("_key", keep="last")
    merged = merged.drop(columns=["_key"])

    merged.to_csv(OUTCOMES_CSV, index=False)
    print(f"Updated {OUTCOMES_CSV} with {len(merged)} total rows")


if __name__ == "__main__":
    main()

