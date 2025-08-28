# scripts/score_history.py
# Score historical PASS rows: did price reach TP (or upper strike) before the window ended?

import os
import glob
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf

HIST_DIR = "data/history"
OUT_DIR  = "data/outcomes"
OUT_FILE = os.path.join(OUT_DIR, "outcomes.csv")

os.makedirs(OUT_DIR, exist_ok=True)

def bday_add(d: pd.Timestamp, n: int) -> pd.Timestamp:
    return (pd.to_datetime(d) + pd.tseries.offsets.BDay(n)).normalize()

def pick_target_level(row) -> tuple[float, str]:
    tp = row.get("TP", np.nan)
    sellk = row.get("SellK", np.nan)
    try: tp = float(tp)
    except Exception: tp = np.nan
    try: sellk = float(sellk)
    except Exception: sellk = np.nan

    if np.isfinite(tp) and np.isfinite(sellk):
        return (sellk, "SellK") if sellk >= tp else (tp, "TP")
    if np.isfinite(tp):    return (tp, "TP")
    if np.isfinite(sellk): return (sellk, "SellK")
    return (np.nan, "TP")

def score_one(row, today_utc: pd.Timestamp):
    tkr = str(row.get("Ticker","")).upper()
    eval_date = pd.to_datetime(row.get("EvalDate"), errors="coerce")
    run_file  = str(row.get("RunFile",""))
    if not tkr or pd.isna(eval_date):
        return None

    # target = max(TP, SellK)
    target_level, target_type = pick_target_level(row)
    if not np.isfinite(target_level):
        return {
            "Ticker": tkr, "EvalDate": eval_date.date().isoformat(), "RunFile": run_file,
            "Outcome": "PENDING", "Reason": "no_target", "TargetLevel": "", "TargetType": target_type,
            "HitDate": "", "WindowEnd": "", "WindowClosed": False, "MaxHigh": ""
        }

    # window end = min(OptExpiry, EvalDate + 21 business days)
    opt_exp = pd.to_datetime(row.get("OptExpiry", None), errors="coerce") if row.get("OptExpiry", None) else pd.NaT
    end_21b = bday_add(eval_date, 21)
    window_end = min(opt_exp.normalize(), end_21b) if not pd.isna(opt_exp) else end_21b

    start = eval_date + pd.Timedelta(days=1)
    end   = window_end + pd.Timedelta(days=1)

    try:
        df = yf.Ticker(tkr).history(start=start.strftime("%Y-%m-%d"),
                                    end=end.strftime("%Y-%m-%d"),
                                    auto_adjust=False, actions=False)
        if df is None or df.empty:
            max_high = np.nan
            hit_date = ""
        else:
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            highs = pd.to_numeric(df["High"], errors="coerce")
            max_high = float(np.nanmax(highs.values)) if len(highs) else np.nan
            mask = highs >= target_level
            hit_date = df.index[mask][0].date().isoformat() if mask.any() else ""
    except Exception:
        max_high = np.nan
        hit_date = ""

    window_closed = (pd.Timestamp.utcnow().normalize() > window_end)
    outcome = "YES" if hit_date else ("NO" if window_closed else "PENDING")

    return {
        "Ticker": tkr,
        "EvalDate": eval_date.date().isoformat(),
        "RunFile": run_file,
        "Outcome": outcome,
        "Reason": "",
        "TargetLevel": round(float(target_level), 4),
        "TargetType": target_type,
        "HitDate": hit_date,
        "WindowEnd": window_end.date().isoformat(),
        "WindowClosed": bool(window_closed),
        "MaxHigh": ("" if not np.isfinite(max_high) else round(float(max_high), 4)),
    }

def main():
    files = sorted(glob.glob(os.path.join(HIST_DIR, "pass_*.csv")))
    if not files:
        print("No history files found; nothing to score.")
        return

    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["RunFile"] = os.path.basename(f)
            frames.append(df)
        except Exception:
            continue
    if not frames:
        print("No readable history CSVs; nothing to score.")
        return

    hist = pd.concat(frames, ignore_index=True)
    rows = []
    today_utc = pd.Timestamp.utcnow()
    for _, r in hist.iterrows():
        s = score_one(r, today_utc)
        if s is not None:
            rows.append(s)

    out = pd.DataFrame(rows)
    if os.path.exists(OUT_FILE):
        prev = pd.read_csv(OUT_FILE)
        key = ["Ticker","EvalDate","RunFile"]
        out = (prev.set_index(key).combine_first(out.set_index(key))).reset_index()
        out = out.drop_duplicates(subset=key, keep="last")

    out.sort_values(["EvalDate","Ticker"], inplace=True)
    out.to_csv(OUT_FILE, index=False)
    print(f"Wrote outcomes -> {OUT_FILE}  ({len(out)} rows)")

if __name__ == "__main__":
    main()
