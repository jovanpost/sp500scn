# check_hits.py
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf

HIST_FILE = os.path.join("history", "passes.csv")

def to_date(s):
    try:
        return datetime.strptime(str(s), "%Y-%m-%d").date()
    except Exception:
        try:
            return pd.to_datetime(s).date()
        except Exception:
            return None

def first_hit_date(high_series: pd.Series, level: float):
    """Return the first date where High >= level, else None."""
    if not np.isfinite(level): return None
    hits = high_series[high_series >= level]
    if hits.empty: return None
    return hits.index[0].date().isoformat()

def fetch_daily_highs(ticker: str, start_date: datetime):
    try:
        df = yf.Ticker(ticker).history(start=start_date, auto_adjust=False, actions=False)
        if df is None or df.empty: return None
        # ensure tz-naive index
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df.index = df.index.tz_convert(None)
        return df["High"]
    except Exception:
        return None

def main():
    if not os.path.exists(HIST_FILE):
        print("No history file to check.")
        return

    hist = pd.read_csv(HIST_FILE)
    if hist.empty:
        print("History empty.")
        return

    # Only rows without hits yet
    pending = hist[(hist["HitShortStrike"] == "") | (hist["HitTP"] == "")]
    if pending.empty:
        print("No pending rows to check.")
        return

    updates = 0
    for idx, row in pending.iterrows():
        tkr = row.get("Ticker")
        eval_date = to_date(row.get("EvalDate"))
        if not tkr or eval_date is None:
            continue

        # We check highs from the next day after the eval date through today
        start = eval_date + timedelta(days=1)

        highs = fetch_daily_highs(tkr, start)
        if highs is None or highs.empty:
            continue

        # Short strike and TP from the row
        sell_k = row.get("SellK")
        tp_val = row.get("TP")

        # Short strike hit
        if row.get("HitShortStrike","") == "" and pd.notna(sell_k):
            try:
                sell_k = float(sell_k)
                hdate = first_hit_date(highs, sell_k)
                if hdate:
                    hist.at[idx, "HitShortStrike"] = "YES"
                    hist.at[idx, "HitShortStrikeDate"] = hdate
                    updates += 1
            except Exception:
                pass

        # TP hit
        if row.get("HitTP","") == "" and pd.notna(tp_val):
            try:
                tp_val = float(tp_val)
                hdate = first_hit_date(highs, tp_val)
                if hdate:
                    hist.at[idx, "HitTP"] = "YES"
                    hist.at[idx, "HitTPDate"] = hdate
                    updates += 1
            except Exception:
                pass

    if updates > 0:
        hist.to_csv(HIST_FILE, index=False)
        print(f"Updated {updates} hit flags.")
    else:
        print("No new hits.")

if __name__ == "__main__":
    main()
