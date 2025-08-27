# run_and_log.py
import os
from datetime import datetime, timezone
import pandas as pd

import swing_options_screener as sos

HIST_DIR = "history"
HIST_FILE = os.path.join(HIST_DIR, "passes.csv")

os.makedirs(HIST_DIR, exist_ok=True)

def load_history():
    if os.path.exists(HIST_FILE):
        return pd.read_csv(HIST_FILE)
    return pd.DataFrame()

def main():
    # Run with your defaults; include options suggestion columns
    out = sos.run_scan(
        tickers=None,  # your code already handles default universe
        res_days=sos.RES_LOOKBACK_DEFAULT,
        rel_vol_min=sos.REL_VOL_MIN_DEFAULT,
        relvol_median=False,
        rr_min=sos.RR_MIN_DEFAULT,
        stop_mode="safest",
        with_options=True,
        opt_days=getattr(sos, "TARGET_OPT_DAYS_DEFAULT", 30),
    )
    df = out.get("pass_df", pd.DataFrame())
    if df is None or df.empty:
        print("No PASS tickers this run.")
        return

    # Minimal normalized schema we’ll keep for history
    cols = [
        "Ticker","EvalDate","EntryTimeET","Price","Change%","RelVol(TimeAdj63d)",
        "Resistance","TP","RR_to_Res","RR_to_TP","SupportType","SupportPrice","Risk$",
        "TPReward$","TPReward%","ResReward$","ResReward%","DailyATR","DailyCap",
        "Hist21d_PassCount","Hist21d_Max%","ResLookbackDays",
        "OptExpiry","BuyK","SellK","Width","DebitMid","DebitCons","MaxProfitMid",
        "MaxProfitCons","RR_Spread_Mid","RR_Spread_Cons","BreakevenMid"
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = None

    # Add run metadata
    run_ts_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
    df["RunUTC"] = run_ts_utc

    # Hit tracking placeholders
    if "HitShortStrike" not in df.columns: df["HitShortStrike"] = ""
    if "HitShortStrikeDate" not in df.columns: df["HitShortStrikeDate"] = ""
    if "HitTP" not in df.columns: df["HitTP"] = ""
    if "HitTPDate" not in df.columns: df["HitTPDate"] = ""

    # Append to CSV history
    hist = load_history()
    hist = pd.concat([hist, df[cols + ["RunUTC","HitShortStrike","HitShortStrikeDate","HitTP","HitTPDate"]]], ignore_index=True)
    hist.to_csv(HIST_FILE, index=False)
    print(f"Appended {len(df)} PASS rows to {HIST_FILE}")

if __name__ == "__main__":
    main()
