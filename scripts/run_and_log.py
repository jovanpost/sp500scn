# scripts/run_and_log.py
import os
import io
from datetime import datetime
import pandas as pd

from swing_options_screener import run_scan

# Run the scan (same defaults as your app)
out = run_scan(
    tickers=None,
    res_days=21,
    rel_vol_min=1.10,
    rr_min=2.0,
    with_options=True,
    opt_days=30,
)

df = out.get("pass_df", pd.DataFrame())
run_time_et = datetime.now().strftime("%Y-%m-%d %H:%M:%S ET")
run_tag = datetime.now().strftime("%Y-%m-%d_%H%M")   # e.g., 2025-08-27_1030
os.makedirs("history", exist_ok=True)

# Ensure consistent columns even if empty
if df is None or df.empty:
    pass_df = pd.DataFrame(columns=[
        'Ticker','EvalDate','Price','EntryTimeET','Change%','RelVol(TimeAdj63d)',
        'Resistance','TP','RR_to_Res','RR_to_TP','SupportType','SupportPrice','Risk$',
        'TPReward$','TPReward%','ResReward$','ResReward%','DailyATR','DailyCap',
        'Hist21d_PassCount','Hist21d_Max%','Hist21d_Examples','ResLookbackDays','Prices',
        'Session','EntrySrc','VolSrc',
        'OptExpiry','BuyK','SellK','Width','DebitMid','DebitCons','MaxProfitMid',
        'MaxProfitCons','RR_Spread_Mid','RR_Spread_Cons','BreakevenMid','PricingNote'
    ])
else:
    pass_df = df.copy()

# Annotate with run time
pass_df["RunTimeET"] = run_time_et

# Write per-run CSV
csv_rel_path = f"history/{run_tag}.csv"
pass_df.to_csv(csv_rel_path, index=False)

# Update index.csv (append a row)
index_path = "history/index.csv"
idx_cols = ["RunTimeET", "CSVPath", "PassCount"]

# Load existing index if present
if os.path.exists(index_path):
    idx = pd.read_csv(index_path)
else:
    idx = pd.DataFrame(columns=idx_cols)

new_row = pd.DataFrame([{
    "RunTimeET": run_time_et,
    "CSVPath": csv_rel_path,
    "PassCount": int(len(pass_df))
}])

idx = pd.concat([idx, new_row], ignore_index=True)
idx.to_csv(index_path, index=False)

print(f"[run_and_log] Wrote {csv_rel_path} with {len(pass_df)} rows; updated {index_path}")
