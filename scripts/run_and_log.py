# scripts/run_and_log.py
import argparse
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

import swing_options_screener as sos  # uses your module


def run_and_save(universe: str, with_options: bool):
    # Run the screener via the module API (same logic as the UI)
    out = sos.run_scan(
        tickers=None if universe == "sp500" else None,  # None -> default list inside module
        with_options=with_options,
    )
    df = out.get("pass_df", pd.DataFrame())

    # Folder for today's run (ET date)
    now_et = datetime.now(ZoneInfo("America/New_York"))
    day_str = now_et.strftime("%Y-%m-%d")
    out_dir = os.path.join("data", "history", day_str)
    os.makedirs(out_dir, exist_ok=True)

    # Save files
    csv_path = os.path.join(out_dir, f"pass_tickers_{day_str}.csv")
    psv_path = os.path.join(out_dir, f"pass_tickers_{day_str}.psv")

    if df.empty:
        # Create empty files with headers so downstream code is stable
        cols = [
            'Ticker','EvalDate','Price','EntryTimeET','Change%','RelVol(TimeAdj63d)',
            'Resistance','TP','RR_to_Res','RR_to_TP','SupportType','SupportPrice','Risk$',
            'TPReward$','TPReward%','ResReward$','ResReward%','DailyATR','DailyCap',
            'Hist21d_PassCount','Hist21d_Max%','Hist21d_Examples','ResLookbackDays','Prices',
            'Session','EntrySrc','VolSrc',
            'OptExpiry','BuyK','SellK','Width','DebitMid','DebitCons','MaxProfitMid',
            'MaxProfitCons','RR_Spread_Mid','RR_Spread_Cons','BreakevenMid','PricingNote'
        ]
        pd.DataFrame(columns=cols).to_csv(csv_path, index=False)
        pd.DataFrame(columns=cols).to_csv(psv_path, index=False, sep="|")
    else:
        # Sort by current price ascending (your preference)
        df = df.sort_values(by=["Price", "Ticker"], ascending=[True, True])
        df.to_csv(csv_path, index=False)
        df.to_csv(psv_path, index=False, sep="|")

    # Update a tiny index file for quick “what happened when”
    idx_path = os.path.join("data", "history", "index.csv")
    row = {
        "date": day_str,
        "processed_at_et": now_et.strftime("%Y-%m-%d %H:%M:%S"),
        "num_pass": 0 if df is None else int(len(df)),
        "csv": os.path.relpath(csv_path),
        "psv": os.path.relpath(psv_path),
    }
    if os.path.exists(idx_path):
        idx = pd.read_csv(idx_path)
        # Replace existing row for today if present
        idx = idx[idx["date"] != day_str]
        idx = pd.concat([idx, pd.DataFrame([row])], ignore_index=True)
    else:
        os.makedirs(os.path.dirname(idx_path), exist_ok=True)
        idx = pd.DataFrame([row])
    idx.sort_values("date").to_csv(idx_path, index=False)

    # Commit the changes back to the repo
    os.system('git config user.name "github-actions"')
    os.system('git config user.email "actions@github.com"')
    os.system('git add -A')
    os.system(f'git commit -m "[skip ci] Scan results for {day_str}" || echo "Nothing to commit"')
    os.system('git push || echo "Nothing to push"')


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--universe", default="sp500", choices=["sp500", "custom"])
    p.add_argument("--with-options", action="store_true")
    args = p.parse_args()
    run_and_save(args.universe, args.with_options)

