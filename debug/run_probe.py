import argparse
import pandas as pd
from data_lake.provider import get_daily_adjusted


def main():
    parser = argparse.ArgumentParser(description="Single ticker probe")
    parser.add_argument("--date", required=True)
    parser.add_argument("--ticker", required=True)
    args = parser.parse_args()

    D = pd.Timestamp(args.date)
    hist = get_daily_adjusted(
        args.ticker,
        start=(D - pd.Timedelta(days=200)).date(),
        end=(D + pd.Timedelta(days=1)).date(),
    )
    if hist.empty or D not in hist.index:
        print("No data")
        return
    idx = hist.index.get_loc(D)
    d1 = hist.index[idx - 1] if idx > 0 else None
    if d1 is None:
        print("Missing D-1")
        return
    d1_row = hist.loc[d1]
    window = hist.loc[:d1].tail(63)
    close_up = (
        (d1_row["close"] - window.iloc[-2]["close"]) / window.iloc[-2]["close"] * 100.0
        if len(window) >= 2
        else 0.0
    )
    vol_mult = (
        (d1_row["volume"] / window["volume"].mean()) if window["volume"].mean() else 0.0
    )
    d_row = hist.loc[D]
    gap_pct = (d_row["open"] - d1_row["close"]) / d1_row["close"] * 100.0
    prior = hist.loc[:d1].tail(21)
    support = prior["low"].min()
    resistance = prior["high"].max()
    entry = d_row["open"]
    sr_ratio = (
        (resistance - entry) / (entry - support) if entry > support else 0.0
    )
    print(
        f"loaded={len(hist)} close%={close_up:.2f} vol_mult={vol_mult:.2f} gap%={gap_pct:.2f} sr={sr_ratio:.2f}"
    )


if __name__ == "__main__":
    main()
