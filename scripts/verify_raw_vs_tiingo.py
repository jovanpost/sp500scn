#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import typing as t

import numpy as np
import pandas as pd
import pandas_datareader as pdr

from data_lake.storage import Storage


def _mad_pct(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    common = a.index.intersection(b.index)
    if len(common) == 0:
        return float("inf")
    rel = (a.loc[common] - b.loc[common]).abs() / b.loc[common].replace(0, np.nan).abs()
    return float(np.nanmedian(rel))


def main(argv: t.Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compare lake RAW vs Tiingo for O/H/L/C/Vol")
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--threshold", type=float, default=0.001)  # 0.1%
    args = ap.parse_args(argv)

    api_key = os.getenv("TIINGO_API_KEY", "")
    if not api_key:
        print("no TIINGO_API_KEY")
        return 2

    storage = Storage.from_env()
    path = f"prices/{args.ticker.upper()}.parquet"
    if not storage.exists(path):
        print("NO lake file")
        return 2

    df = storage.read_parquet_df(path).copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.set_index("date").sort_index()

    tdf = (
        pdr.get_data_tiingo(
            args.ticker.upper(), start=args.start, end=args.end, api_key=api_key
        )
        .reset_index()
    )
    tdf["date"] = pd.to_datetime(tdf["date"]).dt.tz_localize(None)
    tdf = tdf.set_index("date").sort_index()

    cols_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    diffs = {k: _mad_pct(df[k], tdf[v]) for k, v in cols_map.items()}
    print(
        {
            "ticker": args.ticker.upper(),
            "median_abs_pct_diff": {k: round(v, 6) for k, v in diffs.items()},
            "n_common_days": int(len(df.index.intersection(tdf.index))),
        }
    )
    return 0 if all(v <= args.threshold for v in diffs.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
