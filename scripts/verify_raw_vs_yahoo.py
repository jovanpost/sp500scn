#!/usr/bin/env python3
from __future__ import annotations

import argparse
import typing as t

import numpy as np
import pandas as pd
import yfinance as yf

from data_lake.storage import Storage


def _yahoo_symbol(tkr: str) -> str:
    """Return the Yahoo-compatible ticker symbol."""

    return tkr.replace(".", "-").upper()


def _mad_pct(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    common = a.index.intersection(b.index)
    if len(common) == 0:
        return float("inf")
    rel = (a.loc[common] - b.loc[common]).abs() / b.loc[common].replace(0, np.nan).abs()
    return float(np.nanmedian(rel))


def main(argv: t.Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare lake RAW Close to Yahoo RAW Close",
    )
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--start")
    parser.add_argument("--end")
    args = parser.parse_args(argv)

    storage = Storage.from_env()
    path = f"prices/{args.ticker.upper()}.parquet"
    if not storage.exists(path):
        print("NO lake file")
        return 2

    df = storage.read_parquet_df(path).copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.set_index("date").sort_index()

    yahoo = yf.download(
        _yahoo_symbol(args.ticker),
        start=args.start,
        end=args.end,
        auto_adjust=False,
        actions=True,
        progress=False,
        threads=False,
    )
    if yahoo.empty:
        print("Yahoo empty")
        return 3

    if isinstance(yahoo.columns, pd.MultiIndex):
        yahoo = yahoo.droplevel(1, axis=1)
    yahoo.index = pd.to_datetime(yahoo.index).tz_localize(None)

    cols = ["Open", "High", "Low", "Close", "Volume"]
    diffs: dict[str, float] = {}
    for col in cols:
        diffs[col] = _mad_pct(df[col], yahoo[col])

    print(
        {
            "ticker": args.ticker.upper(),
            "median_abs_pct_diff": {k: round(v, 6) for k, v in diffs.items()},
            "n_common_days": int(len(df.index.intersection(yahoo.index))),
        }
    )
    return 0 if all(value <= 0.001 for value in diffs.values()) else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
