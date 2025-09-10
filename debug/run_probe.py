import argparse
import datetime as dt
from data_lake.storage import Storage
import pandas as pd
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "yday_module", Path(__file__).resolve().parents[1] / "ui/pages/45_YdayVolSignal_Open.py"
)
mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(mod)
_load_prices = mod._load_prices
_compute_metrics = mod._compute_metrics

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--ticker", required=True)
    args = ap.parse_args()

    D = pd.to_datetime(args.date).date()
    s = Storage()
    df = _load_prices(s, args.ticker)
    m = _compute_metrics(df, D, 63) if df is not None else None
    print(args.ticker, D, m)
