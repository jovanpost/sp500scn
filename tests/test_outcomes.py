import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from utils import outcomes


def test_upsert_assigns_run_date_and_dedupes(tmp_path):
    today = date.today().isoformat()
    out_path = tmp_path / "outcomes.csv"

    existing = pd.DataFrame([
        {"Ticker": "AAPL", "EvalDate": today, "Price": 1.0, "run_date": today}
    ])
    outcomes.write_outcomes(existing, out_path)

    df_pass = pd.DataFrame([
        {"Ticker": "AAPL", "EvalDate": today, "Price": 2.0},
        {"Ticker": "MSFT", "EvalDate": today, "Price": 3.0},
    ])

    result = outcomes.upsert_and_backfill_outcomes(df_pass, out_path)

    assert set(result["Ticker"]) == {"AAPL", "MSFT"}
    # Existing AAPL row preserved (price 1.0)
    assert result.loc[result["Ticker"] == "AAPL", "Price"].iloc[0] == 1.0
    # New MSFT row has today's run_date assigned
    assert result.loc[result["Ticker"] == "MSFT", "run_date"].iloc[0] == today
    # Ensure only one entry per ticker per run_date
    assert (
        result.groupby(["Ticker", "run_date"]).size().max() == 1
    )
