import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from utils import outcomes


def test_upsert_assigns_run_date_and_dedupes(tmp_path):
    today = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
    out_path = tmp_path / "outcomes.csv"

    existing = pd.DataFrame([
        {"Ticker": "AAPL", "EvalDate": today, "Price": 1.0, "run_date": today}
    ])
    outcomes.write_outcomes(existing, out_path)

    df_pass = pd.DataFrame([
        {"Ticker": "AAPL", "EvalDate": today, "Price": 2.0},
        {
            "Ticker": "MSFT",
            "EvalDate": today,
            "Price": 3.0,
            "Change%": 4.5,
            "RelVol(TimeAdj63d)": 1.2,
        },
    ])

    result = outcomes.upsert_and_backfill_outcomes(df_pass, out_path)

    assert set(result["Ticker"]) == {"AAPL", "MSFT"}
    # Existing AAPL row preserved (price 1.0)
    assert result.loc[result["Ticker"] == "AAPL", "Price"].iloc[0] == 1.0
    # New MSFT row has today's run_date assigned
    assert result.loc[result["Ticker"] == "MSFT", "run_date"].iloc[0] == today
    # Newly inserted MSFT row retains Change% and RelVol columns
    msft_row = result[result["Ticker"] == "MSFT"].iloc[0]
    assert msft_row["Change%"] == 4.5
    assert msft_row["RelVol(TimeAdj63d)"] == 1.2
    # Ensure only one entry per ticker per run_date
    assert (
        result.groupby(["Ticker", "run_date"]).size().max() == 1
    )
