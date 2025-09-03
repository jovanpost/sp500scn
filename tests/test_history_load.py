import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ui.history as history
from utils import outcomes


def test_load_outcomes_wraps_read_outcomes(monkeypatch):
    sentinel = pd.DataFrame({"a": [1]})
    monkeypatch.setattr(history, "read_outcomes", lambda: sentinel)
    assert history.load_outcomes() is sentinel


def test_load_outcomes_missing_file(tmp_path, monkeypatch):
    dummy = tmp_path / "missing.csv"

    def fake_read_outcomes():
        return outcomes.read_outcomes(dummy)

    monkeypatch.setattr(history, "read_outcomes", fake_read_outcomes)
    df = history.load_outcomes()
    assert df.empty
    assert list(df.columns) == outcomes.OUTCOLS


def test_latest_trading_day_recs_filters_without_dedup():
    df = pd.DataFrame(
        {
            "Ticker": ["AAA", "AAA", "BBB"],
            "run_date": [
                "2023-01-02 10:00",
                "2023-01-02 15:30",
                "2023-01-01 09:00",
            ],
        }
    )

    df_latest, date_str = history.latest_trading_day_recs(df)
    assert date_str == "2023-01-02"
    assert list(df_latest["Ticker"]) == ["AAA", "AAA"]
    assert len(df_latest) == 2
