import sys
import json
from datetime import date
from pathlib import Path

import pandas_market_calendars as mcal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import nyse_calendar


def test_previous_trading_day_across_july4(monkeypatch, tmp_path):
    cache = {
        "2022": [],
        "2023": ["2023-07-04"],
        "2024": [],
    }
    cache_file = tmp_path / "nyse_holidays_cache.json"
    cache_file.write_text(json.dumps(cache))
    override_file = tmp_path / "nyse_holidays_override.json"
    override_file.write_text("[]")

    monkeypatch.setattr(nyse_calendar, "CACHE_FILE", cache_file)
    monkeypatch.setattr(nyse_calendar, "OVERRIDE_FILE", override_file)

    def boom(year: int):
        raise AssertionError("_compute_year should not be called")

    monkeypatch.setattr(nyse_calendar, "_compute_year", boom)

    july4 = date(2023, 7, 4)
    assert nyse_calendar.previous_trading_day(july4) == date(2023, 7, 3)
    assert nyse_calendar.previous_trading_day(date(2023, 7, 5)) == date(2023, 7, 5)

    assert mcal is not None
