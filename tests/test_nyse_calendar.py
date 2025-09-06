import sys
from datetime import date
from pathlib import Path

import pytest
pytest.importorskip("pandas_market_calendars", reason="optional dependency")
import pandas_market_calendars as mcal  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import nyse_calendar


@pytest.mark.skipif(mcal is None, reason="pandas_market_calendars not installed")
def test_compute_year_has_known_holidays():
    hols = nyse_calendar._compute_year(2024)
    assert "2024-07-04" in hols
    assert "2024-12-25" in hols


@pytest.mark.skipif(mcal is None, reason="pandas_market_calendars not installed")
def test_previous_trading_day_across_holidays(monkeypatch, tmp_path):
    cache_file = tmp_path / "nyse_holidays_cache.json"
    override_file = tmp_path / "nyse_holidays_override.json"
    override_file.write_text("[]")

    monkeypatch.setattr(nyse_calendar, "CACHE_FILE", cache_file)
    monkeypatch.setattr(nyse_calendar, "OVERRIDE_FILE", override_file)

    assert nyse_calendar.previous_trading_day(date(2024, 7, 4)) == date(2024, 7, 3)
    assert nyse_calendar.previous_trading_day(date(2024, 7, 5)) == date(2024, 7, 5)
    assert nyse_calendar.previous_trading_day(date(2024, 1, 1)) == date(2023, 12, 29)
