from __future__ import annotations

"""NYSE holiday calendar utilities.

This module dynamically generates NYSE holidays using
:mod:`pandas_market_calendars` when available. Generated holidays are cached
on disk so subsequent calls work without network access. A manual override
file can be used to inject or modify holidays if needed.
"""

from datetime import date, datetime, timedelta
import json
from typing import Set

from .io import DATA_DIR

try:  # pragma: no cover - optional dependency
    import pandas_market_calendars as mcal
except Exception:  # pragma: no cover - library missing
    mcal = None

CACHE_FILE = DATA_DIR / "nyse_holidays_cache.json"
OVERRIDE_FILE = DATA_DIR / "nyse_holidays_override.json"

def _load_cache() -> dict[int, list[str]]:
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {int(y): list(dates) for y, dates in raw.items()}
    except Exception:
        return {}

def _save_cache(cache: dict[int, list[str]]) -> None:
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        serial = {str(y): dates for y, dates in cache.items()}
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(serial, f, indent=2)
    except Exception:
        pass

def _load_overrides() -> Set[str]:
    try:
        with open(OVERRIDE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {str(d) for d in data}
    except Exception:
        return set()

def _compute_year(year: int) -> list[str]:
    if mcal is None:
        return []
    cal = mcal.get_calendar("NYSE")
    start = f"{year}-01-01"
    end = f"{year}-12-31"
    hol = cal.holidays(start=start, end=end)
    return [d.strftime("%Y-%m-%d") for d in hol.to_pydatetime()]

def get_nyse_holidays(start_year: int, end_year: int) -> Set[date]:
    """Return NYSE holidays between ``start_year`` and ``end_year`` inclusive."""
    cache = _load_cache()
    changed = False
    for y in range(start_year, end_year + 1):
        if y not in cache:
            cache[y] = _compute_year(y)
            if cache[y]:
                changed = True
    if changed:
        _save_cache(cache)
    dates: Set[str] = set()
    for y in range(start_year, end_year + 1):
        dates.update(cache.get(y, []))
    dates.update(_load_overrides())
    return {date.fromisoformat(d) for d in dates}

def previous_trading_day(ref: date | None = None) -> date:
    """Return the most recent NYSE trading day on or before ``ref``."""
    if ref is None:
        ref = datetime.now().astimezone().date()
    holidays = get_nyse_holidays(ref.year - 1, ref.year + 1)
    d = ref
    while d.weekday() >= 5 or d in holidays:
        d -= timedelta(days=1)
    return d
