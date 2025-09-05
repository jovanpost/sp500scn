from __future__ import annotations

"""Trading day utilities with optional market calendar support."""

import pandas as pd

try:  # pragma: no cover - optional dependency
    import pandas_market_calendars as mcal  # type: ignore
except Exception:  # pragma: no cover - library missing
    mcal = None


def trading_days_between(start, end, tz: str = "America/New_York") -> pd.DatetimeIndex:
    """Return trading days between ``start`` and ``end`` in timezone ``tz``."""
    if mcal:
        cal = mcal.get_calendar("XNYS")
        sched = cal.schedule(start_date=start, end_date=end)
        return pd.DatetimeIndex(sched.index.tz_convert(tz).normalize().unique())
    return pd.bdate_range(start=start, end=end, tz=tz)


def is_trading_day(ts, tz: str = "America/New_York") -> bool:
    """Return ``True`` if ``ts`` falls on a trading day in timezone ``tz``."""
    days = trading_days_between(
        pd.Timestamp(ts, tz=tz).date(),
        pd.Timestamp(ts, tz=tz).date(),
        tz=tz,
    )
    return len(days) > 0
