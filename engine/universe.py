from __future__ import annotations

import pandas as pd


def members_on_date(m: pd.DataFrame, date) -> pd.DataFrame:
    """Return members active on ``date`` with safe date handling."""
    m = m.copy()

    if not isinstance(date, pd.Timestamp):
        date = pd.to_datetime(date)

    if "start_date" in m.columns:
        m["start_date"] = pd.to_datetime(m["start_date"], errors="coerce")
    else:
        m["start_date"] = pd.NaT

    if "end_date" in m.columns:
        m["end_date"] = pd.to_datetime(m["end_date"], errors="coerce")
    else:
        m["end_date"] = pd.NaT

    mask = (m["start_date"] <= date) & (m["end_date"].isna() | (date <= m["end_date"]))
    return m.loc[mask]
