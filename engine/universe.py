from __future__ import annotations

import pandas as pd


def members_on_date(m: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    """Return members active on ``date`` with safe date handling."""
    m = m.copy()
    m["start_date"] = pd.to_datetime(m["start_date"], errors="coerce")
    if "end_date" in m.columns and m["end_date"].notna().any():
        m["end_date"] = pd.to_datetime(m["end_date"], errors="coerce")
    else:
        m["end_date"] = pd.NaT
    date = pd.to_datetime(date)
    return m[(m["start_date"] <= date) & (m["end_date"].isna() | (date <= m["end_date"]))]
