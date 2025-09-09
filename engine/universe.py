from __future__ import annotations

import pandas as pd


def members_on_date(m: pd.DataFrame, date) -> pd.DataFrame:
    """Return members active on ``date`` with safe date handling."""
    m = m.copy()

    m["start_date"] = pd.to_datetime(m["start_date"], errors="coerce", utc=False)
    if "end_date" in m.columns:
        m["end_date"] = pd.to_datetime(m["end_date"], errors="coerce", utc=False)
    else:
        m["end_date"] = pd.NaT

    d = pd.to_datetime(date).normalize()

    mask = (m["start_date"] <= d) & (m["end_date"].isna() | (d <= m["end_date"]))
    return m.loc[mask]
