from __future__ import annotations
import pandas as pd


def members_on_date(members_df: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    """
    members_df columns: ['ticker','start_date','end_date'] (strings or datetimes).
    Return rows active on 'date'. Treat null end_date as active.
    """
    m = members_df.copy()
    m['start_date'] = pd.to_datetime(m['start_date'], errors='coerce')
    if 'end_date' in m.columns:
        m['end_date'] = pd.to_datetime(m['end_date'], errors='coerce')
    else:
        m['end_date'] = pd.NaT
    return m[
        (m['start_date'] <= date)
        & ((m['end_date'].isna()) | (date <= m['end_date']))
    ]


def missing_dates(df: pd.DataFrame, dates: list[pd.Timestamp]) -> list[pd.Timestamp]:
    """Return list of dates from ``dates`` missing in ``df``'s index."""
    idx = pd.DatetimeIndex(df.index)
    return [pd.Timestamp(d) for d in dates if pd.Timestamp(d) not in idx]
