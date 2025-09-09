from __future__ import annotations

from typing import Union

import pandas as pd


def members_on_date(
    m: pd.DataFrame,
    date: Union[str, "pd.Timestamp", "datetime.date", "datetime.datetime"],
) -> pd.DataFrame:
    """Return members active on ``date``.

    All date-like inputs are coerced to ``pd.Timestamp`` to avoid ``str`` vs
    ``Timestamp`` comparisons that can raise errors.
    """
    df = m.copy()

    # Coerce membership bounds if present, otherwise fill with ``NaT`` so the
    # filtering logic still works.
    if "start_date" in df.columns:
        df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    else:
        df["start_date"] = pd.NaT

    if "end_date" in df.columns:
        df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    else:
        df["end_date"] = pd.NaT

    # Coerce the query date as well
    dt = pd.to_datetime(date, errors="coerce")

    mask = (df["start_date"] <= dt) & (df["end_date"].isna() | (dt <= df["end_date"]))
    return df.loc[mask]
