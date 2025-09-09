from __future__ import annotations
import pandas as pd


def members_on_date(m: pd.DataFrame, date) -> pd.DataFrame:
    """Return members active on `date`. Robust to string date columns."""
    df = m.copy()
    D = pd.to_datetime(date).normalize()
    for col in ("start_date", "end_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        else:
            df[col] = pd.NaT
    mask = (df["start_date"] <= D) & (df["end_date"].isna() | (D <= df["end_date"]))
    return df.loc[mask].copy()
