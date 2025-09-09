from __future__ import annotations
import pandas as pd
import pandas.api.types as pdt


def members_on_date(m: pd.DataFrame, date) -> pd.DataFrame:
    """Return members active on ``date``. Be defensive about dtypes."""
    date = pd.to_datetime(date)
    df = m
    if not pdt.is_datetime64_any_dtype(df["start_date"]):
        df = df.copy()
        df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    if "end_date" not in df.columns:
        if df is m:
            df = df.copy()
        df["end_date"] = pd.NaT
    elif not pdt.is_datetime64_any_dtype(df["end_date"]):
        if df is m:
            df = df.copy()
        df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    mask = (df["start_date"] <= date) & (df["end_date"].isna() | (date <= df["end_date"]))
    return df.loc[mask]
