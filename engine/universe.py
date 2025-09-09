import pandas as pd


def members_on_date(members: pd.DataFrame, date) -> pd.DataFrame:
    """Return rows where ticker is active on ``date`` (inclusive bounds)."""
    m = members.copy()

    m["start_date"] = pd.to_datetime(m["start_date"]).dt.tz_localize(None)
    if "end_date" not in m.columns:
        m["end_date"] = pd.NaT
    m["end_date"] = pd.to_datetime(m["end_date"], errors="coerce").dt.tz_localize(None)

    ts = pd.Timestamp(date).tz_localize(None)

    mask = (m["start_date"] <= ts) & (m["end_date"].isna() | (ts <= m["end_date"]))
    return m.loc[mask]
