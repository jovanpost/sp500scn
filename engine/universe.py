import pandas as pd


def members_on_date(members: pd.DataFrame, date) -> pd.DataFrame:
    """Return rows where ticker is active on ``date`` (inclusive bounds)."""
    m = members.copy()

    # robust, timezone-naive timestamps
    m["start_date"] = pd.to_datetime(m["start_date"], errors="coerce").dt.tz_localize(None)
    if "end_date" in m.columns:
        m["end_date"] = pd.to_datetime(m["end_date"], errors="coerce").dt.tz_localize(None)
    else:
        m["end_date"] = pd.NaT

    ts = pd.to_datetime(date, errors="coerce").tz_localize(None)

    mask = (m["start_date"] <= ts) & (m["end_date"].isna() | (ts <= m["end_date"]))
    return m.loc[mask]
