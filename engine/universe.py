import pandas as pd


def _as_naive(ts):
    ts = pd.to_datetime(ts, errors="coerce")
    if isinstance(ts, pd.Series):
        if getattr(ts.dt, "tz", None) is not None:
            ts = ts.dt.tz_localize(None)
    else:
        if getattr(ts, "tz", None) is not None:
            ts = ts.tz_localize(None)
    return ts


def members_on_date(members: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    """Return rows where ticker is active on ``date`` (inclusive bounds)."""

    d = pd.to_datetime(date)
    if getattr(d, "tz", None) is not None:
        d = d.tz_localize(None)

    m = members.copy()
    m["start_date"] = _as_naive(m["start_date"])
    if "end_date" in m.columns:
        m["end_date"] = _as_naive(m["end_date"])
    else:
        m["end_date"] = pd.NaT

    return m[(m["start_date"] <= d) & (m["end_date"].isna() | (d <= m["end_date"]))]
