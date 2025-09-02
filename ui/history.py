import pandas as pd
import streamlit as st
from utils.io import list_pass_files
from utils.outcomes import read_outcomes
from utils.formatting import _usd, _safe


def load_history_df() -> pd.DataFrame:
    """Return a DataFrame concatenating all historical PASS snapshots."""
    paths = [p for p in list_pass_files() if p.suffix in {".psv", ".csv"}]

    if not paths:
        return pd.DataFrame()

    frames = []
    for p in paths:
        try:
            sep = "|" if p.suffix == ".psv" else ","
            df = pd.read_csv(p, sep=sep)
            df["__source_file"] = p.name
            frames.append(df)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


@st.cache_data(show_spinner=False)
def latest_trading_day_recs(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """Return unique tickers for the most recent ``run_date``.

    Parameters
    ----------
    df:
        Outcomes DataFrame to search. The result is cached based on the
        contents of ``df`` so the table only refreshes after new data is
        written.

    Returns
    -------
    tuple[pd.DataFrame, str | None]
        The filtered DataFrame and the latest trading-day string
        (``YYYY-MM-DD``). ``None`` if no valid ``run_date`` exists.
    """
    if df.empty or "run_date" not in df.columns:
        return pd.DataFrame(), None

    df = df.copy()
    df["run_date"] = pd.to_datetime(df["run_date"], errors="coerce")
    latest = df["run_date"].dropna().max()
    if pd.isna(latest):
        return pd.DataFrame(), None

    df_latest = (
        df[df["run_date"] == latest]
        .drop_duplicates(subset=["Ticker"])
        .reset_index(drop=True)
    )
    return df_latest, latest.date().isoformat()


def load_outcomes():
    """Thin wrapper around :func:`utils.outcomes.read_outcomes`."""
    return read_outcomes()


def outcomes_summary(dfh: pd.DataFrame):
    """Render a concise outcomes table with hit/miss statistics."""
    if dfh.empty:
        st.info("No outcomes yet.")
        return

    dfh = dfh.copy()

    # Ensure expected columns exist so downstream ops don't KeyError
    for c in ["Expiry", "EvalDate", "Notes", "LastPrice", "LastPriceAt", "PctToTarget"]:
        if c not in dfh.columns:
            dfh[c] = pd.NA

    # Prefer result_status then fall back to Status
    status_col = (
        "result_status"
        if "result_status" in dfh.columns
        else ("Status" if "Status" in dfh.columns else None)
    )

    # Helper: coerce to tz-naive pandas Timestamp
    def _to_naive(series: pd.Series) -> pd.Series:
        s = pd.to_datetime(series, errors="coerce", utc=True)
        return s.dt.tz_convert("UTC").dt.tz_localize(None)

    dfh["Expiry_parsed"] = _to_naive(dfh["Expiry"])
    dfh["EvalDate_parsed"] = _to_naive(dfh["EvalDate"])

    # Backfill missing expiry from EvalDate + 30d (display-only)
    need_exp = dfh["Expiry_parsed"].isna() & dfh["EvalDate_parsed"].notna()
    if need_exp.any():
        dfh.loc[need_exp, "Expiry_parsed"] = dfh.loc[need_exp, "EvalDate_parsed"] + pd.Timedelta(
            days=30
        )

    # Robust DTE using nanoseconds to avoid dtype issues
    dfh["DTE"] = pd.Series(pd.NA, index=dfh.index, dtype="Int64")
    mask = dfh["Expiry_parsed"].notna()
    if mask.any():
        base_ns = pd.Timestamp.utcnow().normalize().value  # int64 ns at 00:00 UTC today
        exp_ns = dfh.loc[mask, "Expiry_parsed"].view("int64")
        NS_PER_DAY = 86_400_000_000_000
        dte_days = ((exp_ns - base_ns) // NS_PER_DAY).astype("int64")
        dfh.loc[mask, "DTE"] = pd.array(dte_days, dtype="Int64")

    # Sort: earliest expiry first; for ties, most recent EvalDate first; NaT at end
    exp_key = dfh["Expiry_parsed"].fillna(pd.Timestamp.max)
    dfh_sorted = dfh.assign(_expkey=exp_key).sort_values(
        ["_expkey", "EvalDate_parsed"], ascending=[True, False]
    ).drop(columns=["_expkey"])

    # Summary counts
    notes_up = dfh_sorted["Notes"].astype(str).str.upper()
    hits = int(notes_up.isin(["HIT_BY_SELLK", "HIT_BY_TP"]).sum())
    misses = int((notes_up == "EXPIRED_NO_HIT").sum())

    if status_col:
        s_up = dfh_sorted[status_col].astype(str).str.upper()
        settled = int((s_up == "SETTLED").sum())
        pending = int((s_up != "SETTLED").sum())
    else:
        settled = hits + misses
        pending = int(len(dfh_sorted) - settled)

    st.caption(
        f"Settled: {settled} • Hits: {hits} • Misses: {misses} • Pending: {pending}"
    )

    # Show parsed expiry when original blank; format for display
    df_disp = dfh_sorted.copy()
    use_parsed = df_disp["Expiry"].isna() | (
        df_disp["Expiry"].astype(str).str.strip() == ""
    )
    df_disp.loc[use_parsed, "Expiry"] = df_disp.loc[use_parsed, "Expiry_parsed"].dt.strftime(
        "%Y-%m-%d"
    )

    preferred = [
        "Ticker",
        "EvalDate",
        "Price",
        "RelVol(TimeAdj63d)",
        "LastPrice",
        "LastPriceAt",
        "PctToTarget",
        "EntryTimeET",
        status_col if status_col else "Status",
        "HitDateET",
        "Expiry",
        "DTE",
        "BuyK",
        "SellK",
        "TP",
        "Notes",
    ]
    cols = [c for c in preferred if c in df_disp.columns]
    if cols:
        df_disp = df_disp[cols]
    st.dataframe(df_disp)


def render_history_tab():
    df_out = load_outcomes()

    # --- Latest recommendations based on most recent run_date ---
    df_last, date_str = latest_trading_day_recs(df_out)
    if date_str:
        st.subheader(f"Trading day {date_str} recommendations")
        if df_last.empty:
            st.info("No tickers passed that day.")
        else:
            preferred = [
                "Ticker",
                "Price",
                "TP",
                "RelVol(TimeAdj63d)",
                "EntryTimeET",
                "LastPrice",
                "LastPriceAt",
            ]
            cols = [c for c in preferred if c in df_last.columns]
            st.dataframe(df_last[cols] if cols else df_last)
    else:
        st.subheader("Trading day — recommendations")
        st.info("No pass files yet. Run the scanner (or wait for the next scheduled run).")

    # --- Outcomes, sorted by option expiry (oldest → newest) ---
    st.subheader("Outcomes (sorted by option expiry)")

    outcomes_summary(df_out)
