import glob
import os
import pandas as pd
import streamlit as st
from utils.io import DATA_DIR, HISTORY_DIR, OUTCOMES_CSV, read_csv

PASS_DIR = DATA_DIR / "pass_logs"


def load_history_df() -> pd.DataFrame:
    """Return a DataFrame concatenating all historical PASS snapshots."""
    paths = []
    paths.extend(sorted(glob.glob(str(HISTORY_DIR / "pass_*.psv"))))
    # Legacy fallback
    paths.extend(sorted(glob.glob("history/pass_*.psv")))

    if not paths:
        return pd.DataFrame()

    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p, sep="|")
            df["__source_file"] = os.path.basename(p)
            frames.append(df)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def latest_pass_file():
    """Return the newest pass_*.csv from either pass_logs/ or history/."""
    candidates = []
    for d in [PASS_DIR, HISTORY_DIR]:
        candidates.extend(glob.glob(str(d / "pass_*.csv")))
    return sorted(candidates)[-1] if candidates else None


def load_outcomes():
    if OUTCOMES_CSV.exists():
        return read_csv(OUTCOMES_CSV)
    return pd.DataFrame()


def outcomes_summary(dfh: pd.DataFrame):
    if dfh is None or dfh.empty:
        st.info("No outcomes yet.")
        return

    n = len(dfh)
    # Ensure we always have Series (never scalars) to avoid .sum() errors
    if "result_status" in dfh.columns:
        s_status = dfh["result_status"].astype(str)
    else:
        # Assume not yet settled if the column is missing
        s_status = pd.Series(["PENDING"] * n, index=dfh.index, dtype="string")

    if "hit" in dfh.columns:
        hit_mask = dfh["hit"].astype(bool)
    else:
        hit_mask = pd.Series([False] * n, index=dfh.index, dtype=bool)

    settled_mask = s_status.eq("SETTLED")
    pending_mask = ~settled_mask

    settled = int(settled_mask.sum())
    pending = int(pending_mask.sum())
    hits = int((settled_mask & hit_mask).sum())
    misses = settled - hits

    st.caption(f"Settled: {settled} • Hits: {hits} • Misses: {misses} • Pending: {pending}")

    # Nice sort if the columns exist; otherwise just show as-is
    sort_cols = [c for c in ["run_date", "ticker"] if c in dfh.columns]
    if sort_cols:
        # run_date desc if present
        ascending = [False if c == "run_date" else True for c in sort_cols]
        df_show = dfh.sort_values(sort_cols, ascending=ascending)
    else:
        df_show = dfh

    st.dataframe(df_show, use_container_width=True, height=min(600, 80 + 28 * len(df_show)))


def render_history_tab():
    # --- Latest recommendations (most recent run) ---
    st.subheader("Latest recommendations (most recent run)")
    lastf = latest_pass_file()
    if lastf:
        try:
            df_last = read_csv(lastf)
            st.dataframe(df_last, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not read latest pass file: {e}")
    else:
        st.info("No pass files yet. Run the scanner (or wait for the next scheduled run).")

    # --- Outcomes, sorted by option expiry (oldest → newest) ---
    st.subheader("Outcomes (sorted by option expiry)")

    dfh = load_outcomes()
    if dfh is None or dfh.empty:
        st.info("No outcomes yet.")
    else:
        dfh = dfh.copy()

        # Ensure expected columns exist
        for c in ["Expiry", "EvalDate", "Notes"]:
            if c not in dfh.columns:
                dfh[c] = pd.NA

        # Prefer result_status, then Status
        status_col = "result_status" if "result_status" in dfh.columns else ("Status" if "Status" in dfh.columns else None)

        # Helper: to tz-naive pandas Timestamp
        def _to_naive(series: pd.Series) -> pd.Series:
            s = pd.to_datetime(series, errors="coerce", utc=True)
            return s.dt.tz_convert("UTC").dt.tz_localize(None)

        # Parse & normalize times
        dfh["Expiry_parsed"] = _to_naive(dfh["Expiry"])
        dfh["EvalDate_parsed"] = _to_naive(dfh["EvalDate"])

        # Backfill missing expiry from EvalDate + 30d (display-only)
        need_exp = dfh["Expiry_parsed"].isna() & dfh["EvalDate_parsed"].notna()
        if need_exp.any():
            dfh.loc[need_exp, "Expiry_parsed"] = dfh.loc[need_exp, "EvalDate_parsed"] + pd.Timedelta(days=30)

        # ---- Robust DTE using nanoseconds to avoid dtype issues ----
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

        st.caption(f"Settled: {settled} • Hits: {hits} • Misses: {misses} • Pending: {pending}")

        # Show parsed expiry when original blank; format for display
        df_disp = dfh_sorted.copy()
        use_parsed = df_disp["Expiry"].isna() | (df_disp["Expiry"].astype(str).str.strip() == "")
        df_disp.loc[use_parsed, "Expiry"] = df_disp.loc[use_parsed, "Expiry_parsed"].dt.strftime("%Y-%m-%d")

        preferred = [
            "Ticker","EvalDate","Price","EntryTimeET",
            status_col if status_col else "Status",
            "HitDateET","Expiry","DTE","BuyK","SellK","TP","Notes"
        ]
        cols = [c for c in preferred if c in df_disp.columns]
        if cols:
            df_disp = df_disp[cols]

        st.dataframe(df_disp, use_container_width=True, height=min(600, 80 + 28 * len(df_disp)))
