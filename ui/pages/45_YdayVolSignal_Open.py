import io
import pandas as pd
import streamlit as st
from data_lake.storage import Storage
from engine.universe import members_on_date


@st.cache_resource
def _get_storage() -> Storage:
    return Storage()


# --- cache-safe loader: ignore Storage in hashing ---
@st.cache_data(show_spinner=False, hash_funcs={Storage: lambda _s: "Storage"})
def _load_members(_storage: Storage) -> pd.DataFrame:
    raw = _storage.read_bytes("membership/sp500_members.parquet")
    df = pd.read_parquet(io.BytesIO(raw))
    # normalize dates now to avoid dtype surprises later
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    if "end_date" in df.columns:
        df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    else:
        df["end_date"] = pd.NaT
    return df


def _run_signal_scan(active: pd.DataFrame, *, D, lookback, min_close_up, min_vol_mult,
                     min_gap_next_open, opt) -> tuple[pd.DataFrame, dict]:
    """Return (results_df, debug_counts) and keep simple stage counters so we can
    explain zero results."""
    # lazy imports so the page doesn’t explode from unrelated modules
    from data_lake.provider import get_daily_adjusted

    counts: dict[str, int] = {}
    counts["universe"] = len(active)

    # 1) build ticker list
    tickers = active["ticker"].dropna().unique().tolist()
    if not tickers:
        return pd.DataFrame(), counts

    # 2) fetch D-1 window (enough to compute lookback stats)
    D = pd.to_datetime(D)
    results = []
    passed_signal = 0
    for t in tickers:
        hist = get_daily_adjusted(t, end=D, lookback=max(lookback + 2, 70))
        if hist.empty or D not in hist.index:
            continue
        idx = hist.index.get_loc(D)
        d1 = hist.index[idx - 1] if idx > 0 else None
        if d1 is None:
            continue

        # --- Signal conditions (all as of D-1) ---
        d1_row = hist.loc[d1]
        window = hist.loc[:d1].tail(lookback)
        if window.empty or window["volume"].mean() == 0:
            continue

        close_up = (
            (d1_row["close"] - window.iloc[-2]["close"]) / window.iloc[-2]["close"] * 100.0
            if len(window) >= 2
            else 0.0
        )
        vol_mult = (
            (d1_row["volume"] / window["volume"].mean()) if window["volume"].mean() else 0.0
        )

        if close_up < min_close_up or vol_mult < min_vol_mult:
            continue

        passed_signal += 1

        # --- Next-day open gap (D open vs D-1 close) ---
        d_row = hist.loc[D] if D in hist.index else None
        if d_row is None:
            continue
        gap_pct = (d_row["open"] - d1_row["close"]) / d1_row["close"] * 100.0
        if gap_pct < min_gap_next_open:
            continue

        # Optional filters can be added here (ATR, 21d return, min price, etc.)
        results.append(
            {
                "ticker": t,
                "d1_close_up_pct": close_up,
                "d1_vol_mult": vol_mult,
                "gap_open_pct": gap_pct,
            }
        )

    counts["signal_pass"] = passed_signal
    counts["final"] = len(results)
    return pd.DataFrame(results).sort_values("gap_open_pct", ascending=False), counts


def render_page():
    st.header("⚡ Yesterday Close+Volume → Buy Next Open")
    storage = _get_storage()

    D = st.date_input("Entry day (D)", value=pd.Timestamp.today()).to_pydatetime()
    lookback = int(st.number_input("Lookback", value=63, min_value=1, step=1))
    min_close_up = float(st.number_input("Min close-up on D-1 (%)", value=3.0, step=0.5))
    min_vol_mult = float(st.number_input("Min volume multiple", value=1.5, step=0.1))
    min_gap = float(st.number_input("Min gap open (%)", value=0.0, step=0.1))

    if st.button("Run scan"):
        # IMPORTANT: leading underscore arg here matches the cached signature
        members = _load_members(storage)  # storage is fine; cache ignores its hash
        active = members_on_date(members, D)

        if active.empty:
            st.warning("No active S&P members on selected date.")
            with st.expander("Debug: membership on selected date"):
                st.write({"selected_date": str(D), "active_count": 0})
            return

        # run the signal
        results, counts = _run_signal_scan(
            active,
            D=D,
            lookback=lookback,
            min_close_up=min_close_up,
            min_vol_mult=min_vol_mult,
            min_gap_next_open=min_gap,
            opt={},
        )

        if results.empty:
            st.warning("No matches for the selected filters.")
            with st.expander("Why zero? Show stage counts"):
                st.json(counts)
                st.caption(
                    "universe → passed initial signal (D-1) → final (gap & optional filters)"
                )
            return

        st.success(f"{len(results)} matches")
        st.dataframe(results, use_container_width=True)


def page():
    render_page()
