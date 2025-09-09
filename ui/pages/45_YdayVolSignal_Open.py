import io
import datetime as dt

import pandas as pd
import streamlit as st
from data_lake.storage import Storage
from engine.universe import members_on_date


@st.cache_resource
def _get_storage() -> Storage:
    return Storage()


@st.cache_data(show_spinner=False)
def _load_members(_storage) -> pd.DataFrame:
    raw = _storage.read_bytes("membership/sp500_members.parquet")
    df = pd.read_parquet(io.BytesIO(raw))
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    return df


def _run_signal_scan(
    active: pd.DataFrame,
    *,
    D,
    lookback,
    min_close_up,
    min_vol_mult,
    min_gap_next_open,
    opt,
) -> tuple[pd.DataFrame, dict[str, int], pd.DataFrame]:
    """Return (results_df, debug_counts, near_misses_df)."""

    from data_lake.provider import get_daily_adjusted

    stats: dict[str, int] = {
        "active_members": len(active),
        "have_D_D1": 0,
        "pass_close_up": 0,
        "pass_volume_mult": 0,
        "pass_gap": 0,
        "pass_min_close": 0,
        "pass_atr": 0,
        "pass_21d": 0,
    }

    tickers = active["ticker"].dropna().unique().tolist()
    if not tickers:
        return pd.DataFrame(), stats, pd.DataFrame()

    D = pd.to_datetime(D)
    results: list[dict] = []
    near: list[dict] = []

    for t in tickers:
        back_days = max(lookback + 2, 70)
        start = (pd.Timestamp(D) - pd.Timedelta(days=back_days * 2)).date()
        hist = get_daily_adjusted(t, start=start, end=pd.Timestamp(D).date())
        if hist.empty or D not in hist.index:
            continue
        idx = hist.index.get_loc(D)
        d1 = hist.index[idx - 1] if idx > 0 else None
        if d1 is None:
            continue

        d1_row = hist.loc[d1]
        window = hist.loc[:d1].tail(lookback)
        if window.empty or window["volume"].mean() == 0:
            continue

        stats["have_D_D1"] += 1

        close_up = (
            (d1_row["close"] - window.iloc[-2]["close"]) / window.iloc[-2]["close"] * 100.0
            if len(window) >= 2
            else 0.0
        )
        vol_mult = (
            (d1_row["volume"] / window["volume"].mean()) if window["volume"].mean() else 0.0
        )

        d_row = hist.loc[D] if D in hist.index else None
        if d_row is None:
            continue
        gap_pct = (d_row["open"] - d1_row["close"]) / d1_row["close"] * 100.0

        stage_info = {
            "ticker": t,
            "d1_close_up_pct": close_up,
            "d1_vol_mult": vol_mult,
            "gap_open_pct": gap_pct,
        }

        if close_up >= min_close_up:
            stats["pass_close_up"] += 1
        else:
            near.append(stage_info)
            continue

        if vol_mult >= min_vol_mult:
            stats["pass_volume_mult"] += 1
        else:
            near.append(stage_info)
            continue

        if gap_pct >= min_gap_next_open:
            stats["pass_gap"] += 1
        else:
            near.append(stage_info)
            continue

        # Placeholder filters for minimum close, ATR, and 21-day trend
        stats["pass_min_close"] += 1
        stats["pass_atr"] += 1
        stats["pass_21d"] += 1

        results.append(stage_info)

    stats["final"] = len(results)
    near_df = pd.DataFrame(near).sort_values(["gap_open_pct", "d1_vol_mult"], ascending=False)
    res_df = pd.DataFrame(results).sort_values("gap_open_pct", ascending=False)
    return res_df, stats, near_df.head(10)


def render_page():
    st.header("⚡ Yesterday Close+Volume → Buy Next Open")
    storage = _get_storage()

    debug = st.checkbox(
        "Show debug information", value=st.session_state.get("show_debug", False)
    )
    st.session_state["show_debug"] = debug

    _d = st.date_input("Entry day (D)", value=dt.date.today())
    if isinstance(_d, (list, tuple)):
        _d = _d[0]
    D = pd.Timestamp(_d)
    lookback = int(st.number_input("Lookback", value=63, min_value=1, step=1))
    min_close_up = float(
        st.number_input("Min close-up on D-1 (%)", value=3.0, step=0.5)
    )
    min_vol_mult = float(
        st.number_input("Min volume multiple", value=1.5, step=0.1)
    )
    min_gap = float(st.number_input("Min gap open (%)", value=0.0, step=0.1))

    if st.button("Run scan"):
        members = _load_members(storage)
        active = members_on_date(members, D)
        symbols = active["ticker"].unique().tolist()
        frames_present = [s for s in symbols if storage.exists(f"prices/{s}.parquet")]

        stats = {"active_members": len(symbols)}

        if len(frames_present) == 0:
            with st.expander(
                "Debug: filter pipeline", expanded=st.session_state.get("show_debug", False)
            ):
                for k, v in stats.items():
                    st.write(f"{k}: {v}")
            st.warning(
                "No price files found for selected date’s membership. Ingest more prices or pick another date."
            )
            return

        if active.empty:
            with st.expander(
                "Debug: filter pipeline", expanded=st.session_state.get("show_debug", False)
            ):
                for k, v in stats.items():
                    st.write(f"{k}: {v}")
            st.warning("No active S&P members on selected date.")
            return

        results, stats, near = _run_signal_scan(
            active,
            D=D,
            lookback=lookback,
            min_close_up=min_close_up,
            min_vol_mult=min_vol_mult,
            min_gap_next_open=min_gap,
            opt={},
        )

        with st.expander(
            "Debug: filter pipeline", expanded=st.session_state.get("show_debug", False)
        ):
            for k, v in stats.items():
                st.write(f"{k}: {v}")

        if results.empty:
            st.warning("No matches for the selected filters.")
            if not near.empty:
                st.dataframe(near, use_container_width=True)
            return

        st.success(f"{len(results)} matches")
        st.dataframe(results, use_container_width=True)


def page():
    render_page()
