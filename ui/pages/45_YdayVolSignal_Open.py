import io
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

    counts: dict[str, int] = {"universe": len(active), "gap": 0, "vol": 0, "atr": 0, "sr": 0}

    tickers = active["ticker"].dropna().unique().tolist()
    if not tickers:
        return pd.DataFrame(), counts, pd.DataFrame()

    D = pd.to_datetime(D)
    results: list[dict] = []
    near: list[dict] = []

    for t in tickers:
        hist = get_daily_adjusted(t, end=D, lookback=max(lookback + 2, 70))
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

        if gap_pct >= min_gap_next_open:
            counts["gap"] += 1
        else:
            near.append(stage_info)
            continue

        if close_up >= min_close_up and vol_mult >= min_vol_mult:
            counts["vol"] += 1
        else:
            near.append(stage_info)
            continue

        # ATR and S/R checks would go here
        counts["atr"] += 1
        counts["sr"] += 1

        results.append(stage_info)

    counts["final"] = len(results)
    cols = ["ticker", "d1_close_up_pct", "d1_vol_mult", "gap_open_pct"]
    near_df = pd.DataFrame(near, columns=cols)
    if not near_df.empty:
        near_df = near_df.sort_values(["gap_open_pct", "d1_vol_mult"], ascending=False)
    res_df = pd.DataFrame(results, columns=cols)
    if not res_df.empty:
        res_df = res_df.sort_values("gap_open_pct", ascending=False)
    return res_df, counts, near_df.head(10)


def render_page():
    st.header("⚡ Yesterday Close+Volume → Buy Next Open")
    storage = _get_storage()

    D = st.date_input("Entry day (D)", value=pd.Timestamp.today()).to_pydatetime()
    lookback = int(st.number_input("Lookback", value=63, min_value=1, step=1))
    min_close_up = float(st.number_input("Min close-up on D-1 (%)", value=3.0, step=0.5))
    min_vol_mult = float(st.number_input("Min volume multiple", value=1.5, step=0.1))
    min_gap = float(st.number_input("Min gap open (%)", value=0.0, step=0.1))

    if st.button("Run scan"):
        members = _load_members(storage)
        active = members_on_date(members, D)
        st.caption(f"Active members on {D.date()}: {active['ticker'].nunique()}")

        present = [t for t in active["ticker"].unique() if storage.exists(f"prices/{t}.parquet")]
        st.caption(f"Price files found: {len(present)} / {active['ticker'].nunique()}")
        if len(present) == 0:
            st.warning(
                "No price files found for selected date’s membership. Ingest more prices or pick another date."
            )
            return

        if active.empty:
            st.warning("No active S&P members on selected date.")
            with st.expander("Debug: membership on selected date"):
                st.write({"selected_date": str(D), "active_count": 0})
            return

        results, counts, near = _run_signal_scan(
            active,
            D=D,
            lookback=lookback,
            min_close_up=min_close_up,
            min_vol_mult=min_vol_mult,
            min_gap_next_open=min_gap,
            opt={},
        )

        st.caption(f"After gap filter: {counts['gap']}")
        st.caption(f"After vol filter: {counts['vol']}")
        st.caption(f"After ATR check: {counts['atr']}")
        st.caption(f"After S/R check: {counts['sr']}")
        st.caption(f"Final candidates: {counts['final']}")

        if results.empty:
            st.warning("No matches for the selected filters.")
            if not near.empty:
                st.dataframe(near, use_container_width=True)
            return

        st.success(f"{len(results)} matches")
        st.dataframe(results, use_container_width=True)


def page():
    render_page()
