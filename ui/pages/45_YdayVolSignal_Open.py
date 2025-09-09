from __future__ import annotations
import io
import pandas as pd
import streamlit as st
from data_lake.storage import Storage
from engine.features import atr
from engine.universe import members_on_date
from engine.replay import time_to_hit
from engine.signal_buy_open import (
    support_resistance,
    has_21d_forward_hit,
    is_atr_feasible,
)


@st.cache_data(show_spinner=False)
def _load_members(storage: Storage) -> pd.DataFrame:
    data = storage.read_bytes("membership/sp500_members.parquet")
    df = pd.read_parquet(io.BytesIO(data))
    return df


@st.cache_data(show_spinner=False)
def _load_prices(storage: Storage, ticker: str) -> pd.DataFrame:
    data = storage.read_bytes(f"prices/{ticker}.parquet")
    df = pd.read_parquet(io.BytesIO(data))
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date']).set_index('date').sort_index()
    else:
        df = df.copy()
        df.index = pd.to_datetime(df.index, errors='coerce')
        df = df.sort_index()
    return df


def render_page() -> None:
    st.subheader("⚡ Yesterday Close+Volume → Buy Next Open")
    storage = Storage()
    st.caption(storage.info())

    # ---- Inputs ----
    D = st.date_input("Entry day (D)", value=pd.Timestamp.today().normalize())
    D = pd.Timestamp(D)
    min_close_up_pct = st.number_input("Min close-up on D-1 (%)", value=3.0, step=0.5, format="%.2f") / 100.0
    vol_window = st.number_input("Volume lookback (sessions)", value=63, min_value=5, step=5)
    min_vol_mult = st.number_input(
        "Min volume multiple (vs lookback avg)",
        min_value=0.10,
        value=1.50,
        step=0.10,
        format="%.2f",
        help="1.30 = yesterday's volume ≥ 1.30× average over the lookback window (ending on D-1).",
    )
    min_gap_on_open = st.number_input(
        "Min gap at D open vs D-1 close (%)", value=0.10, step=0.1, format="%.2f"
    ) / 100.0

    st.markdown("**Optional filters (all computed as of D-1)**")
    col1, col2, col3 = st.columns(3)
    with col1:
        use_atr_abs = st.checkbox("Min ATR(21) $", value=False)
        min_atr_abs = (
            st.number_input("ATR(21) ≥ ($)", value=0.50, step=0.10, format="%.2f") if use_atr_abs else None
        )
    with col2:
        use_atr_pct = st.checkbox("Min ATR% of price", value=True)
        min_atr_pct = (
            st.number_input("ATR(21) ≥ (% of close)", value=1.50, step=0.10, format="%.2f") / 100.0
            if use_atr_pct
            else None
        )
    with col3:
        use_ret21 = st.checkbox("Min 21-day return", value=True)
        min_ret21 = (
            st.number_input("Ret21 ≥ (%)", value=0.0, step=0.5, format="%.2f") / 100.0
            if use_ret21
            else None
        )

    col4, col5 = st.columns(2)
    with col4:
        use_dollar_liq = st.checkbox("Min avg $ volume (20d)", value=True)
        min_avg_dol_vol = (
            st.number_input("Avg $ vol 20d ≥", value=10_000_000, step=1_000_000) if use_dollar_liq else None
        )
    with col5:
        min_price = st.number_input("Min close price on D-1 ($)", value=10.0, step=0.5, format="%.2f")

    tps = st.multiselect("TP set", options=[0.02, 0.03, 0.04, 0.05, 0.08, 0.10], default=[0.02, 0.04])
    horizon = st.number_input("Horizon (days)", value=30, step=5)

    if st.button("Run scan"):
        members = _load_members(storage)
        active = members_on_date(members, D)
        if active.empty:
            st.warning("No active S&P members on selected date.")
            return

        rows = []
        tickers = active['ticker'].unique().tolist()
        prog = st.progress(0.0, text=f"Scanning {len(tickers)} tickers…")

        for i, ticker in enumerate(tickers, 1):
            prog.progress(i / len(tickers), text=f"{i}/{len(tickers)} {ticker}")
            try:
                df = _load_prices(storage, ticker)
                if D not in df.index:
                    continue
                s_loc = df.index.get_loc(D) - 1
                if s_loc < 1:
                    continue
                S = df.index[s_loc]  # signal day = D-1

                # --- metrics as of S ---
                close_S = float(df.loc[S, 'close'])
                close_Sm1 = float(df.iloc[s_loc - 1]['close'])
                ret1d_S = (close_S / close_Sm1) - 1.0
                avg_vol_S = df['volume'].rolling(int(vol_window), min_periods=int(vol_window)).mean().loc[S]
                if pd.isna(avg_vol_S) or avg_vol_S <= 0:
                    continue
                vol_mult_S = float(df.loc[S, 'volume']) / float(avg_vol_S)
                atr21 = atr(df, 21).loc[S]
                atr_pct_S = (atr21 / close_S) if close_S > 0 else pd.NA
                ret21_S = df['close'].pct_change(21).loc[S]
                avg_dol_20_S = (df['close'] * df['volume']).rolling(20, min_periods=20).mean().loc[S]

                # --- mandatory filters ---
                if close_S < float(min_price):
                    continue
                if ret1d_S < float(min_close_up_pct):
                    continue
                if vol_mult_S < float(min_vol_mult):
                    continue

                # --- optional filters ---
                if use_atr_abs and (pd.isna(atr21) or atr21 < float(min_atr_abs)):
                    continue
                if use_atr_pct and (pd.isna(atr_pct_S) or atr_pct_S < float(min_atr_pct)):
                    continue
                if use_ret21 and (pd.isna(ret21_S) or ret21_S < float(min_ret21)):
                    continue
                if use_dollar_liq and (
                    pd.isna(avg_dol_20_S) or avg_dol_20_S < float(min_avg_dol_vol)
                ):
                    continue

                # --- entry at D open if gap condition passes ---
                open_D = float(df.loc[D, 'open'])
                if open_D < close_S * (1 + float(min_gap_on_open)):
                    continue

                S_lvl, R_lvl = support_resistance(df, D, open_D)
                if S_lvl is None or R_lvl is None or R_lvl <= open_D or S_lvl >= open_D:
                    continue
                rr = (R_lvl - open_D) / (open_D - S_lvl)
                if rr < 2.0:
                    continue

                tp_halfway_pct = 0.5 * (R_lvl - open_D) / open_D
                if not is_atr_feasible(tp_halfway_pct, float(atr_pct_S)):
                    continue
                if not has_21d_forward_hit(df, s_loc, tp_halfway_pct):
                    continue

                all_tps = list({tp_halfway_pct, *tps})
                hits = time_to_hit(df, D, open_D, all_tps, int(horizon))
                days_half = hits.pop(tp_halfway_pct, None)
                hit_half = days_half is not None

                row = {
                    "symbol": ticker,
                    "date_D": pd.to_datetime(D).date(),
                    "close_Dm1": close_S,
                    "open_D": open_D,
                    "ret_Dm1_%": ret1d_S,
                    "vol_mult": vol_mult_S,
                    "ATR21_%": float(atr_pct_S) if pd.notna(atr_pct_S) else None,
                    "ret_21d_%": float(ret21_S) if pd.notna(ret21_S) else None,
                    "avg_dollar_vol_20d": float(avg_dol_20_S) if pd.notna(avg_dol_20_S) else None,
                    "support_S": S_lvl,
                    "resistance_R": R_lvl,
                    "RR": rr,
                    "tp_halfway_%": tp_halfway_pct,
                    "hit_halfway": hit_half,
                    "days_to_hit_halfway": days_half,
                }
                for tp in tps:
                    row[f"tp_{int(tp*100)}_days"] = hits.get(tp)
                rows.append(row)
            except Exception:
                continue

        if not rows:
            st.warning("No matches for the selected filters.")
        else:
            out = pd.DataFrame(rows).sort_values(["vol_mult", "ret_Dm1_%"], ascending=[False, False])
            st.dataframe(out, use_container_width=True)
            st.download_button(
                "Download CSV",
                out.to_csv(index=False).encode(),
                file_name=f"yday_vol_signal_{D.date()}.csv",
                mime="text/csv",
            )


def page():
    render_page()
