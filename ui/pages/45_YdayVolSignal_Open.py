from __future__ import annotations
import io
import pandas as pd
import streamlit as st
from data_lake.storage import Storage
from engine.features import atr
from engine.universe import members_on_date, missing_dates
from engine.replay import time_to_hit


@st.cache_data(show_spinner=False)
def _members_from_bytes(blob: bytes) -> pd.DataFrame:
    """Cached decode of the membership parquet (cache key = file bytes)."""
    return pd.read_parquet(io.BytesIO(blob))


def _load_members(storage: Storage) -> pd.DataFrame:
    """Read bytes via Storage (not cached) then hit the cached decoder."""
    blob = storage.read_bytes("membership/sp500_members.parquet")
    return _members_from_bytes(blob)


@st.cache_data(show_spinner=False)
def _prices_from_bytes(ticker: str, blob: bytes) -> pd.DataFrame:
    df = pd.read_parquet(io.BytesIO(blob))
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date']).set_index('date').sort_index()
    else:
        df = df.copy()
        df.index = pd.to_datetime(df.index, errors='coerce')
        df = df.sort_index()
    return df


@st.cache_data(show_spinner=False, hash_funcs={Storage: lambda _: None})
def _price_bytes(storage: Storage, ticker: str) -> bytes:
    """Cached download of the price parquet (keyed by ticker)."""
    return storage.read_bytes(f"prices/{ticker}.parquet")


def _load_prices(storage: Storage, ticker: str) -> pd.DataFrame:
    blob = _price_bytes(storage, ticker)
    return _prices_from_bytes(ticker, blob)


def render_page() -> None:
    st.subheader("⚡ Yesterday Close+Volume → Buy Next Open")
    storage = Storage()
    st.caption(storage.info())

    # ---- Inputs ----
    D = st.date_input("Entry day (D)", value=pd.Timestamp.today().normalize())
    D = pd.Timestamp(D)
    min_close_up_pct = st.number_input("Min close-up on D-1 (%)", value=3.0, step=0.5, format="%.2f") / 100.0
    vol_window       = st.number_input("Volume lookback (sessions)", value=63, min_value=5, step=5)
    min_vol_mult     = st.number_input(
        "Min volume multiple (vs lookback avg)",
        min_value=0.10, value=1.50, step=0.10, format="%.2f",
        help="1.30 = yesterday's volume ≥ 1.30× average over the lookback window (ending on D-1)."
    )
    min_gap_on_open  = st.number_input("Min gap at D open vs D-1 close (%)", value=0.0, step=0.1, format="%.2f") / 100.0

    st.markdown("**Optional filters (all computed as of D-1)**")
    col1, col2, col3 = st.columns(3)
    with col1:
        use_atr_abs = st.checkbox("Min ATR(21) $", value=False)
        min_atr_abs = st.number_input("ATR(21) ≥ ($)", value=0.50, step=0.10, format="%.2f") if use_atr_abs else None
    with col2:
        use_atr_pct = st.checkbox("Min ATR% of price", value=False)
        min_atr_pct = st.number_input("ATR(21) ≥ (% of close)", value=0.50, step=0.10, format="%.2f")/100.0 if use_atr_pct else None
    with col3:
        use_ret21 = st.checkbox("Min 21-day return", value=False)
        min_ret21 = st.number_input("Ret21 ≥ (%)", value=0.0, step=0.5, format="%.2f")/100.0 if use_ret21 else None

    col4, col5 = st.columns(2)
    with col4:
        use_dollar_liq = st.checkbox("Min avg $ volume (20d)", value=False)
        min_avg_dol_vol = st.number_input("Avg $ vol 20d ≥", value=10_000_000, step=1_000_000) if use_dollar_liq else None
    with col5:
        min_price = st.number_input("Min close price on D-1 ($)", value=5.0, step=0.5, format="%.2f")

    tps = st.multiselect("TP set", options=[0.02,0.03,0.04,0.05,0.08,0.10], default=[0.02,0.04])
    horizon = st.number_input("Horizon (days)", value=30, step=5)
    debug = st.toggle("Debug mode", value=False, help="Show stage counts and near-misses")

    if st.button("Run scan"):
        members = _load_members(storage)
        active = members_on_date(members, D)
        if active.empty:
            st.warning("No active S&P members on selected date.")
            return

        tickers = active['ticker'].unique().tolist()
        prog = st.progress(0.0, text=f"Scanning {len(tickers)} tickers…")
        rows: list[dict] = []
        missing_prices: list[dict] = []
        price_cache: dict[str, pd.DataFrame] = {}

        for i, ticker in enumerate(tickers, 1):
            prog.progress(i/len(tickers), text=f"{i}/{len(tickers)} {ticker}")
            try:
                df = _load_prices(storage, ticker)
                miss = missing_dates(df, [D, D - pd.Timedelta(days=1)])
                if miss:
                    missing_prices.append({
                        "ticker": ticker,
                        "missing": ", ".join(pd.Timestamp(m).date().isoformat() for m in miss),
                    })
                    continue
                s_loc = df.index.get_loc(D) - 1
                if s_loc < 1:
                    missing_prices.append({
                        "ticker": ticker,
                        "missing": (D - pd.Timedelta(days=2)).date().isoformat(),
                    })
                    continue
                S = df.index[s_loc]
                close_d1 = float(df.loc[S, "close"])
                close_d2 = float(df.iloc[s_loc-1]["close"])
                vol_d1 = float(df.loc[S, "volume"])
                vol_avg_lb = (
                    df["volume"].rolling(int(vol_window), min_periods=int(vol_window)).mean().loc[S]
                )
                if pd.isna(vol_avg_lb) or vol_avg_lb <= 0:
                    continue
                open_d = float(df.loc[D, "open"])
                close_up_pct = 100.0 * (close_d1 / close_d2 - 1.0)
                vol_mult = vol_d1 / vol_avg_lb
                gap_open_pct = 100.0 * (open_d / close_d1 - 1.0)
                atr21 = atr(df, 21).loc[S]
                atr21_pct = (
                    100.0 * atr21 / close_d1 if close_d1 > 0 and pd.notna(atr21) else pd.NA
                )
                ret21 = 100.0 * df["close"].pct_change(21).loc[S]
                avg_dollar_vol_20 = (
                    (df["close"] * df["volume"]).rolling(20, min_periods=20).mean().loc[S]
                )
                price_cache[ticker] = df
                rows.append({
                    "ticker": ticker,
                    "close_d1": close_d1,
                    "close_d2": close_d2,
                    "close_up_pct": close_up_pct,
                    "vol_d1": vol_d1,
                    "vol_avg_lb": float(vol_avg_lb),
                    "vol_mult": vol_mult,
                    "open_d": open_d,
                    "gap_open_pct": gap_open_pct,
                    "close_min_ok": close_d1 >= float(min_price),
                    "atr21_$": float(atr21) if pd.notna(atr21) else pd.NA,
                    "atr21_pct": float(atr21_pct) if pd.notna(atr21_pct) else pd.NA,
                    "avg_$vol_20d": (
                        float(avg_dollar_vol_20) if pd.notna(avg_dollar_vol_20) else pd.NA
                    ),
                    "ret_21d_pct": float(ret21) if pd.notna(ret21) else pd.NA,
                })
            except Exception:
                continue

        if not rows:
            if debug and missing_prices:
                st.caption("Tickers missing prices for D or D-1")
                st.dataframe(pd.DataFrame(missing_prices))
            st.warning("No matches for the selected filters.")
            return

        df = pd.DataFrame(rows).set_index("ticker")
        df["ticker"] = df.index

        # thresholds
        df["thr_close_up"] = float(min_close_up_pct) * 100.0
        df["thr_vol_mult"] = float(min_vol_mult)
        df["thr_gap_open"] = float(min_gap_on_open) * 100.0
        df["thr_min_close"] = float(min_price)
        if use_atr_abs:
            df["thr_atr_abs"] = float(min_atr_abs)
        if use_atr_pct:
            df["thr_atr_pct"] = float(min_atr_pct) * 100.0
        if use_dollar_liq:
            df["thr_avg_dol_vol"] = float(min_avg_dol_vol)
        if use_ret21:
            df["thr_ret21"] = float(min_ret21) * 100.0

        # distances & passes
        df["dist_close_up"] = df["close_up_pct"] - df["thr_close_up"]
        df["dist_vol_mult"] = df["vol_mult"] - df["thr_vol_mult"]
        df["dist_gap_open"] = df["gap_open_pct"] - df["thr_gap_open"]
        df["dist_min_close"] = df["close_d1"] - df["thr_min_close"]
        df["pass_closeup"] = df["close_up_pct"] >= df["thr_close_up"]
        df["pass_volmult"] = df["vol_mult"] >= df["thr_vol_mult"]
        df["pass_gap"] = df["gap_open_pct"] >= df["thr_gap_open"]
        df["pass_minclose"] = df["close_min_ok"]
        if use_atr_abs:
            df["dist_atr_abs"] = df["atr21_$"] - df["thr_atr_abs"]
            df["pass_atr_abs"] = df["atr21_$"] >= df["thr_atr_abs"]
        else:
            df["pass_atr_abs"] = True
        if use_atr_pct:
            df["dist_atr_pct"] = df["atr21_pct"] - df["thr_atr_pct"]
            df["pass_atr_pct"] = df["atr21_pct"] >= df["thr_atr_pct"]
        else:
            df["pass_atr_pct"] = True
        if use_dollar_liq:
            df["dist_avg_dol_vol"] = df["avg_$vol_20d"] - df["thr_avg_dol_vol"]
            df["pass_avgdvol"] = df["avg_$vol_20d"] >= df["thr_avg_dol_vol"]
        else:
            df["pass_avgdvol"] = True
        if use_ret21:
            df["dist_ret21"] = df["ret_21d_pct"] - df["thr_ret21"]
            df["pass_ret21"] = df["ret_21d_pct"] >= df["thr_ret21"]
        else:
            df["pass_ret21"] = True

        stages = [("universe", len(df))]
        fails = []
        s = df
        seq = [
            ("close_up", "pass_closeup", "close_up_pct", "thr_close_up", "dist_close_up", True),
            ("vol_mult", "pass_volmult", "vol_mult", "thr_vol_mult", "dist_vol_mult", True),
            ("open_gap", "pass_gap", "gap_open_pct", "thr_gap_open", "dist_gap_open", True),
            ("min_close", "pass_minclose", "close_d1", "thr_min_close", "dist_min_close", True),
            ("atr_abs", "pass_atr_abs", "atr21_$", "thr_atr_abs", "dist_atr_abs", use_atr_abs),
            ("atr_pct", "pass_atr_pct", "atr21_pct", "thr_atr_pct", "dist_atr_pct", use_atr_pct),
            ("avg_dollar_vol", "pass_avgdvol", "avg_$vol_20d", "thr_avg_dol_vol", "dist_avg_dol_vol", use_dollar_liq),
            ("ret21", "pass_ret21", "ret_21d_pct", "thr_ret21", "dist_ret21", use_ret21),
        ]
        for name, pcol, mcol, tcol, dcol, enabled in seq:
            if not enabled:
                continue
            fails_df = s[~s[pcol]][["ticker", mcol, tcol, dcol]]
            fails.append((name, fails_df, mcol, tcol, dcol))
            s = s[s[pcol]]
            stages.append((name, len(s)))
        final = s

        if debug:
            st.dataframe(pd.DataFrame(stages, columns=["stage", "survivors"]))
            for name, fdf, mcol, tcol, dcol in fails:
                if fdf.empty:
                    continue
                with st.expander(name):
                    st.write("fails", fdf.head(10))
                    near = fdf.dropna(subset=[dcol]).copy()
                    near["abs_dist"] = near[dcol].abs()
                    near = near.sort_values("abs_dist").head(10)
                    st.write("near-misses", near[["ticker", mcol, tcol, dcol]])
            if missing_prices:
                st.caption("Tickers missing prices for D or D-1")
                st.dataframe(pd.DataFrame(missing_prices))

        if final.empty:
            st.warning("No matches for the selected filters.")
            return

        results = []
        for ticker in final.index:
            prices = price_cache[ticker]
            open_D = final.loc[ticker, "open_d"]
            hits = time_to_hit(prices, D, open_D, tps, int(horizon))
            results.append({
                "ticker": ticker,
                "signal_day": (D - pd.Timedelta(days=1)).date(),
                "entry_day": D.date(),
                "ret1d_S": final.loc[ticker, "close_up_pct"] / 100.0,
                "vol_mult_S": final.loc[ticker, "vol_mult"],
                "ATR21_$": (
                    final.loc[ticker, "atr21_$"] if pd.notna(final.loc[ticker, "atr21_$"]) else None
                ),
                "ATR21_%": (
                    final.loc[ticker, "atr21_pct"] / 100.0
                    if pd.notna(final.loc[ticker, "atr21_pct"]) else None
                ),
                "ret21_S": (
                    final.loc[ticker, "ret_21d_pct"] / 100.0
                    if pd.notna(final.loc[ticker, "ret_21d_pct"]) else None
                ),
                "avg_dollar_vol_20_S": (
                    final.loc[ticker, "avg_$vol_20d"]
                    if pd.notna(final.loc[ticker, "avg_$vol_20d"]) else None
                ),
                "close_S": final.loc[ticker, "close_d1"],
                "open_D": open_D,
                **hits,
            })
        out = pd.DataFrame(results).sort_values(
            ["vol_mult_S", "ret1d_S"], ascending=[False, False]
        )
        st.dataframe(out, use_container_width=True)
        st.download_button(
            "Download CSV",
            out.to_csv(index=False).encode(),
            file_name=f"yday_vol_signal_{D.date()}.csv",
            mime="text/csv",
        )


def page():
    render_page()
