import io
import datetime as dt
from typing import Dict

import pandas as pd
import streamlit as st

from data_lake.storage import Storage, load_prices_cached
import engine.signal_scan as sigscan
from engine.signal_scan import ScanParams, members_on_date
from ui.components.progress import status_block
from ui.components.debug import debug_panel, _get_dbg


def _render_df_with_copy(title: str, df: pd.DataFrame, key_prefix: str) -> None:
    st.subheader(title)
    if df is None or df.empty:
        st.info("No data")
        return

    # visible table
    st.dataframe(df, width="stretch")

    # text for controls
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    csv_txt = csv_buf.getvalue()
    md_txt = df.to_markdown(index=False)

    # download
    st.download_button(
        label="\u2b07\ufe0f Download CSV",
        data=csv_txt.encode("utf-8"),
        file_name=f"{key_prefix}.csv",
        mime="text/csv",
        key=f"{key_prefix}_dl",
    )

    # copyable textarea
    st.text_area(
        "Copy Markdown",
        value=md_txt,
        height=160,
        key=f"{key_prefix}_copy",
    )


def render_page() -> None:
    st.header("📅 Backtest (range)")
    storage = Storage()
    dbg = _get_dbg("backtest")
    dbg.set_env(storage_mode=getattr(storage, "mode", "unknown"), bucket=getattr(storage, "bucket", None))
    st.caption(f"storage: {storage.info()} mode={storage.mode}")
    if storage.force_supabase and storage.mode == "local":
        st.error(
            "Supabase required. App is configured to force Supabase; see Data Lake page for self-test."
        )
        return

    def form_submit_wrapper(label: str) -> bool:
        return st.form_submit_button(label)

    with st.form(key="backtest_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            start_date = pd.Timestamp(
                st.date_input(
                    "Start date",
                    value=dt.date.today() - dt.timedelta(days=30),
                    key="bt_start",
                )
            ).tz_localize(None)
            end_date = pd.Timestamp(
                st.date_input("End date", value=dt.date.today(), key="bt_end")
            ).tz_localize(None)
            horizon = int(
                st.number_input(
                    "Horizon (days)",
                    min_value=1,
                    value=30,
                    step=1,
                    key="bt_horizon",
                )
            )
        with col2:
            vol_lookback = int(
                st.number_input(
                    "Volume lookback",
                    min_value=1,
                    value=63,
                    step=1,
                    key="bt_vol_lookback",
                )
            )
            min_close_up_pct = float(
                st.number_input(
                    "Min close-up on D-1 (%)",
                    value=3.0,
                    step=0.5,
                    key="bt_min_close_up",
                )
            )
            min_gap_open_pct = float(
                st.number_input(
                    "Min gap open (%)",
                    value=0.0,
                    step=0.1,
                    key="bt_min_gap_open",
                )
            )
        with col3:
            min_vol_multiple = float(
                st.number_input(
                    "Min volume multiple",
                    value=1.5,
                    step=0.1,
                    key="bt_min_vol_mult",
                )
            )
            atr_window = int(
                st.number_input(
                    "ATR window",
                    min_value=5,
                    value=21,
                    step=1,
                    key="bt_atr_win",
                )
            )
            sr_min_ratio = float(
                st.number_input(
                    "Min S:R ratio",
                    value=2.0,
                    step=0.1,
                    key="bt_sr_min_ratio",
                )
            )
            sr_lookback = int(
                st.number_input(
                    "S/R lookback (days)",
                    min_value=10,
                    value=21,
                    step=1,
                    key="bt_sr_lb",
                )
            )
            use_precedent = st.checkbox(
                "Require 21-day precedent (lookback 252d, window 21d)",
                value=True,
                key="bt_req_prec",
            )
            use_atr_feasible = st.checkbox(
                "Require ATR×N feasibility (at D-1)",
                value=True,
                key="bt_req_atr",
            )
            precedent_lookback = int(
                st.number_input(
                    "Precedent lookback (days)",
                    min_value=21,
                    value=252,
                    step=1,
                    key="bt_prec_lookback",
                )
            )
            precedent_window = int(
                st.number_input(
                    "Precedent window (days)",
                    min_value=5,
                    value=21,
                    step=1,
                    key="bt_prec_window",
                )
            )
        save_outcomes = st.checkbox(
            "Save outcomes to lake", value=False, key="bt_save_outcomes"
        )
        run = form_submit_wrapper("Run backtest")

    if run:
        st.session_state["bt_running"] = True
        status, prog, log = status_block("Backtest running…", key_prefix="bt")

        try:
            params: ScanParams = {
                "min_close_up_pct": min_close_up_pct,
                "min_vol_multiple": min_vol_multiple,
                "min_gap_open_pct": min_gap_open_pct,
                "atr_window": atr_window,
                "lookback_days": vol_lookback,
                "horizon_days": horizon,
                "sr_min_ratio": sr_min_ratio,
                "sr_lookback": sr_lookback,
                "use_precedent": use_precedent,
                "use_atr_feasible": use_atr_feasible,
                "precedent_lookback": precedent_lookback,
                "precedent_window": precedent_window,
            }
            dbg.set_params(
                start=str(start_date),
                end=str(end_date),
                horizon=horizon,
                vol_lookback=vol_lookback,
                min_close_up_pct=min_close_up_pct,
                min_gap_open_pct=min_gap_open_pct,
                min_volume_multiple=min_vol_multiple,
                atr_window=atr_window,
                sr_min_ratio=sr_min_ratio,
                sr_lookback=sr_lookback,
                use_precedent=use_precedent,
                use_atr_feasible=use_atr_feasible,
                precedent_lookback=precedent_lookback,
                precedent_window=precedent_window,
            )

            start_ts = start_date.normalize()
            end_ts = end_date.normalize()
            df_days = load_prices_cached(
                storage,
                ["AAPL"],
                start_ts,
                end_ts,
                cache_salt=storage.cache_salt(),
            )
            df_days = df_days[df_days.get("Ticker") == "AAPL"].drop(columns=["Ticker"], errors="ignore")
            if df_days.empty:
                st.info("No data loaded for backtest.")
                debug_panel("backtest")
                return
            date_list = list(pd.DatetimeIndex(df_days.index).sort_values())

            members = sigscan._load_members(storage, cache_salt=storage.cache_salt())

            active_tickers = set()
            for D in date_list:
                active_tickers.update(members_on_date(members, D)["ticker"].tolist())
            active_tickers = sorted(active_tickers)

            log(f"Preloading {len(active_tickers)} tickers…")
            with dbg.step("preload_prices"):
                prices_df = load_prices_cached(
                    storage,
                    active_tickers,
                    start_ts,
                    end_ts,
                    cache_salt=storage.cache_salt(),
                )
            loaded = prices_df.get("Ticker").nunique() if not prices_df.empty else 0
            dbg.event(
                "prices_loaded",
                requested=len(active_tickers),
                loaded=loaded,
                index_dtype=str(getattr(prices_df.index, "dtype", "")),
                tz=str(getattr(getattr(prices_df.index, "tz", None), "zone", None)),
                min=str(prices_df.index.min() if not prices_df.empty else None),
                max=str(prices_df.index.max() if not prices_df.empty else None),
            )
            with st.expander("\U0001F50E Data preflight (debug)"):
                st.write(f"Tickers requested: {len(active_tickers)}")
                sample_check = active_tickers[:5]
                presence = {}
                for t in sample_check:
                    p = f"prices/{t}.parquet"
                    try:
                        presence[t] = bool(getattr(storage, "exists", None) and storage.exists(p))
                    except Exception as e:
                        presence[t] = f"exists() error: {e}"
                st.write({"presence": presence})
                loaded = prices_df.get("Ticker").unique().tolist() if not prices_df.empty else []
                st.write(f"Loaded series: {len(loaded)}")
                if not prices_df.empty:
                    st.write(
                        {
                            "index_dtype": str(prices_df.index.dtype),
                            "tz": getattr(prices_df.index, "tz", None),
                            "min": prices_df.index.min(),
                            "max": prices_df.index.max(),
                        }
                    )
            if prices_df.empty:
                log(
                    "No data loaded for backtest. Check presence above (exists) and list_prefix in Data Lake tab."
                )
                debug_panel("backtest")
                st.stop()

            rows_before = len(prices_df)
            prices_df = prices_df.reset_index(names="date")
            prices_df = (
                prices_df.drop_duplicates(subset=["Ticker", "date"], keep="last")
                .set_index("date")
                .sort_index()
            )
            rows_after = len(prices_df)

            try:
                close_wide = prices_df.pivot_table(
                    index=prices_df.index, columns="Ticker", values="Close", aggfunc="last"
                )
                open_wide = prices_df.pivot_table(
                    index=prices_df.index, columns="Ticker", values="Open", aggfunc="last"
                )
                vol_wide = prices_df.pivot_table(
                    index=prices_df.index, columns="Ticker", values="Volume", aggfunc="last"
                )
            except Exception as e:
                st.error(f"Failed to assemble price matrix: {e}")
                return

            prices: Dict[str, pd.DataFrame] = {}
            for t, df_t in prices_df.reset_index().groupby("Ticker", sort=False):
                df_t = df_t.set_index("date").sort_index()
                df_t = df_t[~df_t.index.duplicated(keep="last")]
                prices[t] = df_t[["Open", "High", "Low", "Close", "Volume"]].rename(
                    columns=str.lower
                )


            prog.progress(100, text=f"Prefetch {len(prices)}/{len(active_tickers)}")
            log("Prefetch complete.")

            def _load_prices_patched(_storage, ticker):
                df = prices.get(ticker)
                if df is None:
                    return None
                out = df.reset_index().rename(columns={"index": "date"})
                out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
                return out[["date", "open", "high", "low", "close", "volume"]].dropna().sort_values("date")

            def _load_members_patched(_storage, cache_salt: str):
                return members

            orig_load_prices = sigscan._load_prices
            orig_load_members = sigscan._load_members
            sigscan._load_prices = _load_prices_patched
            sigscan._load_members = _load_members_patched

            try:
                trades_df = pd.DataFrame()
                summary: Dict[str, float] = {}
                with dbg.step("run_backtest"):
                    results = []
                    days_with_candidates = 0
                    done = 0
                    total_days = len(date_list)
                    for D in date_list:
                        cands, out, _fails, _stats = sigscan.scan_day(
                            storage, pd.Timestamp(D), params
                        )
                        cand_count = int(len(cands))
                        if cand_count:
                            days_with_candidates += 1
                        if not out.empty:
                            tmp = out.copy()
                            tmp["date"] = pd.Timestamp(D).normalize()
                            results.append(tmp)
                        done += 1
                        pct = int(done / max(1, total_days) * 100)
                        prog.progress(pct, text=f"{pct}%  ({pd.Timestamp(D).date()})")
                        if done % 5 == 0:
                            log(f"{pd.Timestamp(D).date()} → {cand_count} candidates")

                    trades_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
                    hits = int(trades_df["hit"].sum()) if not trades_df.empty else 0
                    summary = {
                        "total_days": int(total_days),
                        "days_with_candidates": int(days_with_candidates),
                        "trades": int(len(trades_df)),
                        "hits": hits,
                        "hit_rate": float(hits) / max(1, len(trades_df)),
                        "median_days_to_exit": float(trades_df["days_to_exit"].median()) if not trades_df.empty else float("nan"),
                        "avg_MAE_pct": float(trades_df["mae_pct"].mean()) if not trades_df.empty else float("nan"),
                        "avg_MFE_pct": float(trades_df["mfe_pct"].mean()) if not trades_df.empty else float("nan"),
                    }
                dbg.event(
                    "bt_stats",
                    days=len(date_list),
                    total_trades=int(summary.get("trades", 0)),
                )

                st.session_state["bt_trades"] = trades_df
                st.session_state["bt_summary"] = summary

                if save_outcomes and not trades_df.empty:
                    run_id = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                    buf = io.BytesIO()
                    trades_df.to_parquet(buf, index=False)
                    path = f"backtests/{run_id}.parquet"
                    storage.write_bytes(path, buf.getvalue())
                    st.session_state["bt_save_path"] = path

                status.update(
                    label=f"Backtest complete ✅ ({len(trades_df)} trades)", state="complete"
                )
                st.toast("Backtest finished", icon="✅")
            finally:
                sigscan._load_prices = orig_load_prices
                sigscan._load_members = orig_load_members
        except Exception as e:
            dbg.error("backtest", e)
            log(f"ERROR: {e}")
            status.update(label="Backtest failed ❌", state="error")
        finally:
            st.session_state["bt_running"] = False

    trades_df = st.session_state.get("bt_trades")
    summary = st.session_state.get("bt_summary")
    save_path = st.session_state.get("bt_save_path")

    if summary is not None:
        st.subheader("Summary")
        st.dataframe(pd.DataFrame([summary]), key="bt_summary_df")

    if trades_df is not None:
        if trades_df.empty:
            st.info("No trades found in this range.")
        else:
            _render_df_with_copy("Trades", trades_df, "bt_trades")

    if save_path:
        st.success(f"Saved to lake at {save_path}")

    debug_panel("backtest")


def page() -> None:
    render_page()
