import io
import datetime as dt
from typing import Dict

import pandas as pd
import streamlit as st

from data_lake.storage import Storage, load_prices_cached
import engine.signal_scan as sigscan
from engine.signal_scan import ScanParams, members_on_date
from ui.components.progress import status_block


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

    with st.form("bt_controls"):
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
        run = st.form_submit_button("Run backtest")

    if run:
        st.session_state["bt_running"] = True
        status, prog, log = status_block("Backtest running…", key_prefix="bt")

        try:
            storage = Storage()
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

            start_ts = start_date.normalize()
            end_ts = end_date.normalize()
            df_days = load_prices_cached(storage, ["AAPL"], start_ts, end_ts)
            df_days = df_days[df_days.get("ticker") == "AAPL"].drop(columns=["ticker"], errors="ignore")
            if df_days.empty:
                st.info("No data loaded for backtest.")
                return
            date_list = list(pd.DatetimeIndex(df_days.index).sort_values())

            members = sigscan._load_members(storage)

            active_tickers = set()
            for D in date_list:
                active_tickers.update(members_on_date(members, D)["ticker"].tolist())
            active_tickers = sorted(active_tickers)

            log(f"Preloading {len(active_tickers)} tickers…")
            prices_df = load_prices_cached(storage, active_tickers, start_ts, end_ts)
            if prices_df.empty:
                log("No prices loaded from storage.")
                st.error("No price data loaded.")
                return

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

            stats = st.session_state.get("price_load_stats", {})
            with st.expander("\U0001F50E Data preflight (debug)"):
                st.write(stats)
                st.write(
                    {
                        "rows_before_dedupe": int(rows_before),
                        "rows_after_dedupe": int(rows_after),
                        "dups_dropped": int(rows_before - rows_after),
                        "close_wide_shape": close_wide.shape,
                    }
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

            def _load_members_patched(_storage):
                return members

            orig_load_prices = sigscan._load_prices
            orig_load_members = sigscan._load_members
            sigscan._load_prices = _load_prices_patched
            sigscan._load_members = _load_members_patched

            try:
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


def page() -> None:
    render_page()
