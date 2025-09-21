import io
import datetime as dt
from typing import Dict

import pandas as pd
from pandas.tseries.offsets import BDay
import streamlit as st

from data_lake.storage import ConfigurationError, Storage, load_prices_cached

import engine.signal_scan as sigscan
from engine.signal_scan import ScanParams, members_on_date
from ui.components.progress import status_block
from ui.components.debug import debug_panel, _get_dbg
from ui.components.tables import show_df
from ui.price_filter import (
    CALLOUT_MESSAGE,
    STRUCTURED_ERROR_MESSAGE,
    PriceFilterUnavailableError,
    get_price_filter,
    handle_filter_exception,
)


def page() -> None:
    st.header("📅 Backtest (range)")
    storage = Storage()

    diag_fn = getattr(storage, "diagnostics", None)
    if callable(diag_fn):
        diag = diag_fn()
    else:  # pragma: no cover - defensive fallback for older Storage versions
        diag = {
            "mode": getattr(storage, "mode", "unknown"),
            "bucket": getattr(storage, "bucket", None),
            "local_root": str(getattr(storage, "local_root", "")) if getattr(storage, "mode", None) == "local" else None,
            "supabase_available": getattr(storage, "supabase_available", lambda: False)(),
        }

    try:
        filter_tickers_with_parquet, filter_source = get_price_filter()
    except PriceFilterUnavailableError as exc:
        filter_tickers_with_parquet = None
        filter_source = "unavailable"
        filter_error = exc
    else:
        filter_error = None

    dbg = _get_dbg("backtest")
    dbg.set_env(
        storage_mode=getattr(storage, "mode", "unknown"),
        bucket=getattr(storage, "bucket", None),
        ticker_filter_source=filter_source,
    )
    st.caption(f"storage: mode={diag['mode']} bucket={diag['bucket']}")
    if filter_source == "fallback":
        st.warning(
            "Price availability helper unavailable; ALLOW_FALLBACK enabled — using direct parquet probes."
        )

    if getattr(storage, "force_supabase", False) and storage.mode == "local":
        st.error(
            "Supabase is required but not available. Check secrets: [supabase] url/key, or disable supabase.force."
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
                    value=14,
                    step=1,
                    key="bt_atr_win",
                )
            )
            atr_method = st.selectbox(
                "ATR method",
                options=("wilder", "sma"),
                format_func=lambda opt: "Wilder (RMA)" if opt == "wilder" else opt.upper(),
                index=0,
                key="bt_atr_method",
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
            min_precedent_hits = int(
                st.number_input(
                    "Min precedent hits",
                    min_value=0,
                    value=1,
                    step=1,
                    key="bt_prec_min_hits",
                )
            )
            exit_model = st.selectbox(
                "Exit model",
                options=("pct_tp_only", "sr_tp_vs_stop"),
                index=0,
                format_func=lambda opt: "Percent target only" if opt == "pct_tp_only" else "Legacy S/R TP+stop",
                key="bt_exit_model",
            )
            tp_mode = st.radio(
                "Take-profit mode",
                options=("sr_fraction", "atr_multiple"),
                index=0,
                format_func=lambda opt: "S/R fraction" if opt == "sr_fraction" else "ATR multiple",
                key="bt_tp_mode",
            )
            if "bt_tp_sr_fraction" not in st.session_state:
                st.session_state["bt_tp_sr_fraction"] = 0.50
            if "bt_tp_atr_multiple" not in st.session_state:
                st.session_state["bt_tp_atr_multiple"] = 0.50
            if "bt_tp_atr_preset" not in st.session_state:
                st.session_state["bt_tp_atr_preset"] = 0.50
            if tp_mode == "sr_fraction":
                tp_sr_fraction = float(
                    st.number_input(
                        "Fraction (0–1]",
                        min_value=0.05,
                        max_value=1.0,
                        value=float(st.session_state.get("bt_tp_sr_fraction", 0.50)),
                        step=0.05,
                        key="bt_tp_sr_fraction",
                    )
                )
                tp_atr_multiple = float(st.session_state.get("bt_tp_atr_multiple", 0.50))
            else:
                atr_presets = (0.25, 0.5, 0.75, 1.0)
                preset = st.radio(
                    "ATR multiple presets",
                    options=atr_presets,
                    index=atr_presets.index(0.5),
                    horizontal=True,
                    key="bt_tp_atr_preset",
                )
                if st.session_state.get("_bt_tp_atr_last_preset") != preset:
                    st.session_state["_bt_tp_atr_last_preset"] = preset
                    st.session_state["bt_tp_atr_multiple"] = float(preset)
                tp_atr_multiple = float(
                    st.number_input(
                        "ATR multiple k",
                        min_value=0.05,
                        value=float(st.session_state.get("bt_tp_atr_multiple", preset)),
                        step=0.05,
                        key="bt_tp_atr_multiple",
                    )
                )
                tp_sr_fraction = float(st.session_state.get("bt_tp_sr_fraction", 0.50))
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
                "atr_method": atr_method,
                "lookback_days": vol_lookback,
                "horizon_days": horizon,
                "sr_min_ratio": sr_min_ratio,
                "sr_lookback": sr_lookback,
                "use_precedent": use_precedent,
                "use_atr_feasible": use_atr_feasible,
                "precedent_lookback": precedent_lookback,
                "precedent_window": precedent_window,
                "min_precedent_hits": min_precedent_hits,
                "exit_model": exit_model,
                "tp_mode": tp_mode,
                "tp_sr_fraction": tp_sr_fraction,
                "tp_atr_multiple": tp_atr_multiple,
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
                atr_method=atr_method,
                sr_min_ratio=sr_min_ratio,
                sr_lookback=sr_lookback,
                use_precedent=use_precedent,
                use_atr_feasible=use_atr_feasible,
                precedent_lookback=precedent_lookback,
                precedent_window=precedent_window,
                min_precedent_hits=min_precedent_hits,
                exit_model=exit_model,
                tp_mode=tp_mode,
                tp_sr_fraction=tp_sr_fraction,
                tp_atr_multiple=tp_atr_multiple,
            )

            prior_needed_bdays = int(
                max(
                    63,
                    vol_lookback,
                    atr_window,
                    sr_lookback,
                    (precedent_lookback if use_precedent else 0),
                )
            )
            warmup_start = (start_date - BDay(prior_needed_bdays + 3)).tz_localize(None)
            forward_end = (end_date + BDay(horizon)).tz_localize(None)

            date_index = pd.bdate_range(start_date, end_date)
            if date_index.empty:
                st.info("No business days in selected range.")
                debug_panel("backtest")
                return
            date_list = list(date_index)

            if filter_tickers_with_parquet is None:
                status.update(label="Backtest cancelled ❌", state="error")
                st.error(CALLOUT_MESSAGE)
                log(STRUCTURED_ERROR_MESSAGE)
                dbg.event(
                    "price_filter_unavailable",
                    code=PriceFilterUnavailableError.code,
                    reason=getattr(filter_error, "reason", None),
                )
                debug_panel("backtest")
                return

            members = sigscan._load_members(storage, cache_salt=storage.cache_salt())
            members_ticker_upper = members["ticker"].astype(str).str.upper()

            active_tickers = set()
            for D in date_index:
                daily_members = members_on_date(members, D)["ticker"].dropna().tolist()
                active_tickers.update(str(t).upper() for t in daily_members if t)
            active_tickers = sorted(active_tickers)

            if not active_tickers:
                st.info("No eligible tickers found in selected range.")
                debug_panel("backtest")
                return

            requested_count = len(active_tickers)
            try:
                filtered_tickers, missing_tickers = filter_tickers_with_parquet(
                    storage, active_tickers
                )
            except ConfigurationError as exc:
                error = handle_filter_exception(exc)
                status.update(label="Backtest cancelled ❌", state="error")
                st.error(error.user_message)
                log(error.structured_message)
                dbg.event(
                    "price_filter_unavailable",
                    code=error.code,
                    reason=getattr(error, "reason", str(exc)),
                )
                debug_panel("backtest")
                return
            dbg.event(
                "ticker_filter",
                requested=requested_count,
                available=len(filtered_tickers),
                missing=len(missing_tickers),
            )

            if missing_tickers:
                missing_count = len(missing_tickers)
                available_count = len(filtered_tickers)
                warn_text = (
                    f"Out of {requested_count} S&P tickers in the selected range, "
                    f"{available_count} are available for backtesting. "
                    f"{missing_count} are missing price files and were excluded automatically."
                )
                st.warning(warn_text)
                log(warn_text)

                with st.expander("Missing ticker diagnostics", expanded=False):
                    st.caption(
                        "Tickers listed below were not found in Supabase Storage under "
                        "`prices/*.parquet`."
                    )
                    preview_count = min(50, missing_count)
                    st.code("\n".join(missing_tickers[:preview_count]))
                    if missing_count > preview_count:
                        st.caption(
                            f"Showing the first {preview_count} tickers (of {missing_count} total)."
                        )
                    st.download_button(
                        "Download missing tickers (.txt)",
                        data="\n".join(missing_tickers),
                        file_name="missing_tickers.txt",
                        mime="text/plain",
                        key="bt_missing_download",
                    )

            if not filtered_tickers:
                status.update(label="Backtest cancelled ❌", state="error")
                st.error(
                    "No valid price data found for the provided tickers. "
                    "Please check your list or reload parquet files."
                )
                debug_panel("backtest")
                return

            active_tickers = filtered_tickers

            log(f"Preloading {len(active_tickers)} tickers…")
            with dbg.step("preload_prices"):
                prices_df = load_prices_cached(
                    storage,
                    cache_salt=storage.cache_salt(),
                    tickers=active_tickers,
                    start=warmup_start,
                    end=forward_end,
                )

            coverage_info = {
                "ui_start": str(start_date),
                "ui_end": str(end_date),
                "warmup_start": str(warmup_start),
                "forward_end": str(forward_end),
                "prior_needed_bdays": int(prior_needed_bdays),
                "horizon_bdays": int(horizon),
                "min_loaded": None,
                "max_loaded": None,
            }
            if len(prices_df):
                min_loaded = None
                max_loaded = None
                if isinstance(prices_df.index, pd.MultiIndex):
                    date_levels = [
                        i
                        for i, name in enumerate(prices_df.index.names or [])
                        if name and name.lower() in ("date", "timestamp")
                    ]
                    if date_levels:
                        idx_vals = prices_df.index.get_level_values(date_levels[0])
                        min_loaded = str(idx_vals.min())
                        max_loaded = str(idx_vals.max())
                if min_loaded is None and pd.api.types.is_datetime64_any_dtype(prices_df.index):
                    min_loaded = str(prices_df.index.min())
                    max_loaded = str(prices_df.index.max())
                if min_loaded is None:
                    for dc in ("date", "Date", "timestamp", "Timestamp"):
                        if dc in prices_df.columns:
                            dates = pd.to_datetime(prices_df[dc])
                            min_loaded = str(dates.min())
                            max_loaded = str(dates.max())
                            break
                coverage_info.update(min_loaded=min_loaded, max_loaded=max_loaded)
            dbg.event("coverage", **coverage_info)

            loaded = prices_df.get("Ticker").nunique() if not prices_df.empty else 0
            dbg.event(
                "prices_loaded",
                requested=len(active_tickers),
                loaded=loaded,
                index_dtype=str(prices_df["date"].dtype if not prices_df.empty else ""),
                tz=str(getattr(getattr(prices_df["date"], "dt", None), "tz", None) if not prices_df.empty else None),
                min=str(prices_df["date"].min() if not prices_df.empty else None),
                max=str(prices_df["date"].max() if not prices_df.empty else None),
            )

            with st.expander("\U0001F50E Data preflight (debug)", expanded=False):
                requested = len(active_tickers)
                loaded_list = (
                    sorted(prices_df["Ticker"].dropna().unique().tolist())
                    if not prices_df.empty and "Ticker" in prices_df.columns
                    else []
                )
                st.write(f"Tickers requested: {requested}")
                st.write(f"Loaded series: {len(loaded_list)} / {requested}")

                if prices_df.empty:
                    st.warning("No prices loaded from storage.")
                    missing_examples: list[str] = []
                    for ticker in active_tickers:
                        path = f"prices/{ticker}.parquet"
                        try:
                            if not storage.exists(path):
                                missing_examples.append(path)
                        except Exception:
                            missing_examples.append(path)
                        if len(missing_examples) >= 5:
                            break
                    if missing_examples:
                        st.caption("Missing examples (first 5):")
                        st.code("\n".join(missing_examples))
                else:
                    if loaded_list:
                        sample_ticker = loaded_list[0]
                        df0 = (
                            prices_df.loc[prices_df["Ticker"] == sample_ticker]
                            .drop(columns=["Ticker"], errors="ignore")
                        )
                        last_row = df0.sort_values("date").iloc[-1]
                        sample_out = {
                            "sample": sample_ticker,
                            "rows": int(df0.shape[0]),
                            "range": [
                                str(df0["date"].min()),
                                str(df0["date"].max()),
                            ],
                            "last_bar_date": str(pd.to_datetime(last_row["date"]).date()),
                            "last_bar_close": float(last_row.get("Close", float("nan")))
                            if "Close" in df0.columns
                            else None,
                            "last_bar_adj_close": float(last_row.get("Adj Close", float("nan")))
                            if "Adj Close" in df0.columns
                            else None,
                            "last_bar_dividends": float(last_row.get("Dividends", 0) or 0)
                            if "Dividends" in df0.columns
                            else 0.0,
                            "last_bar_stock_splits": float(last_row.get("Stock Splits", 0) or 0)
                            if "Stock Splits" in df0.columns
                            else 0.0,
                        }
                        st.write(sample_out)

            if prices_df.empty:
                log("No prices loaded—check Storage bucket and paths (prices/*.parquet).")
                return

            def _first_trade_dates(df: pd.DataFrame) -> pd.Series:
                if isinstance(df.index, pd.MultiIndex) and "Ticker" in df.index.names:
                    ticker_level = df.index.names.index("Ticker")
                    date_candidates = [
                        i
                        for i, name in enumerate(df.index.names or [])
                        if i != ticker_level
                        and isinstance(name, str)
                        and name.lower() in ("date", "timestamp")
                    ]
                    if date_candidates:
                        date_level = date_candidates[0]
                    else:
                        date_level = 0 if ticker_level != 0 else 1
                    return df.groupby(level="Ticker").apply(
                        lambda s: s.index.get_level_values(date_level).min()
                    )
                if "Ticker" not in df.columns:
                    raise ValueError(
                        "prices_df must have a 'Ticker' column or MultiIndex with 'Ticker'"
                    )
                if pd.api.types.is_datetime64_any_dtype(df.index):
                    return df.reset_index().groupby("Ticker")["index"].min()
                for dc in ("Date", "date", "Timestamp", "timestamp"):
                    if dc in df.columns:
                        return (
                            df.assign(__d=pd.to_datetime(df[dc]))
                            .groupby("Ticker")["__d"]
                            .min()
                        )
                raise ValueError(
                    "Could not find a date index or date column in prices_df"
                )

            first_dates_raw = _first_trade_dates(prices_df)
            first_dates_df = first_dates_raw.to_frame(name="first_date").dropna()
            first_dates_df["ticker"] = first_dates_df.index.map(lambda x: str(x).upper())
            first_date_by_ticker = (
                first_dates_df.groupby("ticker")["first_date"].min().sort_index()
            )
            requested_tickers = set(active_tickers)
            first_date_by_ticker = first_date_by_ticker.loc[
                ~first_date_by_ticker.index.duplicated()
            ]
            first_date_by_ticker = first_date_by_ticker.loc[
                first_date_by_ticker.index.isin(requested_tickers)
            ]
            loaded_tickers = set(first_date_by_ticker.index)
            missing = sorted(requested_tickers - loaded_tickers)
            dbg.event(
                "history_index",
                requested=len(active_tickers),
                loaded=len(loaded_tickers),
                missing_count=len(missing),
            )
            first_date_map = first_date_by_ticker.to_dict()

            rows_before = len(prices_df)
            prices_df = (
                prices_df.drop_duplicates(subset=["Ticker", "date"], keep="last")
                .set_index("date")
                .sort_index()
            )
            rows_after = len(prices_df)

            trading_index = pd.DatetimeIndex(prices_df.index.unique()).dropna()
            try:
                trading_index = trading_index.tz_localize(None)
            except TypeError:
                pass
            trading_index = trading_index.sort_values()
            date_list = [
                d for d in trading_index if start_date <= d <= end_date
            ]
            if not date_list:
                st.info("No trading days with price data in selected range.")
                debug_panel("backtest")
                return

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
                t_upper = str(t).upper()
                df_t = df_t.set_index("date").sort_index()
                df_t = df_t[~df_t.index.duplicated(keep="last")]
                prices[t_upper] = df_t[["Open", "High", "Low", "Close", "Volume"]].rename(
                    columns=str.lower
                )

            eligible_state = {"tickers": None}

            prog.progress(100, text=f"Prefetch {len(prices)}/{len(active_tickers)}")
            log("Prefetch complete.")

            def _load_prices_patched(_storage, ticker):
                df = prices.get(str(ticker).upper())
                if df is None:
                    return None
                out = df.reset_index().rename(columns={"index": "date"})
                out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
                return out[["date", "open", "high", "low", "close", "volume"]].dropna().sort_values("date")

            def _load_members_patched(_storage, cache_salt: str):
                tickers = eligible_state.get("tickers")
                if tickers is None:
                    return members
                mask = members_ticker_upper.isin(tickers)
                return members.loc[mask]

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
                        current_day = pd.Timestamp(D).tz_localize(None)

                        daily_members = (
                            members_on_date(members, current_day)["ticker"].dropna().astype(str).str.upper()
                        )

                        cutoff = current_day - BDay(prior_needed_bdays)
                        eligible = []
                        for tk in daily_members:
                            first_seen = first_date_map.get(tk)
                            if first_seen is not None and first_seen <= cutoff:
                                eligible.append(tk)
                        eligible_state["tickers"] = set(eligible)
                        eligible_count = len(eligible_state["tickers"])
                        lacking = max(len(daily_members) - eligible_count, 0)
                        dbg.event(
                            "history_guard",
                            date=str(current_day),
                            eligible=eligible_count,
                            lacking=int(lacking),
                            cutoff=str(cutoff),
                        )
                        if not eligible_count:
                            continue

                        cands, out, _fails, _stats = sigscan.scan_day(
                            storage, current_day, params
                        )
                        cand_count = int(len(cands))
                        if cand_count:
                            days_with_candidates += 1
                        if not out.empty:
                            tmp = out.copy()
                            tmp["date"] = current_day.normalize()
                            results.append(tmp)

                        done += 1
                        pct = int(done / max(1, total_days) * 100)
                        prog.progress(pct, text=f"{pct}%  ({current_day.date()})")
                        if done % 5 == 0:
                            log(f"{current_day.date()} → {cand_count} candidates")

                    trades_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
                    trades_df = trades_df.reset_index(drop=True)

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
        summary_df = pd.DataFrame([summary])
        show_df("Summary", summary_df, "bt_summary")

    if trades_df is not None:
        cols = [
            "date",
            "ticker",
            "entry_open",
            "entry_open_2dp",
            "tp_price",
            "tp_price_2dp",
            "tp_price_abs_target",
            "tp_price_abs_target_2dp",
            "tp_price_pct_target",
            "exit_model",
            "exit_date",
            "tp_touch_date",
            "hit",
            "exit_reason",
            "exit_price",
            "exit_price_2dp",
            "exit_bar_high",
            "exit_bar_high_2dp",
            "exit_bar_low",
            "exit_bar_low_2dp",
            "days_to_exit",
            "mae_pct",
            "mae_pct_2dp",
            "mae_date",
            "mfe_pct",
            "mfe_pct_2dp",
            "mfe_date",
            "close_up_pct",
            "close_up_pct_2dp",
            "vol_multiple",
            "vol_multiple_2dp",
            "gap_open_pct",
            "gap_open_pct_2dp",
            "support",
            "support_2dp",
            "resistance",
            "resistance_2dp",
            "sr_support",
            "sr_support_2dp",
            "sr_resistance",
            "sr_resistance_2dp",
            "sr_ratio",
            "sr_ratio_2dp",
            "sr_window_used",
            "sr_ok",
            "tp_frac_used",
            "tp_pct_used",
            "tp_pct_used_2dp",
            "tp_mode",
            "tp_sr_fraction",
            "tp_atr_multiple",
            "tp_halfway_pct",
            "precedent_hits",
            "precedent_ok",
            "precedent_hit_start_dates",
            "precedent_details_hits",
            "precedent_max_hit_date",
            "atr_ok",
            "atr_window",
            "atr_method",
            "atr_value_dm1",
            "atr_value_dm1_2dp",
            "atr_dminus1",
            "atr_dminus1_2dp",
            "atr_budget_dollars",
            "atr_budget_dollars_2dp",
            "tp_required_dollars",
            "tp_required_dollars_2dp",
            "reasons",
        ]

        # Filter to present columns and create a working copy for UI/CSV (minimal change)
        cols_present = [c for c in cols if c in trades_df.columns]
        df_show = trades_df[cols_present].copy()

        # --- Minimal 2-dp rounding for UI table and its downloadable CSV ---
        price_cols = [
            "entry_open", "tp_price", "tp_price_abs_target", "exit_price",
            "exit_bar_high", "exit_bar_low",
            "support", "resistance", "sr_support", "sr_resistance",
            "atr_value_dm1", "atr_dminus1", "atr_budget_dollars", "tp_required_dollars",
            # also round any *_2dp columns if they exist in the view
            "entry_open_2dp", "tp_price_2dp", "tp_price_abs_target_2dp", "exit_price_2dp",
            "exit_bar_high_2dp", "exit_bar_low_2dp",
            "support_2dp", "resistance_2dp", "sr_support_2dp", "sr_resistance_2dp",
            "atr_value_dm1_2dp", "atr_dminus1_2dp", "atr_budget_dollars_2dp", "tp_required_dollars_2dp",
        ]
        pct_cols = [
            "close_up_pct", "gap_open_pct", "tp_pct_used", "mae_pct", "mfe_pct", "tp_price_pct_target",
            # keep if present from metrics:
            "atr21_pct", "ret21_pct",
            # also round any *_2dp columns if they exist in the view
            "close_up_pct_2dp", "gap_open_pct_2dp", "tp_pct_used_2dp", "mae_pct_2dp", "mfe_pct_2dp",
        ]
        ratio_cols = [
            "sr_ratio", "vol_multiple", "tp_sr_fraction", "tp_atr_multiple",
            # also round display variants if present
            "sr_ratio_2dp", "vol_multiple_2dp",
        ]

        def _round2(frame: pd.DataFrame, col_list: list[str]) -> None:
            keep = [c for c in col_list if c in frame.columns]
            if keep:
                frame[keep] = frame[keep].apply(pd.to_numeric, errors="coerce").round(2)

        _round2(df_show, price_cols)
        _round2(df_show, pct_cols)
        _round2(df_show, ratio_cols)
        # -------------------------------------------------------------------

        show_df("Trades", df_show, "bt_trades")

    if save_path:
        st.success(f"Saved to lake at {save_path}")

    debug_panel("backtest")


def render_page() -> None:  # pragma: no cover - compatibility shim
    page()
