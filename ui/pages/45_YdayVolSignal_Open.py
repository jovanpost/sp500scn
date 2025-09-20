import io
import datetime as dt
from typing import Dict

import pandas as pd
import streamlit as st

import engine.signal_scan as sigscan
from engine.signal_scan import ScanParams, members_on_date, scan_day

# --- resilient imports (handles refactors) ---
try:
    # original path used by this page before refactors
    from data_lake.storage import ConfigurationError, Storage, load_prices_cached
except Exception:
    # canonical homes after refactor
    try:
        from data_lake.errors import ConfigurationError
    except Exception:
        # last-resort shim so the page doesn't crash on import
        class ConfigurationError(Exception):
            """Configuration error (temporary shim in page)."""
            pass
    from data_lake.storage import Storage
    from data_lake.prices import load_prices_cached

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
    st.header("⚡ Yesterday Close+Volume → Buy Next Open")

    storage = Storage()
    try:
        filter_tickers_with_parquet, filter_source = get_price_filter()
    except PriceFilterUnavailableError as exc:
        filter_tickers_with_parquet = None
        filter_source = "unavailable"
        filter_error = exc
    else:
        filter_error = None

    dbg = _get_dbg("scan")
    dbg.set_env(
        storage_mode=getattr(storage, "mode", "unknown"),
        bucket=getattr(storage, "bucket", None),
        ticker_filter_source=filter_source,
    )
    st.caption(f"storage: {storage.info()} mode={storage.mode}")
    if filter_source == "fallback":
        st.warning(
            "Price availability helper unavailable; ALLOW_FALLBACK enabled — using direct parquet probes."
        )
    if getattr(storage, "force_supabase", False) and storage.mode == "local":
        st.error(
            "Supabase is required but not available. Check secrets: [supabase] url/key, or disable supabase.force."
        )
        return

    _d = st.date_input("Entry day (D)", value=dt.date.today())
    if isinstance(_d, (list, tuple)):
        _d = _d[0]
    D = pd.Timestamp(_d).normalize()

    vol_lookback = int(st.number_input("Volume lookback", min_value=1, value=63, step=1))
    min_close_up_pct = float(st.number_input("Min close-up on D-1 (%)", value=3.0, step=0.5))
    min_vol_multiple = float(st.number_input("Min volume multiple", value=1.5, step=0.1))
    min_gap_open_pct = float(st.number_input("Min gap open (%)", value=0.0, step=0.1))
    atr_window = int(st.number_input("ATR window", min_value=1, value=14, step=1))
    atr_method = st.selectbox(
        "ATR method",
        options=("wilder", "sma"),
        format_func=lambda opt: "Wilder (RMA)" if opt == "wilder" else opt.upper(),
    )
    horizon = int(st.number_input("Horizon (days)", min_value=1, value=30, step=1))
    sr_min_ratio = float(st.number_input("Min S:R ratio", value=2.0, step=0.1))
    sr_lookback = int(st.number_input("S/R lookback (days)", min_value=10, value=21, step=1))
    exit_model = st.selectbox(
        "Exit model",
        options=("pct_tp_only", "sr_tp_vs_stop"),
        index=0,
        format_func=lambda opt: "Percent target only" if opt == "pct_tp_only" else "Legacy S/R TP+stop",
    )
    save_outcomes = st.checkbox("Save outcomes to lake", value=False)

    if st.button("Run scan", type="primary", key="scan_run"):
        st.session_state["scan_running"] = True
        status, prog, log = status_block("Running filters…", key_prefix="scan")

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
                "exit_model": exit_model,
            }
            dbg.set_params(
                entry_day=str(D.date()),
                vol_lookback=vol_lookback,
                min_close_up_pct=min_close_up_pct,
                min_gap_open_pct=min_gap_open_pct,
                min_volume_multiple=min_vol_multiple,
                atr_window=atr_window,
                atr_method=atr_method,
                horizon=horizon,
                sr_min_ratio=sr_min_ratio,
                sr_lookback=sr_lookback,
                exit_model=exit_model,
            )

            if filter_tickers_with_parquet is None:
                status.update(label="Scan cancelled ❌", state="error")
                st.error(CALLOUT_MESSAGE)
                log(STRUCTURED_ERROR_MESSAGE)
                dbg.event(
                    "price_filter_unavailable",
                    code=PriceFilterUnavailableError.code,
                    reason=getattr(filter_error, "reason", None),
                )
                debug_panel("scan")
                return

            members = sigscan._load_members(storage, cache_salt=storage.cache_salt())
            active_tickers = members_on_date(members, D)["ticker"].dropna().unique().tolist()

            if not active_tickers:
                status.update(label="Scan cancelled ❌", state="error")
                st.info("No eligible tickers for the selected day.")
                debug_panel("scan")
                return

            requested_count = len(active_tickers)
            try:
                filtered_tickers, missing_tickers = filter_tickers_with_parquet(
                    storage, active_tickers
                )
            except ConfigurationError as exc:
                error = handle_filter_exception(exc)
                status.update(label="Scan cancelled ❌", state="error")
                st.error(error.user_message)
                log(error.structured_message)
                dbg.event(
                    "price_filter_unavailable",
                    code=error.code,
                    reason=getattr(error, "reason", str(exc)),
                )
                debug_panel("scan")
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
                    f"Out of {requested_count} S&P tickers for {D.date()}, "
                    f"{available_count} have price files for scanning. "
                    f"{missing_count} are missing and were excluded automatically."
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
                        key="scan_missing_download",
                    )

            if not filtered_tickers:
                status.update(label="Scan cancelled ❌", state="error")
                st.error(
                    "No valid price data found for the provided tickers. "
                    "Please check your list or reload parquet files."
                )
                debug_panel("scan")
                return

            active_tickers = filtered_tickers
            start_date = D - pd.Timedelta(days=365)
            end_date = D + pd.Timedelta(days=horizon)
            log(f"Preloading {len(active_tickers)} tickers…")
            with dbg.step("preload_prices"):
                prices_df = load_prices_cached(
                    storage,
                    cache_salt=storage.cache_salt(),
                    tickers=active_tickers,
                    start=start_date,
                    end=end_date,
                )
            if not prices_df.empty:
                prices_df = prices_df.set_index("date").sort_index()
                prices_df = prices_df.loc[(prices_df.index >= start_date) & (prices_df.index <= end_date)]
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
                if prices_df.empty:
                    st.warning(
                        "No prices loaded from storage. Check bucket paths: lake/prices/{TICKER}.parquet"
                    )
                else:
                    loaded = sorted(
                        set(prices_df.get("Ticker", pd.Series(dtype=str)).tolist())
                    )
                    st.write(
                        f"Loaded series: {len(loaded)} (showing up to 10): {loaded[:10]}"
                    )
                    st.write(
                        {
                            "index_dtype": str(prices_df.index.dtype),
                            "tz": getattr(prices_df.index, "tz", None),
                            "min": prices_df.index.min(),
                            "max": prices_df.index.max(),
                        }
                    )
            if prices_df.empty:
                status.update(label="Scan failed ❌", state="error")
                st.error(
                    "No price data loaded from Supabase Storage. Expected 'lake/prices/{TICKER}.parquet'."
                )
                debug_panel("scan")
                return

            prices: Dict[str, pd.DataFrame] = {}
            for t in active_tickers:
                df_t = prices_df[prices_df.get("Ticker") == t]
                if not df_t.empty:
                    prices[t] = df_t.drop(columns=["Ticker"]).rename(columns=str.lower)

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

            def on_step(i: int, total: int, ticker: str):
                pct = max(0, min(100, int(i / max(1, total) * 100)))
                prog.progress(pct, text=f"{pct}%")
                log(f"{i}/{total} {ticker} ✓")

            try:
                with dbg.step("run_signal_scan"):
                    cand_df, out_df, fails, _dbg = scan_day(
                        storage, D, params, on_step=on_step
                    )
                dbg.event(
                    "scan_outcome",
                    rows=len(cand_df) if cand_df is not None else 0,
                    fails=len(fails) if fails is not None else 0,
                )
            finally:
                sigscan._load_prices = orig_load_prices
                sigscan._load_members = orig_load_members
            if cand_df is not None:
                cand_df = cand_df.reset_index(drop=True)
            if out_df is not None:
                out_df = out_df.reset_index(drop=True)

            st.session_state["cand_df"] = cand_df
            st.session_state["out_df"] = out_df
            st.session_state["fails"] = fails

            if save_outcomes and not out_df.empty:
                buf = io.BytesIO()
                out_df.to_parquet(buf, index=False)
                storage.write_bytes(
                    f"runs/{D.date().isoformat()}/outcomes.parquet",
                    buf.getvalue(),
                )

            status.update(label="Scan complete ✅", state="complete")
            st.toast(f"Scan done: {len(cand_df)} matches", icon="✅")
        except Exception as e:
            dbg.error("scan", e)
            log(f"ERROR: {e}")
            status.update(label="Scan failed ❌", state="error")
        finally:
            st.session_state["scan_running"] = False

    cand_df = st.session_state.get("cand_df")
    out_df = st.session_state.get("out_df")
    fails = st.session_state.get("fails")

    if cand_df is not None:
        summary: Dict[str, float] = {
            "candidates": len(cand_df),
            "fails": fails or 0,
            "hits": int(out_df["hit"].sum()) if out_df is not None and not out_df.empty else 0,
        }
        show_df("Summary", pd.DataFrame([summary]), "scan_summary")
        show_df("✅ Candidates (matches)", cand_df, "matches")
        show_df("🎯 Outcomes", out_df, "outcomes")

    debug_panel("scan")


def render_page() -> None:  # pragma: no cover - compatibility shim
    page()
