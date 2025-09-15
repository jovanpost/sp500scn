import io
import datetime as dt
from typing import Dict

import pandas as pd
import streamlit as st

import engine.signal_scan as sigscan
from engine.signal_scan import scan_day, ScanParams, members_on_date
from data_lake.storage import Storage, load_prices_cached
from ui.components.progress import status_block
from ui.components.debug import debug_panel, _get_dbg


def _render_df_with_copy(title: str, df: pd.DataFrame, key_prefix: str) -> None:
    st.subheader(title)
    if df is None or df.empty:
        st.info("No rows.")
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
    st.header("⚡ Yesterday Close+Volume → Buy Next Open")

    storage = Storage()
    dbg = _get_dbg("scan")
    dbg.set_env(storage_mode=getattr(storage, "mode", "unknown"), bucket=getattr(storage, "bucket", None))
    st.caption(f"storage: {storage.info()} mode={storage.mode}")
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
    atr_window = int(st.number_input("ATR window", min_value=1, value=21, step=1))
    horizon = int(st.number_input("Horizon (days)", min_value=1, value=30, step=1))
    sr_min_ratio = float(st.number_input("Min S:R ratio", value=2.0, step=0.1))
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
                "lookback_days": vol_lookback,
                "horizon_days": horizon,
                "sr_min_ratio": sr_min_ratio,
            }
            dbg.set_params(
                entry_day=str(D.date()),
                vol_lookback=vol_lookback,
                min_close_up_pct=min_close_up_pct,
                min_gap_open_pct=min_gap_open_pct,
                min_volume_multiple=min_vol_multiple,
                atr_window=atr_window,
                horizon=horizon,
                sr_min_ratio=sr_min_ratio,
            )

            members = sigscan._load_members(storage, cache_salt=storage.cache_salt())
            active_tickers = members_on_date(members, D)["ticker"].dropna().unique().tolist()
            start_date = D - pd.Timedelta(days=365)
            end_date = D + pd.Timedelta(days=horizon)
            log(f"Preloading {len(active_tickers)} tickers…")
            with dbg.step("preload_prices"):
                prices_df = load_prices_cached(
                    storage,
                    active_tickers,
                    start_date,
                    end_date,
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
        st.dataframe(pd.DataFrame([summary]))
        _render_df_with_copy("✅ Candidates (matches)", cand_df, "matches")
        _render_df_with_copy("🎯 Outcomes", out_df, "outcomes")

    debug_panel("scan")


def page() -> None:
    render_page()
