from __future__ import annotations

import datetime as dt
from typing import Callable

import pandas as pd
import streamlit as st

from data_lake.storage import Storage
from engine.stocks_only_scanner import (
    DEFAULT_CASH_CAP,
    ScanSummary,
    StocksOnlyScanParams,
    run_scan,
)
from ui.components.progress import status_block


def _default_dates() -> tuple[dt.date, dt.date]:
    today = pd.Timestamp.today().tz_localize(None).normalize()
    start = (today - pd.tseries.offsets.BDay(30)).date()
    return start, today.date()


def _format_currency(value: float) -> str:
    return f"${value:,.2f}"


def _progress_callback(progress_widget, log_fn: Callable[[str], None]):
    def _cb(current: int, total: int, ticker: str) -> None:
        if total <= 0:
            pct = 0.0
        else:
            pct = current / total
        try:
            progress_widget.progress(pct, text=f"{current}/{total} • {ticker}")
        except Exception:
            progress_widget.progress(pct)
        log_fn(f"[{current}/{total}] {ticker}")

    return _cb


def _render_summary(summary: ScanSummary) -> None:
    st.markdown(
        f"**Date span:** {summary.start.date()} → {summary.end.date()}  \\\n+**Tickers scanned:** {summary.tickers_scanned}  \\\n+**Candidates:** {summary.candidates}  \\\n+**Trades taken:** {summary.trades}"
    )

    metrics = st.columns(4)
    metrics[0].metric("Trades", summary.trades)
    win_display = (
        f"{summary.wins} wins" if summary.trades else "0"
    )
    metrics[1].metric("Win rate", f"{summary.win_rate:.1%}", win_display)
    metrics[2].metric("Total capital deployed", _format_currency(summary.total_capital))
    metrics[3].metric("Total P&L", _format_currency(summary.total_pnl))


def _render_ledger(df: pd.DataFrame) -> None:
    st.subheader("Trade ledger")
    if df.empty:
        st.info("No trades met the criteria.")
        return

    display_cols = [
        "entry_date",
        "exit_date",
        "ticker",
        "entry_price",
        "tp_price",
        "sl_price",
        "exit_price",
        "exit_reason",
        "shares",
        "cost",
        "proceeds",
        "pnl",
    ]
    working = df[display_cols].copy()
    working["entry_date"] = pd.to_datetime(working["entry_date"]).dt.date
    working["exit_date"] = pd.to_datetime(working["exit_date"]).dt.date
    numeric_cols = [
        "entry_price",
        "tp_price",
        "sl_price",
        "exit_price",
        "cost",
        "proceeds",
        "pnl",
    ]
    working[numeric_cols] = working[numeric_cols].astype(float).round(2)
    working["shares"] = working["shares"].astype(int)
    working["exit_reason"] = working["exit_reason"].astype(str)

    st.dataframe(working, use_container_width=True)
    csv_bytes = working.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV of the trade ledger",
        csv_bytes,
        file_name="stock_scanner_trades.csv",
        mime="text/csv",
        key="stock_scan_download",
    )


def page() -> None:
    st.header("📊 Stock Scanner (Shares Only)")
    st.caption("Backtest whole-share stock entries with ATR or support/resistance exits.")
    st.info(
        "Entries are capped at $1,000 per trade (whole shares only). Positions exit on TP, SL, or at the day-30 close if neither level is hit."
    )

    storage = Storage()
    st.caption(f"storage: {storage.info()} mode={storage.mode}")
    if getattr(storage, "force_supabase", False) and storage.mode == "local":
        st.error(
            "Supabase is required but not available. Check configuration or disable supabase.force."
        )
        return

    default_start, default_end = _default_dates()

    with st.form("stock_scanner_form"):
        col_dates = st.columns(2)
        start_date = col_dates[0].date_input("Start date", value=default_start)
        end_date = col_dates[1].date_input("End date", value=default_end)

        horizon = int(st.number_input("Horizon (business days)", min_value=1, value=30, step=1))

        sr_cols = st.columns(2)
        sr_lookback = int(
            sr_cols[0].number_input("SR lookback (days)", min_value=5, value=21, step=1)
        )
        sr_min_ratio = float(
            sr_cols[1].number_input("SR min ratio", min_value=0.5, value=2.0, step=0.1)
        )

        filter_cols = st.columns(3)
        min_yup_pct = float(
            filter_cols[0].number_input("Yesterday up % minimum", value=0.0, step=0.1)
        )
        min_gap_pct = float(
            filter_cols[1].number_input("Open gap % minimum", value=0.0, step=0.1)
        )
        min_volume_multiple = float(
            filter_cols[2].number_input("Volume multiple minimum", value=1.0, step=0.1)
        )

        volume_lookback = int(
            st.number_input("Volume lookback (days)", min_value=5, value=20, step=1)
        )

        exit_model = st.radio(
            "Exit model",
            options=("atr", "sr"),
            index=0,
            format_func=lambda v: "ATR multiples" if v == "atr" else "Support/Resistance",
        )

        if exit_model == "atr":
            atr_cols = st.columns(2)
            tp_atr_multiple = float(
                atr_cols[0].number_input("TP ATR multiple", min_value=0.1, value=1.0, step=0.1)
            )
            sl_atr_multiple = float(
                atr_cols[1].number_input("SL ATR multiple", min_value=0.1, value=1.0, step=0.1)
            )
        else:
            tp_atr_multiple = 1.0
            sl_atr_multiple = 1.0

        atr_cols_bottom = st.columns(2)
        atr_window = int(
            atr_cols_bottom[0].number_input("ATR window", min_value=1, value=14, step=1)
        )
        atr_method = atr_cols_bottom[1].selectbox(
            "ATR method",
            options=("wilder", "sma", "ema"),
            index=0,
            format_func=lambda opt: "Wilder" if opt == "wilder" else opt.upper(),
        )

        use_sp_filter = bool(st.checkbox("Use S&P membership filter", value=True))

        st.number_input(
            "$ per trade cap",
            value=int(DEFAULT_CASH_CAP),
            step=100,
            disabled=True,
            help="Whole shares only; trades skip when one share exceeds the cap.",
        )

        run_scan_btn = st.form_submit_button("Run scan", type="primary")

    if not run_scan_btn:
        return

    start_ts = pd.Timestamp(start_date).tz_localize(None)
    end_ts = pd.Timestamp(end_date).tz_localize(None)
    if end_ts < start_ts:
        st.error("End date must be on or after the start date.")
        return

    params: StocksOnlyScanParams = {
        "start": start_ts,
        "end": end_ts,
        "horizon_days": horizon,
        "sr_lookback": sr_lookback,
        "sr_min_ratio": sr_min_ratio,
        "min_yup_pct": min_yup_pct,
        "min_gap_pct": min_gap_pct,
        "min_volume_multiple": min_volume_multiple,
        "volume_lookback": volume_lookback,
        "exit_model": exit_model,
        "atr_window": atr_window,
        "atr_method": atr_method,
        "tp_atr_multiple": tp_atr_multiple,
        "sl_atr_multiple": sl_atr_multiple,
        "use_sp_filter": use_sp_filter,
        "cash_per_trade": DEFAULT_CASH_CAP,
    }

    status, prog_widget, log_fn = status_block("Running stock scan…", key_prefix="stock_scan")
    status.update(label="Loading data…", state="running")

    progress_cb = _progress_callback(prog_widget, log_fn)

    try:
        ledger, summary = run_scan(
            params,
            storage=storage,
            progress=progress_cb,
        )
    except Exception as exc:
        status.update(label="Scan failed ❌", state="error")
        st.error("Scan failed")
        st.exception(exc)
        return

    status.update(label="Scan complete ✅", state="complete")

    _render_summary(summary)
    _render_ledger(ledger)
