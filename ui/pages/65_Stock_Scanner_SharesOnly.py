from __future__ import annotations

import datetime as dt
import json
import platform
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
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


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class ScanDebugCollector:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []
        self.errors: list[dict[str, object]] = []
        self.rejections: Counter[str] = Counter()
        self.tickers: list[str] = []
        self.candidates = 0
        self.trades = 0
        self.tickers_scanned = 0

    def log_event(self, name: str, **data: object) -> None:
        payload = {k: v for k, v in data.items()}
        self.events.append({"t": _utcnow_iso(), "name": name, "data": payload})

    def record_rejection(
        self,
        *,
        ticker: str,
        date: str,
        reasons: list[str],
        details: dict | None = None,
    ) -> None:
        unique = sorted({str(reason) for reason in reasons if reason})
        if unique:
            self.rejections.update(unique)
        payload: dict[str, object] = {"ticker": ticker, "date": date, "reasons": unique}
        if details:
            payload["details"] = details
        self.log_event("reject", **payload)

    def record_candidate(
        self,
        *,
        ticker: str,
        date: str,
        entry_price: float,
        shares: int,
        exit_model: str,
        tp_price: float | None = None,
        sl_price: float | None = None,
    ) -> None:
        self.candidates += 1
        payload: dict[str, object] = {
            "ticker": ticker,
            "date": date,
            "entry_price": float(entry_price),
            "shares": int(shares),
            "exit_model": exit_model,
        }
        if tp_price is not None:
            payload["tp_price"] = float(tp_price)
        if sl_price is not None:
            payload["sl_price"] = float(sl_price)
        self.log_event("candidate", **payload)

    def record_trade_open(
        self,
        *,
        ticker: str,
        date: str,
        entry_price: float,
        tp: float,
        sl: float,
        shares: int,
    ) -> None:
        self.log_event(
            "trade_open",
            ticker=ticker,
            date=date,
            entry_price=float(entry_price),
            tp=float(tp),
            sl=float(sl),
            shares=int(shares),
        )

    def record_trade_exit(
        self,
        *,
        ticker: str,
        exit_date: str,
        exit_reason: str,
        exit_price: float,
        pnl: float,
    ) -> None:
        self.trades += 1
        self.log_event(
            "trade_exit",
            ticker=ticker,
            exit_date=exit_date,
            exit_reason=exit_reason,
            exit_price=float(exit_price),
            pnl=float(pnl),
        )

    def set_tickers(self, tickers: list[str]) -> None:
        self.tickers = list(tickers)

    def set_counts(self, *, tickers: int, candidates: int, trades: int) -> None:
        self.tickers_scanned = int(tickers)
        self.candidates = int(candidates)
        self.trades = int(trades)

    def record_error(self, where: str, exc: BaseException) -> None:
        self.errors.append(
            {
                "t": _utcnow_iso(),
                "where": where,
                "message": str(exc),
                "traceback": "".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                ),
            }
        )


def _render_debug_panel(
    meta: dict[str, object],
    params: dict[str, object],
    env: dict[str, object],
    debug: ScanDebugCollector,
    metrics: dict[str, int],
) -> None:
    with st.expander("🐞 Debug panel", expanded=False):
        st.caption("Everything below is for diagnostics. Safe to share (secrets redacted).")

        tickers_scanned = metrics.get("tickers") or debug.tickers_scanned or len(debug.tickers)
        candidates = metrics.get("candidates") or debug.candidates
        trades = metrics.get("trades") or debug.trades

        c1, c2, c3 = st.columns(3)
        c1.metric("Tickers scanned", int(tickers_scanned))
        c2.metric("Candidates", int(candidates))
        c3.metric("Trades taken", int(trades))

        st.subheader("Why candidates were dropped")
        rej_df = pd.DataFrame(
            [{"reason": reason, "count": count} for reason, count in debug.rejections.items()]
        )
        if not rej_df.empty:
            denom = max(1, int(tickers_scanned))
            rej_df["% of scans"] = (rej_df["count"] / denom) * 100.0
            st.dataframe(rej_df, use_container_width=True, height=220)
        else:
            st.write("No rejections recorded.")

        st.subheader("Event log")
        ev_df = pd.DataFrame(debug.events)
        if not ev_df.empty:
            ev_display = ev_df.copy()
            ev_display["data"] = ev_display["data"].apply(
                lambda val: json.dumps(val, default=str)
                if isinstance(val, (dict, list, tuple))
                else str(val)
            )
            st.dataframe(ev_display.tail(500), use_container_width=True, height=260)
            csv_bytes = ev_display.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download full debug CSV",
                csv_bytes,
                file_name="stock_scanner_debug.csv",
                mime="text/csv",
            )
        else:
            st.write("No events logged.")

        tabs = st.tabs(["meta", "params", "env", "errors"])
        with tabs[0]:
            st.json(meta)
        with tabs[1]:
            st.json(params)
        with tabs[2]:
            st.json(env)
        with tabs[3]:
            st.json(debug.errors)


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

    debug = ScanDebugCollector()
    meta = {
        "session_started": _utcnow_iso(),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": platform.platform(),
        "streamlit": st.__version__,
    }
    params_info = {
        "start": str(start_ts.date()),
        "end": str(end_ts.date()),
        "horizon": horizon,
        "sr_lookback": sr_lookback,
        "sr_min_ratio": sr_min_ratio,
        "yesterday_up_min": min_yup_pct,
        "open_gap_min": min_gap_pct,
        "vol_multiple_min": min_volume_multiple,
        "vol_lookback": volume_lookback,
        "exit_model": exit_model,
        "tp_atr_multiple": tp_atr_multiple,
        "sl_atr_multiple": sl_atr_multiple,
        "atr_window": atr_window,
        "atr_method": atr_method,
        "cap_per_trade": DEFAULT_CASH_CAP,
        "use_sp_filter": use_sp_filter,
    }
    env_info = {
        "storage_mode": getattr(storage, "mode", "unknown"),
        "bucket": getattr(storage, "bucket", None),
        "ticker_filter_source": "sp500_membership" if use_sp_filter else "none",
    }
    debug.log_event("scanner:params", params=params_info, env=env_info)

    metrics_info = {"tickers": 0, "candidates": 0, "trades": 0}

    status, prog_widget, log_fn = status_block("Running stock scan…", key_prefix="stock_scan")
    status.update(label="Loading data…", state="running")

    progress_cb = _progress_callback(prog_widget, log_fn)

    try:
        ledger, summary = run_scan(
            params,
            storage=storage,
            progress=progress_cb,
            debug=debug,
        )
    except Exception as exc:
        status.update(label="Scan failed ❌", state="error")
        st.error("Scan failed")
        st.exception(exc)
        debug.record_error("run_scan", exc)
        debug.log_event("scan:error", error=str(exc))
        _render_debug_panel(meta, params_info, env_info, debug, metrics_info)
        return

    status.update(label="Scan complete ✅", state="complete")

    metrics_info = {
        "tickers": summary.tickers_scanned,
        "candidates": summary.candidates,
        "trades": summary.trades,
    }

    _render_summary(summary)
    _render_ledger(ledger)
    _render_debug_panel(meta, params_info, env_info, debug, metrics_info)
