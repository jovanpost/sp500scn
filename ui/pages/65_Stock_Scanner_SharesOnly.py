from __future__ import annotations

import datetime as dt
import json
import platform
import re
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

import pandas as pd
import streamlit as st

from data_lake.storage import Storage
from engine.scan_shared.precursor_flags import DEFAULT_PARAMS as PRECURSOR_DEFAULTS
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


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _lab_flag_to_condition(flag: str, meta: dict[str, Any]) -> tuple[str, dict[str, float]] | None:
    flag = str(flag or "").strip()
    if not flag:
        return None

    def _extract(pattern: str) -> float | None:
        match = re.search(pattern, flag)
        if not match:
            return None
        return _safe_float(match.group(1))

    flag_lower = flag.lower()
    if flag_lower.startswith("ema_cross_"):
        parts = flag_lower.split("_")
        if len(parts) >= 5:
            fast = _safe_float(parts[2])
            slow = _safe_float(parts[3])
            if fast == 20 and slow == 50:
                return "ema_20_50_cross_up", {}
        return None

    if flag_lower.startswith("rsi_cross_"):
        level = _safe_float(re.sub(r"[^0-9]", "", flag_lower))
        if level in {50, 60}:
            return f"rsi_cross_{int(level)}", {}
        return None

    if flag_lower.startswith("atr_squeeze_le_"):
        threshold = meta.get("threshold")
        if threshold is None:
            threshold = _extract(r"atr_squeeze_le_([0-9]+(?:\.[0-9]+)?)p")
        value = _safe_float(threshold)
        if value is None:
            return None
        return "atr_squeeze_pct", {"max_percentile": float(value)}

    if flag_lower.startswith("bb_squeeze_le_"):
        threshold = meta.get("threshold")
        if threshold is None:
            threshold = _extract(r"bb_squeeze_le_([0-9]+(?:\.[0-9]+)?)p")
        value = _safe_float(threshold)
        if value is None:
            return None
        return "bb_squeeze_pct", {"max_percentile": float(value)}

    if flag_lower.startswith("nr7"):
        return "nr7", {}

    if flag_lower.startswith("gap_prior_ge_"):
        threshold = meta.get("threshold_pct")
        if threshold is None:
            threshold = _extract(r"gap_prior_ge_([0-9]+(?:\.[0-9]+)?)pct")
        value = _safe_float(threshold)
        if value is None:
            return None
        return "gap_up_ge_gpct_prev", {"min_gap_pct": float(value)}

    if flag_lower.startswith("vol_day1_ge_"):
        threshold = meta.get("threshold")
        if threshold is None:
            threshold = _extract(r"vol_day1_ge_([0-9]+(?:\.[0-9]+)?)x")
        value = _safe_float(threshold)
        if value is None:
            return None
        return "vol_mult_d1_ge_x", {"min_mult": float(value)}

    if flag_lower.startswith("vol_day2_ge_"):
        threshold = meta.get("threshold")
        if threshold is None:
            threshold = _extract(r"vol_day2_ge_([0-9]+(?:\.[0-9]+)?)x")
        value = _safe_float(threshold)
        if value is None:
            return None
        return "vol_mult_d2_ge_x", {"min_mult": float(value)}

    if flag_lower.startswith("sr_ratio_ge_"):
        threshold = meta.get("threshold")
        if threshold is None:
            threshold = _extract(r"sr_ratio_ge_([0-9]+(?:\.[0-9]+)?)")
        value = _safe_float(threshold)
        if value is not None and value >= 2.0:
            return "sr_ratio_ge_2", {}
        return None

    if flag_lower.startswith("new_high_") and flag_lower.endswith("_any"):
        match = re.search(r"new_high_(\d+)_any", flag_lower)
        if not match:
            return None
        window = int(match.group(1))
        if window in {20, 63}:
            return f"new_high_{window}", {}
        return None

    return None


def _parse_lab_preset(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], int | None, str | None]:
    raw_conditions = payload.get("conditions") or []
    parsed: list[dict[str, Any]] = []
    within_candidates: list[int] = []

    for item in raw_conditions:
        if not isinstance(item, dict):
            continue
        flag_name = item.get("flag") or item.get("id")
        mapped = _lab_flag_to_condition(flag_name, item)
        if not mapped:
            continue
        canonical_flag, params = mapped
        condition_payload: dict[str, Any] = {"flag": canonical_flag}
        condition_payload.update(params)
        parsed.append(condition_payload)
        within_val = item.get("within_days")
        if within_val is not None:
            try:
                within_candidates.append(int(within_val))
            except (TypeError, ValueError):
                continue

    preset_within = payload.get("within_days") or payload.get("lookback_days")
    if preset_within is not None:
        try:
            within_days = int(preset_within)
        except (TypeError, ValueError):
            within_days = None
    elif within_candidates:
        within_days = max(within_candidates)
    else:
        within_days = None

    logic = payload.get("logic")
    logic_normalized: str | None
    if isinstance(logic, str) and logic.strip():
        logic_normalized = logic.strip().upper()
    else:
        logic_normalized = None

    return parsed, within_days, logic_normalized


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
    def _cb(current, total, ticker: str) -> None:
        # Coerce to numeric defensively
        def _as_int(x, default=0):
            try:
                # Prefer Python ints; handle numpy/pandas scalars
                return int(x)
            except Exception:
                return default

        cur_i = _as_int(current, 0)
        tot_i = _as_int(total, 0)

        pct = 0.0
        if tot_i > 0:
            try:
                pct = float(cur_i) / float(tot_i)
            except Exception:
                pct = 0.0

        # Clamp to [0, 1]
        if not (0.0 <= pct <= 1.0):
            pct = max(0.0, min(1.0, pct))

        try:
            progress_widget.progress(pct, text=f"{cur_i}/{tot_i} • {ticker}")
        except Exception:
            # Older Streamlit versions might not accept text kwarg
            progress_widget.progress(pct)

        log_fn(f"[{cur_i}/{tot_i}] {ticker}")

    return _cb


def _render_summary(summary: ScanSummary) -> None:
    st.markdown(
        f"**Date span:** {summary.start.date()} → {summary.end.date()}  \\\n"
        f"**Tickers scanned:** {summary.tickers_scanned}  \\\n"
        f"**Candidates:** {summary.candidates}  \\\n"
        f"**Trades taken:** {summary.trades}"
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

    show_precursors = False
    precursor_cols = {
        "precursor_score",
        "precursor_flags_hit",
        "precursor_last_seen_days_ago",
    }
    if precursor_cols.issubset(df.columns):
        show_precursors = st.checkbox(
            "Show precursor score/flags", value=False, key="scanner_show_precursors"
        )
        if show_precursors:
            display_cols.extend(
                ["precursor_score", "precursor_flags_hit", "precursor_last_seen_days_ago"]
            )

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

    if show_precursors:
        working["precursor_flags_hit"] = working["precursor_flags_hit"].apply(
            lambda val: ", ".join(sorted(val)) if isinstance(val, Iterable) else ""
        )
        working["precursor_last_seen_days_ago"] = working[
            "precursor_last_seen_days_ago"
        ].apply(
            lambda val: ", ".join(
                f"{flag}:{int(days)}"
                for flag, days in sorted(val.items())
                if days is not None
            )
            if isinstance(val, dict)
            else ""
        )

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

    session = st.session_state
    session.setdefault("scanner_precursor_enabled", False)
    session.setdefault("scanner_precursor_within", PRECURSOR_DEFAULTS["lookback_days"])
    session.setdefault("scanner_precursor_logic", "ANY")
    session.setdefault("scanner_precursor_atr_threshold", PRECURSOR_DEFAULTS["atr_pct_threshold"])
    session.setdefault("scanner_precursor_bb_threshold", PRECURSOR_DEFAULTS["bb_pct_threshold"])
    session.setdefault("scanner_precursor_gap_threshold", PRECURSOR_DEFAULTS["gap_min_pct"])
    session.setdefault("scanner_precursor_vol_threshold", PRECURSOR_DEFAULTS["vol_min_mult"])

    default_start, default_end = _default_dates()

    precursors_enabled = False
    precursor_within_days = PRECURSOR_DEFAULTS["lookback_days"]
    precursor_logic_choice = "ANY"
    precursor_atr_threshold = float(PRECURSOR_DEFAULTS["atr_pct_threshold"])
    precursor_bb_threshold = float(PRECURSOR_DEFAULTS["bb_pct_threshold"])
    precursor_gap_threshold = float(PRECURSOR_DEFAULTS["gap_min_pct"])
    precursor_vol_threshold = float(PRECURSOR_DEFAULTS["vol_min_mult"])
    ema_selected = False
    rsi50_selected = False
    rsi60_selected = False
    atr_selected = False
    bb_selected = False
    nr7_selected = False
    gap_selected = False
    vol_d1_selected = False
    vol_d2_selected = False
    sr_selected = False
    new_high_20_selected = False
    new_high_63_selected = False
    preset_conditions: list[dict[str, Any]] | None = None
    preset_within_override: int | None = None
    preset_logic_override: str | None = None

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

        with st.expander("Spike Precursor Filters (optional)", expanded=False):
            precursors_enabled = st.checkbox(
                "Enable Spike Precursor filters",
                key="scanner_precursor_enabled",
                value=bool(session.get("scanner_precursor_enabled", False)),
            )
            precursor_within_days = int(
                st.slider(
                    "Look back within N business days",
                    min_value=1,
                    max_value=60,
                    value=int(session.get("scanner_precursor_within", PRECURSOR_DEFAULTS["lookback_days"]))
                    if session.get("scanner_precursor_within")
                    else PRECURSOR_DEFAULTS["lookback_days"],
                    key="scanner_precursor_within",
                    disabled=not precursors_enabled,
                )
            )
            logic_options = ("ANY", "ALL")
            logic_default = str(session.get("scanner_precursor_logic", "ANY"))
            if logic_default not in logic_options:
                logic_default = "ANY"
            logic_index = logic_options.index(logic_default)
            precursor_logic_choice = st.radio(
                "Logic mode",
                options=logic_options,
                index=logic_index,
                key="scanner_precursor_logic",
                disabled=not precursors_enabled,
                horizontal=True,
            )

            st.markdown("**Trend & Momentum**")
            trend_cols = st.columns(3)
            ema_selected = trend_cols[0].checkbox(
                "EMA 20/50 cross up",
                key="scanner_precursor_ema",
                disabled=not precursors_enabled,
            )
            rsi50_selected = trend_cols[1].checkbox(
                "RSI cross ≥ 50",
                key="scanner_precursor_rsi50",
                disabled=not precursors_enabled,
            )
            rsi60_selected = trend_cols[2].checkbox(
                "RSI cross ≥ 60",
                key="scanner_precursor_rsi60",
                disabled=not precursors_enabled,
            )

            st.markdown("**Volatility squeezes**")
            squeeze_cols = st.columns(2)
            with squeeze_cols[0]:
                atr_selected = st.checkbox(
                    "ATR percentile ≤",
                    key="scanner_precursor_atr",
                    disabled=not precursors_enabled,
                )
                precursor_atr_threshold = float(
                    st.number_input(
                        "ATR percentile",
                        min_value=1.0,
                        max_value=100.0,
                        step=1.0,
                        key="scanner_precursor_atr_threshold",
                        value=float(
                            session.get(
                                "scanner_precursor_atr_threshold",
                                PRECURSOR_DEFAULTS["atr_pct_threshold"],
                            )
                        ),
                        disabled=not (precursors_enabled and atr_selected),
                    )
                )
            with squeeze_cols[1]:
                bb_selected = st.checkbox(
                    "BB width percentile ≤",
                    key="scanner_precursor_bb",
                    disabled=not precursors_enabled,
                )
                precursor_bb_threshold = float(
                    st.number_input(
                        "BB percentile",
                        min_value=1.0,
                        max_value=100.0,
                        step=1.0,
                        key="scanner_precursor_bb_threshold",
                        value=float(
                            session.get(
                                "scanner_precursor_bb_threshold",
                                PRECURSOR_DEFAULTS["bb_pct_threshold"],
                            )
                        ),
                        disabled=not (precursors_enabled and bb_selected),
                    )
                )

            st.markdown("**Range & breakouts**")
            range_cols = st.columns(3)
            nr7_selected = range_cols[0].checkbox(
                "NR7",
                key="scanner_precursor_nr7",
                disabled=not precursors_enabled,
            )
            new_high_20_selected = range_cols[1].checkbox(
                "New high 20",
                key="scanner_precursor_high20",
                disabled=not precursors_enabled,
            )
            new_high_63_selected = range_cols[2].checkbox(
                "New high 63",
                key="scanner_precursor_high63",
                disabled=not precursors_enabled,
            )

            sr_selected = st.checkbox(
                "Support/resistance ratio ≥ 2",
                key="scanner_precursor_sr",
                disabled=not precursors_enabled,
            )

            st.markdown("**Gaps & volume**")
            gv_cols = st.columns(2)
            with gv_cols[0]:
                gap_selected = st.checkbox(
                    "Prior-day gap ≥ %",
                    key="scanner_precursor_gap",
                    disabled=not precursors_enabled,
                )
                precursor_gap_threshold = float(
                    st.number_input(
                        "Gap percent",
                        min_value=0.0,
                        step=0.5,
                        key="scanner_precursor_gap_threshold",
                        value=float(
                            session.get(
                                "scanner_precursor_gap_threshold",
                                PRECURSOR_DEFAULTS["gap_min_pct"],
                            )
                        ),
                        disabled=not (precursors_enabled and gap_selected),
                    )
                )
            with gv_cols[1]:
                precursor_vol_threshold = float(
                    st.number_input(
                        "Volume multiple",
                        min_value=0.1,
                        step=0.1,
                        key="scanner_precursor_vol_threshold",
                        value=float(
                            session.get(
                                "scanner_precursor_vol_threshold",
                                PRECURSOR_DEFAULTS["vol_min_mult"],
                            )
                        ),
                        disabled=not precursors_enabled,
                    )
                )
                vol_d1_selected = st.checkbox(
                    "Day -1 volume ≥ threshold",
                    key="scanner_precursor_vol_d1",
                    disabled=not precursors_enabled,
                )
                vol_d2_selected = st.checkbox(
                    "Day -2 volume ≥ threshold",
                    key="scanner_precursor_vol_d2",
                    disabled=not precursors_enabled,
                )

            preset_upload = st.file_uploader(
                "Import from Spike Lab preset",
                type=["json"],
                key="scanner_precursor_preset",
            )
            if preset_upload is not None:
                try:
                    preset_data = json.load(preset_upload)
                    preset_upload.seek(0)
                    parsed_conditions, preset_within_override, preset_logic_override = _parse_lab_preset(
                        preset_data
                    )
                    if parsed_conditions:
                        st.caption(
                            f"Preset loaded: {len(parsed_conditions)} supported precursor conditions."
                        )
                        preset_conditions = parsed_conditions
                    else:
                        st.warning("Preset contained no supported precursor flags.")
                        preset_conditions = []
                except Exception as exc:
                    st.error(f"Could not parse preset: {exc}")
                    preset_conditions = []

        run_scan_btn = st.form_submit_button("Run scan", type="primary")

    if not run_scan_btn:
        return

    start_ts = pd.Timestamp(start_date).tz_localize(None)
    end_ts = pd.Timestamp(end_date).tz_localize(None)
    if end_ts < start_ts:
        st.error("End date must be on or after the start date.")
        return

    manual_conditions: list[dict[str, Any]] = []
    if precursors_enabled:
        if ema_selected:
            manual_conditions.append({"flag": "ema_20_50_cross_up"})
        if rsi50_selected:
            manual_conditions.append({"flag": "rsi_cross_50"})
        if rsi60_selected:
            manual_conditions.append({"flag": "rsi_cross_60"})
        if atr_selected:
            manual_conditions.append(
                {"flag": "atr_squeeze_pct", "max_percentile": float(precursor_atr_threshold)}
            )
        if bb_selected:
            manual_conditions.append(
                {"flag": "bb_squeeze_pct", "max_percentile": float(precursor_bb_threshold)}
            )
        if nr7_selected:
            manual_conditions.append({"flag": "nr7"})
        if gap_selected:
            manual_conditions.append(
                {"flag": "gap_up_ge_gpct_prev", "min_gap_pct": float(precursor_gap_threshold)}
            )
        if vol_d1_selected:
            manual_conditions.append(
                {"flag": "vol_mult_d1_ge_x", "min_mult": float(precursor_vol_threshold)}
            )
        if vol_d2_selected:
            manual_conditions.append(
                {"flag": "vol_mult_d2_ge_x", "min_mult": float(precursor_vol_threshold)}
            )
        if sr_selected:
            manual_conditions.append({"flag": "sr_ratio_ge_2"})
        if new_high_20_selected:
            manual_conditions.append({"flag": "new_high_20"})
        if new_high_63_selected:
            manual_conditions.append({"flag": "new_high_63"})

    use_preset = bool(preset_conditions)
    selected_conditions = manual_conditions
    if use_preset:
        selected_conditions = list(preset_conditions)
        precursors_enabled = True
        if preset_within_override is not None:
            precursor_within_days = int(preset_within_override)
        if preset_logic_override:
            precursor_logic_choice = str(preset_logic_override)

    precursor_logic_choice = str(precursor_logic_choice or "ANY").upper()
    if precursor_logic_choice not in {"ANY", "ALL"}:
        precursor_logic_choice = "ANY"

    session["scanner_precursor_enabled"] = bool(selected_conditions)
    session["scanner_precursor_within"] = precursor_within_days
    session["scanner_precursor_logic"] = precursor_logic_choice
    session["scanner_precursor_atr_threshold"] = precursor_atr_threshold
    session["scanner_precursor_bb_threshold"] = precursor_bb_threshold
    session["scanner_precursor_gap_threshold"] = precursor_gap_threshold
    session["scanner_precursor_vol_threshold"] = precursor_vol_threshold

    precursors_payload: dict[str, Any] | None = None
    if selected_conditions:
        precursors_payload = {
            "enabled": True,
            "within_days": int(precursor_within_days),
            "logic": precursor_logic_choice,
            "conditions": selected_conditions,
            "atr_pct_threshold": float(precursor_atr_threshold),
            "bb_pct_threshold": float(precursor_bb_threshold),
            "gap_min_pct": float(precursor_gap_threshold),
            "vol_min_mult": float(precursor_vol_threshold),
            "lookback_days": int(precursor_within_days),
        }

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

    if precursors_payload:
        params["precursors"] = precursors_payload

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
        "precursors_enabled": bool(precursors_payload),
        "precursors_logic": precursors_payload["logic"] if precursors_payload else "OFF",
        "precursors_within": precursors_payload["within_days"] if precursors_payload else 0,
        "precursors_conditions": [
            cond.get("flag") for cond in (precursors_payload["conditions"] if precursors_payload else [])
        ],
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
