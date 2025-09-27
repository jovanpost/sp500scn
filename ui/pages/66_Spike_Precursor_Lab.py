from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import pandas as pd
import streamlit as st
from pandas.tseries.offsets import BDay

from backtest.precursor_eval import (
    build_diagnostic_table,
    evaluate_precursors_naive,
    evaluate_precursors_scanner_aligned,
)
from data_lake.membership import load_membership
from data_lake.storage import Storage, load_prices_cached
from engine.precursor_rules import build_conditions_from_session
from engine.scan_runner import StocksOnlyScanParams
from engine.scan_shared.indicators import IndicatorConfig, ensure_datetime_index
from engine.scan_shared.precursor_flags import (
    DEFAULT_PARAMS as PRECURSOR_DEFAULTS,
    build_precursor_flags,
)
from engine.stocks_only_scanner import DEFAULT_CASH_CAP
from ui.components.precursor_controls import render_precursor_section
from utils.io_export import export_diagnostics, export_trades


@dataclass(frozen=True)
class SpikeDefinition:
    mode: str
    pct_threshold: float | None
    atr_multiple: float | None
    volume_filter_enabled: bool
    volume_threshold: float


def _utcnow_iso() -> str:
    return pd.Timestamp.utcnow().isoformat()


def _default_dates() -> tuple[pd.Timestamp, pd.Timestamp]:
    end = pd.Timestamp.today().tz_localize(None).normalize()
    start = end - pd.DateOffset(months=6)
    return start, end


def _normalize_timestamp(ts: pd.Timestamp | str | None) -> pd.Timestamp | None:
    if ts is None:
        return None
    if isinstance(ts, pd.Timestamp):
        return ts.tz_localize(None) if ts.tzinfo else ts
    return pd.Timestamp(ts).tz_localize(None)


def _filter_membership_by_dates(
    membership: pd.DataFrame, start: pd.Timestamp | None, end: pd.Timestamp | None
) -> pd.DataFrame:
    if membership is None or membership.empty:
        return membership
    if start is None and end is None:
        return membership

    starts = pd.to_datetime(membership.get("start_date"), errors="coerce")
    ends = pd.to_datetime(membership.get("end_date"), errors="coerce")

    mask = pd.Series(True, index=membership.index)
    if end is not None:
        mask &= starts.isna() | (starts <= end)
    if start is not None:
        mask &= ends.isna() | (ends >= start)

    return membership.loc[mask]


def _resolve_universe(
    storage: Storage,
    *,
    use_sp_filter: bool,
    extra_tickers: Sequence[str] | None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> list[str]:
    tickers: set[str] = set()

    try:
        membership = load_membership(storage, cache_salt=storage.cache_salt())
    except Exception:
        membership = pd.DataFrame()

    start = _normalize_timestamp(start)
    end = _normalize_timestamp(end)

    active_membership = _filter_membership_by_dates(membership, start, end)

    if use_sp_filter and active_membership is not None and not active_membership.empty:
        tickers.update(
            active_membership["ticker"]
            .astype(str)
            .str.upper()
            .str.strip()
            .dropna()
            .tolist()
        )

    if extra_tickers:
        tickers.update(str(t).upper().strip() for t in extra_tickers if t)

    if not tickers and not use_sp_filter:
        fallback = active_membership if active_membership is not None else membership
        if fallback is not None and not fallback.empty:
            tickers.update(
                fallback["ticker"].astype(str).str.upper().str.strip().dropna().tolist()
            )

    return sorted(tickers)


def _prepare_precursor_params(raw: dict[str, float | int]) -> dict[str, float]:
    return {
        "atr_pct_threshold": float(
            raw.get("atr_pct_threshold", PRECURSOR_DEFAULTS["atr_pct_threshold"])
        ),
        "bb_pct_threshold": float(
            raw.get("bb_pct_threshold", PRECURSOR_DEFAULTS["bb_pct_threshold"])
        ),
        "gap_min_pct": float(raw.get("gap_min_pct", PRECURSOR_DEFAULTS["gap_min_pct"])),
        "vol_min_mult": float(raw.get("vol_min_mult", PRECURSOR_DEFAULTS["vol_min_mult"])),
        "lookback_days": int(raw.get("lookback_days", PRECURSOR_DEFAULTS["lookback_days"])),
    }


def _find_spike(
    panel: pd.DataFrame,
    event_date: pd.Timestamp,
    *,
    within_days: int,
    definition: SpikeDefinition,
) -> pd.Timestamp | pd.NaT:
    if event_date not in panel.index:
        return pd.NaT

    window_end = event_date + BDay(max(1, within_days))
    future = panel.loc[(panel.index > event_date) & (panel.index <= window_end)].copy()
    if future.empty:
        return pd.NaT

    event_close = float(panel.loc[event_date, "close"])
    if not math.isfinite(event_close) or event_close <= 0:
        return pd.NaT

    if definition.mode == "pct":
        target_pct = float(definition.pct_threshold or 0.0)
        pct_change = (future["close"] / event_close - 1.0) * 100.0
        mask = pct_change >= target_pct
    else:
        atr_multiple = float(definition.atr_multiple or 0.0)
        atr_value = float(panel.loc[event_date, "atr_value"]) if "atr_value" in panel else float("nan")
        if not math.isfinite(atr_value) or atr_value <= 0:
            return pd.NaT
        diff = future["close"] - event_close
        mask = diff >= (atr_multiple * atr_value)

    if definition.volume_filter_enabled:
        vol_col = future.get("vol_mult_raw")
        if vol_col is not None:
            mask &= vol_col >= float(definition.volume_threshold)

    hits = future.loc[mask]
    if hits.empty:
        return pd.NaT
    return pd.to_datetime(hits.index[0])


def _generate_precursor_events(
    storage: Storage,
    *,
    tickers: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    precursors: dict[str, object] | None,
    definition: SpikeDefinition,
) -> pd.DataFrame:
    columns = ["ticker", "signal_date", "spike_date", "flags_fired"]
    if not tickers or precursors is None:
        return pd.DataFrame(columns=columns)

    within_days = int(precursors.get("within_days", PRECURSOR_DEFAULTS["lookback_days"]))
    base_params = _prepare_precursor_params(precursors)

    config = IndicatorConfig()
    pad_days = max(config.max_lookback, within_days + 5)
    fetch_start = (start - BDay(int(pad_days))).tz_localize(None)
    fetch_end = (end + BDay(int(within_days + 5))).tz_localize(None)

    try:
        prices = load_prices_cached(
            storage,
            tickers=list(tickers),
            start=fetch_start,
            end=fetch_end,
        )
    except Exception:
        return pd.DataFrame(columns=columns)

    if prices is None or prices.empty:
        return pd.DataFrame(columns=columns)

    events: list[dict[str, object]] = []
    flag_conditions = precursors.get("conditions", []) or []
    logic = str(precursors.get("logic", "ANY") or "ANY").upper()

    for ticker, frame in prices.groupby("Ticker"):
        ticker = str(ticker).upper()
        panel = ensure_datetime_index(frame)
        if panel.empty:
            continue
        flags_panel, _ = build_precursor_flags(panel, params=base_params)
        if flags_panel.empty:
            continue
        flags_panel = flags_panel.sort_index()
        window = flags_panel.loc[(flags_panel.index >= start) & (flags_panel.index <= end)]
        if window.empty:
            continue

        for event_date, row in window.iterrows():
            hit_flags: list[str] = []
            for condition in flag_conditions:
                flag = str(condition.get("flag", ""))
                if not flag:
                    continue
                if bool(row.get(flag, False)):
                    hit_flags.append(flag)

            if logic == "ALL":
                if len(hit_flags) < len(flag_conditions):
                    continue
            else:
                if not hit_flags:
                    continue

            spike_date = _find_spike(
                flags_panel,
                event_date,
                within_days=within_days,
                definition=definition,
            )
            events.append(
                {
                    "ticker": ticker,
                    "signal_date": pd.to_datetime(event_date),
                    "spike_date": pd.to_datetime(spike_date) if spike_date is not pd.NaT else pd.NaT,
                    "flags_fired": sorted(set(hit_flags)),
                }
            )

    if not events:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(events, columns=columns)


def page() -> None:
    st.header("🚀 Spike Precursor Lab")
    st.caption(
        "Evaluate precursor presets against real trades. Compare naive spike counts with scanner-aligned results."
    )

    storage = Storage()
    st.caption(f"storage: {storage.info()} mode={storage.mode}")

    default_start, default_end = _default_dates()

    render_precursor_section(st.session_state)

    with st.form("precursor_lab_form"):
        date_cols = st.columns(2)
        start_date = date_cols[0].date_input("Start date", value=default_start.date())
        end_date = date_cols[1].date_input("End date", value=default_end.date())

        use_sp_filter = st.checkbox(
            "Use S&P 500 membership filter",
            value=True,
            help="When enabled, restrict analysis to tickers present in the S&P 500 universe.",
        )
        extra_ticker_text = st.text_input(
            "Additional tickers (comma separated)",
            help="Optional: evaluate extra tickers alongside the membership universe.",
        )
        extra_tickers = [
            token.strip()
            for token in extra_ticker_text.split(",")
            if token.strip()
        ]

        st.subheader("Scanner alignment settings")
        horizon = int(st.number_input("Horizon (business days)", min_value=1, value=30, step=1))
        sr_cols = st.columns(2)
        sr_lookback = int(
            sr_cols[0].number_input("SR lookback (days)", min_value=5, value=21, step=1)
        )
        sr_min_ratio = float(
            sr_cols[1].number_input("Minimum SR ratio", min_value=0.5, value=2.0, step=0.1)
        )

        gap_cols = st.columns(3)
        min_yup_pct = float(
            gap_cols[0].number_input("Yesterday % up minimum", value=0.0, step=0.1)
        )
        min_gap_pct = float(
            gap_cols[1].number_input("Open gap minimum %", value=0.0, step=0.1)
        )
        min_volume_multiple = float(
            gap_cols[2].number_input("Volume multiple minimum", value=1.0, step=0.1)
        )

        volume_lookback = int(
            st.number_input("Volume lookback (days)", min_value=5, value=63, step=1)
        )

        exit_model = st.selectbox(
            "Exit model",
            options=("atr", "sr"),
            index=0,
            format_func=lambda opt: "ATR targets" if opt == "atr" else "Support/Resistance",
        )
        atr_cols = st.columns(3)
        atr_window = int(atr_cols[0].number_input("ATR window", min_value=2, value=14, step=1))
        atr_method = atr_cols[1].selectbox(
            "ATR method",
            options=("wilder", "sma", "ema"),
            index=0,
            format_func=lambda opt: "Wilder" if opt == "wilder" else opt.upper(),
        )
        tp_atr_multiple = float(atr_cols[2].number_input("TP ATR multiple", value=1.0, step=0.1))
        sl_atr_multiple = float(
            st.number_input("SL ATR multiple", min_value=0.1, value=1.0, step=0.1)
        )

        st.subheader("Spike definition")
        spike_mode = st.radio(
            "Spike mode",
            options=("pct", "atr"),
            index=0,
            format_func=lambda opt: "Percent spike" if opt == "pct" else "ATR multiple",
        )
        pct_threshold = None
        atr_multiple = None
        if spike_mode == "pct":
            pct_threshold = float(
                st.number_input(
                    "Close change ≥ X% vs prior close",
                    min_value=0.1,
                    value=8.0,
                    step=0.1,
                )
            )
        else:
            atr_multiple = float(
                st.number_input("Close change ≥ K × ATR", min_value=0.1, value=3.0, step=0.1)
            )

        volume_filter_enabled = st.checkbox(
            "Require spike-day volume multiple",
            value=False,
            help="When enabled, spike days must exceed the selected volume multiple threshold.",
        )
        volume_filter_threshold = float(
            st.number_input("Volume multiple threshold", min_value=0.5, value=1.5, step=0.1)
        )

        run_btn = st.form_submit_button("Run diagnostics", type="primary")

    if not run_btn:
        return

    start_ts = pd.Timestamp(start_date).tz_localize(None)
    end_ts = pd.Timestamp(end_date).tz_localize(None)
    if end_ts < start_ts:
        st.error("End date must be on or after the start date.")
        return

    precursors_payload = build_conditions_from_session(st.session_state)
    tickers = _resolve_universe(
        storage,
        use_sp_filter=use_sp_filter,
        extra_tickers=extra_tickers,
        start=start_ts,
        end=end_ts,
    )

    spike_definition = SpikeDefinition(
        mode=spike_mode,
        pct_threshold=pct_threshold,
        atr_multiple=atr_multiple,
        volume_filter_enabled=volume_filter_enabled,
        volume_threshold=volume_filter_threshold,
    )

    events_df = _generate_precursor_events(
        storage,
        tickers=tickers,
        start=start_ts,
        end=end_ts,
        precursors=precursors_payload,
        definition=spike_definition,
    )

    within_days = int(
        precursors_payload.get("within_days", PRECURSOR_DEFAULTS["lookback_days"])
        if precursors_payload
        else PRECURSOR_DEFAULTS["lookback_days"]
    )
    logic = str(
        precursors_payload.get("logic", "ANY") if precursors_payload else "ANY"
    ).upper()

    naive_result = evaluate_precursors_naive(events_df, within_days=within_days, logic=logic)

    scan_params = StocksOnlyScanParams(
        start=start_ts,
        end=end_ts,
        horizon_days=horizon,
        sr_lookback=sr_lookback,
        sr_min_ratio=sr_min_ratio,
        min_yup_pct=min_yup_pct,
        min_gap_pct=min_gap_pct,
        min_volume_multiple=min_volume_multiple,
        volume_lookback=volume_lookback,
        exit_model=exit_model,
        atr_window=atr_window,
        atr_method=atr_method,
        tp_atr_multiple=tp_atr_multiple,
        sl_atr_multiple=sl_atr_multiple,
        use_sp_filter=use_sp_filter,
        cash_per_trade=DEFAULT_CASH_CAP,
        precursors=precursors_payload,
    )

    aligned_result = evaluate_precursors_scanner_aligned(events_df, scan_params)

    summary = aligned_result.metrics.get("summary", {})
    trades_df = aligned_result.metrics.get("trades", pd.DataFrame())

    diag_table = build_diagnostic_table(naive_result, aligned_result)

    st.subheader("Diagnostic comparison")
    st.dataframe(diag_table, use_container_width=True)

    naive_precision = float(naive_result.metrics.get("precision", 0.0))
    win_rate = float(aligned_result.metrics.get("win_rate", 0.0))
    if naive_precision - win_rate >= 0.30:
        st.warning(
            "Precursor-only stats overstate success. Use scanner-aligned results for trading decisions."
        )

    metrics_cols = st.columns(3)
    metrics_cols[0].metric("Signals evaluated", int(naive_result.metrics.get("total", 0)))
    metrics_cols[1].metric("Scanner trades", int(summary.get("trades", 0)))
    metrics_cols[2].metric("Win rate", f"{win_rate:.1%}")

    if not events_df.empty:
        st.subheader("Sample precursor signals")
        st.dataframe(events_df.head(200), use_container_width=True)
    else:
        st.info("No precursor signals matched the current configuration.")

    if not trades_df.empty:
        st.subheader("Scanner-aligned trades")
        st.dataframe(trades_df, use_container_width=True)
    else:
        st.info("Scanner produced no trades for this configuration.")

    diag_base = f"precursor_lab_{start_ts.strftime('%Y%m%d')}_{end_ts.strftime('%Y%m%d')}"
    diag_path = export_diagnostics(aligned_result.diagnostics, diag_base)
    trades_path = export_trades(trades_df, diag_base)
    st.caption(f"Saved diagnostics CSV to {diag_path}")
    st.caption(f"Saved trades CSV to {trades_path}")

    st.caption(f"Session recorded at {_utcnow_iso()}")


if __name__ == "__main__":  # pragma: no cover
    page()
