from __future__ import annotations

import json
import platform
import traceback
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import streamlit as st

from data_lake.membership import load_membership
from data_lake.storage import Storage

from engine.spike_precursor import analyze_spike_precursors
from ui.components.progress import status_block


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class SpikeDebugCollector:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.errors: list[dict[str, Any]] = []
        self.metrics: dict[str, Any] = {}

    def log_event(self, name: str, **data: Any) -> None:
        self.events.append({"t": _utcnow_iso(), "name": name, "data": data})

    def add_error(self, where: str, exc: BaseException) -> None:
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

    def set_metrics(self, **metrics: Any) -> None:
        self.metrics.update(metrics)


def _render_debug_panel(
    meta: dict[str, Any],
    params: dict[str, Any],
    env: dict[str, Any],
    debug: SpikeDebugCollector,
) -> None:
    with st.expander("🐞 Debug panel", expanded=False):
        st.caption("Everything below is for diagnostics. Safe to share (secrets redacted).")

        c1, c2, c3 = st.columns(3)
        c1.metric("Tickers analyzed", int(debug.metrics.get("tickers", 0)))
        c2.metric("Spikes found", int(debug.metrics.get("spikes", 0)))
        c3.metric("Flags tracked", int(debug.metrics.get("flags", 0)))

        st.subheader("Event log")
        events_df = pd.DataFrame(debug.events)
        if not events_df.empty:
            events_disp = events_df.copy()
            events_disp["data"] = events_disp["data"].apply(
                lambda val: json.dumps(val, default=str)
                if isinstance(val, (dict, list))
                else str(val)
            )
            st.dataframe(events_disp.tail(500), use_container_width=True, height=260)
            st.download_button(
                "Download events CSV",
                events_disp.to_csv(index=False).encode("utf-8"),
                file_name="spike_lab_events.csv",
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


def _default_dates() -> tuple[pd.Timestamp, pd.Timestamp]:
    end = pd.Timestamp.today().tz_localize(None).normalize()
    start = end - pd.DateOffset(years=1)
    return start, end


def _load_universe(
    storage: Storage, start: pd.Timestamp, end: pd.Timestamp
) -> list[str]:
    try:
        members = load_membership(storage, cache_salt=storage.cache_salt())
    except Exception:
        return []

    if members is None or members.empty:
        return []

    members = members.copy()
    members["start_date"] = pd.to_datetime(
        members["start_date"], errors="coerce"
    ).dt.tz_localize(None)
    members["end_date"] = pd.to_datetime(
        members.get("end_date"), errors="coerce"
    ).dt.tz_localize(None)

    mask = (members["start_date"] <= end) & (
        members["end_date"].isna() | (start <= members["end_date"])
    )
    filtered = members.loc[mask]
    if filtered.empty:
        return []
    tickers = (
        filtered["ticker"].astype(str).str.upper().str.strip().dropna().unique().tolist()
    )
    return sorted(tickers)


def _clean_levels(text: str) -> list[float]:
    parts = [p.strip() for p in (text or "").split(",") if p.strip()]
    levels: list[float] = []
    for part in parts:
        try:
            levels.append(float(part))
        except Exception:
            continue
    return levels


def page() -> None:
    st.header("🚀 Spike Precursor Lab")
    st.caption(
        "Reverse-engineer what tends to precede big spikes, then export a rule preset to try in the shares-only scanner."
    )

    storage = Storage()
    st.caption(f"storage: {storage.info()} mode={storage.mode}")

    default_start, default_end = _default_dates()

    debug = SpikeDebugCollector()

    debug_meta: dict[str, Any] = {}
    debug_env: dict[str, Any] = {"storage_mode": storage.mode, "storage_info": storage.info()}

    with st.form("spike_precursor_form"):
        date_cols = st.columns(2)
        start_date = date_cols[0].date_input("Start date", value=default_start.date())
        end_date = date_cols[1].date_input("End date", value=default_end.date())

        use_sp_filter = bool(
            st.checkbox("S&P 500 membership filter", value=True, help="Limit to tickers with membership overlaps the date range if data is available.")
        )

        spike_mode = st.radio(
            "Spike definition",
            options=("pct", "atr"),
            index=0,
            format_func=lambda opt: "Percent spike" if opt == "pct" else "ATR multiple",
        )

        pct_threshold = None
        atr_multiple = None
        atr_window = 14
        atr_method = "wilder"

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
            atr_cols = st.columns(3)
            atr_multiple = float(
                atr_cols[0].number_input(
                    "Close change ≥ K × ATR",
                    min_value=0.1,
                    value=3.0,
                    step=0.1,
                )
            )
            atr_window = int(
                atr_cols[1].number_input("ATR window", min_value=2, value=14, step=1)
            )
            atr_method = atr_cols[2].selectbox(
                "ATR method",
                options=("wilder", "sma", "ema"),
                index=0,
                format_func=lambda opt: opt.upper() if opt != "wilder" else "Wilder",
            )

        st.subheader("Optional spike-day filters")
        volume_filter_enabled = st.checkbox(
            "Require volume spike vs trailing window", value=False
        )
        volume_filter_threshold = 1.5
        volume_filter_lookback = 63
        if volume_filter_enabled:
            vf_cols = st.columns(2)
            volume_filter_threshold = float(
                vf_cols[0].number_input(
                    "Volume multiple threshold", min_value=0.5, value=1.5, step=0.1
                )
            )
            volume_filter_lookback = int(
                vf_cols[1].number_input(
                    "Volume lookback (days)", min_value=5, value=63, step=1
                )
            )

        lookback_days = int(
            st.number_input("Precursor window (business days)", min_value=1, value=20, step=1)
        )

        st.subheader("Indicator panel")

        trend_enabled = st.checkbox("Trend: EMA fast/slow cross", value=True)
        trend_pair_text = "20,50"
        if trend_enabled:
            trend_pair_text = st.text_input(
                "EMA periods (fast,slow)",
                value="20,50",
                help="Comma separated values like 20,50",
            )
        trend_fast, trend_slow = 20, 50
        if trend_pair_text:
            vals = _clean_levels(trend_pair_text)
            if len(vals) >= 2:
                trend_fast, trend_slow = int(vals[0]), int(vals[1])

        rsi_enabled = st.checkbox("Momentum: RSI crosses", value=True)
        rsi_levels_text = "50,60"
        if rsi_enabled:
            rsi_levels_text = st.text_input(
                "RSI levels (comma separated)", value="50,60"
            )
        rsi_levels = _clean_levels(rsi_levels_text)
        if not rsi_levels:
            rsi_levels = [50.0, 60.0]

        atr_squeeze_enabled = st.checkbox("Volatility: ATR compression", value=True)
        atr_percentile = 25.0
        atr_pct_window = 63
        if atr_squeeze_enabled:
            atr_sq_cols = st.columns(2)
            atr_percentile = float(
                atr_sq_cols[0].number_input(
                    "ATR percentile threshold", min_value=1.0, value=25.0, step=1.0
                )
            )
            atr_pct_window = int(
                atr_sq_cols[1].number_input(
                    "ATR percentile lookback", min_value=10, value=63, step=1
                )
            )

        bb_enabled = st.checkbox("Bollinger: band width squeeze", value=True)
        bb_percentile = 20.0
        bb_pct_window = 126
        bb_period = 20
        if bb_enabled:
            bb_cols = st.columns(3)
            bb_period = int(
                bb_cols[0].number_input("Bollinger period", min_value=5, value=20, step=1)
            )
            bb_percentile = float(
                bb_cols[1].number_input(
                    "Width percentile threshold", min_value=1.0, value=20.0, step=1.0
                )
            )
            bb_pct_window = int(
                bb_cols[2].number_input(
                    "Percentile lookback", min_value=20, value=126, step=1
                )
            )

        nr7_enabled = st.checkbox("Range: NR7 pattern", value=True)
        nr7_window = 7
        if nr7_enabled:
            nr7_window = int(
                st.number_input("NR7 lookback", min_value=3, value=7, step=1)
            )

        gap_enabled = st.checkbox("Gaps: prior-day gap ≥ g%", value=True)
        gap_threshold = 3.0
        if gap_enabled:
            gap_threshold = float(
                st.number_input("Gap threshold (%)", min_value=0.5, value=3.0, step=0.5)
            )

        volume_enabled = st.checkbox("Volume: day-1/day-2 multiples", value=True)
        volume_threshold = 1.5
        volume_lookback = 63
        if volume_enabled:
            vol_cols = st.columns(2)
            volume_threshold = float(
                vol_cols[0].number_input(
                    "Volume multiple threshold", min_value=0.5, value=1.5, step=0.1
                )
            )
            volume_lookback = int(
                vol_cols[1].number_input(
                    "Volume lookback", min_value=10, value=63, step=1
                )
            )

        sr_enabled = st.checkbox("Support/Resistance ratio", value=True)
        sr_threshold = 2.0
        sr_lookback = 63
        if sr_enabled:
            sr_cols = st.columns(2)
            sr_threshold = float(
                sr_cols[0].number_input(
                    "SR ratio threshold", min_value=0.5, value=2.0, step=0.1
                )
            )
            sr_lookback = int(
                sr_cols[1].number_input(
                    "SR lookback", min_value=10, value=63, step=1
                )
            )

        new_high_enabled = st.checkbox("New high breakout", value=True)
        new_high_windows_text = "20,63"
        if new_high_enabled:
            new_high_windows_text = st.text_input(
                "New high windows", value="20,63"
            )
        new_high_windows = [int(v) for v in _clean_levels(new_high_windows_text)] or [20, 63]

        run_btn = st.form_submit_button("Run analysis", type="primary")

    analysis_ran = False
    spikes_df = pd.DataFrame()
    summary: dict[str, Any] | None = None

    params = {
        "start_date": str(start_date),
        "end_date": str(end_date),
        "spike_mode": spike_mode,
        "pct_threshold": pct_threshold,
        "atr_multiple": atr_multiple,
        "atr_window": atr_window,
        "atr_method": atr_method,
        "lookback_days": lookback_days,
        "use_sp_filter": use_sp_filter,
    }

    indicator_config = {
        "volume_filter": {
            "enabled": volume_filter_enabled,
            "threshold": volume_filter_threshold,
            "lookback": volume_filter_lookback,
        },
        "trend": {
            "enabled": trend_enabled,
            "fast": trend_fast,
            "slow": trend_slow,
        },
        "rsi": {
            "enabled": rsi_enabled,
            "period": 14,
            "levels": rsi_levels,
        },
        "atr_squeeze": {
            "enabled": atr_squeeze_enabled,
            "percentile": atr_percentile,
            "window": atr_pct_window,
        },
        "bb": {
            "enabled": bb_enabled,
            "percentile": bb_percentile,
            "pct_window": bb_pct_window,
            "period": bb_period,
        },
        "nr7": {
            "enabled": nr7_enabled,
            "window": nr7_window,
        },
        "gap": {
            "enabled": gap_enabled,
            "threshold": gap_threshold,
        },
        "volume": {
            "enabled": volume_enabled,
            "threshold": volume_threshold,
            "lookback": volume_lookback,
        },
        "sr": {
            "enabled": sr_enabled,
            "threshold": sr_threshold,
            "lookback": sr_lookback,
        },
        "new_high": {
            "enabled": new_high_enabled,
            "windows": new_high_windows,
        },
    }

    if run_btn:
        analysis_ran = True
        start_ts = pd.Timestamp(start_date).tz_localize(None)
        end_ts = pd.Timestamp(end_date).tz_localize(None)
        if start_ts > end_ts:
            start_ts, end_ts = end_ts, start_ts

        universe: list[str] | None = None
        if use_sp_filter:
            universe = _load_universe(storage, start_ts, end_ts)
            if not universe:
                st.warning(
                    "S&P membership table unavailable or empty. Falling back to all available tickers."
                )
                universe = None
        debug_env.update(
            {
                "requested_universe": len(universe) if universe else 0,
                "use_membership_filter": use_sp_filter,
            }
        )

        debug_meta = {
            "started": _utcnow_iso(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "streamlit": getattr(st, "__version__", "unknown"),
        }

        status, prog, log_fn = status_block("Scanning price history", key_prefix="spike_lab")
        try:
            prog.progress(0.05)
        except Exception:
            pass
        log_fn("Loading data and computing indicators…")

        try:
            spikes_df, summary = analyze_spike_precursors(
                storage,
                start=start_ts,
                end=end_ts,
                universe=universe,
                spike_mode=spike_mode,
                pct_threshold=pct_threshold,
                atr_multiple=atr_multiple,
                atr_window=atr_window,
                atr_method=atr_method,
                lookback_days=lookback_days,
                indicator_config=indicator_config,
                debug=debug,
            )
            try:
                prog.progress(1.0)
                status.update(label="Analysis complete", state="complete")
            except Exception:
                pass
            log_fn(f"Detected {len(spikes_df)} spike events")
        except Exception as exc:
            debug.add_error("analysis", exc)
            st.error(f"Analysis failed: {exc}")
            log_fn(f"Error: {exc}")
            summary = None
        finally:
            counts = summary.get("counts", {}) if summary else {}
            debug.set_metrics(
                tickers=int(counts.get("tickers", 0)),
                spikes=int(counts.get("spikes", 0)),
                flags=len(summary.get("flag_metadata", {})) if summary else 0,
            )

    if analysis_ran and summary:
        counts = summary.get("counts", {})
        st.markdown(
            f"**Date span:** {pd.Timestamp(counts.get('start')).date()} → {pd.Timestamp(counts.get('end')).date()}  \\\n+**Tickers analyzed:** {counts.get('tickers', 0)}"
        )
        metrics_cols = st.columns(3)
        metrics_cols[0].metric("Spikes found", counts.get("spikes", 0))
        if not spikes_df.empty:
            bool_cols = [
                flag
                for flag in summary.get("flag_metadata", {}).keys()
                if flag in spikes_df.columns
            ]
            if bool_cols:
                hit_counts = spikes_df[bool_cols].sum(axis=1)
                median_flags = float(hit_counts.median())
            else:
                median_flags = 0.0
        else:
            median_flags = 0.0
        metrics_cols[1].metric(
            "Median flags per spike", f"{median_flags:.1f}" if median_flags else "0"
        )
        metrics_cols[2].metric("Lookback (days)", summary.get("lookback_days", lookback_days))

        freq_df = summary.get("frequency_table")
        if isinstance(freq_df, pd.DataFrame) and not freq_df.empty:
            freq_display = freq_df.copy()
            freq_display["hit_rate_%"] = freq_display["hit_rate"].apply(lambda x: round(x * 100.0, 2))
            freq_display = freq_display.drop(columns=["hit_rate"])
            st.subheader("Precursor frequencies")
            st.dataframe(freq_display, use_container_width=True)
        else:
            st.info("No precursor flags met the criteria.")

        combos_df = summary.get("combos")
        if isinstance(combos_df, pd.DataFrame) and not combos_df.empty:
            combos_disp = combos_df.copy()
            combos_disp["lift"] = combos_disp["lift"].round(2)
            st.subheader("Top 10 combos (pairs)")
            st.dataframe(combos_disp, use_container_width=True)

        if not spikes_df.empty:
            st.subheader("Spike events (latest 500)")
            display_df = spikes_df.sort_values("spike_date", ascending=False).head(500)
            display_df = display_df.copy()
            display_df["spike_date"] = pd.to_datetime(display_df["spike_date"]).dt.date
            st.dataframe(display_df, use_container_width=True)
            st.download_button(
                "Download full CSV",
                spikes_df.to_csv(index=False).encode("utf-8"),
                file_name="spike_precursors.csv",
                mime="text/csv",
            )

        freq_df = summary.get("frequency_table")
        if isinstance(freq_df, pd.DataFrame) and not freq_df.empty:
            st.subheader("Export rule preset")
            threshold_pct = st.slider(
                "Minimum hit rate for inclusion (%)", min_value=5, max_value=100, value=30, step=5
            )
            eligible = freq_df[freq_df["hit_rate"] >= threshold_pct / 100.0]
            metadata = summary.get("flag_metadata", {})
            lead_stats = summary.get("lead_stats", {})
            if eligible.empty:
                st.info("No flags meet the selected hit-rate threshold.")
            else:
                within_default = int(summary.get("lookback_days", lookback_days))
                selected_flags: list[str] = []
                for _, row in eligible.iterrows():
                    flag = row["flag"]
                    label = f"{flag} ({row['hit_rate'] * 100:.1f}% hits)"
                    if st.checkbox(label, key=f"preset_{flag}"):
                        selected_flags.append(flag)

                if selected_flags:
                    conditions = []
                    for flag in selected_flags:
                        meta = dict(metadata.get(flag, {"type": "custom", "flag": flag}))
                        within_days = within_default
                        stats = lead_stats.get(flag)
                        if stats and stats.get("max"):
                            within_days = max(
                                within_days,
                                int(max(1, round(float(stats.get("max")))))
                            )
                        meta["within_days"] = within_days
                        if "flag" not in meta:
                            meta["flag"] = flag
                        conditions.append(meta)

                    preset = {
                        "name": datetime.utcnow().strftime(
                            "precursor_preset_%Y%m%d_%H%M"
                        ),
                        "conditions": conditions,
                    }
                    preset_json = json.dumps(preset, indent=2)
                    st.download_button(
                        "Download preset JSON",
                        preset_json.encode("utf-8"),
                        file_name=f"{preset['name']}.json",
                        mime="application/json",
                    )
                    st.caption(
                        "You can load this preset from the Stock Scanner (Shares Only) pre-filters (if/when that page supports it)."
                    )
                else:
                    st.info("Select at least one flag to export a preset.")

    _render_debug_panel(debug_meta, params, debug_env, debug)

