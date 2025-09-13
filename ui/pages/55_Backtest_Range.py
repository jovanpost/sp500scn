import io
import datetime as dt
from typing import Dict

import pandas as pd
import streamlit as st

from data_lake.storage import Storage
from engine.signal_scan import ScanParams
from backtest.run_range import run_range


def _df_to_markdown_safe(df: pd.DataFrame) -> str:
    """Return a Markdown table for df without relying on pandas.to_markdown."""
    try:
        from tabulate import tabulate  # optional dependency

        return tabulate(
            df.values.tolist(),
            headers=list(df.columns),
            tablefmt="pipe",
            floatfmt="g",
        )
    except Exception:
        cols = [str(c) for c in df.columns]
        lines = []
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for row in df.itertuples(index=False, name=None):
            vals = []
            for v in row:
                if pd.isna(v):
                    vals.append("")
                elif isinstance(v, (int, float)):
                    vals.append(f"{v:.6g}")
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)


def _df_to_csv_text(df: pd.DataFrame) -> str:
    return df.to_csv(index=False)


def _render_df_with_copy(title: str, df: pd.DataFrame, key_prefix: str) -> None:
    st.subheader(title)
    st.dataframe(df, use_container_width=True, hide_index=False)
    fmt = st.radio(
        "Copy format", ["Markdown", "CSV"], horizontal=True, key=f"{key_prefix}_fmt"
    )
    txt = _df_to_markdown_safe(df) if fmt == "Markdown" else _df_to_csv_text(df)
    st.text_area("Copy this", txt, height=180, key=f"{key_prefix}_copybox")
    mime = "text/markdown" if fmt == "Markdown" else "text/csv"
    ext = "md" if fmt == "Markdown" else "csv"
    st.download_button(
        f"Download table ({ext.upper()})",
        txt.encode("utf-8"),
        file_name=f"{key_prefix}.{ext}",
        mime=mime,
        key=f"{key_prefix}_dl",
    )


def render_page() -> None:
    st.header("📅 Backtest (range)")

    with st.form("range_controls"):
        col1, col2, col3 = st.columns(3)
        with col1:
            start = st.date_input(
                "Start date",
                value=dt.date.today() - dt.timedelta(days=30),
                key="range_start",
            )
            end = st.date_input("End date", value=dt.date.today(), key="range_end")
            horizon = int(
                st.number_input(
                    "Horizon (days)",
                    min_value=1,
                    value=30,
                    step=1,
                    key="range_horizon",
                )
            )
        with col2:
            vol_lookback = int(
                st.number_input(
                    "Volume lookback",
                    min_value=1,
                    value=63,
                    step=1,
                    key="range_vol_lb",
                )
            )
            min_close_up_pct = float(
                st.number_input(
                    "Min close-up on D-1 (%)",
                    value=3.0,
                    step=0.5,
                    key="range_min_close_up",
                )
            )
            min_gap_open_pct = float(
                st.number_input(
                    "Min gap open (%)",
                    value=0.0,
                    step=0.1,
                    key="range_min_gap",
                )
            )
        with col3:
            min_vol_multiple = float(
                st.number_input(
                    "Min volume multiple",
                    value=1.5,
                    step=0.1,
                    key="range_min_vol_mult",
                )
            )
            atr_window = int(
                st.number_input(
                    "ATR window",
                    min_value=5,
                    value=21,
                    step=1,
                    key="range_atr_win",
                )
            )
            sr_min_ratio = float(
                st.number_input(
                    "Min S:R ratio",
                    value=2.0,
                    step=0.1,
                    key="range_sr_min_ratio",
                )
            )
            sr_lookback = int(
                st.number_input(
                    "S/R lookback (days)",
                    min_value=10,
                    value=21,
                    step=1,
                    key="range_sr_lb",
                )
            )
            use_precedent = st.checkbox(
                "Require 21-day precedent (lookback 252d, window 21d)",
                value=True,
                key="range_use_precedent",
            )
            use_atr_feasible = st.checkbox(
                "Require ATR×N feasibility (at D-1)",
                value=True,
                key="range_use_atr_ok",
            )
            precedent_lookback = int(
                st.number_input(
                    "Precedent lookback (days)",
                    min_value=21,
                    value=252,
                    step=1,
                    key="range_prec_lookback",
                )
            )
            precedent_window = int(
                st.number_input(
                    "Precedent window (days)",
                    min_value=5,
                    value=21,
                    step=1,
                    key="range_prec_window",
                )
            )
        save_outcomes = st.checkbox(
            "Save outcomes to lake", value=False, key="range_save_outcomes"
        )
        run = st.form_submit_button("Run backtest", use_container_width=True)

    if isinstance(start, (list, tuple)):
        start = start[0]
    if isinstance(end, (list, tuple)):
        end = end[0]

    if run:
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
        prog = st.progress(0.0)
        status = st.empty()

        def _cb(i: int, total: int, day: pd.Timestamp, cands: int, hits: int) -> None:
            prog.progress(i / total if total else 0.0)
            status.text(f"{day.date()}: {cands} matches, {hits} hits")

        trades_df, summary = run_range(
            storage, str(start), str(end), params, progress_cb=_cb
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

    trades_df = st.session_state.get("bt_trades")
    summary = st.session_state.get("bt_summary")
    save_path = st.session_state.get("bt_save_path")

    if summary is not None:
        st.subheader("Summary")
        st.dataframe(pd.DataFrame([summary]), key="range_summary_df")

    if trades_df is not None:
        if trades_df.empty:
            st.info("No trades found in this range.")
        else:
            _render_df_with_copy("Trades", trades_df, "range_trades")

    if save_path:
        st.success(f"Saved to lake at {save_path}")


def page() -> None:
    render_page()
