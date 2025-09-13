import io
import datetime as dt
from typing import Dict

import pandas as pd
import streamlit as st

from engine.signal_scan import scan_day, ScanParams
from data_lake.storage import Storage
from ui.components.progress import status_block


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

    _d = st.date_input("Entry day (D)", value=dt.date.today())
    if isinstance(_d, (list, tuple)):
        _d = _d[0]
    D = pd.to_datetime(_d).date()

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
            storage = Storage()
            params: ScanParams = {
                "min_close_up_pct": min_close_up_pct,
                "min_vol_multiple": min_vol_multiple,
                "min_gap_open_pct": min_gap_open_pct,
                "atr_window": atr_window,
                "lookback_days": vol_lookback,
                "horizon_days": horizon,
                "sr_min_ratio": sr_min_ratio,
            }

            def on_step(i: int, total: int, ticker: str):
                pct = max(0, min(100, int(i / max(1, total) * 100)))
                prog.progress(pct, text=f"{pct}%")
                log(f"{i}/{total} {ticker} ✓")

            cand_df, out_df, fails, _dbg = scan_day(
                storage, pd.to_datetime(D), params, on_step=on_step
            )
            st.session_state["cand_df"] = cand_df
            st.session_state["out_df"] = out_df
            st.session_state["fails"] = fails

            if save_outcomes and not out_df.empty:
                buf = io.BytesIO()
                out_df.to_parquet(buf, index=False)
                storage.write_bytes(
                    f"runs/{pd.to_datetime(D).date().isoformat()}/outcomes.parquet",
                    buf.getvalue(),
                )

            status.update(label="Scan complete ✅", state="complete")
            st.toast(f"Scan done: {len(cand_df)} matches", icon="✅")
        except Exception as e:
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


def page() -> None:
    render_page()
