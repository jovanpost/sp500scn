import io
import datetime as dt
from typing import Dict

import pandas as pd
import streamlit as st

from engine.signal_scan import scan_day, ScanParams
from data_lake.storage import Storage


def _df_to_markdown_safe(df: pd.DataFrame) -> str:
    """Return a Markdown table for df without relying on pandas.to_markdown."""
    try:
        from tabulate import tabulate  # optional dependency
        return tabulate(df.values.tolist(), headers=list(df.columns), tablefmt="pipe", floatfmt="g")
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
    fmt = st.radio("Copy format", ["Markdown", "CSV"], horizontal=True, key=f"{key_prefix}_fmt")
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

    if st.button("Run scan", type="primary"):
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
        cand_df, out_df, fails, _dbg = scan_day(storage, pd.to_datetime(D), params)
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
        if not cand_df.empty:
            _render_df_with_copy("✅ Candidates (matches)", cand_df, "matches")
        if out_df is not None and not out_df.empty:
            _render_df_with_copy("🎯 Outcomes", out_df, "outcomes")
        else:
            st.info("No outcomes to display.")


def page() -> None:
    render_page()
