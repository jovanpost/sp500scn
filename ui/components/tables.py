from __future__ import annotations

import pandas as pd
import streamlit as st


def show_df(
    title: str,
    df: pd.DataFrame | None,
    key: str,
    height: int | None = None,
    csv_name: str | None = None,
) -> None:
    """Render a dataframe with consistent controls for download/copy."""
    if title:
        st.subheader(title)

    if df is None or df.empty:
        st.info("No rows.")
        return

    normalized_df = df.reset_index(drop=True)
    st.dataframe(normalized_df, use_container_width=True, height=height)

    csv_text = normalized_df.to_csv(index=False)
    st.download_button(
        "Download CSV",
        csv_text.encode("utf-8"),
        file_name=csv_name or f"{key}.csv",
        mime="text/csv",
        key=f"{key}_dl",
    )

    with st.expander("Copy table (CSV)"):
        st.code(csv_text, language="text", wrap_lines=False)
