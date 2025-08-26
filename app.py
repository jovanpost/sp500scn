# app.py
import io
import sys
import time
import contextlib
from datetime import datetime, timezone, timedelta

import pandas as pd
import streamlit as st

# ------------ utils ------------
ET = timezone(timedelta(hours=-5), name="ET")  # crude ET; Streamlit Cloud runs UTC
def now_et() -> datetime:
    return datetime.utcnow().replace(tzinfo=timezone.utc).astimezone(ET)

def read_csv_if_exists(path: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
        if df.empty:
            return None
        return df
    except Exception:
        return None

def df_to_pipe(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["|".join(cols)]
    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r.get(c)
            if pd.isna(v):
                vals.append("")
            else:
                s = str(v).replace("|", "¦")
                vals.append(s)
        lines.append("|".join(vals))
    return "\n".join(lines)

@contextlib.contextmanager
def capture_stdout():
    buf = io.StringIO()
    old = sys.stdout
    try:
        sys.stdout = buf
        yield buf
    finally:
        sys.stdout = old

def invoke_run():
    """
    Calls swing_options_screener.run_scan(); falls back to CSVs if needed.
    Returns (df_pass, near_df, console_text).
    """
    try:
        from swing_options_screener import run_scan
    except Exception as e:
        return None, None, f"Import error: {e}"

    with capture_stdout() as out:
        try:
            result = run_scan()  # your function returns {'pass_df': df} (per our last change)
        except Exception as e:
            print(f"[run_scan error] {e}", file=sys.stderr)
            result = None

    console_text = out.getvalue()

    df_pass = None
    near_df = None
    if isinstance(result, dict):
        df_pass = result.get("pass_df")

    if df_pass is None:
        # Try files your script writes
        df_pass = read_csv_if_exists("pass_tickers.csv")
        if df_pass is None:
            df_pass = read_csv_if_exists("pass_tickers_unadjusted.psv")

    # Near-misses optional file (only if your backend writes it)
    near_df = read_csv_if_exists("near_misses.csv")

    return df_pass, near_df, console_text

def explain_one(ticker: str) -> str:
    """Call swing_options_screener.explain_ticker(t) and return printed output."""
    try:
        from swing_options_screener import explain_ticker
    except Exception as e:
        return f"Import error: {e}"

    with capture_stdout() as out:
        try:
            explain_ticker(ticker.strip().upper())
        except Exception as e:
            print(f"[explain_ticker error] {e}")
    return out.getvalue()

def render_full_table(df: pd.DataFrame, title: str, key_prefix: str):
    st.subheader(title)
    # SHOW EVERYTHING. No compact view, no expander.
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Copy/download
    st.markdown("**Copy (pipe-delimited for Google Sheets)**")
    st.code(df_to_pipe(df), language="text")
    st.download_button(
        "Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"{key_prefix}.csv",
        mime="text/csv",
        key=f"dl_{key_prefix}"
    )

# ------------ UI ------------
st.set_page_config(page_title="S&P 500 Options Screener", layout="wide")
st.title("📈 S&P 500 Options Screener — Full Columns")

st.caption(f"UI started: {now_et().strftime('%Y-%m-%d %H:%M:%S ET')}")

col_left, col_right = st.columns([2.3, 1])

with col_left:
    if st.button("Run Screener", use_container_width=True):
        with st.status("Running screener…", expanded=True) as status:
            t0 = time.time()
            df_pass, near_df, console_text = invoke_run()
            status.update(state="complete", expanded=False)

        with st.expander("Console output", expanded=False):
            st.code(console_text or "(no console output)")

        if df_pass is None or df_pass.empty:
            st.warning("No PASS tickers found (or output not produced).")
            if isinstance(near_df, pd.DataFrame) and not near_df.empty:
                # If your backend provides near misses, show top 3
                top3 = near_df.head(3)
                render_full_table(top3, "Closest 3 (failed) — reasons included", "near_misses")
        else:
            render_full_table(df_pass, "PASS tickers — ALL columns", "pass_tickers")

with col_right:
    st.header("Explain a ticker")
    t = st.text_input("Ticker", placeholder="e.g., WMT, INTC, MOS")
    if st.button("Explain", use_container_width=True, type="primary", disabled=not t.strip()):
        with st.status(f"Explaining {t.strip().upper()}…"):
            txt = explain_one(t)
        st.subheader(f"🔍 Debug: {t.strip().upper()}")
        st.code(txt or "(no output)")

st.caption(
    "Notes: Yahoo prices ~15-min delayed. All columns (Hist21d_*, ResLookbackDays, Prices, options fields) "
    "are shown directly in the table and in the copy/download."
)

