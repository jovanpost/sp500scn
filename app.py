import os, sys, subprocess, time, io
import pandas as pd
import streamlit as st

st.set_page_config(page_title="S&P 500 Options Screener", layout="wide")

APP_TS = time.strftime("%Y-%m-%d %H:%M:%S ET", time.localtime())
st.title("📈 S&P 500 Options Screener")
st.caption(f"UI started: {APP_TS}")

CSV_PATH = "pass_tickers.csv"
SCRIPT = "swing_options_screener.py"

def run_subprocess(args):
    """
    Run your screener script with args and return (rc, stdout, stderr).
    Uses the same Python interpreter Streamlit runs with.
    """
    try:
        proc = subprocess.run(
            [sys.executable, SCRIPT] + args,
            capture_output=True,
            text=True,
            timeout=600,   # 10 min hard cap
        )
        return proc.returncode, proc.stdout, proc.stderr
    except Exception as e:
        return 1, "", f"Exception launching subprocess: {e}"

def run_screener():
    st.info("Running screener… this may take a bit on first run.")
    rc, out, err = run_subprocess([])
    with st.expander("Console output"):
        st.code(out or "(no stdout)", language="bash")
        if err:
            st.error("stderr:")
            st.code(err, language="bash")

    if os.path.exists(CSV_PATH):
        try:
            df = pd.read_csv(CSV_PATH)
        except Exception as e:
            st.error(f"Could not read {CSV_PATH}: {e}")
            return pd.DataFrame()
        return df
    else:
        return pd.DataFrame()

def explain_ticker(ticker: str):
    """Call your script with --explain <TICKER> and show raw output."""
    if not ticker:
        st.warning("Type a ticker first.")
        return
    st.info(f"Explaining {ticker}…")
    rc, out, err = run_subprocess(["--explain", ticker.upper()])
    st.subheader(f"🔍 Debug: {ticker.upper()}")
    if out:
        st.code(out, language="bash")
    if err:
        st.error("stderr:")
        st.code(err, language="bash")

# --- Layout ---
left, right = st.columns([2, 1])

with left:
    if st.button("Run Screener", use_container_width=True):
        df = run_screener()
        if df.empty:
            st.warning("No PASS tickers found (or CSV not produced).")
        else:
            st.success(f"Found {len(df)} PASS tickers")
            st.dataframe(df, use_container_width=True)
            st.download_button(
                "Download CSV",
                df.to_csv(index=False).encode("utf-8"),
                file_name="pass_tickers.csv",
                mime="text/csv",
                use_container_width=True,
            )

with right:
    st.markdown("### Explain a ticker")
    x_ticker = st.text_input("Ticker", placeholder="e.g., WMT, INTC, MOS")
    if st.button("Explain", use_container_width=True):
        explain_ticker(x_ticker)

st.divider()
st.caption(
    "Notes: Yahoo prices are ~15-min delayed. ‘Explain’ runs the same "
    "logic as your CLI (`--explain TICKER`) and prints the exact gate that failed."
)
