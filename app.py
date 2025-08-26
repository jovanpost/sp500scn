import os, sys, subprocess, time
import pandas as pd
import streamlit as st

APP_TS = time.strftime("%Y-%m-%d %H:%M:%S ET", time.localtime())

st.title("📈 S&P 500 Options Screener")
st.caption(f"UI started: {APP_TS}")

def run_screener_script():
    """
    Runs swing_options_screener.py as a subprocess.
    Assumes the script writes pass_tickers.csv in the repo root.
    Returns a pandas DataFrame (or empty DF if none).
    """
    st.info("Running screener… this may take a bit on first run.")
    # Call your script with the same interpreter Streamlit is using
    cmd = [sys.executable, "swing_options_screener.py"]
    # If your script supports flags, append them here, e.g.:
    # cmd += ["--no-explain"]
    try:
        subprocess.run(cmd, check=False)
    except Exception as e:
        st.error(f"Failed to run screener: {e}")
        return pd.DataFrame()

    csv_path = "pass_tickers.csv"
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            return df
        except Exception as e:
            st.error(f"Could not read {csv_path}: {e}")
            return pd.DataFrame()
    else:
        return pd.DataFrame()

if st.button("Run Screener"):
    df = run_screener_script()
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
        )
