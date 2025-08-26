import streamlit as st
from swing_options_screener import run_scan

st.title("📈 S&P 500 Options Screener")

if st.button("Run Screener"):
    results = run_scan()
    if results.empty:
        st.warning("No PASS tickers found.")
    else:
        st.success(f"Found {len(results)} PASS tickers")
        st.dataframe(results)
