# app.py — Streamlit UI for swing_options_screener

import streamlit as st
import pandas as pd
from swing_options_screener import run_scan, explain_ticker, parse_ticker_text, DEFAULT_TICKERS
from sp_universe import get_sp500_tickers

st.set_page_config(page_title="Swing Options Screener", layout="wide")

st.title("📊 Swing Options Screener (Finviz-style, Unadjusted)")

# Sidebar controls
st.sidebar.header("Settings")

universe = st.sidebar.radio(
    "Choose ticker universe:",
    options=["sp500", "custom"],
    index=0,
    help="Select 'sp500' for live S&P 500 from Wikipedia or 'custom' to type tickers below"
)

ticker_text = ""
if universe == "custom":
    ticker_text = st.sidebar.text_area(
        "Custom tickers (comma/space/newline separated):",
        value=",".join(DEFAULT_TICKERS[:10]),
        height=100
    )

res_days = st.sidebar.number_input("Resistance Lookback Days", min_value=10, max_value=60, value=21, step=1)
relvol_min = st.sidebar.number_input("Minimum RelVol", min_value=0.5, max_value=3.0, value=1.10, step=0.05)
rr_min = st.sidebar.number_input("Minimum RR (to Resistance)", min_value=1.0, max_value=5.0, value=2.0, step=0.1)
stop_mode = st.sidebar.selectbox("Stop Mode", options=["safest","structure"], index=0)
opt_days = st.sidebar.number_input("Target Option Expiry Days", min_value=10, max_value=60, value=30, step=1)

if st.sidebar.button("Run Scan"):
    # Choose ticker universe
    if universe == "sp500":
        tickers = get_sp500_tickers()
        if not tickers:
            st.warning("Could not fetch S&P 500 from Wikipedia; falling back to custom tickers.")
            tickers = parse_ticker_text(ticker_text) if ticker_text else list(DEFAULT_TICKERS)
    else:
        tickers = parse_ticker_text(ticker_text) if ticker_text else list(DEFAULT_TICKERS)

    with st.spinner("Running scan..."):
        result = run_scan(
            tickers=tickers,
            res_days=res_days,
            rel_vol_min=relvol_min,
            rr_min=rr_min,
            stop_mode=stop_mode,
            with_options=True,
            opt_days=opt_days,
        )
        df = result['pass_df']

    if not df.empty:
        # Sort by Price ascending
        df = df.sort_values("Price", ascending=True)

        st.success(f"Found {len(df)} passing tickers")
        st.dataframe(df, use_container_width=True)

        # Copy-to-clipboard text box
        csv_str = df.to_csv(sep="|", index=False)
        st.text_area("Copy-paste for Google Sheets (pipe-delimited)", csv_str, height=200)
    else:
        st.error("No PASS tickers found.")

# Debug / Explain section
st.sidebar.header("Explain a ticker")
explain_symbol = st.sidebar.text_input("Enter ticker to explain:")

if st.sidebar.button("Run Explain") and explain_symbol:
    st.write(f"### Debug for {explain_symbol.upper()}")
    explain_ticker(
        explain_symbol.upper(),
        res_days=res_days,
        rel_vol_min=relvol_min,
        rr_min=rr_min,
        stop_mode=stop_mode
    )
