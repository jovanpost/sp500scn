# app.py
import streamlit as st
import pandas as pd
from swing_options_screener import run_scan, explain_ticker, parse_ticker_text
from sp_universe import get_sp500_tickers

st.set_page_config(page_title="Swing Options Screener", layout="wide")

st.title("Swing Options Screener (Unadjusted, Finviz-style RelVol)")

with st.sidebar:
    st.header("Scan Settings")

    # Universe
    uni = st.selectbox("Universe", ["Custom list", "Live S&P 500 (Wikipedia)"], index=1)
    tickers_text = ""
    if uni == "Custom list":
        tickers_text = st.text_area("Tickers (space/comma/newline separated)", value="WMT INTC MOS", height=90)

    # Core params
    res_days = st.number_input("Resistance lookback (days)", min_value=10, max_value=252, value=21, step=1)
    rr_min = st.number_input("Min RR to Resistance", min_value=1.0, max_value=10.0, value=2.0, step=0.1)
    relvol_min = st.number_input("Min RelVol (time-adjusted 63d)", min_value=0.5, max_value=5.0, value=1.10, step=0.05)
    relvol_median = st.checkbox("Use median (not mean) for 63d volume", value=False)
    stop_mode = st.selectbox("Stop preference", ["safest", "structure"], index=0)

    # Options block
    with_options = st.checkbox("Suggest bull call spread near TP", value=True)
    opt_days = st.number_input("Target days to expiry", min_value=7, max_value=90, value=30, step=1)

    # Optional price cap (OFF by default)
    cap_under_100 = st.checkbox("Cap display to Price ≤ $100", value=False)
    price_cap_value = st.number_input("Price cap ($)", min_value=1, max_value=5000, value=100, step=1, help="Used only if the checkbox above is ON")

    run_btn = st.button("Run Scan", type="primary", use_container_width=True)

# Explain panel
with st.expander("Explain a ticker (debug)"):
    ex_ticker = st.text_input("Ticker to explain (e.g., WMT, NOW)")
    ex_go = st.button("Explain")
    if ex_go and ex_ticker.strip():
        st.code(f"=== DEBUG {ex_ticker.upper()} ===", language="text")
        explain_ticker(ex_ticker.upper(),
                       res_days=res_days,
                       rel_vol_min=relvol_min,
                       relvol_median=relvol_median,
                       rr_min=rr_min,
                       stop_mode=stop_mode)
        st.info("Explanation printed to app logs (Streamlit Cloud logs).")

# Run scan
if run_btn:
    if uni == "Live S&P 500 (Wikipedia)":
        tickers = get_sp500_tickers()
        if not tickers:
            st.warning("Could not fetch S&P 500 from Wikipedia; falling back to text box (if provided).")
            tickers = parse_ticker_text(tickers_text)
    else:
        tickers = parse_ticker_text(tickers_text)

    res = run_scan(
        tickers=tickers,
        res_days=res_days,
        rel_vol_min=relvol_min,
        relvol_median=relvol_median,
        rr_min=rr_min,
        stop_mode=stop_mode,
        with_options=with_options,
        opt_days=opt_days,
    )
    df = res.get("pass_df", pd.DataFrame())

    # Optional UI-only price cap
    if cap_under_100 and not df.empty and "Price" in df.columns:
        df = df[df["Price"] <= float(price_cap_value)]

    # Always sort by Price ascending
    if not df.empty and "Price" in df.columns:
        df = df.sort_values("Price", ascending=True)

    st.subheader(f"Passes ({len(df)})")
    if df.empty:
        st.info("No PASS tickers.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Provide a Sheets-friendly copy (pipe-delimited)
        st.subheader("Copy/Paste (Google Sheets)")
        cols = list(df.columns)
        psv = "|".join(cols) + "\n" + "\n".join(
            "|".join("" if pd.isna(v) else str(v).replace("|","/") for v in df.loc[i, cols].values)
            for i in df.index
        )
        st.code(psv, language="text")

