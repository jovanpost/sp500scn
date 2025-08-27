# app.py — Streamlit UI for swing_options_screener (mobile-friendly, no sidebar)
import streamlit as st
import pandas as pd

from swing_options_screener import (
    run_scan, explain_ticker, parse_ticker_text, DEFAULT_TICKERS
)
from sp_universe import get_sp500_tickers, LAST_ERROR

st.set_page_config(page_title="Swing Options Screener", layout="wide")

st.title("📊 Swing Options Screener (Finviz-style, Unadjusted)")

# ----------------------------
# Top: Quick-start controls
# ----------------------------
col1, col2 = st.columns([1, 2])

with col1:
    universe = st.radio(
        "Universe",
        options=["Live S&P 500 (Wikipedia)", "Custom list"],
        index=0,
        help="Live list is scraped and cached for 6h."
    )

with col2:
    if universe == "Custom list":
        tickers_text = st.text_input(
            "Tickers (comma/space/newline)",
            value="WMT INTC MOS",
            placeholder="e.g., AAPL MSFT NVDA"
        )
    else:
        tickers_text = ""  # ignored

# Big Run button
run_btn = st.button("▶️ Run Scan", type="primary", use_container_width=True)

# ----------------------------
# Collapsible: Filters (default collapsed)
# ----------------------------
with st.expander("Filters (optional)", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        res_days = st.number_input(
            "Resistance lookback (days)", min_value=10, max_value=252, value=21, step=1
        )
    with c2:
        rr_min = st.number_input(
            "Min RR to Resistance", min_value=1.0, max_value=10.0, value=2.0, step=0.1
        )
    with c3:
        relvol_min = st.number_input(
            "Min RelVol (time-adjusted 63d)", min_value=0.5, max_value=5.0, value=1.10, step=0.05
        )
    with c4:
        relvol_median = st.checkbox("Use median vol (63d)", value=False,
                                    help="Median instead of mean for 63-day volume baseline")

    # Optional UI-only price cap (OFF by default)
    c5, c6 = st.columns(2)
    with c5:
        cap_under_100 = st.checkbox("Cap display to Price ≤ $100 (UI only)", value=False)
    with c6:
        price_cap_value = st.number_input("Price cap ($)", min_value=1, max_value=5000, value=100, step=1)

# ----------------------------
# Collapsible: Options settings (default collapsed)
# ----------------------------
with st.expander("Options (bull call spread suggestion)", expanded=False):
    c7, c8 = st.columns(2)
    with c7:
        with_options = st.checkbox("Suggest bull call spread near TP", value=True)
    with c8:
        opt_days = st.number_input("Target days to expiry", min_value=7, max_value=90, value=30, step=1)

# ----------------------------
# Collapsible: Explain a ticker (debug)
# ----------------------------
with st.expander("Explain a ticker (debug)", expanded=False):
    ex_ticker = st.text_input("Ticker to explain (e.g., WMT, NOW)", key="explain_tkr")
    ex_go = st.button("Explain", key="explain_go")
    if ex_go and ex_ticker.strip():
        st.code(f"=== DEBUG {ex_ticker.upper()} ===", language="text")
        # Prints detailed debug to logs; also executes the same checks
        explain_ticker(
            ex_ticker.upper(),
            res_days=locals().get("res_days", 21),
            rel_vol_min=locals().get("relvol_min", 1.10),
            relvol_median=locals().get("relvol_median", False),
            rr_min=locals().get("rr_min", 2.0),
            stop_mode="safest",
        )
        st.info("Explanation printed to app logs (Streamlit Cloud → Logs).")

# ----------------------------
# RUN
# ----------------------------
if run_btn:
    # Use defaults if user didn't open the expanders
    res_days = locals().get("res_days", 21)
    rr_min = locals().get("rr_min", 2.0)
    relvol_min = locals().get("relvol_min", 1.10)
    relvol_median = locals().get("relvol_median", False)
    cap_under_100 = locals().get("cap_under_100", False)
    price_cap_value = float(locals().get("price_cap_value", 100))
    with_options = locals().get("with_options", True)
    opt_days = locals().get("opt_days", 30)

    # Build universe
    if universe == "Live S&P 500 (Wikipedia)":
        tickers = get_sp500_tickers()
        if not tickers:
            warn = "Could not fetch S&P 500 from Wikipedia; using Custom list if provided, else defaults."
            if LAST_ERROR:
                warn += f"\n\nDetails: {LAST_ERROR}"
            st.warning(warn)
            tickers = parse_ticker_text(tickers_text) if tickers_text else list(DEFAULT_TICKERS)
    else:
        tickers = parse_ticker_text(tickers_text) if tickers_text else list(DEFAULT_TICKERS)

    # Execute scan
    with st.spinner("Scanning…"):
        res = run_scan(
            tickers=tickers,
            res_days=res_days,
            rel_vol_min=relvol_min,
            relvol_median=relvol_median,
            rr_min=rr_min,
            stop_mode="safest",
            with_options=with_options,
            opt_days=opt_days,
        )
        df = res.get("pass_df", pd.DataFrame())

    # Optional UI-only cap
    if cap_under_100 and not df.empty and "Price" in df.columns:
        df = df[df["Price"] <= price_cap_value]

    # Sort by Price ascending
    if not df.empty and "Price" in df.columns:
        df = df.sort_values("Price", ascending=True)

    st.subheader(f"Passes ({len(df)})")
    if df.empty:
        st.info("No PASS tickers.")
    else:
        # Clean table, scrollable, shows all columns
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Copy/paste (Google Sheets) – pipe-delimited
        st.markdown("**Copy for Google Sheets (pipe-delimited)**")
        cols = list(df.columns)
        psv = "|".join(cols) + "\n" + "\n".join(
            "|".join("" if pd.isna(v) else str(v).replace("|","/") for v in df.loc[i, cols].values)
            for i in df.index
        )
        st.code(psv, language="text")

        # Optional: raw CSV download
        st.download_button(
            label="Download CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="pass_tickers.csv",
            mime="text/csv",
            use_container_width=True
        )

