# app.py
import os
import pandas as pd
import numpy as np
import streamlit as st

from swing_options_screener import (
    run_scan,
    explain_ticker,   # prints console diagnostics
    compute_relvol_time_adjusted,  # for plain-English debug text if needed
)

st.set_page_config(page_title="Swing Options Screener", layout="wide")

# ---------- Styles ----------
RED_BTN = """
<style>
div.stButton > button:first-child {
  background-color: #d90429 !important;
  color: white !important;
  border: 0px;
  padding: 0.6rem 1.1rem;
  font-weight: 700;
  border-radius: 8px;
}
.kbd {
  background: #111; color: #eee; padding: 2px 6px; border-radius: 4px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, "Courier New", monospace;
}
.smallnote { color:#666; font-size: 0.85rem; }
.whybuy p { margin: 0.2rem 0 0.6rem 0; line-height: 1.35rem; }
.whybuy li { margin: 0.2rem 0; }
.strong { font-weight: 700; }
</style>
"""
st.markdown(RED_BTN, unsafe_allow_html=True)

# ---------- Helpers ----------
def _num(v, nd=2, prefix="", suffix=""):
    if v is None or v == "":
        return ""
    try:
        f = float(v)
        if np.isnan(f):
            return ""
        s = f"{f:.{nd}f}"
        if prefix: s = f"{prefix}{s}"
        if suffix: s = f"{s}{suffix}"
        return s
    except Exception:
        return ""

def _pct(v, nd=2):
    return _num(v, nd=nd, suffix="%")

def _usd(v, nd=2):
    return _num(v, nd=nd, prefix="$")

def _bold(txt):
    return f"<span class='strong'>{txt}</span>"

def _relvol_detail(row):
    rv = row.get("RelVol(TimeAdj63d)", "")
    return _num(rv, nd=2, suffix="×") if rv != "" else ""

def _vol_sentence(row):
    # we can’t reconstruct expected volume precisely without full series here.
    # keep concise and consistent: just show the relvol × and today vs avg surrogates if present.
    rv = row.get("RelVol(TimeAdj63d)", "")
    if rv == "" or np.isnan(rv):
        return "Volume is in line with recent activity."
    return f"Volume is running at about {_bold(_num(rv,2,'','×'))} the 63-day average for this point in the session."

def _why_buy_text(row):
    t = row["Ticker"]
    price = _usd(row.get("Price"))
    tp     = _usd(row.get("TP"))
    res    = _usd(row.get("Resistance"))
    rr_res = _num(row.get("RR_to_Res"))
    rr_tp  = _num(row.get("RR_to_TP"))
    chg    = _pct(row.get("Change%"))
    tp_dol = _usd(row.get("TPReward$"))
    tp_pct = _pct(row.get("TPReward%"))
    d_atr  = _usd(row.get("DailyATR"))
    d_cap  = _usd(row.get("DailyCap"))
    hcnt   = str(row.get("Hist21d_PassCount", ""))
    hmax   = _pct(row.get("Hist21d_Max%"))
    hexam  = row.get("Hist21d_Examples", "")
    sup_type = row.get("SupportType","")
    sup_px   = _usd(row.get("SupportPrice"))
    opt_exp  = row.get("OptExpiry","")
    buyk     = _num(row.get("BuyK"),0)
    sellk    = _num(row.get("SellK"),0)

    parts = []

    # First sentence, plain English
    if opt_exp and buyk and sellk:
        parts.append(
            f"{t} is a buy via a bull call spread {_bold(buyk)} / {_bold(sellk)} expiring {_bold(opt_exp)} "
            f"because it recently traded up to {_bold(res)} (recent high / “resistance”) and now trades around {_bold(price)}, "
            f"putting the intermediate target at {_bold(tp)} well within reach."
        )
    else:
        parts.append(
            f"{t} looks buyable because it recently traded up to {_bold(res)} (recent high / “resistance”) and now trades around {_bold(price)}, "
            f"so the intermediate target at {_bold(tp)} is realistic."
        )

    # RR
    parts.append(
        f"The reward-to-risk based on the distance to resistance is approximately {_bold(rr_res)} : 1 "
        f"(to the interim target: {_bold(rr_tp)} : 1)."
    )

    # Up on day + volume
    parts.append(
        f"Today the stock is {_bold(chg)} and {_vol_sentence(row)}"
    )

    # TP distance + ATR capacity
    parts.append(
        f"The move to the target is about {_bold(tp_dol)} (≈ {_bold(tp_pct)}). "
        f"The daily ATR is around {_bold(d_atr)}, implying ≈ {_bold(d_cap)} of potential movement over ~21 trading days."
    )

    # History realism
    if hcnt and hcnt != "0":
        hx = f"Within the past year there were {_bold(hcnt)} separate 21-trading-day windows that exceeded the needed move; best observed ≈ {_bold(hmax)}."
        parts.append(hx)
        if hexam:
            # turn examples into list
            ex = [e.strip() for e in hexam.split(";") if e.strip()]
            ex = ex[:3]
            if ex:
                parts.append("Examples:")
                for e in ex:
                    parts.append(f"- {e}")

    # Support/stop mention
    if sup_type and sup_px:
        parts.append(f"Stop reference: {_bold(sup_type)} near {_bold(sup_px)}.")

    html = "<div class='whybuy'>" + "".join(f"<p>{p}</p>" if not p.startswith("- ") else f"<li>{p[2:]}</li>" for p in parts) + "</div>"
    # Ensure list items wrapped properly
    if any(p.startswith("- ") for p in parts):
        html = "<div class='whybuy'><ul>" + "".join(
            f"{('<li>'+p[2:]+'</li>') if p.startswith('- ') else ('<p>'+p+'</p>')}"
            for p in parts
        ) + "</ul></div>"
    return html

def _pipe_copy_block(df):
    cols = list(df.columns)
    lines = ["|".join(cols)]
    for _, r in df.iterrows():
        lines.append("|".join(str(r.get(c, "")) for c in cols))
    txt = "\n".join(lines)
    return txt

# ---------- Layout ----------
tab_scan, tab_hist = st.tabs(["📊 Scanner", "📜 History"])

with tab_scan:
    st.title("Swing Options Screener")
    st.caption("Unadjusted daily; Finviz-style rel vol (time-adjusted).")

    # Run controls
    c1, c2, c3, c4, c5 = st.columns([1,1,1,1,4])
    with c1:
        st.markdown("&nbsp;")
        run_clicked = st.button("RUN", type="primary")
    with c2:
        res_days = st.number_input("Resistance lookback (days)", min_value=10, max_value=60, value=21, step=1)
    with c3:
        rr_min = st.number_input("Min RR to resistance", min_value=1.0, max_value=10.0, value=2.0, step=0.1)
    with c4:
        relvol_min = st.number_input("Min RelVol (time-adj)", min_value=0.5, max_value=5.0, value=1.10, step=0.05)
    with c5:
        st.write("")  # spacer

    if run_clicked:
        out = run_scan(
            tickers=None,           # your default list or sp500 via backend if you wire it
            res_days=res_days,
            rel_vol_min=relvol_min,
            relvol_median=False,
            rr_min=rr_min,
            stop_mode="safest",
            with_options=True,
            opt_days=30,
        )
        df = out['pass_df'].copy()

        if df.empty:
            st.info("No PASS tickers found on this run.")
        else:
            # main table first (sorted low→high price)
            df_display = df.sort_values(["Price","Ticker"]).reset_index(drop=True)
            st.dataframe(df_display, use_container_width=True)

            # WHY BUY (per-row expander)
            st.subheader("Why Buy (plain English)")
            for _, row in df_display.iterrows():
                with st.expander(f"{row['Ticker']} — details"):
                    st.markdown(_why_buy_text(row), unsafe_allow_html=True)

            # Copy to Google Sheets (pipe-separated)
            with st.expander("Copy for Google Sheets (pipe-separated)"):
                st.code(_pipe_copy_block(df_display), language="text")

    # Debugger section
    with st.expander("Debugger"):
        t = st.text_input("Ticker to diagnose (e.g., WMT, INTC)")
        if st.button("Explain Ticker"):
            if not t.strip():
                st.warning("Enter a ticker.")
            else:
                # we reuse the console-based diagnostics but also provide a human-readable summary:
                st.write(f"Console diagnostics for **{t.upper()}**:")
                explain_ticker(t.upper(), res_days=res_days, rel_vol_min=relvol_min, rr_min=rr_min, stop_mode="safest")
                st.caption("Scroll up to the app logs (Manage app ▸ Logs) if you don't see the console output here.")

with tab_hist:
    st.header("History of Passes")
    hist_path = "history/passes.csv"
    if os.path.exists(hist_path):
        hist_df = pd.read_csv(hist_path)
        st.dataframe(hist_df, use_container_width=True)
    else:
        st.info("No history yet — GitHub Actions scheduler will append runs here once you push the workflow.")

