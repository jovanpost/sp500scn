# app.py — UI for Swing Options Screener
# - Mobile-first, no sidebar
# - One big RUN button
# - Compact results table (Date/Time/Entry/TP/Options)
# - Per-row "WHY BUY" narrative
# - Full table expander (all columns)
# - Copy-to-Google-Sheets expander (pipe-separated + download)

import os
import io
import pandas as pd
import numpy as np
import streamlit as st

from swing_options_screener import run_scan  # uses UNADJUSTED logic & options sugg.

# -------------- Page setup
st.set_page_config(page_title="Swing Options Screener", layout="wide")

TITLE = "Swing Options Screener (Unadjusted) — Live-ish"
st.markdown(f"### {TITLE}")

# -------------- Controls (compact, inline)
with st.expander("Filters & Settings", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        relvol_min = st.number_input("Min RelVol (time-adjusted 63d)", value=1.10, min_value=0.5, step=0.05)
        rr_min     = st.number_input("Min R:R to Resistance", value=2.0, min_value=0.5, step=0.25)
    with c2:
        res_days   = st.number_input("Resistance lookback (days)", value=21, min_value=10, max_value=252, step=1)
        opt_days   = st.number_input("Target option DTE (≈ days)", value=30, min_value=7, max_value=90, step=1)
    with c3:
        hist_basis = st.selectbox("History 21d basis (for pass examples)", options=["tp", "res"], index=0,
                                  help="tp = entry→TP (halfway); res = entry→Resistance")
        stop_mode  = st.selectbox("Stop preference", options=["safest","structure"], index=0,
                                  help="safest = highest support; structure = pivot > swing > ATR")

    with st.popover("Advanced toggles"):
        relvol_median = st.checkbox("Use median (not mean) for RelVol base", value=False)
        with_options  = st.checkbox("Include options suggestion", value=True)

# -------------- RUN
run = st.button("▶️ Run scan", use_container_width=True, type="primary")

if run:
    st.session_state["scan_params"] = dict(
        res_days=int(res_days),
        rel_vol_min=float(relvol_min),
        relvol_median=bool(relvol_median),
        rr_min=float(rr_min),
        stop_mode=str(stop_mode),
        with_options=bool(with_options),
        opt_days=int(opt_days),
        history_basis=str(hist_basis),
    )

if "scan_params" in st.session_state:
    p = st.session_state["scan_params"]

    with st.spinner("Scanning…"):
        out = run_scan(
            tickers=None,  # default universe in the screener
            res_days=p["res_days"],
            rel_vol_min=p["rel_vol_min"],
            relvol_median=p["relvol_median"],
            rr_min=p["rr_min"],
            stop_mode=p["stop_mode"],
            with_options=p["with_options"],
            opt_days=p["opt_days"],
            history_basis=p["history_basis"],
        )

    df = out.get("pass_df", pd.DataFrame())
    if df.empty:
        st.info("No PASS tickers right now.")
        st.stop()

    # Ensure types and missing columns handled
    def col(df, name, default=""):
        if name not in df.columns: df[name] = default
        return df[name]

    # ---------- Compact summary table (per your spec)
    compact_cols = [
        "Ticker", "EvalDate", "EntryTimeET", "Price", "TP",
        "OptExpiry", "BuyK", "SellK"
    ]
    for cc in compact_cols:
        col(df, cc)

    compact = df[compact_cols].copy()
    # Format numeric columns
    for nc in ["Price","TP","BuyK","SellK"]:
        compact[nc] = pd.to_numeric(compact[nc], errors="coerce").round(2)

    st.markdown("#### Passed — Summary")
    st.dataframe(compact.sort_values(["Price","Ticker"]), use_container_width=True, hide_index=True)

    # ---------- WHY BUY narratives
    st.markdown("#### Why Buy (per pass)")
    def fmt_pct(x):
        try:
            return f"{float(x):.2f}%"
        except Exception:
            return ""

    def _safe(v, nd=2):
        try:
            f = float(v)
            return round(f, nd)
        except Exception:
            return v

    def build_why_buy(row: pd.Series) -> str:
        # Core fields
        tkr   = row.get("Ticker","")
        price = _safe(row.get("Price",""))
        tp    = _safe(row.get("TP",""))
        res   = _safe(row.get("Resistance",""))
        rr_r  = _safe(row.get("RR_to_Res",""))
        rr_t  = _safe(row.get("RR_to_TP",""))
        sup_t = row.get("SupportType","")
        sup_p = _safe(row.get("SupportPrice",""))

        # Distances
        tp_reward$ = _safe(row.get("TPReward$",""))
        tp_reward% = fmt_pct(row.get("TPReward%",""))
        res_reward$ = _safe(row.get("ResReward$",""))
        res_reward% = fmt_pct(row.get("ResReward%",""))

        # ATR capacities
        d_atr = _safe(row.get("DailyATR",""), 4)
        d_cap = _safe(row.get("DailyCap",""), 2)

        # History
        h_basis = row.get("Hist21d_CheckBasis","TP")
        h_req   = fmt_pct(row.get("Hist21d_Req%",""))
        h_cnt   = int(row.get("Hist21d_PassCount", 0) or 0)
        h_max   = fmt_pct(row.get("Hist21d_Max%",""))
        h_ex    = row.get("Hist21d_Examples","")

        # Options
        oxp = row.get("OptExpiry","")
        bk  = row.get("BuyK","")
        sk  = row.get("SellK","")
        deb_mid = row.get("DebitMid","")
        rr_sp_mid = row.get("RR_Spread_Mid","")

        # Session/debug
        sess = row.get("Session","")
        esrc = row.get("EntrySrc","")
        vsrc = row.get("VolSrc","")

        # Narrative
        txt = []
        txt.append(f"**{tkr}** — entry ~ **{price}**, TP **{tp}**, resistance **{res}**.")
        if oxp and pd.notna(bk) and pd.notna(sk):
            txt.append(f"Suggested **bull call** ≈{oxp}: **{bk}/{sk}** (debit≈{deb_mid}, spread RR≈{rr_sp_mid}).")

        txt.append(
            f"Rationale: price sits above support (**{sup_t} @ {sup_p}**) with **R:R to resistance ≈ {rr_r}:1** "
            f"(to TP ≈ {rr_t}:1). Upside to TP is **${tp_reward$} ({tp_reward%})**; "
            f"to resistance **${res_reward$} ({res_reward%})**."
        )

        if d_atr and d_cap:
            txt.append(
                f"Volatility capacity: Daily ATR ≈ **{d_atr}**, implying ≈ **{d_cap}** possible over ~21 trading days."
            )

        txt.append(
            f"History check (21d, basis **{h_basis.upper()}**, needs ≥ {h_req}): "
            f"**{h_cnt}** instances passed; best ≈ **{h_max}**."
        )
        if h_ex:
            txt.append(f"Examples: {h_ex}")

        txt.append(f"_Session:_ **{sess}** · _EntrySrc:_ **{esrc}** · _VolSrc:_ **{vsrc}**")
        return "\n\n".join(txt)

    for _, r in df.sort_values(["Price","Ticker"]).iterrows():
        with st.expander(f"WHY BUY — {r.get('Ticker','')}  @ {r.get('Price','')}  (TP {r.get('TP','')})", expanded=False):
            st.markdown(build_why_buy(r))

    # ---------- Full table (all columns)
    with st.expander("Show complete table (all columns)", expanded=False):
        st.dataframe(df, use_container_width=True, hide_index=True)

    # ---------- Copy to Google Sheets (pipe-separated) + download
    with st.expander("Copy for Google Sheets", expanded=False):
        # Build PSV with every column currently in df
        cols = list(df.columns)
        header = "|".join(cols)
        lines = [header]
        for _, row in df.iterrows():
            vals = []
            for c in cols:
                v = row.get(c, "")
                if pd.isna(v): v = ""
                s = str(v)
                # Replace pipe in data to avoid breaking the delimiter
                s = s.replace("|", "/")
                vals.append(s)
            lines.append("|".join(vals))
        psv = "\n".join(lines)

        st.caption("Select-all and copy. This format pastes cleanly into Google Sheets (use 'Split text to columns' with '|' if needed).")
        st.text_area("Pipe-separated output", value=psv, height=200)
        st.download_button("Download .psv", data=psv.encode("utf-8"), file_name="pass_tickers.psv", mime="text/plain")

else:
    st.info("Set filters (optional) and click **Run scan**.")

