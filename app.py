# app.py — UI for Swing Options Screener (Unadjusted)
# - Mobile-first: one RUN button, no sidebar required
# - Compact summary table
# - Per-row “WHY BUY” narrative
# - Full table expander (all columns)
# - Copy-to-Google-Sheets (pipe-separated) + download

import pandas as pd
import numpy as np
import streamlit as st

from swing_options_screener import run_scan  # uses your UNADJUSTED logic

# ---------- Page ----------
st.set_page_config(page_title="Swing Options Screener", layout="wide")
st.markdown("### Swing Options Screener (Unadjusted) — Live-ish")

# ---------- Filters (collapsible) ----------
with st.expander("Filters & Settings", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        relvol_min = st.number_input("Min RelVol (time-adjusted 63d)", value=1.10, min_value=0.5, step=0.05)
        rr_min     = st.number_input("Min R:R to Resistance", value=2.0, min_value=0.5, step=0.25)
    with c2:
        res_days   = st.number_input("Resistance lookback (days)", value=21, min_value=10, max_value=252, step=1)
        opt_days   = st.number_input("Target option DTE (≈ days)", value=30, min_value=7, max_value=90, step=1)
    with c3:
        stop_mode  = st.selectbox("Stop preference", options=["safest","structure"], index=0,
                                  help="safest = highest support; structure = pivot > swing > ATR")
    with st.popover("Advanced toggles"):
        relvol_median = st.checkbox("Use median (not mean) for RelVol base", value=False)
        with_options  = st.checkbox("Include options suggestion", value=True)

# ---------- Run ----------
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
    )

if "scan_params" not in st.session_state:
    st.info("Set filters (optional) and click **Run scan**.")
    st.stop()

p = st.session_state["scan_params"]

with st.spinner("Scanning…"):
    # IMPORTANT: do NOT pass history_basis — your screener doesn't support it
    out = run_scan(
        tickers=None,
        res_days=p["res_days"],
        rel_vol_min=p["rel_vol_min"],
        relvol_median=p["relvol_median"],
        rr_min=p["rr_min"],
        stop_mode=p["stop_mode"],
        with_options=p["with_options"],
        opt_days=p["opt_days"],
    )

df = out.get("pass_df", pd.DataFrame())
if df.empty:
    st.info("No PASS tickers right now.")
    st.stop()

# Ensure columns exist
def need(df, name, default=""):
    if name not in df.columns:
        df[name] = default

for c in [
    "Ticker","EvalDate","EntryTimeET","Price","TP","Resistance","RR_to_Res","RR_to_TP",
    "SupportType","SupportPrice","Risk$","TPReward$","TPReward%","ResReward$","ResReward%",
    "DailyATR","DailyCap","Hist21d_PassCount","Hist21d_Max%","Hist21d_Examples",
    "RelVol(TimeAdj63d)","Change%","Session","EntrySrc","VolSrc",
    "OptExpiry","BuyK","SellK","Width","DebitMid","DebitCons","MaxProfitMid","MaxProfitCons",
    "RR_Spread_Mid","RR_Spread_Cons","BreakevenMid","PricingNote"
]:
    need(df, c)

# ---------- Compact summary table ----------
compact_cols = ["Ticker","EvalDate","EntryTimeET","Price","TP","OptExpiry","BuyK","SellK"]
compact = df[compact_cols].copy()
for nc in ["Price","TP","BuyK","SellK"]:
    compact[nc] = pd.to_numeric(compact[nc], errors="coerce").round(2)

st.markdown("#### Passed — Summary")
st.dataframe(compact.sort_values(["Price","Ticker"]), use_container_width=True, hide_index=True)

# ---------- WHY BUY ----------
st.markdown("#### Why Buy (per pass)")

def fmt_pct(x):
    try: return f"{float(x):.2f}%"
    except: return ""

def _num(v, nd=2):
    try: return round(float(v), nd)
    except: return v

def why_buy_text(row: pd.Series) -> str:
    tkr   = row["Ticker"]
    price = _num(row["Price"])
    tp    = _num(row["TP"])
    res   = _num(row["Resistance"])
    rr_r  = _num(row["RR_to_Res"])
    rr_t  = _num(row["RR_to_TP"])
    sup_t = row["SupportType"]
    sup_p = _num(row["SupportPrice"])

    tp_d  = _num(row["TPReward$"])
    tp_p  = fmt_pct(row["TPReward%"])
    res_d = _num(row["ResReward$"])
    res_p = fmt_pct(row["ResReward%"])

    d_atr = _num(row["DailyATR"], 4)
    d_cap = _num(row["DailyCap"], 2)

    # History: your backend currently computes TP-basis req (from TPReward%)
    h_cnt = int(row["Hist21d_PassCount"] or 0)
    h_max = fmt_pct(row["Hist21d_Max%"])
    h_ex  = row["Hist21d_Examples"] or ""
    h_req = tp_p  # requirement is TP distance %

    oxp = row["OptExpiry"]
    bk  = row["BuyK"]
    sk  = row["SellK"]
    deb_mid   = row["DebitMid"]
    rr_sp_mid = row["RR_Spread_Mid"]

    parts = []
    parts.append(f"**{tkr}** — entry ~ **{price}**, TP **{tp}**, resistance **{res}**.")
    if oxp and pd.notna(bk) and pd.notna(sk):
        parts.append(f"Suggested **bull call** (≈{oxp}): **{bk}/{sk}** · debit≈{deb_mid} · spread RR≈{rr_sp_mid}.")

    parts.append(
        f"Rationale: price above support (**{sup_t} @ {sup_p}**) with **R:R to resistance ≈ {rr_r}:1** "
        f"(to TP ≈ {rr_t}:1). Upside to TP **${tp_d} ({tp_p})**; to resistance **${res_d} ({res_p})**."
    )

    if d_atr and d_cap:
        parts.append(f"Volatility: Daily ATR ≈ **{d_atr}**, implying ≈ **{d_cap}** over ~21 trading days.")

    parts.append(
        f"History check (21d, TP basis; need ≥ {h_req}): **{h_cnt}** instances passed; best ≈ **{h_max}**."
    )
    if h_ex:
        parts.append(f"Examples: {h_ex}")

    parts.append(
        f"_Session:_ **{row['Session']}** · _EntrySrc:_ **{row['EntrySrc']}** · _VolSrc:_ **{row['VolSrc']}**"
    )
    return "\n\n".join(parts)

for _, r in df.sort_values(["Price","Ticker"]).iterrows():
    label = f"WHY BUY — {r['Ticker']}  @ {r['Price']}  (TP {r['TP']})"
    with st.expander(label, expanded=False):
        st.markdown(why_buy_text(r))

# ---------- Full table ----------
with st.expander("Show complete table (all columns)", expanded=False):
    st.dataframe(df, use_container_width=True, hide_index=True)

# ---------- Copy to Google Sheets ----------
with st.expander("Copy for Google Sheets", expanded=False):
    cols = list(df.columns)
    header = "|".join(cols)
    lines = [header]
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row.get(c, "")
            if pd.isna(v): v = ""
            s = str(v).replace("|", "/")
            vals.append(s)
        lines.append("|".join(vals))
    psv = "\n".join(lines)

    st.text_area("Pipe-separated output", value=psv, height=200)
    st.download_button("Download .psv", data=psv.encode("utf-8"),
                       file_name="pass_tickers.psv", mime="text/plain")

