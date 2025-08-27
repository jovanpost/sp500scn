# app.py — UI for Swing Options Screener (Unadjusted)
# - One-click RUN
# - Summary table
# - Per-row WHY BUY in your style (clean $, %, and volume comparison)
# - Full table expander + copy/download (pipe-separated for Sheets)

import pandas as pd
import numpy as np
import streamlit as st

from swing_options_screener import run_scan

st.set_page_config(page_title="Swing Options Screener", layout="wide")
st.markdown("### Swing Options Screener (Unadjusted) — Live-ish")

# ---------- Filters ----------
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
    with st.popover("Advanced"):
        relvol_median = st.checkbox("Use median (not mean) for RelVol base", value=False)
        with_options  = st.checkbox("Include options suggestion", value=True)

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
def ensure(df, name, default=""):
    if name not in df.columns:
        df[name] = default

needed = [
    "Ticker","EvalDate","EntryTimeET","Price","TP","Resistance",
    "RR_to_Res","RR_to_TP","SupportType","SupportPrice","Risk$",
    "TPReward$","TPReward%","ResReward$","ResReward%","Change%",
    "RelVol(TimeAdj63d)","TodayVol","RelVolBase63","RelVolExpectedByNow","SessionProgressPct",
    "DailyATR","DailyCap",
    "Hist21d_PassCount","Hist21d_Max%","Hist21d_Examples","Hist21d_Req%","Hist21d_CheckBasis",
    "Session","EntrySrc","VolSrc",
    "OptExpiry","BuyK","SellK","Width","DebitMid","DebitCons","MaxProfitMid","MaxProfitCons",
    "RR_Spread_Mid","RR_Spread_Cons","BreakevenMid","PricingNote"
]
for c in needed:
    ensure(df, c)

# ---------- Summary table ----------
compact_cols = ["Ticker","EvalDate","EntryTimeET","Price","TP","OptExpiry","BuyK","SellK"]
compact = df[compact_cols].copy()
for nc in ["Price","TP","BuyK","SellK"]:
    compact[nc] = pd.to_numeric(compact[nc], errors="coerce").round(2)

st.markdown("#### Passed — Summary")
st.dataframe(compact.sort_values(["Price","Ticker"]), use_container_width=True, hide_index=True)

# ---------- WHY BUY (your style) ----------
st.markdown("#### Why Buy (details)")

def fmt_money(x, nd=2):
    try:
        return "$" + f"{float(x):,.{nd}f}"
    except Exception:
        return ""

def fmt_pct(x, nd=2):
    try:
        return f"{float(x):.{nd}f}%"
    except Exception:
        return ""

def fmt_int(x):
    try:
        return f"{int(round(float(x))):,}"
    except Exception:
        return ""

def why_buy(row: pd.Series) -> str:
    tkr   = str(row["Ticker"])
    px    = fmt_money(row["Price"])
    tp    = fmt_money(row["TP"])
    res   = fmt_money(row["Resistance"])

    chg   = fmt_pct(row["Change%"])
    relv  = row["RelVol(TimeAdj63d)"]
    relv_txt = f"{float(relv):.2f}×" if str(relv) != "" else ""

    # Volume details
    today_vol   = row["TodayVol"]
    base_63     = row["RelVolBase63"]
    exp_by_now  = row["RelVolExpectedByNow"]
    prog_pct    = fmt_pct(row["SessionProgressPct"])
    vol_line = ""
    if str(today_vol) != "" and str(base_63) != "" and str(exp_by_now) != "":
        vol_line = f"Volume: {fmt_int(today_vol)} vs usual-by-now {fmt_int(exp_by_now)} (base {fmt_int(base_63)} · progress {prog_pct}) → RelVol {relv_txt}."
    elif relv_txt:
        vol_line = f"Volume: RelVol {relv_txt} (time-adjusted)."

    # Rewards / risk
    tp_d  = fmt_money(row["TPReward$"])
    tp_p  = fmt_pct(row["TPReward%"])
    res_d = fmt_money(row["ResReward$"])
    res_p = fmt_pct(row["ResReward%"])
    rr_r  = row["RR_to_Res"]
    rr_t  = row["RR_to_TP"]

    sup_t = str(row["SupportType"])
    sup_p = fmt_money(row["SupportPrice"])

    d_atr = fmt_money(row["DailyATR"], 4) if str(row["DailyATR"]) != "" else ""
    d_cap = fmt_money(row["DailyCap"], 2)

    # History
    h_req = fmt_pct(row["Hist21d_Req%"])
    h_cnt = int(row["Hist21d_PassCount"] or 0)
    h_max = fmt_pct(row["Hist21d_Max%"])
    h_ex  = str(row["Hist21d_Examples"] or "")

    # Options
    oxp = str(row["OptExpiry"] or "")
    bk  = row["BuyK"]; sk = row["SellK"]
    deb = row["DebitMid"]; rr_sp = row["RR_Spread_Mid"]
    opt_line = ""
    if oxp and str(bk) != "" and str(sk) != "":
        opt_line = f"Suggested bull call (~{oxp}): {fmt_money(bk,0).replace('$','')}/{fmt_money(sk,0).replace('$','')} · debit ≈ {fmt_money(deb)} · spread RR ≈ {rr_sp}."

    # Build narrative (tight, no fluff)
    lines = []
    lines.append(f"**{tkr}** — up **{chg}** today. Entry **{px}**, TP **{tp}**, resistance **{res}**.")
    if vol_line: lines.append(vol_line)
    if opt_line: lines.append(opt_line)
    lines.append(
        f"Setup: above **{sup_t} @ {sup_p}**. Reward/Risk ≈ **{rr_r}:1** to resistance, **{rr_t}:1** to TP. "
        f"Upside to TP **{tp_d} ({tp_p})**; to resistance **{res_d} ({res_p})**."
    )
    if d_atr and d_cap:
        lines.append(f"Volatility: Daily ATR ≈ **{d_atr}**, implying ≈ **{d_cap}** in ~21 trading days.")
    lines.append(
        f"History (21d, TP-basis; need ≥ {h_req}): **{h_cnt}** instances passed; best **{h_max}**."
    )
    if h_ex:
        lines.append(f"Examples: {h_ex}")
    lines.append(
        f"_Session:_ **{row['Session']}** · _EntrySrc:_ **{row['EntrySrc']}** · _VolSrc:_ **{row['VolSrc']}**"
    )
    return "\n\n".join(lines)

for _, r in df.sort_values(["Price","Ticker"]).iterrows():
    with st.expander(f"WHY BUY — {r['Ticker']}  @ {r['Price']}  (TP {r['TP']})", expanded=False):
        st.markdown(why_buy(r))

# ---------- Full table (all columns) ----------
with st.expander("Show complete table (all columns)", expanded=True):
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

