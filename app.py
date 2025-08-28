# app.py — Streamlit UI for Swing Options Screener (UNADJUSTED)
import os, glob, io, textwrap
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st

import swing_options_screener as sos  # your module (evaluate_ticker, run_scan, etc.)

st.set_page_config(page_title="Swing Options Screener (UNADJUSTED)", layout="wide")

# ----------- tiny style: red primary button & compact table -----------
st.markdown("""
<style>
div.stButton > button[kind="primary"] { background: #e63946; color: white; border: 0; }
div.stButton > button[kind="primary"]:hover { background: #c92e3f; }
.dataframe td, .dataframe th { font-size: 0.90rem; }
</style>
""", unsafe_allow_html=True)

# ----------- small format helpers -----------
def _safe(x, default=""):
    if x is None: return default
    return x

def _usd(x, nd=2):
    try:
        if pd.isna(x): return ""
        return f"${float(x):,.{nd}f}"
    except Exception:
        return ""

def _pct(x, nd=2):
    try:
        if pd.isna(x): return ""
        return f"{float(x):.{nd}f}%"
    except Exception:
        return ""

def _relvol_human(x):
    try:
        if pd.isna(x): return ""
        f = float(x)
        # convert multiplier to % over typical pace
        return f"+{int(round((f-1.0)*100))}%"
    except Exception:
        return ""

def _bold(s): return f"**{s}**"
def _mk_li(s): return f"- {s}"

# ----------- WHY BUY builder (plain English) -----------
def build_why_buy_html(row: dict) -> str:
    tkr = _safe(row.get("Ticker",""))
    price = _usd(row.get("Price"))
    tp = _usd(row.get("TP"))
    res = _usd(row.get("Resistance"))
    rr_res = _safe(row.get("RR_to_Res",""))
    rr_tp  = _safe(row.get("RR_to_TP",""))
    change = _pct(row.get("Change%"))
    relvol = _relvol_human(row.get("RelVol(TimeAdj63d)"))
    daily_atr = row.get("DailyATR", None)
    daily_atr_s = _usd(daily_atr, nd=4 if isinstance(daily_atr,float) and daily_atr < 1 else 2)
    daily_cap = _usd(row.get("DailyCap"))
    tp_reward = _usd(row.get("TPReward$"))
    tp_reward_pct = _pct(row.get("TPReward%"))
    support_type = _safe(row.get("SupportType",""))
    support_price = _usd(row.get("SupportPrice"))
    hist_cnt = _safe(row.get("Hist21d_PassCount",""))
    hist_ex  = _safe(row.get("Hist21d_Examples",""))
    entry_src = _safe(row.get("EntrySrc",""))
    session = _safe(row.get("Session",""))
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S ET")

    header = (
        f"{_bold(tkr)} is a buy via the vertical call spread {_bold(_safe(row.get('BuyK','')))} / "
        f"{_bold(_safe(row.get('SellK','')))} expiring {_bold(_safe(row.get('OptExpiry','')))}, "
        f"because it recently topped around {_bold(res)} (resistance) and now trades near "
        f"{_bold(price)} (current price). That makes a target at {_bold(tp)} feel realistic."
    )

    bullets = [
        _mk_li(
            f"**Reward vs. risk:** from support at {_bold(support_price)} "
            f"to resistance is about {_bold(str(rr_res))}:1 "
            f"(to the nearer TP it’s {_bold(str(rr_tp))}:1)."),
        _mk_li(
            f"**Move needed to TP:** {_bold(tp_reward)} (≈ {_bold(tp_reward_pct)})."),
        _mk_li(
            f"**Volatility runway (ATR):** daily ATR is {_bold(daily_atr_s)}, so a typical month "
            f"(~21 trading days) allows up to {_bold(daily_cap)} of movement."),
        _mk_li(
            f"**Today’s tone & volume:** price is {_bold(change)} on the day and "
            f"volume is running about {_bold(relvol)} vs typical pace (time-adjusted)."),
        _mk_li(
            f"**History check (21 trading days):** need about {_bold(tp_reward_pct)} to reach TP. "
            f"Over the past year, {_bold(hist_cnt)} separate 21-day windows met or exceeded that move. "
            f"Examples: {hist_ex}")
    ]

    html = f"""
<div style="line-height:1.35">
<p>{header}</p>
<p><strong>Why this setup makes sense</strong></p>
<ul>
<li>{bullets[0][2:]}</li>
<li>{bullets[1][2:]}</li>
<li>{bullets[2][2:]}</li>
<li>{bullets[3][2:]}</li>
<li>{bullets[4][2:]}</li>
</ul>
<p><em>Data as of {now}. Session: {session}; EntrySrc: {entry_src}.</em></p>
</div>
"""
    return html

# ----------- Debugger (plain English with numbers) -----------
def diagnose_ticker(ticker: str, **kwargs) -> str:
    # Grab helpers from module safely
    _get_history = getattr(sos, "_get_history", None)
    get_eptv     = getattr(sos, "get_entry_prevclose_todayvol", None)
    comp_relvol  = getattr(sos, "compute_relvol_time_adjusted", None)
    eval_tick    = getattr(sos, "evaluate_ticker", None)
    if not (_get_history and get_eptv and comp_relvol and eval_tick):
        return "Debugger unavailable: internal helpers not found."

    df = _get_history(ticker)
    if df is None or df.empty:
        return f"**{ticker}** — no daily history available."

    entry, prev_close, today_vol, src, entry_ts = get_eptv(df, ticker)
    change = (entry - prev_close) / prev_close if (entry and prev_close) else np.nan
    relvol = comp_relvol(df, today_vol, use_median=kwargs.get("use_relvol_median", False))

    row, reason = eval_tick(ticker, **kwargs)

    src_s = f"session={src.get('session')} · entry_src={src.get('entry_src')} · vol_src={src.get('vol_src')}"
    base = (
        f"**{ticker}** — price {_usd(entry)} vs prior close {_usd(prev_close)} "
        f"({ _pct(change*100) if not pd.isna(change) else '' }). "
        f"Today vol ≈ {today_vol:,.0f} (relVol {_relvol_human(relvol)}). "
        f"[{src_s}]"
    )
    if reason is None and row:
        return base + "\n\n**Result:** PASS — meets all gates."
    else:
        # Friendly reasons
        reasons = {
            "not_up_on_day": "Not up on the day.",
            "relvol_low_timeadj": "Relative volume too low for the time of day.",
            "no_upside_to_resistance": "No reasonable upside to resistance.",
            "atr_capacity_short_vs_tp": "ATR capacity too small to reach TP.",
            "history_21d_zero_pass": "No similar 21-day windows reached the needed move.",
            "no_valid_support": "No valid support below entry.",
            "non_positive_risk": "Computed risk was non-positive.",
            "rr_to_res_below_min": "Reward-to-risk to resistance below minimum.",
            "bad_entry_prevclose": "Could not compute a valid entry or prior close.",
            "insufficient_rows": "Not enough daily rows.",
            "no_data": "No data for ticker.",
        }
        friendly = reasons.get(reason, reason)
        return base + f"\n\n**Result:** FAIL — {friendly}"

# =================== UI ===================
st.title("Swing Options Screener\n(UNADJUSTED)")

tab_scan, tab_hist, tab_dbg = st.tabs(["🔎 Scanner", "🧺 History", "🧪 Debugger"])

with tab_scan:
    st.caption("Click **Run Screener** to evaluate the universe and show PASS tickers. Sorted by price (low → high).")

    col_btn, _ = st.columns([1,5])
    run = col_btn.button("Run Screener", type="primary")
    if run:
        with st.info("Running screener… this may take a bit on first run."):
            out = sos.run_scan(
                tickers=None,            # use default / S&P per your module
                res_days=21,
                rel_vol_min=1.10,
                relvol_median=False,
                rr_min=2.0,
                stop_mode="safest",
                with_options=True,
                opt_days=30,
            )

        df = out.get("pass_df", pd.DataFrame())
        if df.empty:
            st.warning("No PASS tickers found (or CSV not produced).")
        else:
            # Sort by current price, lowest first
            if "Price" in df.columns:
                df = df.sort_values("Price", ascending=True).reset_index(drop=True)

            st.subheader("PASS tickers")
            st.dataframe(df, use_container_width=True, hide_index=True)

            with st.expander("Copy table (pipe-delimited for Google Sheets)"):
                cols = list(df.columns)
                s = io.StringIO()
                s.write("|".join(cols) + "\n")
                for _, r in df.iterrows():
                    line = "|".join(str(_safe(r.get(c,""))).replace("|","/") for c in cols)
                    s.write(line + "\n")
                st.code(s.getvalue(), language="text")

            st.subheader("Explain each PASS (WHY BUY)")
            for _, r in df.iterrows():
                tkr = r.get("Ticker","")
                with st.expander(f"WHY BUY — {tkr}"):
                    st.markdown(build_why_buy_html(r.to_dict()), unsafe_allow_html=True)

with tab_hist:
    st.caption("This tab shows saved daily runs and their outcomes.")

    hist_files = sorted(glob.glob("data/history/pass_*.csv"))
    if not hist_files:
        st.warning("No history file found at history/pass_*.csv yet.")
    else:
        frames = []
        for f in hist_files:
            try:
                d = pd.read_csv(f)
                d["RunFile"] = os.path.basename(f)
                frames.append(d)
            except Exception:
                continue
        if not frames:
            st.warning("Could not read any history CSVs.")
        else:
            hist_df = pd.concat(frames, ignore_index=True)

            # Merge outcomes if present
            out_path = "data/outcomes/outcomes.csv"
            if os.path.exists(out_path):
                outc = pd.read_csv(out_path)
                # Ensure keys exist
                if "RunFile" not in hist_df.columns:
                    hist_df["RunFile"] = ""
                merged = hist_df.merge(outc, on=["Ticker","EvalDate","RunFile"], how="left", suffixes=("","_out"))
            else:
                merged = hist_df.copy()

            # KPI for closed windows only
            if "WindowClosed" in merged.columns and "Outcome" in merged.columns:
                closed = merged[merged["WindowClosed"] == True]
                if not closed.empty:
                    yes = (closed["Outcome"] == "YES").sum()
                    total = len(closed)
                    win = (yes/total)*100.0
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Closed windows", f"{total}")
                    c2.metric("Hit (YES)", f"{yes}")
                    c3.metric("Win rate", f"{win:.1f}%")
                else:
                    st.info("No closed windows yet — outcomes will appear as trades finish their window.")

            # Put outcome columns up front if available
            front_cols = [c for c in ["Outcome","TargetType","TargetLevel","HitDate","WindowEnd","WindowClosed","MaxHigh"] if c in merged.columns]
            rest = [c for c in merged.columns if c not in front_cols]
            merged = merged[front_cols + rest]

            st.dataframe(merged, use_container_width=True, hide_index=True)

with tab_dbg:
    st.caption("Type a ticker to see **plain-English** explanation of why it passes or which gate fails (with numbers).")
    dbg_ticker = st.text_input("Ticker", placeholder="e.g., WMT, INTC, MOS")
    if st.button("Explain"):
        if not dbg_ticker.strip():
            st.warning("Enter a ticker first.")
        else:
            html = diagnose_ticker(
                dbg_ticker.strip().upper(),
                res_days=21,
                rel_vol_min=1.10,
                use_relvol_median=False,
                rr_min=2.0,
                prefer_stop="safest",
            )
            st.markdown(html)

