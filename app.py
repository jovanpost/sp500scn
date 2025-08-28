# app.py — Streamlit UI for Swing Options Screener + History reader (GitHub data branch)
# NOTE: set GITHUB_OWNER below to your GitHub username.

import io
import sys
from datetime import datetime
import pandas as pd
import streamlit as st

from swing_options_screener import (
    run_scan,          # returns {'pass_df': DataFrame}
    explain_ticker,    # prints CLI-style debug
    diagnose_ticker    # prints plain-english debug (numbers + reasons)
)

from app_history_reader import (
    fetch_history_index,
    fetch_run_csv,
    fetch_latest,
)

# === GitHub data branch config ===
GITHUB_OWNER = "jovanpost"   # <-- CHANGE THIS
GITHUB_REPO = "sp500scn"
GITHUB_DATA_BRANCH = "data"

# ----------- Small helpers (formatting) -----------
def _num(x, nd=2):
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return ""
        return f"{float(x):.{nd}f}"
    except Exception:
        return ""

def _usd(x, nd=2):
    s = _num(x, nd)
    return f"${s}" if s != "" else ""

def _pct(x, nd=2):
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return ""
        return f"{float(x):.{nd}f}%"
    except Exception:
        return ""

def _safe(val):
    return "" if val is None else str(val)

def _relvol_human(rv):
    # Show multiple: value, and "above avg by X%"
    if rv is None or pd.isna(rv) or rv == "":
        return "—"
    try:
        rvf = float(rv)
        if rvf <= 0:
            return f"{rvf:.2f}×"
        delta = (rvf - 1.0) * 100.0
        sign = "+" if delta >= 0 else ""
        return f"{rvf:.2f}× ({sign}{delta:.0f}%)"
    except Exception:
        return str(rv)

def _bold(s):
    return f"<strong>{s}</strong>"

def _mk_bullet(s):
    return f"<li style='margin: 0.15rem 0'>{s}</li>"
def build_why_buy_html(row: dict) -> str:
    tkr = _safe(row.get("Ticker",""))
    price = _usd(row.get("Price"))
    tp = _usd(row.get("TP"))
    res = _usd(row.get("Resistance"))
    rr_res = _safe(row.get("RR_to_Res",""))
    rr_tp = _safe(row.get("RR_to_TP",""))
    change_pct = _pct(row.get("Change%"))
    relvol = _relvol_human(row.get("RelVol(TimeAdj63d)"))

    tp_reward_val = row.get("TPReward$", None)
    tp_reward_pct_val = row.get("TPReward%", None)
    tp_reward = _usd(tp_reward_val)
    tp_reward_pct_s = _pct(tp_reward_pct_val)

    daily_atr = _usd(row.get("DailyATR", None), nd=4 if isinstance(row.get("DailyATR"), float) and row.get("DailyATR")<1 else 2)
    daily_cap = _usd(row.get("DailyCap"))

    hist_cnt = _safe(row.get("Hist21d_PassCount",""))
    hist_ex = _safe(row.get("Hist21d_Examples",""))
    support_type = _safe(row.get("SupportType",""))
    support_price = _usd(row.get("SupportPrice"))
    session = _safe(row.get("Session",""))
    entry_src = _safe(row.get("EntrySrc",""))
    vol_src = _safe(row.get("VolSrc",""))

    header = (
        f"{_bold(tkr)} looks attractive here: it last traded near {_bold(price)}. "
        f"We’re aiming for a take-profit around {_bold(tp)} (halfway to the recent high at {_bold(res)}), "
        f"giving a reward-to-risk of about {_bold(str(rr_res))}:1 to the recent high "
        f"and {_bold(str(rr_tp))}:1 to the take-profit."
    )

    bullets = []
    bullets.append(
        _mk_bullet(
            f"Momentum & liquidity: price is {_bold(change_pct)} on the day and "
            f"relative volume is {_bold(relvol)} vs the last 63 days (time-adjusted)."
        )
    )
    bullets.append(
        _mk_bullet(
            f"Distance to TP: about {_bold(tp_reward)} ({_bold(tp_reward_pct_s)}). "
            f"Daily ATR is around {_bold(daily_atr)}, which implies up to {_bold(daily_cap)} "
            f"of typical movement over ~21 trading days."
        )
    )
    bullets.append(
        _mk_bullet(
            f"1-month history check: {_bold(str(hist_cnt))} instances in the last year where a 21-trading-day move "
            f"met or exceeded the required % to TP. Examples: {_bold(hist_ex)}."
        )
    )
    bullets.append(
        _mk_bullet(
            f"Support: using {_bold(support_type)} around {_bold(support_price)} for risk management."
        )
    )
    bullets.append(
        _mk_bullet(
            f"Data basis: session {_bold(session)}, price source {_bold(entry_src)}, volume source {_bold(vol_src)}."
        )
    )

    bullets_html = "<ul style='padding-left: 1.1rem; margin-top: 0.35rem'>" + "".join(bullets) + "</ul>"
    return f"<div style='line-height:1.35'>{header}{bullets_html}</div>"


def build_google_sheet_psv(df: pd.DataFrame) -> str:
    # Mirror the PSV (pipe-separated) table with all columns we emit from run_scan
    cols = [
        'Ticker','EvalDate','Price','EntryTimeET','Change%','RelVol(TimeAdj63d)',
        'Resistance','TP','RR_to_Res','RR_to_TP','SupportType','SupportPrice','Risk$',
        'TPReward$','TPReward%','ResReward$','ResReward%','DailyATR','DailyCap',
        'Hist21d_PassCount','Hist21d_Max%','Hist21d_Examples','ResLookbackDays','Prices',
        'Session','EntrySrc','VolSrc',
        'OptExpiry','BuyK','SellK','Width','DebitMid','DebitCons','MaxProfitMid',
        'MaxProfitCons','RR_Spread_Mid','RR_Spread_Cons','BreakevenMid','PricingNote'
    ]
    # Ensure columns exist
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    lines = []
    lines.append("|".join(cols))
    for _, r in df[cols].iterrows():
        row_vals = []
        for c in cols:
            v = "" if pd.isna(r[c]) else str(r[c])
            row_vals.append(v.replace("|","/"))
        lines.append("|".join(row_vals))
    return "\n".join(lines)

# ----------- Page config & styles -----------
st.set_page_config(page_title="Swing Options Screener", page_icon="📈", layout="wide")

# Red RUN button theme (scoped CSS)
st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #d62828;
    color: white;
    font-weight: 700;
    border-radius: 6px;
    border: 1px solid #aa1f1f;
}
div.stButton > button:first-child:hover {
    background-color: #bb2424;
    color: #f8f9fa;
}
.kpi-card {
    border: 1px solid #e5e7eb; padding: 0.75rem 1rem; border-radius: 8px; background: #fff;
}
.explain-card {
    border: 1px solid #e5e7eb; padding: 0.9rem 1rem; border-radius: 8px; background: #fcfcff;
}
.small-note { color: #6b7280; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

st.title("📈 Swing Options Screener")

# ============ PRIMARY ACTIONS ============
col_run, col_info = st.columns([1,4])
with col_run:
    do_run = st.button("RUN", use_container_width=True)
with col_info:
    st.write("Runs the screener with current market state (intraday-aware, unadjusted data).")

# On click, execute scan and persist in session
if do_run or "last_df" not in st.session_state:
    out = run_scan(
        tickers=None,            # default universe (or sp500 if your backend is wired)
        res_days=21,
        rel_vol_min=1.10,
        rr_min=2.0,
        with_options=True,
        opt_days=30,
    )
    df = out.get("pass_df", pd.DataFrame())
    # Sort lowest price first (user preference)
    if not df.empty:
        sort_cols = [c for c in ["Price","Ticker"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols)
    st.session_state["last_df"] = df

# ============ RESULTS TABLE FIRST ============
df = st.session_state.get("last_df", pd.DataFrame())
st.subheader("Passes")
if df is None or df.empty:
    st.info("No PASS tickers found this run.")
else:
    # Clean numeric columns for display (don’t alter underlying data)
    show_cols = [
        'Ticker','EvalDate','Price','EntryTimeET','Change%','RelVol(TimeAdj63d)',
        'Resistance','TP','RR_to_Res','RR_to_TP','SupportType','SupportPrice','Risk$',
        'TPReward$','TPReward%','ResReward$','ResReward%','DailyATR','DailyCap',
        'Hist21d_PassCount','Hist21d_Max%','ResLookbackDays','Session'
    ]
    for c in show_cols:
        if c not in df.columns:
            df[c] = ""
    st.dataframe(df[show_cols], hide_index=True, use_container_width=True)

    # WHY BUY (expander per-row)
    st.subheader("Why buy (plain English)")
    for _, row in df.iterrows():
        with st.expander(f"{row.get('Ticker','')} — explain"):
            html = build_why_buy_html(row.to_dict())
            st.markdown(f"<div class='explain-card'>{html}</div>", unsafe_allow_html=True)

    # Copy for Google Sheets — hidden by default
    with st.expander("Copy for Google Sheets (pipe-separated)"):
        psv = build_google_sheet_psv(df)
        st.code(psv, language="text")

# ============ HISTORY TAB ============
st.markdown("---")
st.header("🗂️ History")
try:
    idx = fetch_history_index(GITHUB_OWNER, GITHUB_REPO, GITHUB_DATA_BRANCH)
    if idx.empty:
        st.info("No history yet. The scheduler will populate results after the first run.")
    else:
        st.subheader("Run Index")
        st.dataframe(idx.sort_values("RunTimeET"), use_container_width=True)

        st.subheader("Latest Run (details)")
        latest_df = fetch_latest(GITHUB_OWNER, GITHUB_REPO, GITHUB_DATA_BRANCH)
        if latest_df is not None and not latest_df.empty:
            st.dataframe(latest_df, use_container_width=True, hide_index=True)
        else:
            st.info("Latest run has no PASS rows.")
except Exception as e:
    st.warning(f"Could not load history from GitHub: {e}")

# ============ DEBUGGER (separate toggle) ============
st.markdown("---")
with st.expander("🛠️ Debug a ticker (plain English + numbers)"):
    dbg_col1, dbg_col2 = st.columns([2,1])
    with dbg_col1:
        dbg_ticker = st.text_input("Ticker to diagnose", value="", placeholder="e.g., WMT")
    with dbg_col2:
        do_dbg = st.button("Run Debug")

    if do_dbg and dbg_ticker.strip():
        # Capture printed output from diagnose_ticker (which prints to stdout)
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            # These kwargs mirror the main run
            diagnose_ticker(
                dbg_ticker.strip().upper(),
                res_days=21,
                rel_vol_min=1.10,
                relvol_median=False,
                rr_min=2.0,
                stop_mode="safest",
            )
        except Exception as e:
            sys.stdout = old_stdout
            st.error(f"Debugger error: {e}")
        else:
            sys.stdout = old_stdout
            out_text = buf.getvalue().strip()
            if not out_text:
                st.info("No output produced by debugger.")
            else:
                # Show raw and “explainer”
                st.markdown("**Raw diagnostic output:**")
                st.code(out_text, language="text")
                st.markdown(
                    "<div class='small-note'>Tip: if this says something like "
                    "<em>relvol_low_timeadj</em> or <em>not_up_on_day</em>, it will also print the exact numbers "
                    "(entry/prev_close/vol/relvol) right above.</div>",
                    unsafe_allow_html=True
                )

# Footer / timestamp
st.caption(f"Rendered at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

