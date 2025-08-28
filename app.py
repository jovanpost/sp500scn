# ============================================================
# TABLE OF CONTENTS for app.py
#
#  1. Imports & Safe third-party glue
#  2. App constants (paths, titles, etc.)
#  3. Streamlit page config + global CSS
#  4. Small formatting helpers
#  5. WHY BUY explanation builder (plain English)
#  6. CSV helpers (latest pass file, outcomes)
#  7. Outcomes counters (robust to minimal/extended schemas)
#  8. Debugger (plain-English reasons with numbers)
#  9. UI – Tabs
# 10. TAB – Scanner (table → WHY BUY → Sheets export)
# 11. TAB – History & Outcomes
# 12. TAB – Debugger (plain English + numbers)
# ============================================================

# ============================================================
# 1. Imports & Safe third-party glue
# ============================================================
import os, glob, io, pandas as pd, streamlit as st
from datetime import datetime
import swing_options_screener as sos  # core engine

# ============================================================
# 2. App constants (paths, titles, etc.)
# ============================================================
PASS_DIR = "data/pass_logs"
HIST_DIR = "data/history"
OUT_FILE = os.path.join(HIST_DIR, "outcomes.csv")

st.set_page_config(page_title="Swing Options Scanner", layout="wide")

# ============================================================
# 3. Streamlit page config + global CSS
# ============================================================
st.markdown(
    """
    <style>
    div.stButton > button:first-child {
        background-color: red;
        color:white;
        font-weight: bold;
    }
    .whybuy {font-size: 16px; line-height: 1.5;}
    .debugbox {background:#222; color:#eee; padding:10px; border-radius:8px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 4. Small formatting helpers
# ============================================================
def _bold(x): return f"<b>{x}</b>"
def _usd(x, nd=2):
    try: return f"${x:.{nd}f}"
    except: return str(x)
def _pct(x):
    try: return f"{float(x):.2f}%"
    except: return str(x)
def _safe(x): return str(x) if x is not None else ""

# ============================================================
# 5. WHY BUY explanation builder (plain English)
# ============================================================
def build_why_buy_html(row: dict) -> str:
    tkr = _safe(row.get("Ticker",""))
    price = _usd(row.get("Price"))
    tp = _usd(row.get("TP"))
    res = _usd(row.get("Resistance"))
    rr_res = _safe(row.get("RR_to_Res",""))
    rr_tp = _safe(row.get("RR_to_TP",""))
    change_pct = _pct(row.get("Change%"))
    relvol = _safe(row.get("RelVol(TimeAdj63d)"))
    tp_reward = _usd(row.get("TPReward$", None))
    tp_reward_pct = _pct(row.get("TPReward%", None))
    daily_atr = _usd(row.get("DailyATR", None))
    daily_cap = _usd(row.get("DailyCap", None))
    hist_cnt = _safe(row.get("Hist21d_PassCount",""))
    hist_ex = _safe(row.get("Hist21d_Examples",""))
    support_type = _safe(row.get("SupportType",""))
    support_price = _usd(row.get("SupportPrice"))
    session = _safe(row.get("Session",""))
    entry_src = _safe(row.get("EntrySrc",""))
    vol_src = _safe(row.get("VolSrc",""))

    header = (
        f"{_bold(tkr)} is a buy because it last traded near {_bold(price)} "
        f"with a target around {_bold(tp)} (halfway to the recent high at {_bold(res)}). "
        f"This gives about {_bold(rr_res)}:1 reward-to-risk to the high and {_bold(rr_tp)}:1 to the target."
    )
    bullets = [
        f"- The stock is up {_bold(change_pct)} today with relative volume {_bold(relvol)} vs its 63-day average.",
        f"- The move to target is about {_bold(tp_reward)} ({_bold(tp_reward_pct)}). "
        f"Daily ATR is {_bold(daily_atr)}, so a typical month (~21 trading days) allows {_bold(daily_cap)} of movement.",
        f"- In the past year, there were {_bold(hist_cnt)} instances where the 21-day move beat this requirement. Examples: {hist_ex}.",
        f"- Support is {_bold(support_type)} at {_bold(support_price)}.",
        f"- Session={session}, EntrySrc={entry_src}, VolSrc={vol_src}."
    ]
    return "<div class='whybuy'>" + "<br>".join(bullets) + "</div>"

# ============================================================
# 6. CSV helpers (latest pass file, outcomes)
# ============================================================
def latest_pass_file():
    files = sorted(glob.glob(os.path.join(PASS_DIR, "*.csv")))
    return files[-1] if files else None

def load_outcomes():
    if os.path.exists(OUT_FILE):
        return pd.read_csv(OUT_FILE)
    return pd.DataFrame()

# ============================================================
# 7. Outcomes counters (robust to minimal/extended schemas)
# ============================================================
def outcomes_summary(dfh: pd.DataFrame):
    if dfh is None or dfh.empty:
        st.info("No outcomes yet.")
        return

    n = len(dfh)
    # Ensure we always have Series (never scalars) to avoid .sum() errors
    if "result_status" in dfh.columns:
        s_status = dfh["result_status"].astype(str)
    else:
        # Assume not yet settled if the column is missing
        s_status = pd.Series(["PENDING"] * n, index=dfh.index, dtype="string")

    if "hit" in dfh.columns:
        hit_mask = dfh["hit"].astype(bool)
    else:
        hit_mask = pd.Series([False] * n, index=dfh.index, dtype=bool)

    settled_mask = s_status.eq("SETTLED")
    pending_mask = ~settled_mask

    settled = int(settled_mask.sum())
    pending = int(pending_mask.sum())
    hits    = int((settled_mask & hit_mask).sum())
    misses  = settled - hits

    st.caption(f"Settled: {settled} • Hits: {hits} • Misses: {misses} • Pending: {pending}")

    # Nice sort if the columns exist; otherwise just show as-is
    sort_cols = [c for c in ["run_date", "ticker"] if c in dfh.columns]
    if sort_cols:
        # run_date desc if present
        ascending = [False if c == "run_date" else True for c in sort_cols]
        df_show = dfh.sort_values(sort_cols, ascending=ascending)
    else:
        df_show = dfh

    st.dataframe(df_show, use_container_width=True, height=min(600, 80 + 28 * len(df_show)))
    
# ============================================================
# 8. Debugger (plain-English reasons with numbers)
# ============================================================
def diagnose_ticker(ticker, **kwargs):
    df = sos._get_history(ticker)
    entry, prev_close, today_vol, src, entry_ts = sos.get_entry_prevclose_todayvol(df, ticker) if df is not None else (None, None, None, {}, None)
    row, reason = sos.evaluate_ticker(ticker, **kwargs)
    if reason is None:
        return f"{ticker} PASSED ✅", row
    return f"{ticker} FAILED ❌ because {reason}", {
        "entry": entry, "prev_close": prev_close, "today_vol": today_vol,
        "src": src, "entry_ts": entry_ts
    }

# ============================================================
# 9. UI – Tabs
# ============================================================
tab1, tab2, tab3 = st.tabs(["Scanner","History & Outcomes","Debugger"])

# ============================================================
# 10. TAB – Scanner (table → WHY BUY → Sheets export)
# ============================================================
with tab1:
    st.header("Swing Options Scanner")
    if st.button("RUN SCAN"):
        out = sos.run_scan(universe="sp500", with_options=True)
        df = out["pass_df"]
        if df.empty:
            st.warning("No PASS tickers found.")
        else:
            st.dataframe(df, use_container_width=True)
            for _, r in df.iterrows():
                with st.expander(f"WHY BUY {r['Ticker']}"):
                    st.markdown(build_why_buy_html(r), unsafe_allow_html=True)

# ============================================================
# 11. TAB – History & Outcomes
# ============================================================
with tab2:
    st.header("History & Outcomes")
    lastf = latest_pass_file()
    if lastf:
        st.success(f"Last run file: {lastf}")
        st.dataframe(pd.read_csv(lastf), use_container_width=True)
    dfh = load_outcomes()
    outcomes_summary(dfh)

# ============================================================
# 12. TAB – Debugger (plain English + numbers)
# ============================================================
with tab3:
    st.header("Debugger")
    dbg_ticker = st.text_input("Enter ticker to debug")
    if dbg_ticker:
        msg, details = diagnose_ticker(dbg_ticker.strip().upper())
        st.subheader(msg)
        st.json(details)

