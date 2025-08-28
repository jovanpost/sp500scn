# app.py — Streamlit UI for the Swing Options Screener
# - Big red RUN button
# - Main results table first
# - Per-row "WHY BUY" expander (plain-English)
# - Hidden "Copy to Google Sheets" block (pipe-separated)
# - History tab with robust outcomes summary (safe when columns missing)
# - Debug tab to explain a single ticker in plain-English with numbers

import os
import io
import glob
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st

# --- Import the backend ---
# We only call run_scan() and explain_ticker(); everything else is UI-formatting here
import swing_options_screener as sos

# -----------------------------
# UI helpers: formatting & HTML
# -----------------------------
def _num(x, nd=2):
    try:
        if x is None or (isinstance(x, float) and (np.isnan(x))):
            return ""
        return f"{float(x):.{nd}f}"
    except Exception:
        return ""

def _usd(x, nd=2):
    try:
        if x is None or (isinstance(x, float) and (np.isnan(x))):
            return ""
        return f"${float(x):.{nd}f}"
    except Exception:
        return ""

def _pct(x, nd=2):
    try:
        if x is None or (isinstance(x, float) and (np.isnan(x))):
            return ""
        return f"{float(x):.{nd}f}%"
    except Exception:
        return ""

def _safe(x):
    return "" if x is None else str(x)

def _bold(s):
    return f"<b>{_safe(s)}</b>"

def _relvol_human(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return ""
        return f"{float(x):.2f}×"
    except Exception:
        return ""

def _mk_li(s):
    return f"<li>{s}</li>"

def _mk_ul(items):
    return f"<ul style='margin-top:0.25rem;margin-bottom:0.25rem'>{''.join(items)}</ul>"

def _mk_p(s):
    return f"<p style='margin:.25rem 0'>{s}</p>"

def _mk_small(s):
    return f"<div style='font-size:0.9rem;opacity:.85'>{s}</div>"

def build_why_buy_html(row: dict) -> str:
    """
    Plain-English WHY BUY block with dollars, percents, ATR & rel vol, examples.
    """
    tkr = _safe(row.get("Ticker",""))
    price_s = _usd(row.get("Price"))
    tp_s    = _usd(row.get("TP"))
    res_s   = _usd(row.get("Resistance"))

    rr_res  = _safe(row.get("RR_to_Res",""))
    rr_tp   = _safe(row.get("RR_to_TP",""))

    change_pct_s = _pct(row.get("Change%"))
    relvol_s     = _relvol_human(row.get("RelVol(TimeAdj63d)"))

    tp_reward_s     = _usd(row.get("TPReward$"))
    tp_reward_pct_s = _pct(row.get("TPReward%"))

    daily_atr_val = row.get("DailyATR", None)
    # show 4 decimals if < 1 for readability
    daily_atr_s  = _usd(daily_atr_val, nd=(4 if isinstance(daily_atr_val, (int,float)) and daily_atr_val < 1 else 2))
    daily_cap_s  = _usd(row.get("DailyCap"))

    hist_cnt      = _safe(row.get("Hist21d_PassCount",""))
    hist_examples = _safe(row.get("Hist21d_Examples",""))

    support_type  = _safe(row.get("SupportType",""))
    support_price = _usd(row.get("SupportPrice"))

    session = _safe(row.get("Session",""))
    entry_src = _safe(row.get("EntrySrc",""))
    vol_src   = _safe(row.get("VolSrc",""))

    # Header in plain, convincing English
    header = (
        f"{_bold(tkr)} looks attractive: it’s trading near {_bold(price_s)}."
        f" We’re targeting a take-profit around {_bold(tp_s)} (about halfway toward the recent high at {_bold(res_s)})."
        f" The reward-to-risk is roughly {_bold(rr_res)}:1 to the recent high and {_bold(rr_tp)}:1 to the take-profit."
    )

    bullets = []

    # Momentum + liquidity
    bullets.append(_mk_li(
        f"Momentum & liquidity: the stock is {_bold(change_pct_s)} on the day"
        f" and relative volume is {_bold(relvol_s)} versus the last 63 sessions (time-adjusted)."
    ))

    # TP distance and ATR-based feasibility
    bullets.append(_mk_li(
        f"Distance to TP: {_bold(tp_reward_s)} ({_bold(tp_reward_pct_s)})."
        f" The daily ATR is around {_bold(daily_atr_s)},"
        f" which implies roughly {_bold(daily_cap_s)} of typical movement over ~21 trading days."
    ))

    # Support basis
    bullets.append(_mk_li(
        f"Support: using {_bold(support_type)} at {_bold(support_price)} as the protective stop."
    ))

    # History realism
    bullets.append(_mk_li(
        f"History check (last 12 months, 21-trading-day windows): there are {_bold(hist_cnt)} instances"
        f" where the forward 21-day move exceeded the required TP move. Examples: {hist_examples}"
    ))

    # Sources
    sourcing = _mk_small(
        f"Session: {_bold(session)} &nbsp;·&nbsp; EntrySrc: {_bold(entry_src)} &nbsp;·&nbsp; VolSrc: {_bold(vol_src)}"
    )

    html = (
        "<div style='line-height:1.35'>"
        + _mk_p(header)
        + _mk_ul(bullets)
        + sourcing +
        "</div>"
    )
    return html

# -----------------------------
# Page config & styles
# -----------------------------
st.set_page_config(page_title="Swing Options Screener", layout="wide")

# Red primary button styling (keep your look)
st.markdown("""
<style>
div.stButton > button[kind="primary"] {
  background-color: #d32f2f !important;
  color: white !important;
  border: 0 !important;
}
</style>
""", unsafe_allow_html=True)

st.title("Swing Options Screener")

tab1, tab2, tab3 = st.tabs(["🔎 Scanner", "📜 History / Outcomes", "🧪 Explain / Debug"])

# -----------------------------
# TAB 1 — SCANNER
# -----------------------------
with tab1:
    st.write("Click **RUN** to scan. Results show below. Per-row you can expand **WHY BUY** for plain-English rationale. A copy-to-Sheets block is hidden under each row.")

    run = st.button("RUN", type="primary", use_container_width=True)

    if run:
        with st.spinner("Scanning…"):
            # Use backend defaults; with_options=True so we always show option columns
            out = sos.run_scan(
                tickers=None,
                res_days=sos.RES_LOOKBACK_DEFAULT,
                rel_vol_min=sos.REL_VOL_MIN_DEFAULT,
                relvol_median=False,
                rr_min=sos.RR_MIN_DEFAULT,
                stop_mode="safest",
                with_options=True,
                opt_days=sos.TARGET_OPT_DAYS_DEFAULT,
            )
        df = out.get("pass_df", pd.DataFrame())

        if df.empty:
            st.info("No PASS tickers found.")
        else:
            # Main table first (show all columns; no truncation)
            st.dataframe(df, use_container_width=True, height=min(600, 80 + 28*len(df)))

            # Per-row WHY BUY and Copy blocks
            for idx, row in df.iterrows():
                with st.expander(f"WHY BUY — {row.get('Ticker','')}"):
                    st.markdown(build_why_buy_html(row), unsafe_allow_html=True)

                    # Copy-to-Sheets block (pipe-separated)
                    cols = list(df.columns)
                    # make TSV/PSV string for this one row
                    line = "|".join(str(row.get(c,"")).replace("|","/") for c in cols)
                    st.caption("Copy row (pipe-separated, Google Sheets-friendly):")
                    st.code("|".join(cols) + "\n" + line, language="text")

    st.caption("Tip: If a stock you expected to pass didn’t appear, open the **Explain / Debug** tab and run an explain on that ticker.")

# -----------------------------
# TAB 2 — HISTORY / OUTCOMES
# -----------------------------
with tab2:
    st.write("This shows stored historical runs (from the scheduled job) and summarized outcomes.")

    # Recent history runs (pass_*.csv)
    hist_files = sorted(glob.glob("data/history/pass_*.csv"))
    if not hist_files:
        st.info("No history yet. Once the scheduled job writes data/history/pass_YYYYMMDD.csv, it will appear here.")
    else:
        latest = hist_files[-1]
        st.subheader("Most recent run")
        st.caption(os.path.basename(latest))
        try:
            dfr = pd.read_csv(latest)
            st.dataframe(dfr, use_container_width=True, height=min(500, 80 + 28*len(dfr)))
        except Exception as e:
            st.error(f"Failed to read {latest}: {e}")

    st.divider()
    st.subheader("Outcomes summary")

    # outcomes.csv (can be missing initially)
    outcomes_path = "data/history/outcomes.csv"
    if not os.path.exists(outcomes_path):
        st.info("No outcomes yet. Once the scheduler writes data/history/outcomes.csv, it will appear here.")
    else:
        try:
            dfh = pd.read_csv(outcomes_path)
        except Exception as e:
            st.error(f"Failed to read outcomes: {e}")
            dfh = pd.DataFrame()

        if dfh.empty:
            st.info("Outcomes file is empty for now.")
        else:
            # ---- safe column helper ----
            def _scol(df: pd.DataFrame, name: str, dtype=None):
                if name in df.columns:
                    return df[name]
                return pd.Series([], dtype=(dtype if dtype is not None else object))

            s_status = _scol(dfh, "result_status", dtype="string")
            s_hit    = _scol(dfh, "tp_hit")

            # normalize booleans
            hit_mask = (s_hit == True) | (s_hit.astype("string").str.lower() == "true")

            settled_mask = (s_status == "SETTLED")
            pending_mask = (s_status != "SETTLED")

            settled = int(settled_mask.sum())
            pending = int(pending_mask.sum())
            hits    = int((settled_mask & hit_mask).sum())
            misses  = int((settled_mask & ~hit_mask).sum())

            st.caption(f"Settled: {settled} • Hits: {hits} • Misses: {misses} • Pending: {pending}")
            st.dataframe(dfh, use_container_width=True, height=min(600, 80 + 28*len(dfh)))

# -----------------------------
# TAB 3 — EXPLAIN / DEBUG
# -----------------------------
with tab3:
    st.write("Explain a single ticker: see exactly which gate failed (or why it passed).")
    dbg_ticker = st.text_input("Ticker to explain", "", placeholder="e.g., WMT")
    if st.button("Explain", use_container_width=True):
        if dbg_ticker.strip():
            # The backend's explain prints to stdout; we’ll run a small 'diagnose' wrapper for plain-English too
            # Use the same params the scanner uses
            params = dict(
                res_days=sos.RES_LOOKBACK_DEFAULT,
                rel_vol_min=sos.REL_VOL_MIN_DEFAULT,
                relvol_median=False,
                rr_min=sos.RR_MIN_DEFAULT,
                stop_mode="safest",
            )

            # Run the backend 'check' (prints to logs)
            sos.explain_ticker(dbg_ticker.strip().upper(), **params)

            # Also evaluate to build a plain-English summary with numbers
            row, reason = sos.evaluate_ticker(dbg_ticker.strip().upper(), **params)

            st.subheader(dbg_ticker.strip().upper())
            # Show raw gate result first
            if reason is None:
                st.success("PASS ✅")
                st.markdown(build_why_buy_html(row), unsafe_allow_html=True)
            else:
                st.error(f"FAIL ❌  {reason}")
                # Annotate common reasons with numbers where helpful
                if reason == "not_up_on_day":
                    st.caption("Gate: Up on day — failed. The current price is not above the previous close.")
                elif reason == "relvol_low_timeadj":
                    st.caption("Gate: Relative volume — failed. Time-adjusted rel vol is below the minimum threshold.")
                elif reason == "no_upside_to_resistance":
                    st.caption("Gate: Resistance — failed. Entry is already at/above the prior lookback high.")
                elif reason == "atr_capacity_short_vs_tp":
                    st.caption("Gate: ATR capacity — failed. 21-day ATR capacity is below the TP distance.")
                elif reason == "history_21d_zero_pass":
                    st.caption("Gate: History realism — failed. No 21-day window in the past 12 months exceeded the required TP move.")
                elif reason == "rr_to_res_below_min":
                    st.caption("Gate: Reward-to-risk — failed. R:R to resistance is below your minimum.")

            # Dump the backend’s current numbers (entry, vol sources) for audit
            try:
                dfh = sos._get_history(dbg_ticker.strip().upper())
                entry, prev_close, today_vol, src, entry_ts = sos.get_entry_prevclose_todayvol(dfh, dbg_ticker.strip().upper())
                now_et = datetime.now().strftime("%Y-%m-%d %H:%M:%S ET")
                age_min = ""
                if isinstance(entry_ts, datetime):
                    # the entry_ts from backend is tz-aware; for display we just show the string they provide too
                    pass
                st.code(
                    f"session={src.get('session')} entry_src={src.get('entry_src')} vol_src={src.get('vol_src')}\n"
                    f"entry_used={entry} prev_close_used={prev_close} today_vol={today_vol}\n"
                    f"EntryTimeET={sos._fmt_ts(entry_ts)} Now={now_et}"
                )
            except Exception:
                pass

# ------------- End of file -------------
