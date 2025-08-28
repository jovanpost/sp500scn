# app.py
# Streamlit UI for Swing Options Screener (UNADJUSTED)
# - Minimal Scanner: red RUN button only
# - PASS table first, then WHY BUY expanders, then Sheets copy (hidden)
# - Debugger expander (plain English + numbers)
# - History tab: reads data/history/outcomes.csv robustly

import os
import io
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st

# ---- Import backend functions from your module ----
from swing_options_screener import (
    run_scan,
    evaluate_ticker,
    _get_history,
    get_entry_prevclose_todayvol,
    compute_relvol_time_adjusted,
    _now_et,
)

# ---------- Page / Styles ----------
st.set_page_config(page_title="Swing Options Screener", page_icon="📈", layout="wide")

st.markdown("""
<style>
/* Keep the classic red RUN button */
div.stButton > button[kind="primary"] {
  background-color: #CC2B2B !important;
  color: white !important;
  border: none !important;
}
/* Copy block and WHY BUY cosmetics */
.small-muted { color:#6b7280; font-size:0.9rem; }
.why p { margin: 0.2rem 0 0.6rem 0; }
.why li { margin: 0.15rem 0; }
.why .tag { background:#eef2ff; padding:2px 6px; border-radius:4px; font-size:0.85rem; }
</style>
""", unsafe_allow_html=True)

# ---------- Tiny format helpers ----------
def _usd(x, nd=2):
    try:
        v = float(x);  return f"${v:,.{nd}f}"
    except Exception:
        return ""

def _pct(x, nd=2):
    try:
        v = float(x);  return f"{v:.{nd}f}%"
    except Exception:
        return ""

def _safe(x): return "" if x is None else str(x)
def _bold(x): return f"**{_safe(x)}**"

def _relvol_human(x):
    try: return f"{float(x):.2f}×"
    except Exception: return ""

def _mk_ul(items):
    if not items: return ""
    return "<ul>" + "".join(f"<li>{it}</li>" for it in items) + "</ul>"

# ---------- WHY BUY renderer ----------
def build_why_buy_html(row: dict) -> str:
    tkr   = _safe(row.get("Ticker",""))
    price = _usd(row.get("Price"))
    tp    = _usd(row.get("TP"))
    res   = _usd(row.get("Resistance"))
    rr_res = _safe(row.get("RR_to_Res",""))
    rr_tp  = _safe(row.get("RR_to_TP",""))

    change_pct = _pct(row.get("Change%"))
    relvol     = _relvol_human(row.get("RelVol(TimeAdj63d)"))

    tp_reward_val     = row.get("TPReward$", None)
    tp_reward_pct_val = row.get("TPReward%", None)
    tp_reward     = _usd(tp_reward_val)
    tp_reward_pct = _pct(tp_reward_pct_val)

    atr_val = row.get("DailyATR", None)
    daily_atr = _usd(atr_val, nd=4 if isinstance(atr_val,(int,float)) and atr_val < 1 else 2)
    daily_cap = _usd(row.get("DailyCap"))

    hist_cnt = _safe(row.get("Hist21d_PassCount",""))
    hist_ex  = _safe(row.get("Hist21d_Examples",""))
    support_type  = _safe(row.get("SupportType",""))
    support_price = _usd(row.get("SupportPrice"))
    session  = _safe(row.get("Session",""))
    entry_src= _safe(row.get("EntrySrc",""))
    vol_src  = _safe(row.get("VolSrc",""))

    header = (
        f"{_bold(tkr)} looks attractive: it last traded near {_bold(price)}. "
        f"We’re targeting a take-profit around {_bold(tp)} (halfway toward the recent high at {_bold(res)}), "
        f"so reward-to-risk is about {_bold(rr_res)}:1 to the recent high and {_bold(rr_tp)}:1 to the take-profit."
    )

    bullets = [
        f"Momentum & liquidity: price is {_bold(change_pct)} on the day and "
        f"relative volume is {_bold(relvol)} versus the 63-day average (time-adjusted).",
        f"Distance to TP: {_bold(tp_reward)} ({_bold(tp_reward_pct)}). "
        f"Daily ATR is around {_bold(daily_atr)}, which implies roughly {_bold(daily_cap)} "
        f"of typical movement across ~21 trading days.",
        f"Support: using {_bold(support_type)} at {_bold(support_price)}; this is the stop reference.",
        f"History check: across the last year, there were {_bold(hist_cnt)} monthly (21-trading-day) moves "
        f"that met or exceeded the TP distance requirement. Examples: {hist_ex}.",
        f"<span class='tag'>Session</span> {session} "
        f"<span class='tag'>EntrySrc</span> {entry_src} "
        f"<span class='tag'>VolSrc</span> {vol_src}",
    ]
    return "<div class='why' style='line-height:1.35'>" + f"<p>{header}</p>{_mk_ul(bullets)}</div>"

# ---------- Debugger (plain English + numbers) ----------
def diagnose_ticker_plain_english(ticker: str,
                                  res_days: int = 21,
                                  rel_vol_min: float = 1.10,
                                  use_relvol_median: bool = False,
                                  rr_min: float = 2.0,
                                  prefer_stop: str = "safest") -> str:
    df = _get_history(ticker)
    if df is None or df.empty:
        return f"**{ticker}** — No historical data from Yahoo Finance."

    entry, prev_close, today_vol, src, entry_ts = get_entry_prevclose_todayvol(df, ticker)
    now = _now_et()
    age_min = None
    if isinstance(entry_ts, datetime):
        age_min = round((now - entry_ts).total_seconds()/60.0, 2)

    row, reason = evaluate_ticker(
        ticker, res_days=res_days, rel_vol_min=rel_vol_min,
        use_relvol_median=use_relvol_median, rr_min=rr_min, prefer_stop=prefer_stop
    )

    lines = []
    lines.append(f"**{ticker} — diagnostic**")
    lines.append(f"- Session: `{src.get('session')}` · EntrySrc: `{src.get('entry_src')}` · VolSrc: `{src.get('vol_src')}`")
    lines.append(f"- Entry used: {_usd(entry)} · Prev close used: {_usd(prev_close)} · Today volume: {int(today_vol) if today_vol==today_vol else 'NA'}")
    if entry_ts:
        lines.append(f"- Entry time (ET): {entry_ts.strftime('%Y-%m-%d %H:%M:%S ET')} · Data age: {age_min} min")

    if reason is None and row:
        lines += [
            "",
            "**PASS ✅** (all gates satisfied)",
            f"- Change on day: {_pct(row.get('Change%'))} · RelVol(63d, time-adj): {_relvol_human(row.get('RelVol(TimeAdj63d)'))}",
            f"- Resistance: {_usd(row.get('Resistance'))} · TP: {_usd(row.get('TP'))}",
            f"- RR to Resistance: **{row.get('RR_to_Res')}** : 1 · RR to TP: **{row.get('RR_to_TP')}** : 1",
            f"- Support type: **{row.get('SupportType')}** at {_usd(row.get('SupportPrice'))} · Risk: {_usd(row.get('Risk$'))}",
            f"- TP distance: {_usd(row.get('TPReward$'))} ({_pct(row.get('TPReward%'))})",
            f"- Daily ATR: {_usd(row.get('DailyATR'), nd=4 if isinstance(row.get('DailyATR'), (int,float)) and row.get('DailyATR')<1 else 2)} "
            f"· 21-day capacity: {_usd(row.get('DailyCap'))}",
            f"- History: pass_count={row.get('Hist21d_PassCount')} · Examples: {row.get('Hist21d_Examples')}",
        ]
    else:
        lines += ["", "**FAIL ❌** — first blocking gate:"]
        reason_map = {
            "no_data": "No Yahoo data.",
            "insufficient_rows": "Not enough daily bars to compute resistance/ATR reliably.",
            "bad_entry_prevclose": "Could not determine intraday/daily entry or previous close.",
            "not_up_on_day": "Price is not up vs previous close.",
            "relvol_low_timeadj": "Relative volume is below the required threshold.",
            "no_upside_to_resistance": "Entry is not below the prior lookback high (no upside room).",
            "atr_capacity_short_vs_tp": "ATR capacity (~21 days) is not enough for the TP distance.",
            "insufficient_past_for_21d": "Not enough past data to test 21-trading-day history.",
            "history_21d_zero_pass": "No 21-day windows hit the required TP distance in the past year.",
            "no_valid_support": "Couldn't find a valid support below entry.",
            "non_positive_risk": "Support stop is not below entry (non-positive risk).",
            "rr_to_res_below_min": "Reward-to-risk to resistance is below the minimum.",
        }
        lines.append(f"- Code: `{reason}` · Meaning: {reason_map.get(reason, 'See code for details.')}")
        # Enrich with numbers
        try:
            change = (entry - prev_close) / prev_close if (entry==entry and prev_close==prev_close and prev_close!=0) else np.nan
        except Exception:
            change = np.nan
        if change==change: lines.append(f"- Change on day (computed): {_pct(change*100)}")
        relv = compute_relvol_time_adjusted(df, today_vol, use_median=use_relvol_median)
        if relv==relv: lines.append(f"- RelVol(63d, time-adj): {_relvol_human(relv)} (min required: {rel_vol_min:.2f}×)")
        try:
            rolling_high = df['High'].rolling(window=res_days, min_periods=res_days).max()
            resistance = float(rolling_high.shift(1).iloc[-1])
            lines.append(f"- Resistance (prior {res_days}-day high excl. today): {_usd(resistance)}")
        except Exception:
            pass

    return "\n".join(lines)

# ---------- Tabs ----------
tab_scan, tab_history = st.tabs(["🔴 Scanner", "📜 History"])

# ================== SCANNER ==================
with tab_scan:
    st.title("Swing Options Screener (UNADJUSTED)")

    # Minimal UI: just the red RUN button
    run_clicked = st.button("RUN", type="primary")

    pass_df = None
    if run_clicked:
        # Use defaults you specified in the backend
        out = run_scan(
            tickers=None,            # S&P 500 live (UI doesn't expose toggles)
            res_days=21,
            rel_vol_min=1.10,
            relvol_median=False,
            rr_min=2.0,
            stop_mode="safest",
            with_options=True,
            opt_days=30,
        )
        pass_df = out.get("pass_df", pd.DataFrame())

        st.subheader("PASS results")
        if pass_df is None or pass_df.empty:
            st.info("No PASS tickers found.")
        else:
            if "Price" in pass_df.columns:
                pass_df = pass_df.sort_values(["Price","Ticker"], ascending=[True, True])
            st.dataframe(pass_df, use_container_width=True, hide_index=True)

            st.subheader("WHY BUY (plain English)")
            for _, r in pass_df.iterrows():
                tkr = str(r.get("Ticker",""))
                with st.expander(f"{tkr} — Why this makes sense"):
                    st.markdown(build_why_buy_html(r), unsafe_allow_html=True)
                    if all(c in pass_df.columns for c in ["OptExpiry","BuyK","SellK","Width","DebitMid","DebitCons","RR_Spread_Mid","RR_Spread_Cons"]):
                        exp = str(r.get("OptExpiry",""))
                        buyk = r.get("BuyK",""); sellk = r.get("SellK",""); width = r.get("Width","")
                        rr_m = r.get("RR_Spread_Mid",""); rr_c = r.get("RR_Spread_Cons","")
                        st.caption(
                            f"Options idea (closest to 30d): {tkr} "
                            f"{_usd(buyk,0)} / {_usd(sellk,0)} (width {_usd(width,0)}) · "
                            f"RR(mid)={rr_m} · RR(cons)={rr_c} · Exp: {exp}"
                        )

            with st.expander("Copy for Google Sheets"):
                cols_all = list(pass_df.columns)
                buf = io.StringIO()
                buf.write("|".join(cols_all) + "\n")
                for _, rr in pass_df[cols_all].iterrows():
                    row_txt = "|".join("" if pd.isna(rr[c]) else str(rr[c]).replace("|","/") for c in cols_all)
                    buf.write(row_txt + "\n")
                txt = buf.getvalue()
                st.code(txt, language=None)
                st.download_button("Download .psv", data=txt, file_name="pass_tickers_unadjusted.psv", mime="text/plain")

    # Debugger expander (separate, not filters)
    with st.expander("🧪 Debug a ticker"):
        dbg_t = st.text_input("Ticker", value="WMT").strip().upper()
        if st.button("Explain"):
            pe = diagnose_ticker_plain_english(dbg_t)
            st.markdown(f"```\n{pe}\n```")

# ================== HISTORY ==================
with tab_history:
    st.title("History & Outcomes")

    # Resolve outcomes.csv robustly from this file’s directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    outcomes_fp = os.path.join(base_dir, "data", "history", "outcomes.csv")

    if not os.path.exists(outcomes_fp):
        st.info("No history yet. Once the scheduled job writes `data/history/outcomes.csv`, it will appear here.")
    else:
        try:
            dfh = pd.read_csv(outcomes_fp)
        except Exception as e:
            st.error(f"Could not read history file: {e}")
            dfh = None

        if dfh is not None and not dfh.empty:
            # Parse dates if present
            for c in ["run_date","expiry","hit_date"]:
                if c in dfh.columns:
                    dfh[c] = pd.to_datetime(dfh[c], errors="coerce")

            # KPIs
            total   = len(dfh)
            settled = int((dfh.get("result_status","")=="SETTLED").sum())
            pending = int((dfh.get("result_status","")=="PENDING").sum())
            hit     = int((dfh.get("tp_hit","")=="YES").sum())
            miss    = int((dfh.get("tp_hit","")=="NO").sum())
            c1,c2,c3,c4,c5 = st.columns(5)
            c1.metric("Total rows", total)
            c2.metric("Settled", settled)
            c3.metric("Pending", pending)
            c4.metric("TP Hit (YES)", hit)
            c5.metric("TP Miss (NO)", miss)

            st.subheader("Pending (not yet expired)")
            st.dataframe(
                dfh[dfh["result_status"]=="PENDING"].sort_values(["run_date","ticker"]),
                use_container_width=True, hide_index=True
            )
            st.subheader("Settled (expired) — results")
            st.dataframe(
                dfh[dfh["result_status"]=="SETTLED"].sort_values(["run_date","ticker"]),
                use_container_width=True, hide_index=True
            )
        else:
            st.info("History file is present but empty.")
