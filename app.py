# app.py — Streamlit UI for Swing Options Screener (UNADJUSTED)
# - Red RUN button
# - Settings hidden in a collapsible expander (defaults used otherwise)
# - Tabs: Scanner | History | Debugger
# - Scanner: PASS table -> WHY BUY expanders -> Copy-to-Sheets
# - Debugger: plain-English + raw numbers
# - History: reads history/pass_history.csv if available

import importlib
import traceback
from datetime import datetime
import os
import pandas as pd
import numpy as np
import streamlit as st

# ---------- Safe import of engine ----------
try:
    sos = importlib.import_module("swing_options_screener")
except Exception as e:
    st.error("Could not import swing_options_screener.py. Fix syntax/errors and redeploy.")
    st.exception(e)
    st.stop()

def _fn(name, required=True):
    fn = getattr(sos, name, None)
    if fn is None and required:
        st.error(f"Function '{name}' not found in swing_options_screener.py")
        st.stop()
    return fn

run_scan = _fn("run_scan", required=True)
evaluate_ticker = _fn("evaluate_ticker", required=True)
_get_history = _fn("_get_history", required=True)
get_entry_prevclose_todayvol = _fn("get_entry_prevclose_todayvol", required=True)
compute_relvol_time_adjusted = _fn("compute_relvol_time_adjusted", required=True)

# ---------- Format helpers ----------
def _safe(x): return "" if x is None else str(x)

def _usd(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)): return ""
    try: return f"${float(x):,.{nd}f}"
    except Exception: return _safe(x)

def _pct(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)): return ""
    try: return f"{float(x):.{nd}f}%"
    except Exception: return _safe(x)

def _bold(s): return f"<b>{_safe(s)}</b>"
def _mk_bullet(s): return f"<li style='margin: 0.15rem 0'>{s}</li>"

def _relvol_human(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)): return ""
    try: return f"{float(x):.2f}×"
    except Exception: return _safe(x)

# ---------- WHY BUY builder ----------
def build_why_buy_html(row: dict) -> str:
    tkr = _safe(row.get("Ticker",""))
    price = _usd(row.get("Price"))
    tp = _usd(row.get("TP"))
    res = _usd(row.get("Resistance"))
    rr_res = _safe(row.get("RR_to_Res",""))
    rr_tp = _safe(row.get("RR_to_TP",""))
    change_pct = _pct(row.get("Change%"))
    relvol = _relvol_human(row.get("RelVol(TimeAdj63d)"))
    tp_reward = _usd(row.get("TPReward$", None))
    tp_reward_pct_s = _pct(row.get("TPReward%", None))
    daily_atr_val = row.get("DailyATR", None)
    daily_atr = _usd(daily_atr_val, nd=4 if isinstance(daily_atr_val, float) and daily_atr_val < 1 else 2)
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
    if hist_cnt:
        bullets.append(
            _mk_bullet(
                f"1-month history check: {_bold(str(hist_cnt))} instances in the last year where a 21-trading-day move "
                f"met or exceeded the required % to TP. Examples: {_bold(hist_ex)}."
            )
        )
    bullets.append(_mk_bullet(f"Support: using {_bold(support_type)} around {_bold(support_price)} for risk management."))
    bullets.append(_mk_bullet(f"Data basis: session {_bold(session)}, price source {_bold(entry_src)}, volume source {_bold(vol_src)}."))

    bullets_html = "<ul style='padding-left: 1.1rem; margin-top: 0.35rem'>" + "".join(bullets) + "</ul>"
    return f"<div style='line-height:1.35; font-size:0.98rem'>{header}{bullets_html}</div>"

# ---------- Debugger ----------
def diagnose_ticker(ticker: str,
                    res_days: int,
                    rel_vol_min: float,
                    relvol_median: bool,
                    rr_min: float,
                    stop_mode: str):
    df = _get_history(ticker)
    if df is None or df.empty:
        return {"ok": False, "reason": "no_data", "details": "No daily history."}

    entry, prev_close, today_vol, src, entry_ts = get_entry_prevclose_todayvol(df, ticker)
    details = {
        "Ticker": ticker,
        "Session": src.get("session",""),
        "EntrySrc": src.get("entry_src",""),
        "VolSrc": src.get("vol_src",""),
        "Entry": entry,
        "PrevClose": prev_close,
        "TodayVol": today_vol,
        "EntryTimeET": entry_ts.strftime("%Y-%m-%d %H:%M:%S ET") if isinstance(entry_ts, datetime) else "",
    }

    if not (np.isfinite(prev_close) and np.isfinite(entry)):
        return {"ok": False, "reason": "bad_entry_prevclose", "details": details}

    change = (entry - prev_close) / prev_close
    details["Change%"] = change * 100.0
    if change <= 0:
        return {"ok": False, "reason": "not_up_on_day", "details": details}

    relvol = compute_relvol_time_adjusted(df, today_vol, use_median=relvol_median)
    details["RelVol(TimeAdj63d)"] = relvol
    if not (np.isfinite(relvol) and relvol >= rel_vol_min):
        return {"ok": False, "reason": "relvol_low_timeadj", "details": details}

    rolling_high = df['High'].rolling(window=res_days, min_periods=res_days).max()
    resistance = float(rolling_high.shift(1).iloc[-1]) if np.isfinite(rolling_high.shift(1).iloc[-1]) else np.nan
    details["Resistance"] = resistance
    if not (np.isfinite(resistance) and resistance > entry):
        return {"ok": False, "reason": "no_upside_to_resistance", "details": details}

    row, reason = evaluate_ticker(
        ticker,
        res_days=res_days,
        rel_vol_min=rel_vol_min,
        use_relvol_median=relvol_median,
        rr_min=rr_min,
        prefer_stop=stop_mode
    )
    if reason is None:
        return {"ok": True, "reason": None, "row": row, "details": details}
    return {"ok": False, "reason": reason, "details": details}

# ---------- Page setup & CSS ----------
st.set_page_config(page_title="Swing Options Screener", layout="wide")
st.markdown(
    """
    <style>
      /* Make ALL buttons red/white & bold so RUN is definitely red */
      .stButton > button {
          background-color: #d81b60 !important;
          color: #ffffff !important;
          border: none !important;
          font-weight: 700 !important;
      }
      .small-muted { color:#666; font-size:0.9rem; }
      .whybuy-card { padding: 0.35rem 0.75rem; border-radius: 6px; background: #fafafa; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Swing Options Screener (UNADJUSTED)")

tabs = st.tabs(["🔎 Scanner", "📜 History", "🧪 Debugger"])

# ---------------- Scanner ----------------
with tabs[0]:
    # Settings hidden by default
    with st.expander("Settings (optional)", expanded=False):
        colA, colB, colC, colD, colE = st.columns([1,1,1,1,1])
        res_days = colA.number_input("Resistance lookback (days)", min_value=10, max_value=60, value=21, step=1)
        rel_vol_min = colB.number_input("RelVol (time-adj) min", min_value=0.5, max_value=5.0, value=1.10, step=0.05)
        relvol_median = colC.checkbox("Use median for RelVol", value=False)
        rr_min = colD.number_input("Min R:R to resistance", min_value=1.0, max_value=10.0, value=2.0, step=0.25)
        stop_mode = colE.selectbox("Stop preference", ["safest","structure"], index=0)
    # Defaults if user never opens settings
    if "res_days" not in locals(): res_days = 21
    if "rel_vol_min" not in locals(): rel_vol_min = 1.10
    if "relvol_median" not in locals(): relvol_median = False
    if "rr_min" not in locals(): rr_min = 2.0
    if "stop_mode" not in locals(): stop_mode = "safest"

    run_clicked = st.button("RUN", use_container_width=True, key="run_main")

    results_state = st.session_state.get("last_results", None)
    if run_clicked:
        try:
            out = run_scan(
                tickers=None,
                res_days=res_days,
                rel_vol_min=rel_vol_min,
                relvol_median=relvol_median,
                rr_min=rr_min,
                stop_mode=stop_mode,
                with_options=True,
                opt_days=getattr(sos, "TARGET_OPT_DAYS_DEFAULT", 30),
            )
            df = out.get("pass_df", pd.DataFrame())
            # sort by current price ascending
            if "Price" in df.columns:
                df = df.sort_values(["Price","Ticker"], ascending=[True, True])
            st.session_state["last_results"] = {"df": df, "when": datetime.utcnow().isoformat()+"Z"}
            results_state = st.session_state["last_results"]
        except Exception:
            st.error("Scan failed.")
            st.code(traceback.format_exc())

    if results_state and isinstance(results_state.get("df", None), pd.DataFrame) and not results_state["df"].empty:
        df = results_state["df"].copy()
        st.subheader("PASS tickers")
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.subheader("Why Buy (per ticker)")
        for _, row in df.iterrows():
            t = row.get("Ticker","")
            with st.expander(f"WHY BUY — {t}"):
                html = build_why_buy_html(row.to_dict())
                st.markdown(f"<div class='whybuy-card'>{html}</div>", unsafe_allow_html=True)

        # Copy-to-Google Sheets (pipe-delimited)
        st.subheader("Copy to Google Sheets")
        cols = list(df.columns)
        lines = ["|".join(cols)]
        for _, r in df.iterrows():
            parts = []
            for c in cols:
                v = r.get(c, "")
                s = "" if pd.isna(v) else str(v)
                parts.append(s.replace("|","/"))
            lines.append("|".join(parts))
        tsv_text = "\n".join(lines)
        st.text_area("Pipe-separated output", tsv_text, height=200)
    else:
        st.info("No PASS tickers yet. Click RUN.")

# ---------------- History ----------------
with tabs[1]:
    st.write("This tab shows saved daily runs if a scheduler/write step is storing them.")
    hist_path = "history/pass_history.csv"
    if os.path.exists(hist_path):
        try:
            dfh = pd.read_csv(hist_path)
            st.dataframe(dfh, use_container_width=True, hide_index=True)
        except Exception:
            st.error("Found history/pass_history.csv but failed to read.")
            st.code(traceback.format_exc())
    else:
        st.warning("No history file found at history/pass_history.csv yet.")

# ---------------- Debugger ----------------
with tabs[2]:
    st.markdown("Plain-English reasons + raw numbers for a single ticker.")
    with st.expander("Explain a ticker", expanded=False):
        dbg_tkr = st.text_input("Ticker (e.g., WMT)", value="")
        # Use current settings for consistency
        dbg_res_days = res_days
        dbg_relvol_min = rel_vol_min
        dbg_relvol_median = relvol_median
        dbg_rr_min = rr_min
        dbg_stop_mode = stop_mode
        if st.button("Explain", key="dbg_explain"):
            if not dbg_tkr.strip():
                st.warning("Enter a ticker.")
            else:
                try:
                    d = diagnose_ticker(
                        dbg_tkr.strip().upper(),
                        res_days=dbg_res_days,
                        rel_vol_min=dbg_relvol_min,
                        relvol_median=dbg_relvol_median,
                        rr_min=dbg_rr_min,
                        stop_mode=dbg_stop_mode
                    )
                    if d["ok"]:
                        st.success(f"{dbg_tkr.upper()} PASSED all filters.")
                        st.json(d.get("row", {}))
                    else:
                        reason = d.get("reason","")
                        details = d.get("details", {})
                        reasons = {
                            "no_data": "No daily history available.",
                            "insufficient_rows": "Not enough rows to compute indicators.",
                            "bad_entry_prevclose": "Entry/previous close not available.",
                            "not_up_on_day": "Price is not up on the day.",
                            "relvol_low_timeadj": "Relative volume (time-adjusted) is below the minimum.",
                            "no_upside_to_resistance": "No headroom to the recent high (resistance).",
                            "atr_capacity_short_vs_tp": "ATR capacity is insufficient vs distance to TP.",
                            "insufficient_past_for_21d": "Not enough past data to run 21-day history check.",
                            "history_21d_zero_pass": "No 21-day forward windows meeting the required % to TP.",
                            "no_valid_support": "No valid support (SwingLow21/Pivot/ATR stop) below entry.",
                            "non_positive_risk": "Computed risk is non-positive.",
                            "rr_to_res_below_min": "Reward-to-risk to resistance is below the minimum."
                        }
                        st.error(f"FAIL — {reasons.get(reason, reason)}")
                        pretty = {k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in details.items()}
                        # Plain English summary with numbers
                        if reason == "not_up_on_day":
                            st.write(f"Price change today: {_pct(pretty.get('Change%'))} (must be > 0).")
                        elif reason == "relvol_low_timeadj":
                            st.write(f"RelVol(time-adj 63d): {pretty.get('RelVol(TimeAdj63d)')} (min: {dbg_relvol_min}).")
                        elif reason == "no_upside_to_resistance":
                            st.write(f"Resistance: {pretty.get('Resistance')}, Entry: {pretty.get('Entry')} (resistance must be above entry).")
                        st.json(pretty)
                except Exception:
                    st.error("Debugger crashed.")
                    st.code(traceback.format_exc())

