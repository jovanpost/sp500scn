# app.py — Streamlit UI for Swing Options Screener
# - Robust import of swing_options_screener (module-level import, no fragile named imports)
# - Preserves red RUN button flow
# - Shows full pass table first, then per-row WHY BUY (collapsible)
# - Copy-to-Google-Sheets TSV output
# - Separate Debugger toggle with plain-English + raw numbers
# ------------------------------------------------------------

import importlib
import traceback
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st

# ---------- Safe import of the engine module ----------
try:
    sos = importlib.import_module("swing_options_screener")
except Exception as e:
    st.error("Could not import swing_options_screener.py. Make sure the file exists and has no syntax errors.")
    st.exception(e)
    st.stop()

def _fn(name, required=True):
    fn = getattr(sos, name, None)
    if fn is None and required:
        st.error(f"Function '{name}' not found in swing_options_screener.py")
        st.stop()
    return fn

run_scan = _fn("run_scan", required=True)  # main engine
evaluate_ticker = _fn("evaluate_ticker", required=True)
_get_history = _fn("_get_history", required=True)
get_entry_prevclose_todayvol = _fn("get_entry_prevclose_todayvol", required=True)
compute_relvol_time_adjusted = _fn("compute_relvol_time_adjusted", required=True)

# ---------- Small format helpers ----------
def _safe(x):
    return "" if x is None else str(x)

def _usd(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    try:
        return f"${float(x):,.{nd}f}"
    except Exception:
        return _safe(x)

def _pct(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    try:
        return f"{float(x):.{nd}f}%"
    except Exception:
        return _safe(x)

def _bold(s):
    return f"<b>{_safe(s)}</b>"

def _mk_bullet(s):
    return f"<li style='margin: 0.15rem 0'>{s}</li>"

def _relvol_human(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    try:
        return f"{float(x):.2f}×"
    except Exception:
        return _safe(x)

# ---------- WHY BUY builder (plain English, clean HTML) ----------
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
    return f"<div style='line-height:1.35; font-size:0.98rem'>{header}{bullets_html}</div>"

# ---------- Debugger (plain English + raw) ----------
def diagnose_ticker(ticker: str,
                    res_days: int,
                    rel_vol_min: float,
                    relvol_median: bool,
                    rr_min: float,
                    stop_mode: str):
    # Recompute inputs to explain failure reasons
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
    if change <= 0:
        details["Change%"] = change * 100.0
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

    # If it passed to here, call the real evaluator to capture the exact reason (if any)
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

# ---------- Minimal page setup ----------
st.set_page_config(page_title="Swing Options Screener", layout="wide")
st.markdown(
    """
    <style>
    .run-btn > button {
        background-color: #d81b60 !important; /* red-ish */
        color: white !important;
        border: none !important;
        font-weight: 700 !important;
    }
    .small-muted { color:#666; font-size:0.9rem; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Swing Options Screener (UNADJUSTED)")

# Controls (kept simple & visible)
colA, colB, colC, colD, colE = st.columns([1,1,1,1,1])
res_days = colA.number_input("Resistance lookback (days)", min_value=10, max_value=60, value=21, step=1)
rel_vol_min = colB.number_input("RelVol (time-adj) min", min_value=0.5, max_value=5.0, value=1.10, step=0.05)
relvol_median = colC.checkbox("Use median for RelVol", value=False)
rr_min = colD.number_input("Min R:R to resistance", min_value=1.0, max_value=10.0, value=2.0, step=0.25)
stop_mode = colE.selectbox("Stop preference", ["safest","structure"], index=0)

# RUN
run_col = st.container()
with run_col:
    c1, c2 = st.columns([1,5])
    run_clicked = c1.button("RUN", use_container_width=True, key="run", help="Run the scanner now")
    st.write("")

results_state = st.session_state.get("last_results", None)

if run_clicked:
    try:
        out = run_scan(
            tickers=None,                # engine will use defaults/universe internally
            res_days=res_days,
            rel_vol_min=rel_vol_min,
            relvol_median=relvol_median,
            rr_min=rr_min,
            stop_mode=stop_mode,
            with_options=True,
            opt_days=getattr(sos, "TARGET_OPT_DAYS_DEFAULT", 30),
        )
        df = out.get("pass_df", pd.DataFrame())
        st.session_state["last_results"] = {"df": df, "when": datetime.utcnow().isoformat()+"Z"}
        results_state = st.session_state["last_results"]
    except Exception:
        st.error("Scan failed.")
        st.code(traceback.format_exc())

# ---------- Show results ----------
if results_state and isinstance(results_state.get("df", None), pd.DataFrame) and not results_state["df"].empty:
    df = results_state["df"].copy()

    # Sort by current price ascending (as requested)
    if "Price" in df.columns:
        df = df.sort_values(["Price","Ticker"], ascending=[True, True])

    st.subheader("PASS tickers")
    st.dataframe(df, use_container_width=True, hide_index=True)

    # WHY BUY per row
    st.subheader("Why Buy (per ticker)")
    for _, row in df.iterrows():
        t = row.get("Ticker","")
        with st.expander(f"WHY BUY — {t}"):
            html = build_why_buy_html(row.to_dict())
            st.markdown(html, unsafe_allow_html=True)

    # Copy-to-Sheets TSV (pipe-separated)
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
    st.info("No PASS tickers yet. Adjust filters or click RUN.")

# ---------- Debugger toggle ----------
with st.expander("Debugger"):
    st.markdown("Enter a ticker to see **plain-English** reasons and raw numbers for pass/fail.")
    dbg_tkr = st.text_input("Ticker", value="")
    if st.button("Explain", key="explain_btn"):
        if not dbg_tkr.strip():
            st.warning("Enter a ticker, e.g., WMT")
        else:
            try:
                d = diagnose_ticker(
                    dbg_tkr.strip().upper(),
                    res_days=res_days,
                    rel_vol_min=rel_vol_min,
                    relvol_median=relvol_median,
                    rr_min=rr_min,
                    stop_mode=stop_mode
                )
                if d["ok"]:
                    st.success(f"{dbg_tkr.upper()} PASSED all filters.")
                    st.json(d.get("row", {}))
                else:
                    reason = d.get("reason","")
                    details = d.get("details", {})
                    # Plain-English mapping
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
                    # Numbers
                    pretty = {k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in details.items()}
                    st.json(pretty)
            except Exception:
                st.error("Debugger crashed.")
                st.code(traceback.format_exc())

