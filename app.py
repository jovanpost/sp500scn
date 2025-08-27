# app.py
import math
from datetime import datetime, time
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# === bring in your screener logic ===
from swing_options_screener import (
    run_scan,
    evaluate_ticker,
    _get_history,
    get_entry_prevclose_todayvol,
    compute_relvol_time_adjusted,
    _atr_from_ohlc,
    _recent_pivot_low,
    # constants (we reuse them so our debugger matches the gates)
    RES_LOOKBACK_DEFAULT,
    REL_VOL_MIN_DEFAULT,
    RR_MIN_DEFAULT,
    SUPPORT_LOOKBACK_DAYS,
    PIVOT_K,
    PIVOT_LOOKBACK_DAYS,
    ATR_STOP_MULT,
)

# -----------------
# formatting helpers
# -----------------
def _num(x, d=2):
    try:
        if pd.isna(x): return ""
        return f"{float(x):,.{d}f}"
    except Exception:
        return ""

def _usd(x, d=2):
    try:
        if pd.isna(x): return ""
        return f"${float(x):,.{d}f}"
    except Exception:
        return ""

def _pct(x, d=2):
    try:
        if pd.isna(x): return ""
        return f"{float(x):.{d}f}%"
    except Exception:
        return ""

def _xmult(x, d=2):
    try:
        if pd.isna(x): return ""
        return f"{float(x):.{d}f}×"
    except Exception:
        return ""

def _is_open_now_et() -> Tuple[bool, datetime, datetime, datetime]:
    """Market open status in ET."""
    now = datetime.now(tz=pd.Timestamp.now(tz="America/New_York").tz)
    open_t = now.replace(hour=9, minute=30, second=0, microsecond=0)
    close_t = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return (open_t < now < close_t), now, open_t, close_t

# ---------------------------------------
# Plain-English debugger for a single ticker
# ---------------------------------------
def diagnose_ticker(
    ticker: str,
    res_days: int = RES_LOOKBACK_DEFAULT,
    rel_vol_min: float = REL_VOL_MIN_DEFAULT,
    rr_min: float = RR_MIN_DEFAULT,
    prefer_stop: str = "safest",
    use_relvol_median: bool = False,
) -> Dict[str, Any]:
    """
    Re-run the same gates step-by-step, return a dict with numbers + English text.
    """
    out = {"ticker": ticker}

    df = _get_history(ticker)
    if df is None or df.empty:
        out["status"] = "FAIL"
        out["reason_code"] = "no_data"
        out["explain"] = "No price history available to evaluate this ticker."
        return out

    if len(df) < max(22, res_days + 1):
        out["status"] = "FAIL"
        out["reason_code"] = "insufficient_rows"
        out["explain"] = f"Not enough rows ({len(df)}) to compute a {res_days}-day resistance and 21-day history checks."
        return out

    entry, prev_close, today_vol, src, entry_ts = get_entry_prevclose_todayvol(df, ticker)
    out["session"] = src.get("session", "")
    out["entry_src"] = src.get("entry_src", "")
    out["vol_src"] = src.get("vol_src", "")
    out["entry_ts"] = entry_ts

    if not (np.isfinite(entry) and np.isfinite(prev_close)):
        out["status"] = "FAIL"
        out["reason_code"] = "bad_entry_prevclose"
        out["explain"] = "Could not reliably determine today’s entry price or yesterday’s close."
        return out

    change = (entry - prev_close) / prev_close
    out["entry"] = float(entry)
    out["prev_close"] = float(prev_close)
    out["change_pct"] = float(change * 100)

    # Gate 1: up on the day
    if change <= 0:
        out["status"] = "FAIL"
        out["reason_code"] = "not_up_on_day"
        out["explain"] = f"The stock is not up today: change = {_pct(out['change_pct'])} vs requirement > 0%."
        return out

    # RelVol (time adjusted)
    relvol = compute_relvol_time_adjusted(df, today_vol, use_median=use_relvol_median)
    # Also surface components for clarity
    base_series = df["Volume"].iloc[-64:-1]
    avg_63 = float((base_series.median() if use_relvol_median else base_series.mean())) if not base_series.empty else float("nan")

    is_open, now_et, open_t, close_t = _is_open_now_et()
    if now_et <= open_t:
        progress = 0.0
    elif now_et >= close_t:
        progress = 1.0
    else:
        progress = (now_et - open_t).total_seconds() / (close_t - open_t).total_seconds()
        progress = max(progress, 1 / 390)

    out["today_vol"] = float(today_vol) if np.isfinite(today_vol) else float("nan")
    out["avg_63"] = avg_63
    out["session_progress"] = float(progress)
    out["relvol"] = float(relvol) if np.isfinite(relvol) else float("nan")

    if not (np.isfinite(relvol) and relvol >= rel_vol_min):
        # compute expected volume by now for the explanation
        expected_by_now = avg_63 * progress if (np.isfinite(avg_63) and progress > 0) else float("nan")
        out["status"] = "FAIL"
        out["reason_code"] = "relvol_low_timeadj"
        out["explain"] = (
            f"Today’s trading volume is too weak. Relative volume = "
            f"{_xmult(out['relvol'])} vs threshold ≥ {_xmult(rel_vol_min)}. "
            f"(63-day {'median' if use_relvol_median else 'average'}: {_num(avg_63,0)}; "
            f"expected by now: {_num(expected_by_now,0)}; actual today: {_num(today_vol,0)})."
        )
        return out

    # Resistance: prior N-day high (exclude today)
    rolling_high = df["High"].rolling(window=res_days, min_periods=res_days).max()
    resistance = rolling_high.shift(1).iloc[-1]
    out["resistance"] = float(resistance) if np.isfinite(resistance) else float("nan")
    if not (np.isfinite(resistance) and resistance > entry):
        out["status"] = "FAIL"
        out["reason_code"] = "no_upside_to_resistance"
        out["explain"] = (
            f"No upside: prior {res_days}-day high (resistance) = {_usd(resistance)}, "
            f"which is not above entry {_usd(entry)}."
        )
        return out

    tp = entry + 0.5 * (resistance - entry)
    tp_reward = tp - entry
    res_reward = resistance - entry
    out["tp"] = float(tp)
    out["tp_reward"] = float(tp_reward)
    out["res_reward"] = float(res_reward)
    out["tp_req_pct"] = float((tp_reward / entry) * 100.0)

    # ATR capacity (daily only, consistent with your script)
    daily_atr = _atr_from_ohlc(df, 14)
    daily_cap = (daily_atr * 21.0) if np.isfinite(daily_atr) else 0.0
    out["daily_atr"] = float(daily_atr) if np.isfinite(daily_atr) else float("nan")
    out["daily_cap"] = float(daily_cap)

    if not (daily_cap > tp_reward):
        out["status"] = "FAIL"
        out["reason_code"] = "atr_capacity_short_vs_tp"
        out["explain"] = (
            f"Volatility looks too small: TP needs {_usd(tp_reward)} ({_pct(out['tp_req_pct'])}), "
            f"but 14-day ATR projects only about {_usd(daily_cap)} over ~21 trading days."
        )
        return out

    # Support selection (same logic as screener)
    swing_low_21 = float(df["Low"].iloc[-SUPPORT_LOOKBACK_DAYS:].min()) if len(df) >= SUPPORT_LOOKBACK_DAYS else float("nan")
    pivot_low = _recent_pivot_low(df, k=PIVOT_K, lookback_days=PIVOT_LOOKBACK_DAYS)
    atr_stop = entry - ATR_STOP_MULT * (daily_atr if np.isfinite(daily_atr) else 0.0)

    supports = {
        "SwingLow21": swing_low_21 if np.isfinite(swing_low_21) and swing_low_21 < entry else np.nan,
        "PivotLow": pivot_low if np.isfinite(pivot_low) and pivot_low < entry else np.nan,
        f"ATR{ATR_STOP_MULT}x": atr_stop if np.isfinite(atr_stop) and atr_stop < entry else np.nan,
    }
    candidates = [(k, v) for k, v in supports.items() if np.isfinite(v)]

    if not candidates:
        out["status"] = "FAIL"
        out["reason_code"] = "no_valid_support"
        out["explain"] = "No valid support found below the entry price (swing low, pivot, or ATR-based)."
        return out

    if prefer_stop == "structure":
        for k in ["PivotLow", "SwingLow21", f"ATR{ATR_STOP_MULT}x"]:
            if np.isfinite(supports.get(k, np.nan)):
                support_type = k
                stop = supports[k]
                break
        else:
            support_type, stop = max(candidates, key=lambda kv: kv[1])
    else:
        # 'safest': choose the deepest (largest) support below entry
        support_type, stop = max(candidates, key=lambda kv: kv[1])

    risk = entry - stop
    out["support_type"] = support_type
    out["stop"] = float(stop)
    out["risk"] = float(risk)

    if risk <= 0:
        out["status"] = "FAIL"
        out["reason_code"] = "non_positive_risk"
        out["explain"] = (
            f"Calculated stop {_usd(stop)} is not below entry {_usd(entry)} — risk would be ≤ 0."
        )
        return out

    rr_to_res = res_reward / risk
    rr_to_tp = tp_reward / risk
    out["rr_to_res"] = float(rr_to_res)
    out["rr_to_tp"] = float(rr_to_tp)

    if rr_to_res < rr_min:
        out["status"] = "FAIL"
        out["reason_code"] = "rr_to_res_below_min"
        out["explain"] = (
            f"Risk-to-reward to resistance is too low: R = {_usd(risk)}, Reward = {_usd(res_reward)} "
            f"→ R:R = {rr_to_res:.2f}:1 vs required ≥ {rr_min:.2f}:1."
        )
        return out

    # If you get here, it passes all gates.
    out["status"] = "PASS"
    out["reason_code"] = ""
    out["explain"] = (
        f"PASS — Up on day ({_pct(out['change_pct'])}); RelVol {_xmult(relvol)} ≥ {_xmult(rel_vol_min)}; "
        f"clear upside to resistance {_usd(resistance)}; TP at {_usd(tp)}; "
        f"ATR capacity {_usd(daily_cap)} > TP distance {_usd(tp_reward)}; "
        f"support ({support_type}) at {_usd(stop)} gives R:R to resistance ≈ {rr_to_res:.2f}:1."
    )
    return out

# =========
# UI Layout
# =========
st.set_page_config(page_title="Swing Options Screener", page_icon="📈", layout="wide")

st.title("📈 Swing Options Screener (Unadjusted, Finviz-style)")

with st.container():
    cols = st.columns([1, 1, 1, 1, 1, 2, 2])
    with cols[0]:
        res_days = st.number_input("Resistance lookback (days)", min_value=10, max_value=60, value=RES_LOOKBACK_DEFAULT, step=1)
    with cols[1]:
        rr_min = st.number_input("Min R:R to resistance", min_value=1.0, max_value=5.0, value=RR_MIN_DEFAULT, step=0.1)
    with cols[2]:
        rel_vol_min = st.number_input("Min RelVol (time-adj)", min_value=0.5, max_value=3.0, value=REL_VOL_MIN_DEFAULT, step=0.05)
    with cols[3]:
        stop_mode = st.selectbox("Stop preference", ["safest", "structure"], index=0)
    with cols[4]:
        relvol_median = st.checkbox("RelVol: use 63-day median", value=False)
    with cols[5]:
        opt_days = st.slider("Options target days", min_value=20, max_value=45, value=30, step=1)
    with cols[6]:
        st.write("")  # spacer

# RUN button (kept red as you wanted)
run_clicked = st.button("RUN", type="primary", help="Run the scan with the settings above", use_container_width=True)

# Storage for last results
if "last_pass_df" not in st.session_state:
    st.session_state["last_pass_df"] = pd.DataFrame()

if run_clicked:
    # Run the scan
    out = run_scan(
        tickers=None,                 # use your DEFAULT_TICKERS (or S&P flow inside the screener if configured)
        res_days=res_days,
        rel_vol_min=rel_vol_min,
        relvol_median=relvol_median,
        rr_min=rr_min,
        stop_mode=stop_mode,
        with_options=True,
        opt_days=opt_days,
    )
    st.session_state["last_pass_df"] = out.get("pass_df", pd.DataFrame())

pass_df = st.session_state["last_pass_df"]

# ====== Results table (always first) ======
st.subheader("Results")
if pass_df is not None and not pass_df.empty:
    # Sort by current price ascending (your request)
    if "Price" in pass_df.columns:
        pass_df = pass_df.sort_values(["Price", "Ticker"], ascending=[True, True])

    st.dataframe(
        pass_df,
        use_container_width=True,
        hide_index=True,
    )

    # ===== WHY BUY per-row (expander) =====
    st.markdown("### WHY BUY (per ticker)")
    for _, row in pass_df.iterrows():
        tkr = row["Ticker"]
        with st.expander(f"Why {tkr}?"):
            # Pull fields safely
            price = _usd(row.get("Price"))
            tp = _usd(row.get("TP"))
            res = _usd(row.get("Resistance"))
            rr_res = row.get("RR_to_Res", "")
            rr_tp = row.get("RR_to_TP", "")
            change_pct = _pct(row.get("Change%"))
            relvol = _xmult(row.get("RelVol(TimeAdj63d)"))
            tp_reward = _usd(row.get("TPReward$"))
            tp_reward_pct = _pct(row.get("TPReward%"))
            daily_atr = _usd(row.get("DailyATR"))
            daily_cap = _usd(row.get("DailyCap"))
            support_type = row.get("SupportType", "")
            support_price = _usd(row.get("SupportPrice"))
            opt_exp = row.get("OptExpiry", "")
            buyk = _num(row.get("BuyK"))
            sellk = _num(row.get("SellK"))

            st.markdown(
                f"""
**{tkr}** looks buyable **now at {price}** with a take-profit **{tp}** (midway to recent high **{res}**).

**Why this makes sense:**
- It’s **up {change_pct} today** and **trading heavier than usual ({relvol})**.
- The **reward** to TP is **{tp_reward} ({tp_reward_pct})**.
- Volatility backs it up: **14-day ATR ≈ {daily_atr}**, which implies ~**{daily_cap}** of potential movement over ~21 trading days.
- Support to lean on: **{support_type} ≈ {support_price}**.

**Risk/Reward:**
- **R:R to resistance ≈ {rr_res}:1**, to TP ≈ **{rr_tp}:1**.

**Nearest 30-day bullish vertical (call) idea:**
- Expiry **{opt_exp}**, **buy {buyk} / sell {sellk}** (see options columns for pricing and max profit).
                """.strip()
            )

    # ===== Google Sheets (copy) — hidden behind expander =====
    with st.expander("Copy for Google Sheets"):
        st.write(
            "Use the button below to copy the full pipe-delimited table (compatible with ‘Split text to columns’ on ‘|’)."
        )
        cols = pass_df.columns.tolist()
        lines = ["|".join(cols)]
        for _, r in pass_df.iterrows():
            vals = []
            for c in cols:
                v = r.get(c, "")
                if pd.isna(v):
                    v = ""
                s = str(v).replace("|", "/")
                vals.append(s)
            lines.append("|".join(vals))
        pipe_blob = "\n".join(lines)
        st.code(pipe_blob, language="text")

else:
    st.info("No PASS tickers yet. Adjust filters or try again during market hours.")

# ======= Debugger (separate toggle) =======
st.markdown("---")
with st.expander("🔎 Debugger (plain English) — explain a single ticker"):
    dbg_cols = st.columns([1, 1, 1, 1])
    with dbg_cols[0]:
        dbg_ticker = st.text_input("Ticker", value="", placeholder="e.g., WMT")
    with dbg_cols[1]:
        dbg_res_days = st.number_input("Res lookback", min_value=10, max_value=60, value=res_days, step=1, key="dbg_res")
    with dbg_cols[2]:
        dbg_relvol_min = st.number_input("Min RelVol", min_value=0.5, max_value=3.0, value=rel_vol_min, step=0.05, key="dbg_rv")
    with dbg_cols[3]:
        dbg_rr_min = st.number_input("Min R:R to res", min_value=1.0, max_value=5.0, value=rr_min, step=0.1, key="dbg_rr")

    dbg_run = st.button("Explain", use_container_width=False)

    if dbg_run and dbg_ticker.strip():
        d = diagnose_ticker(
            dbg_ticker.strip().upper(),
            res_days=int(dbg_res_days),
            rel_vol_min=float(dbg_relvol_min),
            rr_min=float(dbg_rr_min),
            prefer_stop=stop_mode,
            use_relvol_median=relvol_median,
        )

        # Show header
        et = d.get("entry_ts")
        ts_str = et.strftime("%Y-%m-%d %H:%M:%S ET") if isinstance(et, datetime) else ""
        st.markdown(
            f"**{d['ticker']}** — session **{d.get('session','')}**, entry source **{d.get('entry_src','')}**, vol source **{d.get('vol_src','')}** "
            f"{'· Entry '+ts_str if ts_str else ''}"
        )

        # Verdict
        verdict = "✅ PASS" if d.get("status") == "PASS" else "❌ FAIL"
        st.markdown(f"### {verdict}")
        st.markdown(d.get("explain", ""))

        # Numbers table (always show for transparency)
        key_rows = []
        def add_row(k, v):
            key_rows.append({"Metric": k, "Value": v})

        add_row("Entry", _usd(d.get("entry")))
        add_row("Prev close", _usd(d.get("prev_close")))
        add_row("Change today", _pct(d.get("change_pct")))
        add_row("RelVol (time-adj)", _xmult(d.get("relvol")))
        add_row("63-day avg volume", _num(d.get("avg_63"), 0))
        # expected by now:
        try:
            exp_now = float(d.get("avg_63", float("nan"))) * float(d.get("session_progress", 0.0))
        except Exception:
            exp_now = float("nan")
        add_row("Expected volume by now", _num(exp_now, 0))
        add_row("Today volume", _num(d.get("today_vol"), 0))
        add_row("Resistance", _usd(d.get("resistance")))
        add_row("TP", _usd(d.get("tp")))
        add_row("TP distance", _usd(d.get("tp_reward")))
        add_row("TP distance %", _pct(d.get("tp_req_pct")))
        add_row("ATR(14)", _usd(d.get("daily_atr")))
        add_row("ATR capacity (~21d)", _usd(d.get("daily_cap")))
        add_row("Support type", d.get("support_type", ""))
        add_row("Stop", _usd(d.get("stop")))
        add_row("Risk $", _usd(d.get("risk")))
        add_row("R:R to Resistance", f"{d.get('rr_to_res'):.2f}:1" if d.get("rr_to_res") is not None and not pd.isna(d.get("rr_to_res")) else "")
        add_row("R:R to TP", f"{d.get('rr_to_tp'):.2f}:1" if d.get("rr_to_tp") is not None and not pd.isna(d.get("rr_to_tp")) else "")

        st.dataframe(pd.DataFrame(key_rows), hide_index=True, use_container_width=True)

