# app.py
import math
from datetime import datetime
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Import the core screener module robustly
import swing_options_screener as sos

# ---- Defaults in case the module doesn’t export them ----
RES_LOOKBACK_DEFAULT   = getattr(sos, "RES_LOOKBACK_DEFAULT", 21)
REL_VOL_MIN_DEFAULT    = getattr(sos, "REL_VOL_MIN_DEFAULT", 1.10)
RR_MIN_DEFAULT         = getattr(sos, "RR_MIN_DEFAULT", 2.0)
SUPPORT_LOOKBACK_DAYS  = getattr(sos, "SUPPORT_LOOKBACK_DAYS", 21)
PIVOT_K                = getattr(sos, "PIVOT_K", 2)
PIVOT_LOOKBACK_DAYS    = getattr(sos, "PIVOT_LOOKBACK_DAYS", 40)
ATR_STOP_MULT          = getattr(sos, "ATR_STOP_MULT", 1.5)

# ========= Styling / layout =========
st.set_page_config(page_title="Swing Options Screener", page_icon="📈", layout="wide")
st.markdown(
    """
    <style>
      /* Make fonts consistent and readable */
      .buy-blk p { margin: 0.25rem 0; }
      .buy-blk ul { margin: 0.25rem 0 0.5rem 1.2rem; }
      .buy-blk strong { font-weight: 700; }
      .stButton>button[kind="primary"] {
        background: #d62728 !important;  /* red */
        color: white !important;
        border: 0 !important;
      }
      .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📈 Swing Options Screener (Unadjusted • Finviz-style)")

# ========= Small formatting helpers =========
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
    now = datetime.now(tz=pd.Timestamp.now(tz="America/New_York").tz)
    open_t = now.replace(hour=9, minute=30, second=0, microsecond=0)
    close_t = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return (open_t < now < close_t), now, open_t, close_t

# ========= Robust access to internal funcs =========
_get_history                 = getattr(sos, "_get_history")
get_entry_prevclose_todayvol = getattr(sos, "get_entry_prevclose_todayvol")
_atr_from_ohlc               = getattr(sos, "_atr_from_ohlc")
_recent_pivot_low            = getattr(sos, "_recent_pivot_low")

# Fallback for compute_relvol_time_adjusted (some deployments don’t export it)
def _compute_relvol_time_adjusted_fallback(df_daily: pd.DataFrame, today_vol: float, use_median: bool=False):
    if not np.isfinite(today_vol) or today_vol < 0: return np.nan
    base = df_daily["Volume"].iloc[-64:-1]
    if base.empty: return np.nan
    avg_63 = (base.median() if use_median else base.mean())
    if not np.isfinite(avg_63) or avg_63 <= 0: return np.nan
    is_open, now, open_t, close_t = _is_open_now_et()
    if now <= open_t: progress = 0.0
    elif now >= close_t: progress = 1.0
    else:
        progress = (now - open_t).total_seconds() / (close_t - open_t).total_seconds()
        progress = max(progress, 1/390)
    expected = avg_63 * progress
    return today_vol / expected if expected > 0 else np.nan

compute_relvol_time_adjusted = getattr(
    sos, "compute_relvol_time_adjusted", _compute_relvol_time_adjusted_fallback
)

# ==========================================
# Plain-English debugger with numbers
# ==========================================
def diagnose_ticker(
    ticker: str,
    res_days: int = RES_LOOKBACK_DEFAULT,
    rel_vol_min: float = REL_VOL_MIN_DEFAULT,
    rr_min: float = RR_MIN_DEFAULT,
    prefer_stop: str = "safest",
    use_relvol_median: bool = False,
) -> Dict[str, Any]:
    out = {"ticker": ticker}

    df = _get_history(ticker)
    if df is None or df.empty:
        out.update(status="FAIL", reason_code="no_data",
                   explain="No price history available to evaluate this ticker.")
        return out

    if len(df) < max(22, res_days + 1):
        out.update(status="FAIL", reason_code="insufficient_rows",
                   explain=f"Not enough rows ({len(df)}) to compute a {res_days}-day resistance and 21-day checks.")
        return out

    entry, prev_close, today_vol, src, entry_ts = get_entry_prevclose_todayvol(df, ticker)
    out["session"]   = src.get("session", "")
    out["entry_src"] = src.get("entry_src", "")
    out["vol_src"]   = src.get("vol_src", "")
    out["entry_ts"]  = entry_ts

    if not (np.isfinite(entry) and np.isfinite(prev_close)):
        out.update(status="FAIL", reason_code="bad_entry_prevclose",
                   explain="Could not determine today’s entry price or yesterday’s close.")
        return out

    change = (entry - prev_close) / prev_close
    out["entry"] = float(entry)
    out["prev_close"] = float(prev_close)
    out["change_pct"] = float(change * 100)

    if change <= 0:
        out.update(status="FAIL", reason_code="not_up_on_day",
                   explain=f"Not up today: change = {_pct(out['change_pct'])} (needs > 0%).")
        return out

    # RelVol time-adjusted
    relvol = compute_relvol_time_adjusted(df, today_vol, use_median=use_relvol_median)
    base_series = df["Volume"].iloc[-64:-1]
    avg_63 = float((base_series.median() if use_relvol_median else base_series.mean())) if not base_series.empty else float("nan")

    is_open, now_et, open_t, close_t = _is_open_now_et()
    if now_et <= open_t:
        progress = 0.0
    elif now_et >= close_t:
        progress = 1.0
    else:
        progress = (now_et - open_t).total_seconds() / (close_t - open_t).total_seconds()
        progress = max(progress, 1/390)

    out["today_vol"] = float(today_vol) if np.isfinite(today_vol) else float("nan")
    out["avg_63"] = avg_63
    out["session_progress"] = float(progress)
    out["relvol"] = float(relvol) if np.isfinite(relvol) else float("nan")

    if not (np.isfinite(relvol) and relvol >= rel_vol_min):
        expected_by_now = avg_63 * progress if (np.isfinite(avg_63) and progress > 0) else float("nan")
        out.update(status="FAIL", reason_code="relvol_low_timeadj",
                   explain=(f"Volume light: RelVol = {_xmult(out['relvol'])} vs threshold ≥ {_xmult(rel_vol_min)}. "
                            f"(63d {'median' if use_relvol_median else 'avg'}: {_num(avg_63,0)} | "
                            f"expected by now: {_num(expected_by_now,0)} | today: {_num(today_vol,0)})."))
        return out

    # Resistance (prior N-day high excluding today)
    rolling_high = df["High"].rolling(window=res_days, min_periods=res_days).max()
    resistance = rolling_high.shift(1).iloc[-1]
    out["resistance"] = float(resistance) if np.isfinite(resistance) else float("nan")
    if not (np.isfinite(resistance) and resistance > entry):
        out.update(status="FAIL", reason_code="no_upside_to_resistance",
                   explain=(f"No upside: prior {res_days}-day high (resistance) = {_usd(resistance)} "
                            f"is not above entry {_usd(entry)}."))
        return out

    tp = entry + 0.5 * (resistance - entry)
    tp_reward = tp - entry
    res_reward = resistance - entry
    out["tp"] = float(tp)
    out["tp_reward"] = float(tp_reward)
    out["res_reward"] = float(res_reward)
    out["tp_req_pct"] = float((tp_reward / entry) * 100.0)

    # ATR capacity (vs TP distance only)
    daily_atr = _atr_from_ohlc(df, 14)
    daily_cap = (daily_atr * 21.0) if np.isfinite(daily_atr) else 0.0
    out["daily_atr"] = float(daily_atr) if np.isfinite(daily_atr) else float("nan")
    out["daily_cap"] = float(daily_cap)

    if not (daily_cap > tp_reward):
        out.update(status="FAIL", reason_code="atr_capacity_short_vs_tp",
                   explain=(f"ATR too small: TP needs {_usd(tp_reward)} ({_pct(out['tp_req_pct'])}); "
                            f"14d ATR projects about {_usd(daily_cap)} over ~21 days."))
        return out

    # Support selection
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
        out.update(status="FAIL", reason_code="no_valid_support",
                   explain="No valid support below entry (swing low, pivot, or ATR-based).")
        return out

    # “Safest” = use the highest support (tightest stop below entry)
    support_type, stop = max(candidates, key=lambda kv: kv[1])
    risk = entry - stop
    out["support_type"] = support_type
    out["stop"] = float(stop)
    out["risk"] = float(risk)

    if risk <= 0:
        out.update(status="FAIL", reason_code="non_positive_risk",
                   explain=(f"Stop {_usd(stop)} isn’t below entry {_usd(entry)} — risk would be ≤ 0."))
        return out

    rr_to_res = res_reward / risk
    rr_to_tp  = tp_reward / risk
    out["rr_to_res"] = float(rr_to_res)
    out["rr_to_tp"]  = float(rr_to_tp)

    if rr_to_res < RR_MIN_DEFAULT:
        out.update(status="FAIL", reason_code="rr_to_res_below_min",
                   explain=(f"R:R to resistance too low: R = {_usd(risk)}, Reward = {_usd(res_reward)} "
                            f"→ {rr_to_res:.2f}:1 vs ≥ {RR_MIN_DEFAULT:.2f}:1."))
        return out

    out.update(status="PASS", reason_code="",
               explain=(f"PASS — Up today ({_pct(out['change_pct'])}); RelVol {_xmult(out['relvol'])}; "
                        f"resistance {_usd(resistance)} above entry; TP {_usd(tp)}; "
                        f"ATR capacity {_usd(daily_cap)} > TP distance {_usd(tp_reward)}; "
                        f"support {support_type} at {_usd(stop)} gives R:R to resistance ≈ {rr_to_res:.2f}:1."))
    return out

# ========= App state & RUN button =========
if "last_pass_df" not in st.session_state:
    st.session_state["last_pass_df"] = pd.DataFrame()

run_clicked = st.button("RUN", type="primary", use_container_width=True)
if run_clicked:
    # Use backend defaults; no top filter bar right now
    out = sos.run_scan(
        tickers=None,
        res_days=RES_LOOKBACK_DEFAULT,
        rel_vol_min=REL_VOL_MIN_DEFAULT,
        relvol_median=False,
        rr_min=RR_MIN_DEFAULT,
        stop_mode="safest",
        with_options=True,
        opt_days=getattr(sos, "TARGET_OPT_DAYS_DEFAULT", 30),
    )
    st.session_state["last_pass_df"] = out.get("pass_df", pd.DataFrame())

pass_df = st.session_state["last_pass_df"]

# ========= Results table first =========
st.subheader("Results")
if pass_df is not None and not pass_df.empty:
    if "Price" in pass_df.columns:
        pass_df = pass_df.sort_values(["Price", "Ticker"], ascending=[True, True])
    st.dataframe(pass_df, use_container_width=True, hide_index=True)

    # ========= WHY BUY per row (expander) =========
    st.markdown("### WHY BUY (per ticker)")
    for _, row in pass_df.iterrows():
        tkr = row["Ticker"]
        with st.expander(f"Why {tkr}?"):
            price           = _usd(row.get("Price"))
            tp              = _usd(row.get("TP"))
            res             = _usd(row.get("Resistance"))
            rr_res          = row.get("RR_to_Res", "")
            rr_tp           = row.get("RR_to_TP", "")
            change_pct      = _pct(row.get("Change%"))
            relvol          = _xmult(row.get("RelVol(TimeAdj63d)"))
            tp_reward       = _usd(row.get("TPReward$"))
            tp_reward_pct   = _pct(row.get("TPReward%"))
            daily_atr       = _usd(row.get("DailyATR"))
            daily_cap       = _usd(row.get("DailyCap"))
            support_type    = row.get("SupportType", "")
            support_price   = _usd(row.get("SupportPrice"))
            opt_exp         = row.get("OptExpiry", "")
            buyk            = _num(row.get("BuyK"))
            sellk           = _num(row.get("SellK"))

            st.markdown(
                f"""
<div class="buy-blk">
  <p><strong>{tkr}</strong> looks buyable <strong>now at {price}</strong> with a take-profit <strong>{tp}</strong> (halfway toward the recent high <strong>{res}</strong>).</p>
  <p><strong>Why this makes sense</strong></p>
  <ul>
    <li>It’s <strong>up {change_pct} today</strong> and trading <strong>heavier than usual ({relvol})</strong>.</li>
    <li>The move to TP is <strong>{tp_reward}</strong> (<strong>{tp_reward_pct}</strong>).</li>
    <li>Volatility backs it: <strong>ATR(14) ≈ {daily_atr}</strong>, which implies about <strong>{daily_cap}</strong> of potential movement over ~21 trading days.</li>
    <li>You’ve got a nearby support to lean on: <strong>{support_type} ≈ {support_price}</strong>.</li>
  </ul>
  <p><strong>Risk / Reward</strong><br/>
     R:R to resistance ≈ <strong>{rr_res}:1</strong>; to TP ≈ <strong>{rr_tp}:1</strong>.
  </p>
  <p><strong>Nearest ~30-day bullish vertical (calls)</strong><br/>
     Expiry <strong>{opt_exp}</strong>, buy <strong>{buyk}</strong> / sell <strong>{sellk}</strong>.
  </p>
</div>
                """,
                unsafe_allow_html=True,
            )

    # ========= Google Sheets copy (expander) =========
    with st.expander("Copy for Google Sheets"):
        st.write("Pipe-delimited table below. In Sheets: Data → Split text to columns → delimiter “|”.")
        cols = pass_df.columns.tolist()
        lines = ["|".join(cols)]
        for _, r in pass_df.iterrows():
            vals = []
            for c in cols:
                v = r.get(c, "")
                if pd.isna(v): v = ""
                vals.append(str(v).replace("|", "/"))
            lines.append("|".join(vals))
        st.code("\n".join(lines), language="text")

else:
    st.info("No PASS tickers yet. Click RUN during market hours or after the close.")

# ========= Debugger (separate toggle) =========
st.markdown("---")
with st.expander("🔎 Debugger — plain English with numbers"):
    dbg_cols = st.columns([1, 1, 1, 1])
    with dbg_cols[0]:
        dbg_ticker = st.text_input("Ticker", value="", placeholder="e.g., WMT")
    with dbg_cols[1]:
        dbg_res_days = st.number_input("Res lookback", min_value=10, max_value=60,
                                       value=RES_LOOKBACK_DEFAULT, step=1, key="dbg_res")
    with dbg_cols[2]:
        dbg_relvol_min = st.number_input("Min RelVol", min_value=0.5, max_value=3.0,
                                         value=REL_VOL_MIN_DEFAULT, step=0.05, key="dbg_rv")
    with dbg_cols[3]:
        dbg_rr_min = st.number_input("Min R:R to res", min_value=1.0, max_value=5.0,
                                     value=RR_MIN_DEFAULT, step=0.1, key="dbg_rr")

    dbg_run = st.button("Explain", use_container_width=False)

    if dbg_run and dbg_ticker.strip():
        d = diagnose_ticker(
            dbg_ticker.strip().upper(),
            res_days=int(dbg_res_days),
            rel_vol_min=float(dbg_relvol_min),
            rr_min=float(dbg_rr_min),
            prefer_stop="safest",
            use_relvol_median=False,
        )

        ts = d.get("entry_ts")
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S ET") if isinstance(ts, datetime) else ""
        st.markdown(
            f"**{d['ticker']}** — session **{d.get('session','')}**, entry source **{d.get('entry_src','')}**, "
            f"vol source **{d.get('vol_src','')}**{' · Entry ' + ts_str if ts_str else ''}"
        )

        verdict = "✅ PASS" if d.get("status") == "PASS" else "❌ FAIL"
        st.markdown(f"### {verdict}")
        st.markdown(d.get("explain", ""))

        # Numbers table
        key_rows = []
        def add_row(k, v): key_rows.append({"Metric": k, "Value": v})

        try:
            exp_now = float(d.get("avg_63", float("nan"))) * float(d.get("session_progress", 0.0))
        except Exception:
            exp_now = float("nan")

        add_row("Entry", _usd(d.get("entry")))
        add_row("Prev close", _usd(d.get("prev_close")))
        add_row("Change today", _pct(d.get("change_pct")))
        add_row("RelVol (time-adj)", _xmult(d.get("relvol")))
        add_row("63-day avg volume", _num(d.get("avg_63"), 0))
        add_row("Expected vol by now", _num(exp_now, 0))
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

