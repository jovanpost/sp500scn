# app.py — Streamlit UI for Swing Options Screener (UNADJUSTED)
# - Clean main view: big Run button
# - Shows ALL columns in a wide, scrollable, styled table
# - For each PASS, a "WHY BUY" explainer in your conversational style
# - Recomputes intraday volume stats so the explainer can show % and raw numbers

import math
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import datetime
try:
    from zoneinfo import ZoneInfo  # py>=3.9
except Exception:
    from backports.zoneinfo import ZoneInfo

# Import the screener library API
from swing_options_screener import run_scan

ET = ZoneInfo("America/New_York")

# ---------- Helpers for narrative ----------

def _safe(x, ndigits=None):
    """Convert values to printable strings with optional rounding."""
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return ""
    try:
        if ndigits is not None and isinstance(x, (int, float, np.floating)):
            return f"{round(float(x), ndigits)}"
        return f"{x}"
    except Exception:
        return f"{x}"

def _dollar(x, ndigits=2):
    s = _safe(x, ndigits)
    return f"${s}" if s != "" else ""

def _pct(x, ndigits=2):
    s = _safe(x, ndigits)
    return f"{s}%" if s != "" else ""

def _fmt_date(dt_iso: str):
    try:
        # Expecting 'YYYY-MM-DD'
        d = datetime.strptime(dt_iso, "%Y-%m-%d").date()
        return d.strftime("%b %d, %Y")
    except Exception:
        return dt_iso or ""

def _fmt_ts(ts: str):
    # Already formatted by screener as "YYYY-MM-DD HH:MM:SS ET"
    return ts or ""

def _get_intraday_volume_stats(ticker: str):
    """
    Returns dict with:
      today_vol, avg63, progress, expected_by_now, relvol_now
    If market closed, progress=1.0 and volumes are 'final'.
    """
    now = datetime.now(ET)
    open_t  = now.replace(hour=9, minute=30, second=0, microsecond=0)
    close_t = now.replace(hour=16, minute=0, second=0, microsecond=0)

    # Daily unadjusted history (16 months—same basis as screener)
    try:
        df = yf.Ticker(ticker).history(period="16mo", auto_adjust=False, actions=False)
        if df is None or df.empty:
            return {}
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
    except Exception:
        return {}

    # 63-session average volume, excluding today
    try:
        base = df['Volume'].iloc[-64:-1]
        if base is None or base.empty:
            return {}
        avg63 = float(np.nanmean(pd.to_numeric(base, errors="coerce")))
        if not np.isfinite(avg63) or avg63 <= 0:
            return {}
    except Exception:
        return {}

    # Today's cumulative volume
    # If today exists in df (partial during session, final after close) use that;
    # otherwise fallback to 1m sum (rare).
    today_vol = np.nan
    try:
        if len(df) >= 1 and df.index[-1].date() == now.date():
            today_vol = float(df['Volume'].iloc[-1])
    except Exception:
        pass

    if not np.isfinite(today_vol):
        # Fallback: 1m bars sum
        try:
            m1 = yf.download(ticker, period="1d", interval="1m", auto_adjust=False, progress=False, prepost=False)
            if m1 is not None and not m1.empty:
                today_vol = float(pd.to_numeric(m1['Volume'], errors="coerce").sum())
        except Exception:
            pass

    if not np.isfinite(today_vol):
        return {}

    # Session progress
    if now <= open_t:
        progress = 0.0
    elif now >= close_t:
        progress = 1.0
    else:
        progress = (now - open_t).total_seconds() / (close_t - open_t).total_seconds()
        progress = max(progress, 1/390)  # avoid 0

    expected_by_now = avg63 * progress if progress > 0 else np.nan
    relvol_now = (today_vol / expected_by_now) if (np.isfinite(expected_by_now) and expected_by_now > 0) else np.nan

    return {
        "today_vol": today_vol,
        "avg63": avg63,
        "progress": progress,
        "expected_by_now": expected_by_now,
        "relvol_now": relvol_now,
    }

def _history_examples_list(row):
    """
    Turn the screener's 'Hist21d_Examples' (e.g. '2025-04-21:+35.06%; ...')
    into bullet-like lines with date and percent. If prices are available,
    we leave as-is (we don't have start/end prices here).
    """
    ex = (row.get("Hist21d_Examples") or "").strip()
    if not ex:
        return []
    parts = [p.strip() for p in ex.split(";") if p.strip()]
    return parts[:3]

def why_buy_paragraph(row: dict) -> str:
    """
    Your narrative explainer, using real numbers. Includes:
      - Option spread mention (if present)
      - Why TP is reachable (recent resistance vs current price)
      - R:R to resistance (and to TP)
      - Move to TP in $ and %
      - ATR-based capacity (daily ~21d)
      - Intraday volume vs usual (with raw numbers)
      - History: pass count and example windows
    """
    tkr = row.get("Ticker", "")
    price = row.get("Price", None)
    tp = row.get("TP", None)
    res = row.get("Resistance", None)
    rr_res = row.get("RR_to_Res", None)
    rr_tp = row.get("RR_to_TP", None)
    support_type = row.get("SupportType", "")
    support_price = row.get("SupportPrice", None)
    change_pct = row.get("Change%", None)
    relvol = row.get("RelVol(TimeAdj63d)", None)
    daily_atr = row.get("DailyATR", None)
    daily_cap = row.get("DailyCap", None)
    eval_date = row.get("EvalDate", "")
    entry_ts = row.get("EntryTimeET", "")

    # TP move in $ and %
    tp_move_d = None
    tp_move_p = None
    try:
        if price is not None and tp is not None:
            tp_move_d = float(tp) - float(price)
            tp_move_p = (tp_move_d / float(price)) * 100.0 if float(price) != 0 else None
    except Exception:
        pass

    # Options chain (if present)
    opt_exp = row.get("OptExpiry", "")
    buy_k = row.get("BuyK", "")
    sell_k = row.get("SellK", "")

    # Volume stats for % and raw numbers
    vol_txt = ""
    vol_stats = _get_intraday_volume_stats(tkr)
    if vol_stats:
        today_vol = vol_stats.get("today_vol")
        avg63 = vol_stats.get("avg63")
        rel_now = vol_stats.get("relvol_now")
        # Build: "up X% on the day and volume is Y% above usual (A vs B by now)"
        if np.isfinite(today_vol) and np.isfinite(avg63) and np.isfinite(rel_now):
            # What would be "usual by now"
            usual_now = vol_stats.get("expected_by_now")
            bump = (rel_now - 1.0) * 100.0
            vol_txt = (f"and volume is {_pct(bump)} vs usual "
                       f"({int(today_vol):,} vs {int(usual_now):,} expected by now)")
        else:
            # Fallback to the screener's relvol multiple
            if relvol is not None and np.isfinite(relvol):
                bump = (float(relvol) - 1.0) * 100.0
                vol_txt = f"and time-adjusted volume is {_pct(bump)} vs usual"
    else:
        if relvol is not None and np.isfinite(relvol):
            bump = (float(relvol) - 1.0) * 100.0
            vol_txt = f"and time-adjusted volume is {_pct(bump)} vs usual"

    # History examples
    examples = _history_examples_list(row)
    hist_count = row.get("Hist21d_PassCount", "")
    hist_best = row.get("Hist21d_Max%", "")

    # ATR narrative (21 trading days ≈ 1 month)
    # We already have DailyCap (= DailyATR * ~21).
    # Present ATR directly and translate to ~monthly potential.
    atr_lines = []
    if daily_atr is not None and np.isfinite(daily_atr):
        atr_lines.append(f"the daily ATR is about {_dollar(daily_atr)}")
    if daily_cap is not None and np.isfinite(daily_cap):
        atr_lines.append(f"which implies roughly {_dollar(daily_cap)} of movement in ~21 trading days")
    atr_text = ", ".join(atr_lines) if atr_lines else ""

    # Build the opening claim (with or without options)
    if opt_exp and buy_k and sell_k:
        headline = (f"{tkr} is a buy via the **{_dollar(buy_k)} / {_dollar(sell_k)}** "
                    f"vertical call spread expiring **{opt_exp}**.")
    else:
        headline = (f"{tkr} looks like a buy setup here.")

    # Why TP looks reachable (connect current vs resistance)
    why_tp = ""
    if res is not None and price is not None:
        why_tp = (f" It recently topped around {_dollar(res)} and now trades near {_dollar(price)}, "
                  f"so the TP at {_dollar(tp)} looks realistic.")

    # Reward-to-risk statement
    rr_txt = ""
    if rr_res is not None and np.isfinite(rr_res):
        if rr_tp is not None and np.isfinite(rr_tp):
            rr_txt = (f" The reward-to-risk is about **{_safe(rr_res, 2)}:1** to resistance "
                      f"(~**{_safe(rr_tp, 2)}:1** to TP) based on support at "
                      f"{support_type} {_dollar(support_price)}.")
        else:
            rr_txt = (f" The reward-to-risk is about **{_safe(rr_res, 2)}:1** to resistance "
                      f"based on support at {support_type} {_dollar(support_price)}.")

    # TP move explained
    tp_txt = ""
    if tp_move_d is not None and tp_move_p is not None:
        tp_txt = (f" The move to TP is **{_dollar(tp_move_d)}** ("
                  f"**{_pct(tp_move_p)}**).")

    # Day change + volume kicker
    day_kicker = ""
    if change := row.get("Change%", None):
        day_kicker = f" It’s up **{_pct(change)}** today {vol_txt}."

    # ATR text
    atr_kicker = f" That’s well within reach — {atr_text}." if atr_text else ""

    # History line
    hist_line = ""
    if (isinstance(hist_count, (int, float)) and hist_count >= 0) or hist_best:
        pieces = []
        if isinstance(hist_count, (int, float)):
            pieces.append(f"{int(hist_count)} historical 21-day windows cleared that required move")
        if hist_best:
            pieces.append(f"best was about **{_pct(hist_best)}**")
        if pieces:
            hist_line = " History backs it up: " + ", ".join(pieces) + "."

    # Examples list
    ex_text = ""
    if examples:
        ex_text = " For example:\n" + "\n".join([f"- {e}" for e in examples])

    # Session timestamp
    timing = ""
    if entry_ts:
        timing = f"\n\n_Data as of **{_fmt_ts(entry_ts)}**._"

    paragraph = (
        f"{headline}{why_tp}{rr_txt}{tp_txt}{day_kicker}{atr_kicker}{hist_line}{ex_text}{timing}"
    )

    return paragraph

# ---------- Streamlit UI ----------

st.set_page_config(
    page_title="Swing Options Screener (Unadjusted)",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Swing Options Screener — Unadjusted (Finviz-style)")
st.caption("Time-adjusted relative volume, prior-high resistance, R:R≥2:1, ATR capacity, 21-day realism, and optional bull call spread near TP.")

# Main controls (clean: one big run)
col_run, col_dummy = st.columns([1, 4])
with col_run:
    run_clicked = st.button("▶️ Run Scan", type="primary")

with st.expander("Settings (advanced)", expanded=False):
    st.write("Tune filters (defaults match our tested setup).")
    universe_choice = st.selectbox(
        "Universe",
        options=["Default list (hardcoded)", "Custom (paste tickers, comma/space/newline-separated)"],
        index=0,
    )
    tickers_text = ""
    if universe_choice.startswith("Custom"):
        tickers_text = st.text_area("Tickers", height=120, placeholder="e.g. WMT, INTC, MRNA ...")

    colA, colB, colC, colD = st.columns(4)
    with colA:
        relvol_min = st.number_input("Min RelVol (time-adj 63d)", min_value=1.00, max_value=5.0, value=1.10, step=0.05)
    with colB:
        rr_min = st.number_input("Min R:R to resistance", min_value=1.0, max_value=10.0, value=2.0, step=0.25)
    with colC:
        res_days = st.number_input("Resistance lookback (days)", min_value=10, max_value=60, value=21, step=1)
    with colD:
        stop_mode = st.selectbox("Stop mode", options=["safest", "structure"], index=0)

    colE, colF = st.columns(2)
    with colE:
        with_options = st.checkbox("Suggest bull call spread (near TP)", value=True)
    with colF:
        opt_days = st.number_input("Target option expiry (days)", min_value=10, max_value=60, value=30, step=1)

# Execute scan
if run_clicked:
    # Prepare tickers argument for run_scan
    tickers_arg = None
    if universe_choice.startswith("Custom"):
        raw = [t.strip().upper() for chunk in tickers_text.replace("\n", ",").replace(" ", ",").split(",") if t.strip()]
        # de-dup while preserving order
        seen, clean = set(), []
        for t in raw:
            if t not in seen:
                seen.add(t); clean.append(t)
        tickers_arg = clean if clean else None  # None => library default

    out = run_scan(
        tickers=tickers_arg,
        res_days=int(res_days),
        rel_vol_min=float(relvol_min),
        relvol_median=False,   # keep mean basis (Finviz-like)
        rr_min=float(rr_min),
        stop_mode=stop_mode,
        with_options=with_options,
        opt_days=int(opt_days),
    )

    df = out.get("pass_df", pd.DataFrame())
    if df is None or df.empty:
        st.warning("No PASS tickers found.")
    else:
        # Sort by current price ascending
        if "Price" in df.columns:
            df = df.sort_values(["Price", "Ticker"])

        st.subheader("PASS Tickers")

        # Always show ALL columns, nicely
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        # Download buttons
        tsv_cols = list(df.columns)
        tsv = df.to_csv(sep="|", index=False)
        csv_std = df.to_csv(index=False)

        st.download_button("⬇️ Copy/Download (Pipe-separated for Google Sheets)", data=tsv, file_name="pass_tickers.psv", mime="text/plain")
        st.download_button("⬇️ CSV (standard)", data=csv_std, file_name="pass_tickers.csv", mime="text/csv")

        st.markdown("---")
        st.subheader("WHY BUY (for each PASS)")

        # One explainer per row (collapsible)
        for _, row in df.iterrows():
            tkr = row.get("Ticker", "")
            with st.expander(f"WHY BUY — {tkr}", expanded=False):
                paragraph = why_buy_paragraph(row.to_dict())
                st.markdown(paragraph)

