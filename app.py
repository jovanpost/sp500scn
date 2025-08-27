# app.py — Streamlit UI for the swing/options screener

import io
import textwrap
import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf

from datetime import datetime
from swing_options_screener import run_scan   # library API you already have


# -----------------------
# Small formatting helpers
# -----------------------
def _safe(x, nd=2):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return ""
    try:
        return round(float(x), nd)
    except Exception:
        return x

def _pct(x, nd=2):
    try:
        return f"{round(float(x), nd)}%"
    except Exception:
        return ""

def _dollar(x, nd=2):
    try:
        return f"${round(float(x), nd)}"
    except Exception:
        return ""

def _history_examples_list(row: dict, max_items: int = 3):
    raw = (row or {}).get("Hist21d_Examples", "")
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(";") if p.strip()]
    return parts[:max_items]


# -----------------------
# Intraday volume snapshot
# -----------------------
def _get_intraday_volume_stats(ticker: str):
    """
    Returns a dict with today_vol, avg_63, expected_by_now, relvol_now.
    Handles both market-open and closed sessions.
    """
    try:
        t = yf.Ticker(ticker)
        # Daily (unadjusted) last ~16mo for avg calculation:
        df = t.history(period="16mo", auto_adjust=False, actions=False)
        if df is None or df.empty or "Volume" not in df.columns:
            return None

        # Last 63 completed days (exclude today if present)
        vol_hist = df["Volume"].iloc[-64:-1]
        if vol_hist.empty:
            return None
        avg_63 = float(np.nanmean(vol_hist.values))

        # Try to get today’s intraday cumulative volume
        m1 = yf.download(
            ticker, period="1d", interval="1m",
            auto_adjust=False, progress=False, prepost=False
        )
        today_vol = None
        if m1 is not None and not m1.empty and "Volume" in m1.columns:
            try:
                today_vol = float(np.nansum(m1["Volume"].values))
            except Exception:
                today_vol = None

        # Progress in session (9:30→16:00 ET ≈ 390 minutes)
        now = pd.Timestamp.now(tz="America/New_York")
        open_t  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
        close_t = now.replace(hour=16, minute=0,  second=0, microsecond=0)

        if now <= open_t:
            progress = 0.0
        elif now >= close_t:
            progress = 1.0
        else:
            progress = (now - open_t) / (close_t - open_t)
            progress = max(progress, pd.Timedelta(minutes=1) / (close_t - open_t))

        expected_by_now = avg_63 * float(progress)

        # If market closed (or 1m unavailable), fall back to today's daily volume
        if today_vol is None:
            try:
                today_vol = float(df["Volume"].iloc[-1])
                # If that bar is actually "yesterday" (pre-open), progress=1
                if progress == 0.0:
                    expected_by_now = avg_63 * 1.0
            except Exception:
                return None

        if expected_by_now <= 0:
            relvol_now = np.nan
        else:
            relvol_now = float(today_vol / expected_by_now)

        return {
            "today_vol": today_vol,
            "avg_63": avg_63,
            "expected_by_now": expected_by_now,
            "relvol_now": relvol_now,
        }
    except Exception:
        return None


# -----------------------
# Narrative “WHY BUY”
# -----------------------
def why_buy_paragraph(row: dict) -> str:
    """
    Narrative explainer in your style:
    - bold all numbers
    - always show $ and %
    - include today's change and volume vs expected
    - include ATR capacity and history examples
    """
    tkr = row.get("Ticker", "")
    price = row.get("Price")
    tp = row.get("TP")
    res = row.get("Resistance")
    rr_res = row.get("RR_to_Res")
    rr_tp = row.get("RR_to_TP")
    support_type = row.get("SupportType", "")
    support_price = row.get("SupportPrice")
    change_pct = row.get("Change%")
    relvol_timeadj = row.get("RelVol(TimeAdj63d)")
    daily_atr = row.get("DailyATR")
    daily_cap = row.get("DailyCap")
    entry_ts = row.get("EntryTimeET", "")
    hist_count = row.get("Hist21d_PassCount", "")
    hist_best = row.get("Hist21d_Max%", "")

    # TP move ($ and %)
    tp_move_d = (tp - price) if (tp is not None and price is not None) else None
    tp_move_p = (tp_move_d / price * 100.0) if (tp_move_d not in (None, 0) and price) else None

    # Options chain (if present)
    opt_exp = row.get("OptExpiry", "")
    buy_k = row.get("BuyK", "")
    sell_k = row.get("SellK", "")

    # Volume snapshot
    vol_stats = _get_intraday_volume_stats(tkr)
    vol_txt = ""
    if vol_stats:
        today_vol = vol_stats.get("today_vol")
        expected_by_now = vol_stats.get("expected_by_now")
        r_now = vol_stats.get("relvol_now")
        if r_now and np.isfinite(r_now):
            bump = (r_now - 1.0) * 100.0
            vol_txt = (f" and volume is **{_pct(bump)}** vs its typical pace "
                       f"(**{int(today_vol):,}** vs **{int(expected_by_now):,}** expected by now)")
    elif relvol_timeadj:
        bump = (relvol_timeadj - 1.0) * 100.0
        vol_txt = f" and time-adjusted volume is **{_pct(bump)}** vs usual"

    # History examples
    examples = _history_examples_list(row)
    examples_md = ""
    if examples:
        bullets = "\n".join([f"- {e}" for e in examples])
        examples_md = f"\n\n**Some recent 21-day examples:**\n{bullets}"

    # Headline sentence
    if opt_exp and buy_k and sell_k:
        headline = (
            f"**{tkr}** is a buy via the vertical call spread "
            f"**{_dollar(buy_k)} / {_dollar(sell_k)}** expiring **{opt_exp}**."
        )
    else:
        headline = f"**{tkr}** looks like a buy setup."

    # “Why TP is reachable”
    why_tp = (
        f" It recently topped around **{_dollar(res)}** and now trades near **{_dollar(price)}**. "
        f"Target price is **{_dollar(tp)}** — a level that looks reasonable given the setup."
    )

    # Reward vs risk
    rr_txt = (
        f" The reward-to-risk is about **{_safe(rr_res,2)}:1** to resistance "
        f"(~**{_safe(rr_tp,2)}:1** to TP) using {support_type} at **{_dollar(support_price)}** as the stop."
    )

    # TP distance
    tp_txt = ""
    if tp_move_d is not None and tp_move_p is not None:
        tp_txt = f" The move to TP is **{_dollar(tp_move_d)}** (**{_pct(tp_move_p)}**)."

    # Today’s change & volume
    day_kicker = f" It’s up **{_pct(change_pct)}** today{vol_txt}." if change_pct not in (None, "") else ""

    # ATR capacity
    atr_txt = (
        f" Daily ATR is **{_dollar(daily_atr)}**, which implies roughly **{_dollar(daily_cap)}** "
        f"of potential movement over ~21 trading days."
    )

    # History validation
    hist_txt = ""
    if hist_count not in (None, "") or hist_best not in (None, ""):
        hist_txt = (
            f" Over the past year, **{int(hist_count) if hist_count not in ('', None) else 0}** 21-day windows "
            f"met or exceeded that required move; the best window was about **{_pct(hist_best)}**."
        )

    stamped = f"\n\n*Data as of **{entry_ts}**.*" if entry_ts else ""

    md = f"{headline}{why_tp}{rr_txt}{tp_txt}{day_kicker}{atr_txt}{hist_txt}{examples_md}{stamped}"
    # Clean tiny double spaces that may creep in
    return " ".join(md.split())


# -----------------------
# UI
# -----------------------
st.set_page_config(page_title="S&P 500 Options Screener", layout="wide")

st.title("📈 S&P 500 Options Screener")
st.caption(f"UI started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S ET')}")

with st.expander("⚙️ Settings (optional)", expanded=False):
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        relvol_min = st.number_input("Min RelVol (time-adjusted)", value=1.10, step=0.05, format="%.2f")
    with c2:
        rr_min = st.number_input("Min RR to Resistance", value=2.0, step=0.1, format="%.2f")
    with c3:
        res_days = st.number_input("Resistance lookback (days)", value=21, step=1, min_value=10, max_value=252)
    with c4:
        opt_days = st.number_input("Target option days", value=30, step=1, min_value=7, max_value=90)

    stop_mode = st.selectbox("Stop preference", ["safest", "structure"], index=0)
    with_options = st.checkbox("Include option-spread suggestion", value=True)
    relvol_median = st.checkbox("Use median volume (63d) for RelVol", value=False)

run_col = st.container()
with run_col:
    run_clicked = st.button("Run Screener", use_container_width=True)

console = st.expander("🖨️ Console output", expanded=False)
pass_area = st.container()

if run_clicked:
    with st.status("Running screener… this may take a bit on first run.", expanded=True) as s:
        # Call the screener library
        out = run_scan(
            tickers=None,
            res_days=int(res_days),
            rel_vol_min=float(relvol_min),
            relvol_median=bool(relvol_median),
            rr_min=float(rr_min),
            stop_mode=stop_mode,
            with_options=with_options,
            opt_days=int(opt_days),
        )
        s.update(label="Screener finished.")

    df = out.get("pass_df", pd.DataFrame())

    # Always sort by current Price, ascending (lowest → highest)
    if not df.empty and "Price" in df.columns:
        df = df.sort_values(["Price", "Ticker"], ascending=[True, True]).reset_index(drop=True)

    with console:
        if df.empty:
            st.write("No PASS tickers found (or CSV not produced).")
        else:
            # Show a one-line preview header so you can see the actual columns
            st.code("|".join(df.columns), language="text")
            # And a couple sample lines
            sample = io.StringIO()
            df.to_csv(sample, sep="|", index=False)
            st.code(sample.getvalue().splitlines()[0: min(6, len(df)+1)], language="text")

    with pass_area:
        if df.empty:
            st.warning("No PASS tickers found (or CSV not produced).")
        else:
            st.subheader("PASS tickers")

            # Nice data table for the web (hide raw options columns at first if there are many)
            show_cols = [
                "Ticker", "EvalDate", "Price", "EntryTimeET",
                "Change%", "RelVol(TimeAdj63d)",
                "Resistance", "TP",
                "RR_to_Res", "RR_to_TP",
                "SupportType", "SupportPrice", "Risk$",
                "TPReward$", "TPReward%", "ResReward$", "ResReward%",
                "DailyATR", "DailyCap",
            ]
            show_cols = [c for c in show_cols if c in df.columns]
            st.dataframe(df[show_cols], use_container_width=True, hide_index=True)

            # Copy-to-Sheets block (pipe-delimited, includes ALL columns)
            st.caption("Copy table (pipe-delimited for Google Sheets)")
            buf = io.StringIO()
            df.to_csv(buf, index=False, sep="|")
            st.code(buf.getvalue(), language="text")

            # WHY BUY expanders per ticker
            st.markdown("---")
            st.subheader("Explain each PASS (WHY BUY)")
            for _, row in df.iterrows():
                tkr = row["Ticker"]
                with st.expander(f"WHY BUY — {tkr}", expanded=False):
                    md = why_buy_paragraph(row.to_dict())
                    # Use simple markdown; Streamlit will bold **numbers** and keep the prose
                    st.markdown(md)


