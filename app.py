# app.py — Streamlit UI for the swing/options screener (polished WHY BUY cards)

import io
import textwrap
import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
from datetime import datetime

from swing_options_screener import run_scan  # your library API


# -----------------------
# Formatting helpers
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

def _int0(x):
    try:
        return int(float(x))
    except Exception:
        return 0

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
    Returns dict: today_vol, avg_63, expected_by_now, relvol_now.
    Uses 1m data if available (market open), otherwise today's daily bar.
    """
    try:
        t = yf.Ticker(ticker)
        df = t.history(period="16mo", auto_adjust=False, actions=False)
        if df is None or df.empty or "Volume" not in df.columns:
            return None

        vol_hist = df["Volume"].iloc[-64:-1]
        if vol_hist.empty:
            return None
        avg_63 = float(np.nanmean(vol_hist.values))

        m1 = yf.download(
            ticker, period="1d", interval="1m",
            auto_adjust=False, progress=False, prepost=False
        )
        today_vol = None
        if m1 is not None and not m1.empty and "Volume" in m1.columns:
            today_vol = float(np.nansum(m1["Volume"].values))

        now = pd.Timestamp.now(tz="America/New_York")
        open_t  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
        close_t = now.replace(hour=16, minute=0,  second=0, microsecond=0)

        if now <= open_t:
            progress = 0.0
        elif now >= close_t:
            progress = 1.0
        else:
            progress = (now - open_t) / (close_t - open_t)
            # never 0% (avoids div-by-zero early)
            progress = max(progress, pd.Timedelta(minutes=1) / (close_t - open_t))

        expected_by_now = avg_63 * float(progress)

        if today_vol is None:
            today_vol = float(df["Volume"].iloc[-1])
            if progress == 0.0:
                expected_by_now = avg_63 * 1.0

        relvol_now = float(today_vol / expected_by_now) if expected_by_now > 0 else np.nan

        return {
            "today_vol": today_vol,
            "avg_63": avg_63,
            "expected_by_now": expected_by_now,
            "relvol_now": relvol_now,
        }
    except Exception:
        return None


# -----------------------
# WHY BUY (HTML card)
# -----------------------
def why_buy_card(row: dict) -> str:
    """
    Produces a styled HTML card with clear bullets and bolded numbers.
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

    opt_exp = row.get("OptExpiry", "")
    buy_k = row.get("BuyK", "")
    sell_k = row.get("SellK", "")

    # TP move ($ and %)
    tp_move_d = (tp - price) if (tp is not None and price is not None) else None
    tp_move_p = (tp_move_d / price * 100.0) if (tp_move_d not in (None, 0) and price) else None

    # Volume snapshot (live)
    vol_stats = _get_intraday_volume_stats(tkr)
    vol_line = ""
    if vol_stats and np.isfinite(vol_stats.get("relvol_now", np.nan)):
        bump = (vol_stats["relvol_now"] - 1.0) * 100.0
        vol_line = (
            f"Up **{_pct(change_pct)}** today, and **volume** is **{_pct(bump)}** vs its usual pace "
            f"(**{int(vol_stats['today_vol']):,}** vs **{int(vol_stats['expected_by_now']):,}** expected by now)."
        )
    else:
        # fallback to time-adjusted relvol from the screener
        if relvol_timeadj not in (None, ""):
            bump = (relvol_timeadj - 1.0) * 100.0
            vol_line = f"Up **{_pct(change_pct)}** today, and **volume** is **{_pct(bump)}** vs usual."

    # History examples
    examples = _history_examples_list(row)
    examples_items = "".join([f"<li>{e}</li>" for e in examples])

    # Headline
    if opt_exp and buy_k and sell_k:
        headline = (
            f"<div class='wb-headline'><b>{tkr}</b> via vertical call spread "
            f"<b>{_dollar(buy_k)}</b> / <b>{_dollar(sell_k)}</b> expiring <b>{opt_exp}</b></div>"
        )
    else:
        headline = f"<div class='wb-headline'><b>{tkr}</b> setup</div>"

    # Content bullets
    bullets = f"""
    <ul class='wb-list'>
      <li><b>Price</b> now <b>{_dollar(price)}</b>, <b>TP</b> <b>{_dollar(tp)}</b>, <b>Resistance</b> <b>{_dollar(res)}</b>.</li>
      <li><b>R:R</b> ≈ <b>{_safe(rr_res,2)}:1</b> to resistance (≈ <b>{_safe(rr_tp,2)}:1</b> to TP) with stop at
          <b>{support_type}</b> <b>{_dollar(support_price)}</b>.</li>
      <li><b>TP distance</b>: <b>{_dollar(tp_move_d)}</b> (≈ <b>{_pct(tp_move_p)}</b>).</li>
      <li><b>Volatility (ATR)</b>: Daily ATR ≈ <b>{_dollar(daily_atr)}</b>, implying ≈ <b>{_dollar(daily_cap)}</b>
          of potential movement over ~21 trading days.</li>
      <li>{vol_line}</li>
      <li><b>History</b>: {_int0(hist_count)} prior 21-day windows met/exceeded the required move; best was about <b>{_pct(hist_best)}</b>.
        {"<ul>"+examples_items+"</ul>" if examples_items else ""}
      </li>
    </ul>
    """

    footer = f"<div class='wb-footer'>Data as of <b>{entry_ts}</b>.</div>" if entry_ts else ""

    html = f"""
    <div class='wb-card'>
      {headline}
      {bullets}
      {footer}
    </div>
    """
    return html


# -----------------------
# Page
# -----------------------
st.set_page_config(page_title="S&P 500 Options Screener", layout="wide")

# Inject lightweight CSS once
if "wb_css" not in st.session_state:
    st.markdown(
        """
        <style>
        .wb-card{
          border:1px solid rgba(0,0,0,.08);
          border-radius:12px;
          padding:16px 18px;
          background:#fff;
          box-shadow:0 1px 2px rgba(0,0,0,.05);
          margin-bottom: 10px;
        }
        .wb-headline{
          font-size:1.05rem;
          margin-bottom:6px;
        }
        .wb-list{
          margin: 0.25rem 0 0.25rem 1.1rem;
          line-height:1.6;
          font-size:0.98rem;
        }
        .wb-list li{ margin-bottom:2px; }
        .wb-footer{
          font-size:.85rem;
          color:#666;
          margin-top:6px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.session_state["wb_css"] = True

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

    # Sort by Price ascending (lowest -> highest)
    if not df.empty and "Price" in df.columns:
        df = df.sort_values(["Price", "Ticker"], ascending=[True, True]).reset_index(drop=True)

    with console:
        if df.empty:
            st.write("No PASS tickers found (or CSV not produced).")
        else:
            st.code("|".join(df.columns), language="text")
            sample = io.StringIO()
            df.to_csv(sample, sep="|", index=False)
            st.code(sample.getvalue().splitlines()[0: min(6, len(df)+1)], language="text")

    with pass_area:
        if df.empty:
            st.warning("No PASS tickers found (or CSV not produced).")
        else:
            st.subheader("PASS tickers")

            # Web-friendly table
            show_cols = [
                "Ticker","EvalDate","Price","EntryTimeET",
                "Change%","RelVol(TimeAdj63d)",
                "Resistance","TP",
                "RR_to_Res","RR_to_TP",
                "SupportType","SupportPrice","Risk$",
                "TPReward$","TPReward%","ResReward$","ResReward%",
                "DailyATR","DailyCap",
                "OptExpiry","BuyK","SellK","Width","DebitMid","DebitCons",
                "MaxProfitMid","MaxProfitCons","RR_Spread_Mid","RR_Spread_Cons","BreakevenMid",
            ]
            show_cols = [c for c in show_cols if c in df.columns]
            st.dataframe(df[show_cols], use_container_width=True, hide_index=True)

            # Copy-to-Sheets (ALL columns)
            st.caption("Copy table (pipe-delimited for Google Sheets)")
            buf = io.StringIO()
            df.to_csv(buf, index=False, sep="|")
            st.code(buf.getvalue(), language="text")

            # WHY BUY section as styled cards
            st.markdown("---")
            st.subheader("Explain each PASS (WHY BUY)")
            for _, row in df.iterrows():
                card_html = why_buy_card(row.to_dict())
                st.markdown(card_html, unsafe_allow_html=True)

