# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo

# Your existing screener module
from swing_options_screener import run_scan

# ---------- Styling (kept same, including red RUN button) ----------
st.set_page_config(page_title="S&P 500 Options Screener", layout="wide")
st.markdown(
    """
    <style>
      .run-btn > button {
        background-color: #e11d48 !important; /* red */
        color: white !important;
        border: 0;
        font-weight: 600;
      }
      .small-note { color: #6b7280; font-size: 0.9rem; }
      .why-card {
        padding: 0.25rem 0.5rem;
      }
      .why-card p { margin: 0.25rem 0; }
      .why-card ul { margin: 0.25rem 0 0.5rem 1.1rem; }
      .why-card li { margin: 0.15rem 0; }
      .why-head { font-size: 1.05rem; font-weight: 700; }
      .pill {
        display:inline-block; padding:0.1rem 0.4rem; border-radius:0.4rem;
        background:#f3f4f6; border:1px solid #e5e7eb; margin-left:0.25rem; font-weight:600;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- small helpers ----------
def _fmt_money(x):
    try:
        if pd.isna(x): return "—"
        return f"${float(x):,.2f}"
    except Exception:
        return "—"

def _num(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def _parse_examples(ex_str: str):
    # expects "YYYY-MM-DD:+12.3%; YYYY-MM-DD:+9.8%; ..."
    if not ex_str:
        return []
    parts = [p.strip() for p in ex_str.split(";") if p.strip()]
    return parts[:3]

# ---------- WHY BUY HTML (FIXED variable names) ----------
def _why_buy_html(row: pd.Series) -> str:
    """
    Plain-English WHY BUY rendered as HTML (no Markdown),
    so symbols never turn into italics/bullets accidentally.
    """
    t          = row.get("Ticker", "")
    price      = _num(row.get("Price"))
    tp         = _num(row.get("TP"))
    res        = _num(row.get("Resistance"))
    rr_res     = _num(row.get("RR_to_Res"))
    rr_tp      = _num(row.get("RR_to_TP"))
    sup_type   = str(row.get("SupportType",""))
    sup_px     = _num(row.get("SupportPrice"))
    tp_dollars = _num(row.get("TPReward$"))
    tp_percent = _num(row.get("TPReward%"))
    d_atr      = _num(row.get("DailyATR"))
    w_atr      = d_atr * 5  if np.isfinite(d_atr) else np.nan
    m_atr      = d_atr * 21 if np.isfinite(d_atr) else np.nan
    relvol     = _num(row.get("RelVol(TimeAdj63d)"))
    chg        = _num(row.get("Change%"))
    exp        = str(row.get("OptExpiry","")).strip()
    buyk       = str(row.get("BuyK","")).strip()
    sellk      = str(row.get("SellK","")).strip()
    examples   = str(row.get("Hist21d_Examples","")).strip()
    pass_ct    = row.get("Hist21d_PassCount", "—")
    ts         = str(row.get("EntryTimeET","")).strip()

    recent_top_txt = f"{_fmt_money(res)} <span class='small-note'>(resistance)</span>" if np.isfinite(res) else "recent highs"
    sup_note = f"{_fmt_money(sup_px)} <span class='small-note'>(support)</span>" if np.isfinite(sup_px) else "a recent support"

    hdr = (
        f"<div class='why-card'>"
        f"<div class='why-head'>{t} — Why this setup makes sense</div>"
        f"<p>{t} is a buy via the "
        f"<strong>{buyk or '—'}/{sellk or '—'}</strong> vertical call spread "
        f"expiring <strong>{exp or '—'}</strong> because it recently reached about <strong>{recent_top_txt}</strong> "
        f"and now trades near <strong>{_fmt_money(price)}</strong>. "
        f"That makes a target at <strong>{_fmt_money(tp)}</strong> feel realistic.</p>"
    )

    # bullets
    items = []

    # Reward vs risk
    rr_res_txt = f"{rr_res:.2f}:1" if np.isfinite(rr_res) else "—"
    rr_tp_txt  = f"{rr_tp:.2f}:1"  if np.isfinite(rr_tp)  else "—"
    items.append(
        f"<strong>Reward vs. risk:</strong> from {sup_note} up to resistance is about "
        f"<strong>{rr_res_txt}</strong> (to the nearer target it’s <strong>{rr_tp_txt}</strong>)."
    )

    # Move needed
    if np.isfinite(tp_dollars) and np.isfinite(tp_percent):
        items.append(
            f"<strong>Move needed to the target:</strong> "
            f"<strong>{_fmt_money(tp_dollars)}</strong> (≈ <strong>{tp_percent:.2f}%</strong>)."
        )

    # ATR runway
    atr_bits = []
    if np.isfinite(d_atr):
        atr_bits.append(f"daily ATR is <strong>{_fmt_money(d_atr)}</strong>")
    if np.isfinite(w_atr):
        atr_bits.append(f"a typical week allows about <strong>{_fmt_money(w_atr)}</strong>")
    if np.isfinite(m_atr):
        atr_bits.append(f"~21 trading days allow roughly <strong>{_fmt_money(m_atr)}</strong>")
    if atr_bits:
        items.append(
            f"<strong>Volatility runway (ATR):</strong> " +
            ", ".join(atr_bits[:-1]) + (", and " if len(atr_bits) > 1 else "") + atr_bits[-1] + "."
            if len(atr_bits) > 1 else f"<strong>Volatility runway (ATR):</strong> {atr_bits[0]}."
        )

    # Tone & volume
    tone = []
    if np.isfinite(chg):
        tone.append(f"price is up <strong>{chg:.2f}%</strong> today")
    if np.isfinite(relvol):
        tone.append(
            "volume is running about "
            f"<strong>{(relvol-1)*100:.0f}%</strong> vs its typical pace "
            "<span class='small-note'>(time-adjusted)</span>"
            if relvol >= 1 else
            "volume is a bit lighter than typical pace"
        )
    if tone:
        items.append(f"<strong>Today’s tone & volume:</strong> " + " and ".join(tone) + ".")

    bullets = "<ul>" + "".join([f"<li>{it}</li>" for it in items]) + "</ul>"

    # History
    hist = ""
    if np.isfinite(tp_percent):
        hist = (
            f"<p><strong>History check (21 trading days):</strong> to reach the target, "
            f"the stock needs about <strong>{tp_percent:.2f}%</strong> from here. "
            f"Over the past year, <strong>{pass_ct}</strong> separate 21-day windows met or exceeded that move."
        )
        ex = _parse_examples(examples)
        if ex:
            hist += " <strong>Examples:</strong></p><ul>" + "".join([f"<li>{e}</li>" for e in ex]) + "</ul>"
        else:
            hist += "</p>"

    footer = (
        f"<p class='small-note'>Data as of <strong>{ts or datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d %H:%M:%S ET')}</strong>.</p>"
        f"</div>"
    )

    return hdr + bullets + hist + footer

# ---------- UI HEADER ----------
st.title("📈 S&P 500 Options Screener")

st.caption(
    f"UI started: {datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d %H:%M:%S ET')}"
)

# ---------- MAIN RUN ----------
col1, col2, col3 = st.columns([1, 1, 3])
with col1:
    run_clicked = st.button("Run Screener", key="run", type="primary", use_container_width=True, help="Run with default settings", args=None)
    st.markdown("<div class='run-btn' style='position:absolute;top:-1000px'></div>", unsafe_allow_html=True)  # keep CSS scope for button

# Keep behavior: single click runs scan with defaults; advanced settings handled elsewhere in your codebase if present
if run_clicked:
    st.info("Running screener… this may take a bit on first run.")
    out = run_scan(
        tickers=None,
        res_days=21,
        rel_vol_min=1.10,
        relvol_median=False,
        rr_min=2.0,
        stop_mode="safest",
        with_options=True,
        opt_days=30,
    )
    df = out.get("pass_df", pd.DataFrame())
else:
    # initial no-run state shows latest from backend if present (non-blocking); otherwise empty table
    out = run_scan(
        tickers=None,
        res_days=21,
        rel_vol_min=1.10,
        relvol_median=False,
        rr_min=2.0,
        stop_mode="safest",
        with_options=True,
        opt_days=30,
    )
    df = out.get("pass_df", pd.DataFrame())

# ---------- RESULTS TABLE (first thing) ----------
st.subheader("PASS tickers")
if df is None or df.empty:
    st.warning("No PASS tickers found (or CSV not produced).")
else:
    # sort by current price ascending
    sort_cols = [c for c in ["Price", "Ticker"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=[True, True])

    # show a nice, readable table
    show_cols = [c for c in [
        "Ticker","EvalDate","Price","EntryTimeET","Change%","RelVol(TimeAdj63d)",
        "Resistance","TP","RR_to_Res","RR_to_TP",
        "SupportType","SupportPrice","Risk$",
        "TPReward$","TPReward%","ResReward$","ResReward%",
        "DailyATR","DailyCap",
        "OptExpiry","BuyK","SellK","Width","DebitMid","DebitCons",
        "MaxProfitMid","MaxProfitCons","RR_Spread_Mid","RR_Spread_Cons","BreakevenMid"
    ] if c in df.columns]

    st.dataframe(
        df[show_cols],
        use_container_width=True,
        hide_index=True
    )

    # ---------- WHY BUY (toggle for each) ----------
    st.subheader("Explain each PASS (WHY BUY)")
    for _, row in df.iterrows():
        tkr = row["Ticker"]
        with st.expander(f"WHY BUY — {tkr}", expanded=False):
            html = _why_buy_html(row)
            st.markdown(html, unsafe_allow_html=True)

    # ---------- Copy for Google Sheets (collapsed) ----------
    with st.expander("Copy table for Google Sheets (pipe-delimited)", expanded=False):
        pipe_cols = [c for c in df.columns]  # keep everything; user asked to include all columns
        header = "|".join(pipe_cols)
        lines = [header]
        for _, r in df.iterrows():
            parts = []
            for c in pipe_cols:
                val = r.get(c, "")
                if pd.isna(val):
                    parts.append("")
                else:
                    s = str(val)
                    parts.append(s.replace("|","/"))
            lines.append("|".join(parts))
        block = "\n".join(lines)
        st.code(block, language="text")

