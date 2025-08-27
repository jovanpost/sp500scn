# app.py
import streamlit as st
import pandas as pd
import numpy as np

from swing_options_screener import run_scan

# ---------------- Page & styles ----------------
st.set_page_config(page_title="S&P 500 Options Screener", layout="wide")

PRIMARY_RED = "#e11d48"
st.markdown(
    f"""
    <style>
      .red-btn>button {{
        background:{PRIMARY_RED} !important;
        color:#fff !important;
        border:0 !important;
        border-radius:10px !important;
        padding:0.65rem 1.25rem !important;
        font-weight:700 !important;
      }}
      .muted {{ color:#6b7280; font-size:0.9rem; }}
      .whybuy h4 {{ margin: 0 0 .4rem 0; }}
      .whybuy p  {{ margin: .25rem 0; line-height:1.55; }}
      .whybuy ul {{ margin:.35rem 0 .5rem 1.1rem; }}
      .whybuy li {{ margin:.18rem 0; }}
      .whybuy strong {{ font-weight:700; }}
      .tbl .stDataFrame {{ border-radius:10px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📈 S&P 500 Options Screener")

# --------------- Helpers ----------------
COLS_ORDER = [
    "Ticker","EvalDate","Price","EntryTimeET","Change%","RelVol(TimeAdj63d)",
    "Resistance","TP","RR_to_Res","RR_to_TP","SupportType","SupportPrice","Risk$",
    "TPReward$","TPReward%","ResReward$","ResReward%","DailyATR","DailyCap",
    "Hist21d_PassCount","Hist21d_Max%","Hist21d_Examples","ResLookbackDays","Prices",
    "Session","EntrySrc","VolSrc",
    "OptExpiry","BuyK","SellK","Width","DebitMid","DebitCons",
    "MaxProfitMid","MaxProfitCons","RR_Spread_Mid","RR_Spread_Cons","BreakevenMid","PricingNote",
]

def _fmt_money(x):
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return str(x)

def _fmt_pct(x):
    try:
        return f"{float(x):.2f}%"
    except Exception:
        return str(x)

def _pipe_df(df: pd.DataFrame) -> str:
    cols = [c for c in COLS_ORDER if c in df.columns]
    out = ["|".join(cols)]
    for _, r in df[cols].iterrows():
        cells = []
        for c in cols:
            v = r.get(c, "")
            s = "" if pd.isna(v) else str(v)
            cells.append(s.replace("|","/"))
        out.append("|".join(cells))
    return "\n".join(out)

def _num(v, default=np.nan):
    try:
        f = float(v)
        if np.isfinite(f):
            return f
        return default
    except Exception:
        return default

def _why_buy_html(row: pd.Series) -> str:
    """
    Plain-English WHY BUY rendered as HTML (no Markdown),
    so symbols never turn into italics/bullets accidentally.
    """
    t        = row["Ticker"]
    price    = _num(row.get("Price"))
    tp       = _num(row.get("TP"))
    res      = _num(row.get("Resistance"))
    rr_res   = _num(row.get("RR_to_Res"))
    rr_tp    = _num(row.get("RR_to_TP"))
    sup_type = str(row.get("SupportType",""))
    sup_px   = _num(row.get("SupportPrice"))
    tp_dollars     = _num(row.get("TPReward$"))
    tp_%     = _num(row.get("TPReward%"))
    d_atr    = _num(row.get("DailyATR"))
    w_atr    = d_atr*5  if np.isfinite(d_atr) else np.nan
    m_atr    = d_atr*21 if np.isfinite(d_atr) else np.nan
    relvol   = _num(row.get("RelVol(TimeAdj63d)"))
    chg      = _num(row.get("Change%"))
    exp      = str(row.get("OptExpiry","")).strip()
    buyk     = str(row.get("BuyK","")).strip()
    sellk    = str(row.get("SellK","")).strip()
    examples = str(row.get("Hist21d_Examples","")).strip()
    pass_ct  = str(row.get("Hist21d_PassCount","—"))
    ts       = str(row.get("EntryTimeET","")).strip()

    opt_line = ""
    if exp and buyk and sellk:
        opt_line = f" via the <strong>${buyk}/{sellk}</strong> vertical call spread expiring <strong>{exp}</strong>"

    vol_clause = ""
    if np.isfinite(relvol):
        vol_clause = f" and volume is running about <strong>{(relvol-1)*100:.0f}%</strong> vs its typical pace (time-adjusted)"

    chg_text = ""
    if np.isfinite(chg):
        chg_text = f"price is up <strong>{chg:.2f}%</strong> today{vol_clause}."

    items = []
    items.append(
        f"<strong>Reward vs. risk:</strong> from <strong>{_fmt_money(sup_px)}</strong> "
        f"(where buyers have stepped in before; <em>support</em>) up to resistance is about "
        f"<strong>{rr_res:.2f}:1</strong>. To the nearer target it’s <strong>{rr_tp:.2f}:1</strong>."
    )
    items.append(
        f"<strong>Move needed to the target:</strong> <strong>{_fmt_money(tp_dollars)}</strong> "
        f"(≈ <strong>{tp_%:.2f}%</strong>)."
    )
    if np.isfinite(d_atr):
        items.append(
            f"<strong>Volatility runway (ATR):</strong> daily ATR is <strong>{_fmt_money(d_atr)}</strong>. "
            f"That suggests roughly <strong>{_fmt_money(w_atr)}</strong> of typical movement over a week "
            f"and <strong>{_fmt_money(m_atr)}</strong> over ~21 trading days."
        )
    if chg_text:
        items.append(f"<strong>Today’s tone &amp; volume:</strong> {chg_text}")

    lis = "".join(f"<li>{it}</li>" for it in items)

    ex_block = ""
    if examples:
        ex_items = "".join(f"<li>{e.strip()}</li>" for e in examples.split(";") if e.strip())
        ex_block = f"<p><strong>Examples:</strong></p><ul>{ex_items}</ul>"

    html = f"""
    <div class="whybuy">
      <p><strong>{t}</strong> is a buy{opt_line} because it recently reached about
         <strong>{_fmt_money(res)}</strong> (resistance) and now trades near
         <strong>{_fmt_money(price)}</strong> (current price). That makes a target at
         <strong>{_fmt_money(tp)}</strong> feel realistic.</p>

      <h4>Why this setup makes sense</h4>
      <ul>
        {lis}
      </ul>

      <p><strong>History check (21 trading days):</strong> to reach the target, the stock needs about
         <strong>{tp_%:.2f}%</strong> from here. Over the past year, <strong>{pass_ct}</strong> separate 21-day
         windows met or exceeded that move.</p>
      {ex_block}

      <p class="muted">Data as of <strong>{ts}</strong>.</p>
    </div>
    """
    return html

# --------------- UI: Run button ---------------
colA, colB, colC = st.columns([1.3,1,1])
with colA:
    st.caption("Click to screen the universe with default rules.")
with colB:
    pass
with colC:
    st.caption("")

st.markdown('<div class="red-btn">', unsafe_allow_html=True)
go = st.button("Run Screener", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

console = st.expander("Console output", expanded=False)
table_area = st.container()
why_area = st.container()
copy_area = st.expander("Copy table (pipe-delimited for Google Sheets)", expanded=False)

# --------------- Run ---------------
if go:
    with st.status("Running screener… this may take a bit on first run.", expanded=False) as status:
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
        status.update(label="Done", state="complete", expanded=False)

    df = out.get("pass_df", pd.DataFrame())

    if df.empty:
        st.warning("No PASS tickers found (or CSV not produced).")
    else:
        # Sort & prettify table
        if "Price" in df.columns:
            df = df.sort_values("Price", ascending=True).reset_index(drop=True)

        show_cols = [c for c in [
            "Ticker","EvalDate","Price","EntryTimeET","Change%","RelVol(TimeAdj63d)",
            "Resistance","TP","RR_to_Res","RR_to_TP","SupportType","SupportPrice",
            "TPReward$","TPReward%","DailyATR","Session"
        ] if c in df.columns]

        fmt = df.copy()
        for c in ["Price","TP","Resistance","SupportPrice","TPReward$","DailyATR"]:
            if c in fmt.columns:
                fmt[c] = fmt[c].map(_fmt_money)
        if "Change%" in fmt.columns:
            fmt["Change%"] = fmt["Change%"].map(_fmt_pct)
        for c in ["RR_to_Res","RR_to_TP","RelVol(TimeAdj63d)"]:
            if c in fmt.columns:
                fmt[c] = fmt[c].map(lambda x: f"{float(x):.2f}" if pd.notna(x) else "")

        st.subheader("PASS tickers")
        with st.container():
            st.dataframe(fmt[show_cols], use_container_width=True, hide_index=True)

        st.subheader("Explain each PASS (WHY BUY)")
        for _, row in df.iterrows():
            with st.expander(f"WHY BUY — {row['Ticker']}", expanded=False):
                st.markdown(_why_buy_html(row), unsafe_allow_html=True)

        with copy_area:
            st.code(_pipe_df(df), language="text")
else:
    st.info("Click **Run Screener** to fetch fresh results.")


