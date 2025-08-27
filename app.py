# app.py
import streamlit as st
import pandas as pd
import numpy as np

# Our scanning library
from swing_options_screener import run_scan

# ---------- Page config & styles ----------
st.set_page_config(page_title="S&P 500 Options Screener", layout="wide")

PRIMARY_RED = "#e11d48"  # Tailwind rose-600 vibe
st.markdown(
    f"""
    <style>
      .red-btn>button {{
        background:{PRIMARY_RED} !important;
        color:white !important;
        border:0 !important;
        border-radius:10px !important;
        padding:0.6rem 1.2rem !important;
        font-weight:600 !important;
      }}
      .stAlert p, .stMarkdown p {{ margin-bottom: 0.3rem; }}
      .muted {{ color:#6b7280; font-size:0.9rem; }}
      .kpi {{ font-weight:700; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📈 S&P 500 Options Screener")

# ---------- Helpers ----------
COLS_ORDER = [
    "Ticker","EvalDate","Price","EntryTimeET","Change%","RelVol(TimeAdj63d)",
    "Resistance","TP","RR_to_Res","RR_to_TP","SupportType","SupportPrice","Risk$",
    "TPReward$","TPReward%","ResReward$","ResReward%","DailyATR","DailyCap",
    "Hist21d_PassCount","Hist21d_Max%","Hist21d_Examples","ResLookbackDays","Prices",
    "Session","EntrySrc","VolSrc",
    "OptExpiry","BuyK","SellK","Width","DebitMid","DebitCons",
    "MaxProfitMid","MaxProfitCons","RR_Spread_Mid","RR_Spread_Cons","BreakevenMid","PricingNote",
]

def _safe(v, default=""):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return default
    return v

def _fmt_money(x):
    try:
        x = float(x)
        return f"${x:,.2f}"
    except Exception:
        return str(x)

def _fmt_pct(x):
    try:
        x = float(x)
        return f"{x:.2f}%"
    except Exception:
        return str(x)

def _pipe_df(df: pd.DataFrame) -> str:
    # produce pipe-delimited text with all known columns in a stable order
    cols = [c for c in COLS_ORDER if c in df.columns]
    out = ["|".join(cols)]
    for _, r in df[cols].iterrows():
        cells = []
        for c in cols:
            v = r.get(c, "")
            s = "" if pd.isna(v) else str(v)
            s = s.replace("|", "/")
            cells.append(s)
        out.append("|".join(cells))
    return "\n".join(out)

def _why_buy_card(row: pd.Series) -> str:
    """
    Plain-English rationale. Uses parentheses to introduce technical terms,
    but keeps the narrative understandable for non-traders.
    """
    t = row["Ticker"]
    price = float(row["Price"])
    tp = float(row["TP"])
    res = float(row["Resistance"])
    rr_res = float(row["RR_to_Res"])
    rr_tp = float(row["RR_to_TP"])
    sup_type = str(row["SupportType"])
    sup_px = float(row["SupportPrice"])
    tp_dollar = float(row["TPReward$"])
    tp_pct = float(row["TPReward%"])
    daily_atr = float(row["DailyATR"]) if pd.notna(row.get("DailyATR")) else np.nan
    weekly_atr = daily_atr * 5 if np.isfinite(daily_atr) else np.nan
    monthly_atr = daily_atr * 21 if np.isfinite(daily_atr) else np.nan

    # volume (Finviz-style time-adjusted relvol)
    relvol = float(row.get("RelVol(TimeAdj63d)")) if "RelVol(TimeAdj63d)" in row else np.nan
    vol_clause = ""
    if np.isfinite(relvol):
        vol_clause = f" and volume is running about **{(relvol-1)*100:.0f}%** vs its typical pace (time-adjusted)"

    # today's change (already in %)
    chg = float(row.get("Change%")) if "Change%" in row else np.nan
    chg_display = f"{chg:.2f}%" if np.isfinite(chg) else None

    # options line if present
    exp = row.get("OptExpiry", "")
    buyk = row.get("BuyK", "")
    sellk = row.get("SellK", "")
    opt_line = ""
    if exp and buyk and sellk:
        opt_line = f" via the **${buyk}/{sellk}** vertical call spread expiring **{exp}**"

    examples = str(row.get("Hist21d_Examples","")).strip()
    pass_count = row.get("Hist21d_PassCount","—")

    # Intro sentence
    md = []
    md.append(
        f"**{t}** is a buy{opt_line} because it recently reached about **${res:.2f}** "
        f"(resistance) and now trades near **${price:.2f}** (current price). "
        f"That makes a target at **${tp:.2f}** feel realistic."
    )

    # Reasons list
    md.append("")
    md.append("**Why this setup makes sense**")

    bullets = []
    bullets.append(
        f"**Reward vs. risk:** from **${sup_px:.2f}** (where buyers stepped in before; *support*) "
        f"up to resistance is about **{rr_res:.2f}:1**. To the nearer target it’s **{rr_tp:.2f}:1**."
    )
    bullets.append(
        f"**Move needed to the target:** **${tp_dollar:.2f}** (≈ **{tp_pct:.2f}%**)."
    )
    if np.isfinite(daily_atr):
        bullets.append(
            f"**Volatility runway (ATR):** daily ATR is **${daily_atr:.2f}**. "
            f"That suggests roughly **${weekly_atr:.2f}** of typical movement over a week "
            f"and **${monthly_atr:.2f}** over ~21 trading days."
        )
    today_line = None
    if chg_display:
        today_line = f"**Today’s tone & volume:** price is up **{chg_display}** today{vol_clause}."
        bullets.append(today_line)

    md.extend([f"- {b}" for b in bullets])

    # History
    md.append("")
    md.append(
        f"**History check (21 trading days):** to reach the target, the stock needs about "
        f"**{tp_pct:.2f}%** from here. Over the past year, **{pass_count}** separate 21-day windows "
        f"met or exceeded that move."
    )
    if examples:
        md.append("**Examples:**")
        for ex in [e.strip() for e in examples.split(";") if e.strip()]:
            md.append(f"- {ex}")

    ts = row.get("EntryTimeET","")
    if ts:
        md.append("")
        md.append(f"<span class='muted'>Data as of **{ts}**.</span>")

    return "\n".join(md)

# ---------- Run button ----------
col_left, col_mid, col_right = st.columns([1.4, 1, 1])
with col_left:
    st.caption("Click to screen the universe with default rules.")
with col_mid:
    pass
with col_right:
    st.caption("")

run_clicked = st.container()
with run_clicked:
    st.markdown('<div class="red-btn">', unsafe_allow_html=True)
    go = st.button("Run Screener", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Output areas ----------
console = st.expander("Console output", expanded=False)
table_area = st.container()
why_area = st.container()
copy_area = st.expander("Copy table (pipe-delimited for Google Sheets)", expanded=False)

if go:
    with st.status("Running screener… first run may take a bit.", expanded=False) as status:
        # Call scanner with defaults; it returns {'pass_df': df}
        out = run_scan(
            tickers=None,                # use default universe from the module
            res_days=21,
            rel_vol_min=1.10,
            relvol_median=False,
            rr_min=2.0,
            stop_mode="safest",
            with_options=True,          # include options columns
            opt_days=30,
        )
        status.update(label="Done", state="complete", expanded=False)

    df = out.get("pass_df", pd.DataFrame())
    if df.empty:
        st.warning("No PASS tickers found (or CSV not produced).")
    else:
        # sort by current price (ascending)
        if "Price" in df.columns:
            df = df.sort_values("Price", ascending=True).reset_index(drop=True)

        # Nicely formatted preview table
        show_cols = [c for c in [
            "Ticker","EvalDate","Price","EntryTimeET","Change%","RelVol(TimeAdj63d)",
            "Resistance","TP","RR_to_Res","RR_to_TP","SupportType","SupportPrice",
            "TPReward$","TPReward%","DailyATR","Session"
        ] if c in df.columns]

        fmt_df = df.copy()
        for c in ["Price","TP","Resistance","SupportPrice","TPReward$","DailyATR"]:
            if c in fmt_df.columns:
                fmt_df[c] = fmt_df[c].map(_fmt_money)
        for c in ["Change%","TPReward%","RR_to_Res","RR_to_TP","RelVol(TimeAdj63d)"]:
            if c in fmt_df.columns:
                if c.endswith("%"):
                    fmt_df[c] = fmt_df[c].map(_fmt_pct)
                else:
                    # Keep RR and RelVol numeric formatting
                    fmt_df[c] = fmt_df[c].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")

        st.subheader("PASS tickers")
        st.dataframe(fmt_df[show_cols], use_container_width=True, hide_index=True)

        # WHY BUY expanders (one per row)
        st.subheader("Explain each PASS (WHY BUY)")
        for _, row in df.iterrows():
            with st.expander(f"WHY BUY — {row['Ticker']}", expanded=False):
                st.markdown(_why_buy_card(row), unsafe_allow_html=True)

        # Copy text (Google Sheets)
        with copy_area:
            st.code(_pipe_df(df), language="text")

else:
    st.info("Click **Run Screener** to fetch fresh results.")

