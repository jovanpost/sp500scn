# app.py — Streamlit UI for the swing/options screener

import os
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st

# Your core scanner (already in the repo)
from swing_options_screener import run_scan

# ---------- Small helpers ----------
def _fmt_usd(x, nd=2):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    return f"${x:,.{nd}f}"

def _fmt_pct(x, nd=2, signed=False):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    sign = "+" if (signed and x >= 0) else ""
    return f"{sign}{x:.{nd}f}%"

def _safe(v, nd=None):
    if v is None:
        return "—"
    if isinstance(v, (float, np.floating)):
        if np.isnan(v) or np.isinf(v):
            return "—"
        return f"{v:.{nd}f}" if nd is not None else f"{v}"
    return str(v)

def _mk_copy_psv(df: pd.DataFrame) -> str:
    # Build pipe-delimited table with all known columns in order if present
    cols = [
        "Ticker","EvalDate","Price","EntryTimeET","Change%","RelVol(TimeAdj63d)",
        "Resistance","TP","RR_to_Res","RR_to_TP","SupportType","SupportPrice","Risk$",
        "TPReward$","TPReward%","ResReward$","ResReward%","DailyATR","DailyCap",
        "Hist21d_PassCount","Hist21d_Max%","Hist21d_Examples","ResLookbackDays","Prices",
        "Session","EntrySrc","VolSrc",
        "OptExpiry","BuyK","SellK","Width","DebitMid","DebitCons","MaxProfitMid",
        "MaxProfitCons","RR_Spread_Mid","RR_Spread_Cons","BreakevenMid","PricingNote",
    ]
    present = [c for c in cols if c in df.columns]
    out_lines = []
    out_lines.append("|".join(present))
    for _, r in df.iterrows():
        row = []
        for c in present:
            val = r[c]
            # keep raw for Sheets (don’t prettify)
            row.append("" if pd.isna(val) else str(val))
        out_lines.append("|".join(row))
    return "\n".join(out_lines)

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
    daily_atr = float(row["DailyATR"]) if row.get("DailyATR", np.nan) == row.get("DailyATR", np.nan) else np.nan
    # simple ATR “week/month” approximations
    weekly_atr = daily_atr * 5 if np.isfinite(daily_atr) else np.nan
    monthly_atr = daily_atr * 21 if np.isfinite(daily_atr) else np.nan

    # volume/relvol
    relvol = float(row.get("RelVol(TimeAdj63d)")) if "RelVol(TimeAdj63d)" in row else np.nan
    vol_note = "—"
    if np.isfinite(relvol):
        # relvol of 1.20 ~ 20% above typical pace
        vol_note = f"about **{_fmt_pct((relvol - 1.0)*100, 0, signed=True)}** vs typical pace (time-adjusted)."

    # intraday color
    chg = float(row.get("Change%")) if "Change%" in row else np.nan

    # options (if present)
    exp = row.get("OptExpiry", "")
    buyk = row.get("BuyK", "")
    sellk = row.get("SellK", "")
    width = row.get("Width", "")

    opt_line = ""
    if all(k in row for k in ["OptExpiry","BuyK","SellK"]):
        if pd.notna(exp) and pd.notna(buyk) and pd.notna(sellk) and str(exp) != "":
            opt_line = f" via the **{_fmt_usd(float(buyk),0)}/{_fmt_usd(float(sellk),0)}** vertical call spread expiring **{exp}**"
        else:
            opt_line = " (vertical call spread suggestion pending quotes)"

    # history examples (already string)
    examples = str(row.get("Hist21d_Examples","")).strip()
    pass_count = row.get("Hist21d_PassCount","—")
    need_pct = tp_pct  # required move basis

    # narrative
    md = []
    md.append(
        f"**{t}** is a buy{opt_line} because it **recently reached about {_fmt_usd(res,0)} (resistance)** "
        f"and now trades near **{_fmt_usd(price,2)} (current price)**. "
        f"That makes the **target at {_fmt_usd(tp,2)}** feel realistic."
    )
    md.append("")
    md.append("**Why this setup makes sense**")
    bullets = [
        f"**Reward vs. risk** from where buyers have stepped in before "
        f"(**{_fmt_usd(sup_px,2)} support**) up to resistance is about "
        f"**{rr_res:.2f}:1** (to the nearer target it’s **{rr_tp:.2f}:1**).",
        f"**Move needed to TP:** **{_fmt_usd(tp_dollar,2)}** (≈ **{_fmt_pct(tp_pct,2)}**).",
    ]
    # ATR trio in plain English
    if np.isfinite(daily_atr):
        bullets.append(
            f"**Volatility runway (ATR):** daily ATR is **{_fmt_usd(daily_atr,2)}**, "
            f"so a typical month (~21 trading days) allows **~{_fmt_usd(monthly_atr,2)}** of movement. "
            f"For context, a typical week allows **~{_fmt_usd(weekly_atr,2)}**."
        )
    # intraday tone + volume
    if np.isfinite(chg):
        bullets.append(
            f"**Today’s tone & volume:** the stock is **{_fmt_pct(chg*100,2,signed=True)}** on the day; "
            f"volume is {vol_note}"
        )

    md.extend([f"- {b}" for b in bullets])

    # History section
    md.append("")
    md.append(
        f"**History check (21 trading days):** need about **{_fmt_pct(need_pct,2)}** to hit TP. "
        f"Over the past year, **{pass_count}** separate 21-day windows met or exceeded that."
    )
    if examples:
        md.append("**Examples:**")
        for ex in [e.strip() for e in examples.split(";") if e.strip()]:
            # ex already like "2025-06-23:+35.06%"
            md.append(f"- {ex}")

    # footer
    ts = row.get("EntryTimeET","")
    if ts:
        md.append("")
        md.append(f"_Data as of **{ts}**._")

    return "\n".join(md)


# ---------- Page & styles ----------
st.set_page_config(page_title="S&P 500 Options Screener", layout="wide")

# Red primary action button (minimal CSS tweak)
st.markdown("""
<style>
button[kind="primary"] { background-color: #dc2626 !important; } /* red-600 */
</style>
""", unsafe_allow_html=True)

st.title("📈 S&P 500 Options Screener")

# Top controls (simple)
colA, colB = st.columns([1,2])
with colA:
    run_clicked = st.button("Run Screener", type="primary")
with colB:
    opt = st.expander("Options (advanced)", expanded=False)
    with opt:
        st.caption("Most users can leave these as-is.")
        rr_min = st.slider("Minimum RR to Resistance", 1.0, 5.0, 2.0, 0.25)
        relvol_min = st.slider("Minimum Relative Volume (time-adjusted)", 0.8, 3.0, 1.10, 0.05)
        res_days = st.slider("Resistance lookback (days)", 10, 60, 21, 1)
        stop_mode = st.selectbox("Stop preference", ["safest", "structure"], index=0)
        opt_days = st.slider("Target option expiry (days)", 10, 60, 30, 1)
        include_options = st.checkbox("Include option spread suggestion", True)
        # optional custom universe
        tickers_text = st.text_area("Tickers (comma/space/newline separated). Leave empty for defaults.",
                                    value="", height=80)

# ---------- Run or warm welcome ----------
if run_clicked:
    with st.status("Running screener…", expanded=True):
        # Forward parameters to your core scanner
        out = run_scan(
            tickers=[t.strip().upper() for t in tickers_text.replace("\n",",").replace(" ", ",").split(",") if t.strip()] if tickers_text.strip() else None,
            res_days=res_days,
            rel_vol_min=relvol_min,
            relvol_median=False,
            rr_min=rr_min,
            stop_mode=stop_mode,
            with_options=include_options,
            opt_days=opt_days,
        )

    df: pd.DataFrame = out.get("pass_df", pd.DataFrame())
    if df is None or df.empty:
        st.warning("No PASS tickers found (or CSV not produced).")
    else:
        # Sort by current price (ascending)
        if "Price" in df.columns:
            df = df.sort_values("Price", ascending=True).reset_index(drop=True)

        # ===== 1) PASS TABLE (first) =====
        st.subheader("PASS tickers")
        st.dataframe(
            df,
            use_container_width=True,
            height=min(600, 48 + 31*max(4, len(df))),
        )

        # ===== 2) WHY BUY (expandable per ticker) =====
        st.subheader("Explain each PASS (WHY BUY)")
        for _, row in df.iterrows():
            with st.expander(f"WHY BUY — {row['Ticker']}", expanded=False):
                md = _why_buy_card(row)
                st.markdown(md)

        # ===== 3) Google Sheets (hidden by default) =====
        with st.expander("Copy for Google Sheets (pipe-delimited)", expanded=False):
            psv = _mk_copy_psv(df)
            st.code(psv, language="text")

else:
    st.info("Click **Run Screener** to fetch the latest candidates. Use **Options** if you need to tweak gates.")

