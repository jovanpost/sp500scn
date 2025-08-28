# app.py — Streamlit UI for Swing Options Screener (UNADJUSTED)
# - Scanner tab: big red RUN button -> PASS table -> WHY BUY expanders -> Copy to Google Sheets
# - History tab: auto-load all CSVs in data/history/pass_*.csv
# - Debugger tab: plain-English explanation + numbers for a single ticker

import os
import io
import glob
import re
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st

# Import your screener as a module
import swing_options_screener as sos  # must be in repo root

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _safe(x):
    return "" if x is None else str(x)

def _usd(x, nd=2):
    try:
        fx = float(x)
        # show more precision for very small ATRs
        if nd is None:
            nd = 4 if abs(fx) < 1 else 2
        return f"${fx:,.{nd}f}"
    except Exception:
        return _safe(x)

def _pct(x, nd=2):
    try:
        fx = float(x)
        return f"{fx:.{nd}f}%"
    except Exception:
        return _safe(x)

def _relvol_human(x):
    try:
        fx = float(x)
        # Express as “+87% vs typical” if >1, else “-30% vs typical”
        delta = (fx - 1.0) * 100.0
        sign = "+" if delta >= 0 else ""
        return f"{sign}{delta:.0f}% vs typical"
    except Exception:
        return _safe(x)

def _mk_bullet(s):
    return f"<li style='margin:0.2rem 0'>{s}</li>"

def _bold(s):
    return f"<strong>{_safe(s)}</strong>"

def _timestamp_et():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S ET")

# -----------------------------------------------------------------------------
# WHY BUY builder (plain English; readable numbers with $/%; includes ATR & volume)
# -----------------------------------------------------------------------------
def build_why_buy_html(row: dict) -> str:
    tkr     = _safe(row.get("Ticker",""))
    entry   = _usd(row.get("Price"))
    tp      = _usd(row.get("TP"))
    res     = _usd(row.get("Resistance"))
    rr_res  = _safe(row.get("RR_to_Res",""))
    rr_tp   = _safe(row.get("RR_to_TP",""))
    change  = _pct(row.get("Change%"))
    relvol  = _relvol_human(row.get("RelVol(TimeAdj63d)"))

    tp_move_abs  = _usd(row.get("TPReward$", None))
    tp_move_pct  = _pct(row.get("TPReward%", None))

    daily_atr    = _usd(row.get("DailyATR", None), nd=None)  # auto nd
    daily_cap    = _usd(row.get("DailyCap", None), nd=2)

    support_type = _safe(row.get("SupportType",""))
    support_px   = _usd(row.get("SupportPrice", None))

    hist_cnt     = _safe(row.get("Hist21d_PassCount",""))
    hist_ex      = _safe(row.get("Hist21d_Examples",""))

    opt_exp      = _safe(row.get("OptExpiry",""))
    buy_k        = _safe(row.get("BuyK",""))
    sell_k       = _safe(row.get("SellK",""))

    header = (
        f"{_bold(tkr)} is a buy via the {_bold(buy_k)}/{_bold(sell_k)} vertical call spread "
        f"expiring {_bold(opt_exp)} because it recently reached a level near {_bold(res)} "
        f"(resistance) and now trades around {_bold(entry)}. "
        f"That makes a target at {_bold(tp)} feel realistic."
    )

    bullets = []
    bullets.append(_mk_bullet(
        f"Reward vs. risk: from where buyers have stepped in before "
        f"({_bold(support_px)} {_bold('support')}) up to resistance is about {_bold(str(rr_res))}:1; "
        f"to the nearer take-profit it’s {_bold(str(rr_tp))}:1."
    ))
    bullets.append(_mk_bullet(
        f"Move needed to TP: {_bold(tp_move_abs)} (≈ {_bold(tp_move_pct)})."
    ))
    bullets.append(_mk_bullet(
        f"Volatility runway (ATR): daily ATR ≈ {_bold(daily_atr)}, which implies roughly "
        f"{_bold(daily_cap)} of potential movement over ~21 trading days."
    ))
    bullets.append(_mk_bullet(
        f"Today’s tone & volume: price is {_bold(change)} on the day; "
        f"volume is running {_bold(relvol)} (time-adjusted)."
    ))

    hist = (
        f"<p style='margin:0.4rem 0 0.2rem 0'><em>History check (21 trading days)</em>: "
        f"over the past year, {_bold(hist_cnt)} separate 21-day windows met or exceeded the required move. "
        f"<em>Examples</em>: { _safe(hist_ex) }</p>"
    )

    bullets_html = "<ul style='padding-left:1.1rem; margin:0.6rem 0 0.2rem 0'>" + "".join(bullets) + "</ul>"

    footer = f"<p style='margin-top:0.3rem; color:#6b7280;'>Data as of {_bold(_timestamp_et())}.</p>"

    return f"<div style='line-height:1.35'>{header}{bullets_html}{hist}{footer}</div>"

# -----------------------------------------------------------------------------
# Streamlit page config & style
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Swing Options Screener (UNADJUSTED)", layout="wide")

# A tiny CSS tweak: red RUN button
st.markdown("""
<style>
div.stButton > button[kind="primary"]{
  background-color:#e11d48 !important; /* rose-600 */
  border-color:#e11d48 !important;
}
</style>
""", unsafe_allow_html=True)

st.title("Swing Options Screener\n(UNADJUSTED)")

tab_scan, tab_hist, tab_dbg = st.tabs(["🔎 Scanner", "📜 History", "🧪 Debugger"])

# -----------------------------------------------------------------------------
# Scanner tab
# -----------------------------------------------------------------------------
with tab_scan:
    c1, c2 = st.columns([1,3])
    with c1:
        run_clicked = st.button("Run Screener", type="primary", use_container_width=True)
    with c2:
        st.caption(f"UI started: {_timestamp_et()}")

    console = st.expander("🖥️ Console output", expanded=False)

    passed_df = None
    pipe_text = ""

    if run_clicked:
        with st.status("Running screener… this may take a bit on first run.", expanded=True):
            # Run the scan — always include options so those columns appear
            out = sos.run_scan(
                tickers=None,
                with_options=True,
            )
            passed_df = out.get("pass_df", pd.DataFrame())

            with console:
                st.write("Processed at", _timestamp_et())

    # If we already have a cached run in memory (e.g., Streamlit rerun), keep it
    if passed_df is None:
        st.info("Click **Run Screener** to generate today’s PASS list.")

    elif passed_df.empty:
        st.warning("No PASS tickers found (or CSV not produced).")

    else:
        # Sort by current Price (lowest -> highest) per your preference
        if "Price" in passed_df.columns:
            try:
                passed_df = passed_df.sort_values(["Price","Ticker"], ascending=[True, True])
            except Exception:
                pass

        st.subheader("PASS tickers")
        st.dataframe(
            passed_df,
            use_container_width=True,
            height=min(520, 220 + 28*len(passed_df))
        )

        # WHY BUY expanders
        st.subheader("Explain each PASS (WHY BUY)")
        for _, row in passed_df.iterrows():
            sym = _safe(row.get("Ticker",""))
            with st.expander(f"WHY BUY — {sym}", expanded=False):
                html = build_why_buy_html(row)
                st.markdown(html, unsafe_allow_html=True)

        # Google Sheets copy expander (pipe-delimited)
        with st.expander("Copy table (pipe-delimited for Google Sheets)", expanded=False):
            cols = list(passed_df.columns)
            buf = io.StringIO()
            buf.write("|".join(cols) + "\n")
            for _, r in passed_df.iterrows():
                line = "|".join(_safe(r.get(c, "")) for c in cols)
                buf.write(line + "\n")
            pipe_text = buf.getvalue()
            st.code(pipe_text, language="text")

# -----------------------------------------------------------------------------
# History tab — auto-load all data/history/pass_*.csv
# -----------------------------------------------------------------------------
HISTORY_DIR = "data/history"

def _parse_run_ts_from_filename(fname: str):
    m = re.search(r"pass_(\d{8})-(\d{6})", fname)
    if not m:
        return ""
    ymd, hms = m.group(1), m.group(2)
    return f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]} {hms[:2]}:{hms[2:4]}:{hms[4:]}"

@st.cache_data(show_spinner=False)
def load_history_df():
    files = sorted(glob.glob(os.path.join(HISTORY_DIR, "pass_*.csv")))
    if not files:
        return None, []

    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            run_file = os.path.basename(f)
            df["RunFile"] = run_file
            df["RunET"]   = _parse_run_ts_from_filename(run_file)  # from filename
            frames.append(df)
        except Exception:
            continue

    if not frames:
        return None, files

    all_hist = pd.concat(frames, ignore_index=True)
    if "RunET" in all_hist.columns:
        all_hist["RunET_sort"] = pd.to_datetime(all_hist["RunET"], errors="coerce")
        all_hist = all_hist.sort_values(["RunET_sort","Ticker"], ascending=[False, True]) \
                           .drop(columns=["RunET_sort"])
    return all_hist, files

with tab_hist:
    st.header("📜 History")
    hist_df, hist_files = load_history_df()

    if hist_df is None:
        st.warning(f"No history files found in `{HISTORY_DIR}` yet.")
        st.caption("Once the GitHub Action writes daily CSVs, they will appear here automatically.")
    else:
        st.caption(f"Loaded **{len(hist_files)}** run file(s), **{len(hist_df)}** rows.")
        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            f_ticker = st.text_input("Filter by ticker (optional):", "").strip().upper()
        with c2:
            f_from = st.date_input("From date (EvalDate):", value=None)
        with c3:
            f_to = st.date_input("To date (EvalDate):", value=None)

        df = hist_df.copy()
        if "EvalDate" in df.columns:
            df["_EvalDate"] = pd.to_datetime(df["EvalDate"], errors="coerce")
        else:
            df["_EvalDate"] = pd.NaT

        if f_ticker:
            df = df[df["Ticker"].astype(str).str.upper() == f_ticker]
        if f_from:
            df = df[df["_EvalDate"] >= pd.to_datetime(f_from)]
        if f_to:
            df = df[df["_EvalDate"] <= pd.to_datetime(f_to)]

        preferred = [
            "RunET","Ticker","EvalDate","Price","TP","Resistance","RR_to_TP","RR_to_Res",
            "Change%","RelVol(TimeAdj63d)","SupportType","SupportPrice","TPReward$","TPReward%",
            "OptExpiry","BuyK","SellK","Width","DebitMid","DebitCons"
        ]
        ordered = [c for c in preferred if c in df.columns] + \
                  [c for c in df.columns if c not in preferred + ["_EvalDate"]]
        df = df[ordered]

        st.dataframe(df, use_container_width=True, height=520)

        csv = df.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download filtered history CSV",
            data=csv,
            file_name="pass_history_filtered.csv",
            mime="text/csv",
            type="secondary"
        )

# -----------------------------------------------------------------------------
# Debugger tab — plain English diagnosis + numbers
# -----------------------------------------------------------------------------
def diagnose_ticker_plain(ticker: str,
                          res_days=sos.RES_LOOKBACK_DEFAULT,
                          rel_vol_min=sos.REL_VOL_MIN_DEFAULT,
                          use_relvol_median=False,
                          rr_min=sos.RR_MIN_DEFAULT,
                          stop_mode="safest"):
    """
    Returns (ok, html) where ok=True means PASS, False means FAIL; html is human readable.
    """
    try:
        # Re-compute core inputs to show numbers
        df = sos._get_history(ticker)
        if df is None or df.empty:
            return False, f"{_bold(ticker)} — could not load price history."

        entry, prev_close, today_vol, src, entry_ts = sos.get_entry_prevclose_todayvol(df, ticker)
        change = (entry - prev_close) / prev_close * 100.0 if (np.isfinite(entry) and np.isfinite(prev_close)) else np.nan
        relvol = sos.compute_relvol_time_adjusted(df, today_vol, use_median=use_relvol_median)

        row, reason = sos.evaluate_ticker(
            ticker,
            res_days=res_days,
            rel_vol_min=rel_vol_min,
            use_relvol_median=use_relvol_median,
            rr_min=rr_min,
            prefer_stop=stop_mode,
        )

        # Build message
        parts = []
        parts.append(f"<p><strong>{ticker}</strong> session={src.get('session','?')} · "
                     f"EntrySrc={_safe(src.get('entry_src',''))} · VolSrc={_safe(src.get('vol_src',''))}</p>")
        parts.append("<ul>")

        if np.isfinite(entry) and np.isfinite(prev_close):
            parts.append(_mk_bullet(f"Entry used: {_usd(entry)}; previous close: {_usd(prev_close)} "
                                    f"({ _pct(change) } on day)."))
        if np.isfinite(relvol):
            parts.append(_mk_bullet(f"Relative volume (time-adjusted 63d): { _relvol_human(relvol) }."))

        if reason is None and row:
            parts.append(_mk_bullet(f"Resistance: {_usd(row.get('Resistance'))}; TP: {_usd(row.get('TP'))}."))
            parts.append(_mk_bullet(f"Reward-to-risk — to resistance: {_safe(row.get('RR_to_Res'))}:1; "
                                    f"to TP: {_safe(row.get('RR_to_TP'))}:1."))
            parts.append(_mk_bullet(f"Move to TP: {_usd(row.get('TPReward$'))} "
                                    f"({ _pct(row.get('TPReward%')) })."))
            parts.append("</ul>")
            parts.append("<p><strong>PASS ✅</strong></p>")
            return True, "".join(parts)
        else:
            # Translate common reasons
            readable = {
                "not_up_on_day": "Price is not up on the day (needs green day).",
                "relvol_low_timeadj": "Relative volume is below the minimum (time-adjusted).",
                "no_upside_to_resistance": "Not enough upside to the prior high (resistance).",
                "atr_capacity_short_vs_tp": "ATR suggests insufficient typical movement to reach TP.",
                "history_21d_zero_pass": "No 21-day windows in the last year met the required move.",
                "no_valid_support": "No valid support below price for a reasonable stop.",
                "rr_to_res_below_min": "Reward-to-risk to resistance is below the minimum.",
                "non_positive_risk": "Calculated risk (entry − stop) is not positive.",
            }.get(reason, f"Failed gate: {reason}")
            parts.append("</ul>")
            parts.append(f"<p><strong>FAIL ❌</strong> — {readable}</p>")
            return False, "".join(parts)

    except Exception as e:
        return False, f"Error diagnosing {ticker}: {e}"

with tab_dbg:
    st.header("🧪 Debugger")
    dbg_ticker = st.text_input("Explain a ticker", placeholder="e.g., WMT, INTC, NOW").strip().upper()
    if st.button("Explain", type="secondary"):
        if not dbg_ticker:
            st.warning("Enter a ticker symbol.")
        else:
            ok, html = diagnose_ticker_plain(dbg_ticker)
            st.markdown(html, unsafe_allow_html=True)

