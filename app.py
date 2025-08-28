# ============================================================
# TABLE OF CONTENTS for app.py
#
#  1. Imports & Safe third-party glue
#  2. App constants (paths, titles, etc.)
#  3. Streamlit page config + global CSS
#  4. Small formatting helpers
#  5. WHY BUY explanation builder (plain English)
#  6. CSV helpers (latest pass file, outcomes)
#  7. Outcomes counters (robust to minimal/extended schemas)
#  8. Debugger (plain-English reasons with numbers)
# 9. TAB – Scanner (table → WHY BUY → Sheets export)
#  10. UI – Tabs
# ============================================================

# ============================================================
# 1. Imports & Safe third-party glue
# ============================================================
import os, glob, io, pandas as pd, streamlit as st
from datetime import datetime
import swing_options_screener as sos  # core engine

# ============================================================
# 2. App constants (paths, titles, etc.)
# ============================================================
PASS_DIR = "data/pass_logs"
HIST_DIR = "data/history"
OUT_FILE = os.path.join(HIST_DIR, "outcomes.csv")

st.set_page_config(page_title="Swing Options Scanner", layout="wide")

# ============================================================
# 3. Streamlit page config + global CSS
# ============================================================
st.markdown(
    """
    <style>
    div.stButton > button:first-child {
        background-color: red;
        color:white;
        font-weight: bold;
    }
    .whybuy {font-size: 16px; line-height: 1.5;}
    .debugbox {background:#222; color:#eee; padding:10px; border-radius:8px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 4. Small formatting helpers
# ============================================================
def _bold(x): return f"<b>{x}</b>"
def _usd(x, nd=2):
    try: return f"${x:.{nd}f}"
    except: return str(x)
def _pct(x):
    try: return f"{float(x):.2f}%"
    except: return str(x)
def _safe(x): return str(x) if x is not None else ""

# ============================================================
# 5. WHY BUY explanation builder (plain English)
# ============================================================
def build_why_buy_html(row: dict) -> str:
    tkr = _safe(row.get("Ticker",""))
    price = _usd(row.get("Price"))
    tp = _usd(row.get("TP"))
    res = _usd(row.get("Resistance"))
    rr_res = _safe(row.get("RR_to_Res",""))
    rr_tp = _safe(row.get("RR_to_TP",""))
    change_pct = _pct(row.get("Change%"))
    relvol = _safe(row.get("RelVol(TimeAdj63d)"))
    tp_reward = _usd(row.get("TPReward$", None))
    tp_reward_pct = _pct(row.get("TPReward%", None))
    daily_atr = _usd(row.get("DailyATR", None))
    daily_cap = _usd(row.get("DailyCap", None))
    hist_cnt = _safe(row.get("Hist21d_PassCount",""))
    hist_ex = _safe(row.get("Hist21d_Examples",""))
    support_type = _safe(row.get("SupportType",""))
    support_price = _usd(row.get("SupportPrice"))
    session = _safe(row.get("Session",""))
    entry_src = _safe(row.get("EntrySrc",""))
    vol_src = _safe(row.get("VolSrc",""))

    header = (
        f"{_bold(tkr)} looks attractive here: it last traded near {_bold(price)}. "
        f"We’re aiming for a take-profit around {_bold(tp)} (halfway to the recent high at {_bold(res)}). "
        f"That sets reward-to-risk at roughly {_bold(rr_res)}:1 to the recent high and {_bold(rr_tp)}:1 to the take-profit."
    )

    bullets = [
        f"- Momentum & liquidity: up {_bold(change_pct)} today with relative volume {_bold(relvol)} (time-adjusted vs 63-day average).",
        f"- Distance to target: {_bold(tp_reward)} ({_bold(tp_reward_pct)}). Daily ATR ≈ {_bold(daily_atr)}, "
        f"so a typical month (~21 trading days) allows about {_bold(daily_cap)} of movement.",
        f"- History check: {_bold(hist_cnt)} instances in the past year where a 21-day move met/exceeded this target. Examples: {hist_ex}.",
        f"- Support: {_bold(support_type)} near {_bold(support_price)}.",
        f"- Data basis: Session={session} • EntrySrc={entry_src} • VolSrc={vol_src}."
    ]

    return "<div class='whybuy'>" + header + "<br>" + "<br>".join(bullets) + "</div>"

# ============================================================
# 6. CSV helpers (latest pass file, outcomes)
# ============================================================
def latest_pass_file():
    files = sorted(glob.glob(os.path.join(PASS_DIR, "*.csv")))
    return files[-1] if files else None

def load_outcomes():
    if os.path.exists(OUT_FILE):
        return pd.read_csv(OUT_FILE)
    return pd.DataFrame()

# ============================================================
# 7. Outcomes counters (robust to minimal/extended schemas)
# ============================================================
def outcomes_summary(dfh: pd.DataFrame):
    if dfh is None or dfh.empty:
        st.info("No outcomes yet.")
        return

    n = len(dfh)
    # Ensure we always have Series (never scalars) to avoid .sum() errors
    if "result_status" in dfh.columns:
        s_status = dfh["result_status"].astype(str)
    else:
        # Assume not yet settled if the column is missing
        s_status = pd.Series(["PENDING"] * n, index=dfh.index, dtype="string")

    if "hit" in dfh.columns:
        hit_mask = dfh["hit"].astype(bool)
    else:
        hit_mask = pd.Series([False] * n, index=dfh.index, dtype=bool)

    settled_mask = s_status.eq("SETTLED")
    pending_mask = ~settled_mask

    settled = int(settled_mask.sum())
    pending = int(pending_mask.sum())
    hits    = int((settled_mask & hit_mask).sum())
    misses  = settled - hits

    st.caption(f"Settled: {settled} • Hits: {hits} • Misses: {misses} • Pending: {pending}")

    # Nice sort if the columns exist; otherwise just show as-is
    sort_cols = [c for c in ["run_date", "ticker"] if c in dfh.columns]
    if sort_cols:
        # run_date desc if present
        ascending = [False if c == "run_date" else True for c in sort_cols]
        df_show = dfh.sort_values(sort_cols, ascending=ascending)
    else:
        df_show = dfh

    st.dataframe(df_show, use_container_width=True, height=min(600, 80 + 28 * len(df_show)))
    
# ============================================================
# 8. Debugger (plain-English reasons with numbers)
# ============================================================
def _fmt_ts_et(ts):
    try:
        # pretty print if it's a tz-aware datetime; otherwise show str
        return ts.strftime("%Y-%m-%d %H:%M:%S %Z") if hasattr(ts, "strftime") else str(ts)
    except Exception:
        return str(ts)

def _num(x, nd=4):
    try:
        return float(x)
    except Exception:
        return None

def _finite(x):
    from math import isfinite
    try:
        return isfinite(float(x))
    except Exception:
        return False

def _mk_reason_expl(reason: str, ctx: dict) -> str:
    """Turn engine reason codes into friendly, numeric explanations."""
    lines = []
    code = (reason or "").strip()

    # frequently-hit context values
    chg = ctx.get("change_pct")
    rel = ctx.get("relvol")
    rel_min = ctx.get("relvol_min")
    entry = ctx.get("entry")
    prev_close = ctx.get("prev_close")
    res = ctx.get("resistance")
    tp = ctx.get("tp")
    daily_atr = ctx.get("daily_atr")
    daily_cap = ctx.get("daily_cap")
    req_tp_pct = ctx.get("tp_req_pct")

    def pct(x):
        try: return f"{float(x):.2f}%"
        except: return str(x)

    def usd(x, nd=2):
        try: return f"${float(x):.{nd}f}"
        except: return str(x)

    if code == "relvol_low_timeadj":
        lines.append(
            f"Relative volume is too low: current RelVol (time-adjusted) is "
            f"**{rel:.2f}×**, but the minimum is **{rel_min:.2f}×**."
        )
    elif code == "not_up_on_day":
        lines.append(
            f"Price isn’t up on the day: change is **{pct(chg)}** from "
            f"yesterday’s close {usd(prev_close)} to entry {usd(entry)}."
        )
    elif code == "no_upside_to_resistance":
        lines.append(
            f"No room to the recent high: resistance {usd(res)} is not above entry {usd(entry)}."
        )
    elif code == "atr_capacity_short_vs_tp":
        lines.append(
            "ATR capacity is too small to reasonably reach the target in a month: "
            f"need ≈ **{pct(req_tp_pct)}** to target (≈ {usd(tp)}), but Daily ATR is "
            f"{usd(daily_atr, nd=4)}, implying about **{usd(daily_cap)}** over ~21 trading days."
        )
    elif code == "history_21d_zero_pass":
        lines.append(
            "History check failed: in the last year there were **0** cases where a 21-trading-day move "
            f"matched or exceeded the required **{pct(req_tp_pct)}**."
        )
    elif code in {"no_valid_support", "non_positive_risk"}:
        lines.append(
            "Couldn’t find a valid support below price to place a stop (risk would be non-positive)."
        )
    elif code == "rr_to_res_below_min":
        lines.append(
            "Reward-to-risk to the recent high is below the minimum (needs ≥ 2:1)."
        )
    elif code in {"insufficient_rows", "insufficient_past_for_21d"}:
        lines.append("Not enough price history to evaluate this ticker robustly.")
    elif code == "bad_entry_prevclose":
        lines.append("Intraday quote/previous close unavailable or inconsistent.")
    else:
        lines.append(f"Engine rejected the setup: **{code}**.")

    # Add a compact numeric snapshot for context
    snap = []
    if _finite(entry) and _finite(prev_close):
        snap.append(f"Entry {usd(entry)} vs prev close {usd(prev_close)} → day change {pct(chg)}.")
    if _finite(res) and _finite(tp):
        snap.append(f"Resistance {usd(res)}, TP {usd(tp)}.")
    if _finite(rel):
        snap.append(f"RelVol (time-adjusted): {rel:.2f}× (min {rel_min:.2f}×).")
    if _finite(daily_atr):
        snap.append(f"Daily ATR {usd(daily_atr, nd=4)} ⇒ ~{usd(daily_cap)} / 21 trading days.")

    if snap:
        lines.append("")
        lines.append("**Snapshot:** " + " ".join(snap))

    return "<br>".join(lines)

def diagnose_ticker(ticker: str,
                    res_days=None,
                    rel_vol_min=None,
                    relvol_median=False,
                    rr_min=None,
                    stop_mode="safest"):
    """
    Returns:
      title (str), details (dict with numbers), and 'explanation_md' (str) for plain-English UI.
    """
    # Pull defaults from the engine when not supplied
    res_days = res_days if res_days is not None else getattr(sos, "RES_LOOKBACK_DEFAULT", 21)
    rel_vol_min = rel_vol_min if rel_vol_min is not None else getattr(sos, "REL_VOL_MIN_DEFAULT", 1.10)
    rr_min = rr_min if rr_min is not None else getattr(sos, "RR_MIN_DEFAULT", 2.0)

    df = sos._get_history(ticker)
    entry = prev_close = today_vol = None
    src = {}
    entry_ts = None

    if df is not None:
        entry, prev_close, today_vol, src, entry_ts = sos.get_entry_prevclose_todayvol(df, ticker)

    row, reason = sos.evaluate_ticker(
        ticker,
        res_days=res_days,
        rel_vol_min=rel_vol_min,
        use_relvol_median=relvol_median,
        rr_min=rr_min,
        prefer_stop=stop_mode
    )

    # If it passed, build a quick friendly summary too
    if reason is None and isinstance(row, dict):
        # quick narrative for passes
        narrative = (
            f"**{ticker} PASSED** ✔️ — price is up **{row.get('Change%', 0):.2f}%** today, "
            f"time-adjusted RelVol **{row.get('RelVol(TimeAdj63d)', 0):.2f}×**. "
            f"Target {row.get('TP')} vs price {row.get('Price')}, "
            f"Daily ATR ≈ {row.get('DailyATR')} (~{row.get('DailyCap')} per month). "
            f"R:R to high ≈ **{row.get('RR_to_Res')}**:1; to TP ≈ **{row.get('RR_to_TP')}**:1."
        )
        details = {
            "entry": entry,
            "prev_close": prev_close,
            "today_vol": today_vol,
            "src": src,
            "entry_ts": _fmt_ts_et(entry_ts),
            "explanation_md": narrative
        }
        return f"{ticker} PASSED ✅", details

    # Build context for a friendly failure explanation
    from math import isfinite
    ctx = {}
    ctx["entry"] = _num(entry)
    ctx["prev_close"] = _num(prev_close)
    ctx["change_pct"] = ((ctx["entry"] - ctx["prev_close"]) / ctx["prev_close"] * 100.0) if (_finite(ctx["entry"]) and _finite(ctx["prev_close"]) and ctx["prev_close"] != 0) else None

    # relvol (time adjusted)
    relvol_val = None
    try:
        if df is not None and _finite(today_vol):
            relvol_val = sos.compute_relvol_time_adjusted(df, today_vol, use_median=relvol_median)
    except Exception:
        relvol_val = None
    ctx["relvol"] = relvol_val
    ctx["relvol_min"] = rel_vol_min

    # resistance & TP from current entry (for context only)
    try:
        if df is not None and len(df) >= max(22, res_days + 1) and _finite(ctx["entry"]):
            rolling_high = df["High"].rolling(window=res_days, min_periods=res_days).max()
            res = float(rolling_high.shift(1).iloc[-1])
            ctx["resistance"] = res
            if isfinite(res) and res > ctx["entry"]:
                ctx["tp"] = ctx["entry"] + 0.5 * (res - ctx["entry"])
                if isfinite(ctx["tp"]):
                    ctx["tp_req_pct"] = (ctx["tp"] - ctx["entry"]) / ctx["entry"] * 100.0
    except Exception:
        pass

    # ATR & monthly capacity
    try:
        if df is not None:
            da = sos._atr_from_ohlc(df, 14)
            ctx["daily_atr"] = da
            if _finite(da):
                ctx["daily_cap"] = da * 21.0
    except Exception:
        pass

    # Compose narrative
    title = f"{ticker} FAILED ❌ — {reason}"
    narrative = _mk_reason_expl(reason, ctx)

    details = {
        "entry": entry,
        "prev_close": prev_close,
        "today_vol": today_vol,
        "src": src,
        "entry_ts": _fmt_ts_et(entry_ts),
        "relvol_time_adj": float(relvol_val) if _finite(relvol_val) else None,
        "resistance": ctx.get("resistance"),
        "tp": ctx.get("tp"),
        "daily_atr": ctx.get("daily_atr"),
        "daily_cap": ctx.get("daily_cap"),
        "explanation_md": narrative
    }
    return title, details
                        
# ─────────────────────────────────────────────────────────────────────────────
# 9. TAB – Scanner (table → WHY BUY → Sheets export)
# ─────────────────────────────────────────────────────────────────────────────
def _safe_run_scan() -> dict:
    """Call sos.run_scan with backward-compatible signatures and normalize outputs
    without using boolean truthiness on DataFrames."""
    import pandas as _pd

    # Try different parameter names used across your versions
    try:
        out = sos.run_scan(market="sp500", with_options=True)
    except TypeError:
        try:
            out = sos.run_scan(universe="sp500", with_options=True)
        except TypeError:
            out = sos.run_scan(with_options=True)

    df_pass, df_scan = None, None

    if isinstance(out, dict):
        p = out.get("pass", None)
        if isinstance(p, _pd.DataFrame):
            df_pass = p
        p2 = out.get("pass_df", None)
        if df_pass is None and isinstance(p2, _pd.DataFrame):
            df_pass = p2
        p3 = out.get("pass_df_unadjusted", None)
        if df_pass is None and isinstance(p3, _pd.DataFrame):
            df_pass = p3

        s = out.get("scan", None)
        if isinstance(s, _pd.DataFrame):
            df_scan = s
        s2 = out.get("scan_df", None)
        if df_scan is None and isinstance(s2, _pd.DataFrame):
            df_scan = s2

    elif isinstance(out, (list, tuple)):
        if len(out) >= 1 and isinstance(out[0], _pd.DataFrame):
            df_pass = out[0]
        if len(out) >= 2 and isinstance(out[1], _pd.DataFrame):
            df_scan = out[1]

    elif isinstance(out, _pd.DataFrame):
        df_pass = out

    return {"pass": df_pass, "scan": df_scan}


def _sheet_friendly(df: pd.DataFrame) -> pd.DataFrame:
    """Produce a sheet-friendly subset (subset of columns)."""
    prefer = [
        "Ticker","EvalDate","Price","EntryTimeET",
        "Change%","RelVol(TimeAdj63d)","Resistance","TP",
        "RR_to_Res","RR_to_TP","SupportType","SupportPrice",
        "Risk$","TPReward$","TPReward%","ResReward$","ResReward%",
        "DailyATR","DailyCap","Hist21d_PassCount"
    ]
    cols = [c for c in prefer if c in df.columns]
    return df.loc[:, cols].copy() if cols else df.copy()


def _render_why_buy_block(df: pd.DataFrame):
    """Render WHY BUY expanders per ticker."""
    if df is None or df.empty:
        return
    st.markdown("### WHY BUY details")
    for _, row in df.iterrows():
        tkr = str(row.get("Ticker", "")).strip() or "—"
        with st.expander(f"WHY BUY — {tkr}", expanded=False):
            html = build_why_buy_html(row)
            st.markdown(html, unsafe_allow_html=True)


def render_scanner_tab():
    st.markdown("#### Scanner")

    # Red RUN button (custom CSS inside this tab to avoid global collisions)
    st.markdown(
        """
        <style>
        div.stButton > button:first-child {
            background-color: red !important;
            color: white !important;
            font-weight: bold !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    run_clicked = st.button("RUN", key="run_scan_btn")

    if run_clicked:
        with st.spinner("Scanning…"):
            out = _safe_run_scan()
        df_pass: pd.DataFrame | None = out.get("pass", None)

        st.session_state["last_pass"] = df_pass

        if df_pass is None or df_pass.empty:
            st.warning("No tickers passed the filters.")
        else:
            st.success(f"Found {len(df_pass)} passing tickers.")
            st.dataframe(df_pass, use_container_width=True, height=min(560, 80+28*len(df_pass)))
            _render_why_buy_block(df_pass)
            with st.expander("Google-Sheet style view (optional)", expanded=False):
                st.dataframe(_sheet_friendly(df_pass), use_container_width=True, height=min(560, 80+28*len(df_pass)))

    elif isinstance(st.session_state.get("last_pass"), pd.DataFrame) and not st.session_state["last_pass"].empty:
        df_pass: pd.DataFrame = st.session_state["last_pass"]
        st.info(f"Showing last run in this session • {len(df_pass)} tickers")
        st.dataframe(df_pass, use_container_width=True, height=min(560, 80+28*len(df_pass)))
        _render_why_buy_block(df_pass)
        with st.expander("Google-Sheet style view (optional)", expanded=False):
            st.dataframe(_sheet_friendly(df_pass), use_container_width=True, height=min(560, 80+28*len(df_pass)))
    else:
        st.caption("No results yet. Press **RUN** to scan.")

# ============================================================
# 10. UI – Tabs
# ============================================================
tab1, tab2, tab3 = st.tabs(["Scanner", "History & Outcomes", "Debugger"])

# Scanner tab (includes red RUN button and results)
with tab1:
    render_scanner_tab()

# History & Outcomes tab (rendered ONCE)
with tab2:
    st.header("History & Outcomes")
    lastf = latest_pass_file()
    if lastf:
        st.success(f"Last run file: {lastf}")
        try:
            st.dataframe(pd.read_csv(lastf), use_container_width=True)
        except Exception:
            st.info("Pass file exists but could not be read.")
    dfh = load_outcomes()
    outcomes_summary(dfh)

# Debugger tab
with tab3:
    st.header("Debugger")
    dbg_ticker = st.text_input("Enter ticker to debug", key="dbg_ticker_input")
    if dbg_ticker:
        title, details = diagnose_ticker(dbg_ticker.strip().upper())
        st.subheader(title)
        # 👉 show the plain-English narrative
        expl = details.get("explanation_md", "")
        if expl:
            st.markdown(expl, unsafe_allow_html=True)
        # raw numbers snapshot
        st.json({k: v for k, v in details.items() if k != "explanation_md"})
