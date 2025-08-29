# ============================================================
# TABLE OF CONTENTS for app.py
#
#  1. Imports & Safe third-party glue
#  2. App constants (paths, titles, etc.)
# 3. Streamlit page config + global CSS
#  4. Small formatting helpers
#  5. WHY BUY explanation builder (plain English)
#  6. CSV helpers (latest pass file, outcomes)
#  7. Outcomes counters (robust to minimal/extended schemas)
#  8. Debugger (plain-English reasons with numbers)
# 9. TAB – Scanner (table → WHY BUY → Sheets export)
#  10. UI – Tabs
# 11. TAB – History & Outcomes
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

st.set_page_config(
    page_title="Edge500",     # Title shown in browser tab
    page_icon="logo.png",     # Favicon (logo.png in repo root)
    layout="wide"
)


# ============================================================
# 3. Streamlit page config + global CSS
# ============================================================
st.markdown(
    """
    <style>
    /* --- Buttons / general --- */
    div.stButton > button:first-child {
        background-color: red !important;
        color: white !important;
        font-weight: 700 !important;
    }

    /* --- WHY BUY text block --- */
    .whybuy { font-size: 16px; line-height: 1.55; }

    /* --- Debugger layout (HTML) --- */
    .dbg-wrap { max-width: 1100px; margin-top: 8px; }
    .dbg-title { font-size: 28px; font-weight: 800; letter-spacing: .2px; margin: 4px 0 12px; }
    .dbg-badge {
        display:inline-block; padding: 4px 10px; margin-left: 10px;
        border-radius: 999px; font-size: 13px; font-weight: 700;
        vertical-align: middle;
    }
    .dbg-badge.fail { background:#ffe6e6; color:#b00020; border:1px solid #ffb3b3; }
    .dbg-badge.pass { background:#e7f6ec; color:#0a7a35; border:1px solid #bfe6cc; }
    .dbg-subtle { color:#666; font-size: 14px; margin-bottom: 10px; }
    .dbg-snapshot {
        background:#f7f7f9; border-left:4px solid #c7c7d1;
        padding:10px 12px; margin: 14px 0 10px; font-size:15px;
    }
    .dbg-snap-kv { display:inline-block; margin-right: 14px; }
    .dbg-snap-kv .k { color:#666; }
    .dbg-snap-kv .v { font-weight:700; color:#111; }
    .dbg-json details { margin-top: 10px; }
    .dbg-json summary { cursor: pointer; font-weight: 700; }
    .dbg-json pre {
        background:#111; color:#f2f2f2; padding:12px; border-radius:8px;
        overflow:auto; font-size:13px; line-height:1.45;
    }
    .em { font-style: italic; }

    /* --- Logo header --- */
    .app-logo {
        display: flex;
        justify-content: center;
        margin: 0.5rem 0 1rem;
    }
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
    """Return the newest pass_*.csv from either pass_logs/ or history/."""
    candidates = []
    for d in [PASS_DIR, HIST_DIR]:
        candidates.extend(glob.glob(os.path.join(d, "pass_*.csv")))
    return sorted(candidates)[-1] if candidates else None

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

# --- Aliases for common company names → tickers ---
_ALIAS_MAP = {
    "NVIDIA": "NVDA", "NVIDIA CORPORATION": "NVDA",
    "TESLA": "TSLA", "TESLA INC": "TSLA",
    "APPLE": "AAPL", "APPLE INC": "AAPL",
    "MICROSOFT": "MSFT", "MICROSOFT CORPORATION": "MSFT",
    "ALPHABET": "GOOGL", "GOOGLE": "GOOGL",
    "META": "META", "META PLATFORMS": "META",
    "AMAZON": "AMZN", "AMAZONCOM": "AMZN", "AMAZON.COM": "AMZN",
    "NETFLIX": "NFLX", "WALMART": "WMT", "WALMART INC": "WMT",
    "JPMORGAN": "JPM", "JPMORGAN CHASE": "JPM",
    "BERKSHIRE": "BRK.B", "BERKSHIRE HATHAWAY": "BRK.B",
    "UNITEDHEALTH": "UNH", "UNITEDHEALTH GROUP": "UNH",
    "COCA COLA": "KO", "COCA-COLA": "KO",
    "PEPSICO": "PEP", "ADOBE": "ADBE", "INTEL": "INTC",
    "AMD": "AMD", "BROADCOM": "AVGO", "SALESFORCE": "CRM",
    "SERVICENOW": "NOW", "SERVICE NOW": "NOW",
    "CROWDSTRIKE": "CRWD", "MCDONALDS": "MCD", "MCDONALD'S": "MCD",
    "COSTCO": "COST", "HOME DEPOT": "HD",
    "PROCTER & GAMBLE": "PG", "PROCTER AND GAMBLE": "PG",
    "ELI LILLY": "LLY", "ABBVIE": "ABBV",
    "EXXON": "XOM", "EXXONMOBIL": "XOM", "CHEVRON": "CVX",
}

def _normalize_brk(s: str) -> str | None:
    s2 = s.replace(" ", "").replace("-", "").replace("_", "").upper()
    if s2 in {"BRKB", "BRK.B"}: return "BRK.B"
    if s2 in {"BRKA", "BRK.A"}: return "BRK.A"
    sU = s.upper().strip()
    if sU in {"BRK B", "BRK-B", "BRK_B"}: return "BRK.B"
    if sU in {"BRK A", "BRK-A", "BRK_A"}: return "BRK.A"
    return None

def _normalize_symbol(inp: str) -> str | None:
    """Best-effort mapping: ticker-looking → upper; else try aliases; else heuristics."""
    if not inp: return None
    s = str(inp).strip()
    if not s: return None

    # Looks like a ticker already?
    if 1 <= len(s) <= 6 and all(c.isalnum() or c == "." for c in s):
        brk = _normalize_brk(s)
        return brk if brk else s.upper()

    # Company name path
    key = s.upper()
    key = key.replace(",", "").replace(".", "")
    for kill in (" INC", " CORPORATION", " COMPANY", " HOLDINGS", " PLC", " LTD"):
        key = key.replace(kill, "")
    key = key.replace(" CLASS A", "").replace(" CLASS B", "")
    key = " ".join(key.split())

    if key in _ALIAS_MAP:
        return _ALIAS_MAP[key]

    # Last chance Berkshire normalization
    brk = _normalize_brk(s)
    return brk if brk else s.upper()

def _mk_reason_expl(reason: str, ctx: dict) -> str:
    """Turn engine reason codes into friendly, numeric explanations."""
    lines = []
    code = (reason or "").strip()

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
            f"Relative volume is too low: current RelVol (time-adjusted) is **{rel:.2f}×**, "
            f"but the minimum is **{rel_min:.2f}×**."
        )
    elif code == "not_up_on_day":
        lines.append(
            f"Price isn’t up on the day: change is **{pct(chg)}** from "
            f"yesterday’s close {usd(prev_close)} to entry {usd(entry)}."
        )
    elif code == "no_upside_to_resistance":
        lines.append(f"No room to the recent high: resistance {usd(res)} is not above entry {usd(entry)}.")
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
        lines.append("Couldn’t find a valid support below price to place a stop (risk would be non-positive).")
    elif code == "rr_to_res_below_min":
        lines.append("Reward-to-risk to the recent high is below the minimum (needs ≥ 2:1).")
    elif code in {"insufficient_rows", "insufficient_past_for_21d"}:
        lines.append("Not enough price history to evaluate this ticker robustly.")
    elif code == "bad_entry_prevclose":
        lines.append("Intraday quote/previous close unavailable or inconsistent.")
    elif code == "no_data":
        lines.append("No price data returned for this input after normalization.")
    else:
        lines.append(f"Engine rejected the setup: **{code}**.")

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

def _yf_fetch_daily(symbol: str):
    """Fallback: pull ~6 months daily bars via yfinance and shape like engine OHLCV."""
    try:
        import yfinance as yf
        df = yf.download(symbol, period="6mo", interval="1d", auto_adjust=False, progress=False)
        if df is None or df.empty:
            return None
        # Ensure expected column names
        cols = {c.lower(): c for c in df.columns}
        for need in ["Open","High","Low","Close","Volume"]:
            if need not in df.columns:
                # try case-insensitive rescue
                for k,v in cols.items():
                    if k == need.lower():
                        break
                else:
                    return None
        df = df[["Open","High","Low","Close","Volume"]].dropna()
        if df.empty:
            return None
        return df
    except Exception:
        return None

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

    original = (ticker or "").strip()
    symbol = _normalize_symbol(original)

    # Try engine history first
    df = sos._get_history(symbol) if symbol else None

    # Fallback to yfinance if engine returned nothing
    if df is None and symbol:
        df = _yf_fetch_daily(symbol)

    entry = prev_close = today_vol = None
    src = {}
    entry_ts = None

    if df is not None:
        try:
            entry, prev_close, today_vol, src, entry_ts = sos.get_entry_prevclose_todayvol(df, symbol)
        except Exception:
            # Light fallback if engine helper not usable
            try:
                # entry = last close; prev_close = previous day's close; volume = last row Volume
                entry = float(df["Close"].iloc[-1])
                prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else None
                today_vol = float(df["Volume"].iloc[-1])
                src = {"session": "UNKNOWN", "entry_src": "fallback", "vol_src": "fallback"}
                entry_ts = df.index[-1].to_pydatetime() if hasattr(df.index, "to_pydatetime") else None
            except Exception:
                df = None  # force no_data path below

    # Still nothing? Explain cleanly.
    if df is None:
        title = f"{original if original else '—'} FAILED ❌ — no_data"
        narrative = ("No price data returned for this input. Try the **ticker symbol** or a known S&P-500 "
                     "company name (the Debugger maps names like ‘NVIDIA’ → ‘NVDA’, ‘Tesla’ → ‘TSLA’).")
        details = {
            "entry": None,
            "prev_close": None,
            "today_vol": None,
            "src": {},
            "entry_ts": _fmt_ts_et(entry_ts),
            "relvol_time_adj": None,
            "resistance": None,
            "tp": None,
            "daily_atr": None,
            "daily_cap": None,
            "explanation_md": narrative,
        }
        return title, details

    # Evaluate via engine
    row, reason = sos.evaluate_ticker(
        symbol,
        res_days=res_days,
        rel_vol_min=rel_vol_min,
        use_relvol_median=relvol_median,
        rr_min=rr_min,
        prefer_stop=stop_mode
    )

    # Passed → short friendly summary
    if reason is None and isinstance(row, dict):
        narrative = (
            f"**{symbol} PASSED** ✔️ — price is up **{row.get('Change%', 0):.2f}%** today, "
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
        return f"{symbol} PASSED ✅", details

    # Failure context for readable explanation
    from math import isfinite
    ctx = {}
    ctx["entry"] = _num(entry)
    ctx["prev_close"] = _num(prev_close)
    ctx["change_pct"] = ((ctx["entry"] - ctx["prev_close"]) / ctx["prev_close"] * 100.0) if (_finite(ctx["entry"]) and _finite(ctx["prev_close"]) and ctx["prev_close"] != 0) else None

    # RelVol (time-adjusted) using engine if possible
    relvol_val = None
    try:
        if df is not None and _finite(today_vol):
            relvol_val = sos.compute_relvol_time_adjusted(df, today_vol, use_median=relvol_median)
    except Exception:
        relvol_val = None
    ctx["relvol"] = relvol_val
    ctx["relvol_min"] = rel_vol_min

    # Resistance & TP (context only)
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

    title = f"{symbol} FAILED ❌ — {reason}"
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
        cand = out.get("pass", None)
        if isinstance(cand, _pd.DataFrame):
            df_pass = cand
        cand = out.get("pass_df", None)
        if df_pass is None and isinstance(cand, _pd.DataFrame):
            df_pass = cand
        cand = out.get("pass_df_unadjusted", None)
        if df_pass is None and isinstance(cand, _pd.DataFrame):
            df_pass = cand

        cand = out.get("scan", None)
        if isinstance(cand, _pd.DataFrame):
            df_scan = cand
        cand = out.get("scan_df", None)
        if df_scan is None and isinstance(cand, _pd.DataFrame):
            df_scan = cand

    elif isinstance(out, (list, tuple)):
        if len(out) >= 1 and isinstance(out[0], _pd.DataFrame):
            df_pass = out[0]
        if len(out) >= 2 and isinstance(out[1], _pd.DataFrame):
            df_scan = out[1]

    elif isinstance(out, _pd.DataFrame):
        # Some versions just return the passing table
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
            st.success(f"Found {len(df_pass)} passing tickers (latest run).")
            st.dataframe(df_pass, use_container_width=True, height=min(560, 80 + 28*len(df_pass)))
            _render_why_buy_block(df_pass)
            with st.expander("Google-Sheet style view (optional)", expanded=False):
                st.dataframe(_sheet_friendly(df_pass), use_container_width=True, height=min(560, 80 + 28*len(df_pass)))

    elif isinstance(st.session_state.get("last_pass"), pd.DataFrame) and not st.session_state["last_pass"].empty:
        df_pass: pd.DataFrame = st.session_state["last_pass"]
        st.info(f"Showing last run in this session • {len(df_pass)} tickers")
        st.dataframe(df_pass, use_container_width=True, height=min(560, 80 + 28*len(df_pass)))
        _render_why_buy_block(df_pass)
        with st.expander("Google-Sheet style view (optional)", expanded=False):
            st.dataframe(_sheet_friendly(df_pass), use_container_width=True, height=min(560, 80 + 28*len(df_pass)))
    else:
        st.caption("No results yet. Press **RUN** to scan.")

# ============================================================
# 10. UI – Tabs
# ============================================================

# ---- Brand header (logo only) ----
st.markdown('<div class="app-logo">', unsafe_allow_html=True)
st.image("logo.png", width=140)
st.markdown('</div>', unsafe_allow_html=True)
st.divider()

# Create tabs once with unique variable names
tab_scanner, tab_history, tab_debug = st.tabs(
    ["Scanner", "History & Outcomes", "Debugger"]
)

# ── TAB 1: Scanner (red RUN button, results table, WHY BUY, Sheets view)
with tab_scanner:
    render_scanner_tab()

# ── TAB 2: History & Outcomes (single clean call)
with tab_history:
    render_history_and_outcomes_tab()

# ── TAB 3: Debugger (plain-English + numbers; styled HTML)
with tab_debug:
    import json, html as _html

    st.markdown("""
    <style>
      .dbg-wrap { margin-top: 0.5rem; }
      .dbg-title { font-weight: 700; font-size: 1.05rem; margin-bottom: .25rem; }
      .dbg-badge { padding: 2px 6px; border-radius: 6px; font-size: .8rem; margin-left: .4rem; }
      .dbg-badge.pass { background: #0f5132; color: #fff; }
      .dbg-badge.fail { background: #842029; color: #fff; }
      .dbg-subtle { margin-bottom: .5rem; line-height: 1.4; }
      .dbg-snapshot { background: #111; color: #eee; padding: 10px; border-radius: 8px; font-size: .9rem; }
      .dbg-snap-kv { display: inline-block; margin-right: 14px; margin-top: 4px; }
      .dbg-snap-kv .k { color: #bbb; }
      .dbg-snap-kv .v { color: #fff; font-weight: 600; }
      .dbg-json details { margin-top: .5rem; }
      .dbg-json pre { background: #0b0b0b; color: #e6e6e6; padding: 10px; border-radius: 8px; overflow-x: auto; }
    </style>
    """, unsafe_allow_html=True)

    st.header("Debugger")
    dbg_ticker = st.text_input("Enter ticker to debug", key="dbg_ticker_input")

    if dbg_ticker:
        title, details = diagnose_ticker(dbg_ticker.strip().upper())
        is_fail = "FAIL" in (title or "").upper()
        badge = '<span class="dbg-badge fail">FAIL</span>' if is_fail else '<span class="dbg-badge pass">PASS</span>'

        def g(d, k, default="—"):
            try:
                v = d.get(k, default)
                return default if v is None else v
            except Exception:
                return default

        entry          = g(details, "entry")
        prev_close     = g(details, "prev_close")
        today_vol      = g(details, "today_vol")
        src            = g(details, "src", {})
        session        = g(src, "session", "—") if isinstance(src, dict) else "—"
        entry_src      = g(src, "entry_src", "—") if isinstance(src, dict) else "—"
        vol_src        = g(src, "vol_src", "—") if isinstance(src, dict) else "—"
        entry_ts       = g(details, "entry_ts")
        resistance     = g(details, "resistance")
        tp             = g(details, "tp")
        relvol_timeadj = g(details, "relvol_time_adj")
        daily_atr      = g(details, "daily_atr")
        daily_cap      = g(details, "daily_cap")

        narrative_html = details.get("explanation_md", "")

        html_top = f"""
        <div class="dbg-wrap">
          <div class="dbg-title">{title} {badge}</div>
          <div class="dbg-subtle">{narrative_html}</div>
        """

        html_snapshot = f"""
          <div class="dbg-snapshot">
            <span class="dbg-snap-kv"><span class="k">Session:</span> <span class="v">{session}</span></span>
            <span class="dbg-snap-kv"><span class="k">Entry src:</span> <span class="v">{entry_src}</span></span>
            <span class="dbg-snap-kv"><span class="k">Vol src:</span> <span class="v">{vol_src}</span></span><br/>
            <span class="dbg-snap-kv"><span class="k">Entry:</span> <span class="v">{entry}</span></span>
            <span class="dbg-snap-kv"><span class="k">Prev Close:</span> <span class="v">{prev_close}</span></span>
            <span class="dbg-snap-kv"><span class="k">Today Vol:</span> <span class="v">{today_vol}</span></span><br/>
            <span class="dbg-snap-kv"><span class="k">Resistance:</span> <span class="v">{resistance}</span></span>
            <span class="dbg-snap-kv"><span class="k">TP:</span> <span class="v">{tp}</span></span>
            <span class="dbg-snap-kv"><span class="k">RelVol(Adj):</span> <span class="v">{relvol_timeadj}</span></span>
            <span class="dbg-snap-kv"><span class="k">Daily ATR:</span> <span class="v">{daily_atr}</span></span>
            <span class="dbg-snap-kv"><span class="k">Daily Cap:</span> <span class="v">{daily_cap}</span></span><br/>
            <span class="dbg-snap-kv"><span class="k">Timestamp:</span> <span class="v">{entry_ts}</span></span>
          </div>
        """

        pretty = json.dumps({k: v for k, v in details.items() if k != "explanation_md"},
                            indent=2, default=str)
        html_json = f"""
          <div class="dbg-json">
            <details>
              <summary>Show raw JSON</summary>
              <pre>{_html.escape(pretty)}</pre>
            </details>
          </div>
        </div>
        """

        st.markdown(html_top, unsafe_allow_html=True)
        st.markdown(html_snapshot, unsafe_allow_html=True)
        st.markdown(html_json, unsafe_allow_html=True)
        
        # ─────────────────────────────────────────────────────────────────────────────
# 11. TAB – History & Outcomes (renderer function used by Section 10)
# ─────────────────────────────────────────────────────────────────────────────
def render_history_and_outcomes_tab():
    # --- Latest recommendations (most recent run) ---
    st.subheader("Latest recommendations (most recent run)")
    lastf = latest_pass_file()
    if lastf:
        try:
            df_last = pd.read_csv(lastf)
            st.dataframe(df_last, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not read latest pass file: {e}")
    else:
        st.info("No pass files yet. Run the scanner (or wait for the next scheduled run).")

    # --- Outcomes, sorted by option expiry (oldest → newest) ---
    st.subheader("Outcomes (sorted by option expiry)")

    dfh = load_outcomes()
    if dfh is None or dfh.empty:
        st.info("No outcomes yet.")
        return

    dfh = dfh.copy()

    # Ensure expected columns exist
    for c in ["Expiry", "EvalDate", "Notes"]:
        if c not in dfh.columns:
            dfh[c] = pd.NA

    # Prefer result_status, then Status
    status_col = "result_status" if "result_status" in dfh.columns else ("Status" if "Status" in dfh.columns else None)

    # Helper: to tz-naive pandas Timestamp
    def _to_naive(series):
        s = pd.to_datetime(series, errors="coerce", utc=True)
        return s.dt.tz_convert("UTC").dt.tz_localize(None)

    # Parse & normalize times
    dfh["Expiry_parsed"]   = _to_naive(dfh["Expiry"])
    dfh["EvalDate_parsed"] = _to_naive(dfh["EvalDate"])

    # Backfill missing expiry from EvalDate + 30d (display-only)
    need_exp = dfh["Expiry_parsed"].isna() & dfh["EvalDate_parsed"].notna()
    if need_exp.any():
        dfh.loc[need_exp, "Expiry_parsed"] = dfh.loc[need_exp, "EvalDate_parsed"] + pd.Timedelta(days=30)

    # ---- Robust DTE (avoid datetime arithmetic edge-cases) ----
    # Work in nanoseconds since epoch to guarantee arithmetic works
    dfh["DTE"] = pd.Series(pd.NA, index=dfh.index, dtype="Int64")
    mask = dfh["Expiry_parsed"].notna()
    if mask.any():
        # base in ns for "today at 00:00 UTC"
        base_ns = pd.Timestamp.utcnow().normalize().value  # int64 ns
        # expiry ns (safe because mask excludes NaT)
        exp_ns = dfh.loc[mask, "Expiry_parsed"].view("int64")
        # ns per day
        NS_PER_DAY = 86_400_000_000_000
        dte_days = ((exp_ns - base_ns) // NS_PER_DAY).astype("int64")
        dfh.loc[mask, "DTE"] = pd.array(dte_days, dtype="Int64")

    # Sort: earliest expiry first; for ties, most recent EvalDate first; NaT at end
    exp_key = dfh["Expiry_parsed"].fillna(pd.Timestamp.max)
    dfh_sorted = dfh.assign(_expkey=exp_key).sort_values(
        ["_expkey", "EvalDate_parsed"], ascending=[True, False]
    ).drop(columns=["_expkey"])

    # Summary counts
    notes_up = dfh_sorted["Notes"].astype(str).str.upper()
    hits   = int(notes_up.isin(["HIT_BY_SELLK", "HIT_BY_TP"]).sum())
    misses = int((notes_up == "EXPIRED_NO_HIT").sum())

    if status_col:
        s_up    = dfh_sorted[status_col].astype(str).str.upper()
        settled = int((s_up == "SETTLED").sum())
        pending = int((s_up != "SETTLED").sum())
    else:
        settled = hits + misses
        pending = int(len(dfh_sorted) - settled)

    st.caption(f"Settled: {settled} • Hits: {hits} • Misses: {misses} • Pending: {pending}")

    # Show parsed expiry when original blank; format for display
    df_disp = dfh_sorted.copy()
    use_parsed = df_disp["Expiry"].isna() | (df_disp["Expiry"].astype(str).str.strip() == "")
    df_disp.loc[use_parsed, "Expiry"] = df_disp.loc[use_parsed, "Expiry_parsed"].dt.strftime("%Y-%m-%d")

    preferred = [
        "Ticker","EvalDate","Price","EntryTimeET",
        status_col if status_col else "Status",
        "HitDateET","Expiry","DTE","BuyK","SellK","TP","Notes"
    ]
    cols = [c for c in preferred if c in df_disp.columns]
    if cols:
        df_disp = df_disp[cols]

    st.dataframe(df_disp, use_container_width=True, height=min(600, 80 + 28 * len(df_disp)))
    
    



    
