# ============================================================
# TABLE OF CONTENTS for app.py
#
#  1. Imports & Safe third-party glue
#  2. App constants (paths, titles, etc.)
# 3. Streamlit page config + global CSS
# 4. TAB – Scanner (table → WHY BUY → Sheets export)
# 5. UI – Tabs
# 6. TAB – History & Outcomes
# ============================================================

# ============================================================
# 1. Imports & Safe third-party glue
# ============================================================
import os, glob, io, pandas as pd, streamlit as st
from datetime import datetime
import swing_options_screener as sos  # core engine
from ui.layout import build_why_buy_html
from ui.debugger import diagnose_ticker
from utils.data import latest_pass_file, load_outcomes

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

# ── TAB 2: History & Outcomes (INLINE — no external function call)
with tab_history:
    # --- Latest recommendations (most recent run) ---
    st.subheader("Latest recommendations (most recent run)")
    lastf = latest_pass_file(PASS_DIR, HIST_DIR)
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

    dfh = load_outcomes(OUT_FILE)
    if dfh is None or dfh.empty:
        st.info("No outcomes yet.")
    else:
        dfh = dfh.copy()

        # Ensure expected columns exist
        for c in ["Expiry", "EvalDate", "Notes"]:
            if c not in dfh.columns:
                dfh[c] = pd.NA

        # Prefer result_status, then Status
        status_col = "result_status" if "result_status" in dfh.columns else ("Status" if "Status" in dfh.columns else None)

        # Helper: to tz-naive pandas Timestamp
        def _to_naive(series: pd.Series) -> pd.Series:
            s = pd.to_datetime(series, errors="coerce", utc=True)
            return s.dt.tz_convert("UTC").dt.tz_localize(None)

        # Parse & normalize times
        dfh["Expiry_parsed"]   = _to_naive(dfh["Expiry"])
        dfh["EvalDate_parsed"] = _to_naive(dfh["EvalDate"])

        # Backfill missing expiry from EvalDate + 30d (display-only)
        need_exp = dfh["Expiry_parsed"].isna() & dfh["EvalDate_parsed"].notna()
        if need_exp.any():
            dfh.loc[need_exp, "Expiry_parsed"] = dfh.loc[need_exp, "EvalDate_parsed"] + pd.Timedelta(days=30)

        # ---- Robust DTE using nanoseconds to avoid dtype issues ----
        dfh["DTE"] = pd.Series(pd.NA, index=dfh.index, dtype="Int64")
        mask = dfh["Expiry_parsed"].notna()
        if mask.any():
            base_ns = pd.Timestamp.utcnow().normalize().value  # int64 ns at 00:00 UTC today
            exp_ns  = dfh.loc[mask, "Expiry_parsed"].view("int64")
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
        
