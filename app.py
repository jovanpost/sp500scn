# ============================================================
# TABLE OF CONTENTS for app.py
#
#  1. Imports & Safe third-party glue
#  2. App constants (paths, titles, etc.)
# 3. WHY BUY explanation builder (plain English)
# 4. CSV helpers (latest pass file, outcomes)
# 5. Outcomes counters (robust to minimal/extended schemas)
# 6. TAB – Scanner (table → WHY BUY → Sheets export)
# 7. UI – Tabs
# 8. TAB – History & Outcomes
# 9. TAB – Debugger
# ============================================================

# ============================================================
# 1. Imports & Safe third-party glue
# ============================================================
import os, glob, io, pandas as pd, streamlit as st
from datetime import datetime
from ui.layout import setup_page, render_header
from ui.debugger import render_debugger_tab
from utils.formatting import _bold, _usd, _pct, _safe
from utils.scan import safe_run_scan

# ============================================================
# 2. App constants (paths, titles, etc.)
# ============================================================
PASS_DIR = "data/pass_logs"
HIST_DIR = "data/history"
OUT_FILE = os.path.join(HIST_DIR, "outcomes.csv")
# Initialize page and global layout/CSS
setup_page()

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


# ─────────────────────────────────────────────────────────────────────────────
# 9. TAB – Scanner (table → WHY BUY → Sheets export)
# ─────────────────────────────────────────────────────────────────────────────
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
            out = safe_run_scan()
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

# ---- Brand header ----
render_header()

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


# ── TAB 3: Debugger
with tab_debug:
    render_debugger_tab()
