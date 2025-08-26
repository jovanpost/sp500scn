import os, sys, subprocess, time, re
import pandas as pd
import streamlit as st

st.set_page_config(page_title="S&P 500 Options Screener", layout="wide")
APP_TS = time.strftime("%Y-%m-%d %H:%M:%S ET", time.localtime())

st.title("📈 S&P 500 Options Screener")
st.caption(f"UI started: {APP_TS}")

SCRIPT = "swing_options_screener.py"
CSV_NAME = "pass_tickers.csv"
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(REPO_ROOT, CSV_NAME)

# -------------------- helpers --------------------
def run_subprocess(args):
    try:
        proc = subprocess.run(
            [sys.executable, os.path.join(REPO_ROOT, SCRIPT)] + args,
            capture_output=True,
            text=True,
            timeout=900,
            cwd=REPO_ROOT,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except Exception as e:
        return 1, "", f"Exception launching subprocess: {e}"

PIPE_HEADER_RE = re.compile(r"^Ticker\|", re.IGNORECASE)

def parse_pipe_stdout_to_df(stdout: str) -> pd.DataFrame:
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    start = None
    for i, ln in enumerate(lines):
        if PIPE_HEADER_RE.match(ln):
            start = i
            break
    if start is None:
        return pd.DataFrame()
    header = [h.strip() for h in lines[start].split("|")]
    rows = []
    for ln in lines[start + 1:]:
        if ln.startswith("Processed at"):
            break
        if "|" not in ln:
            continue
        parts = [p.strip() for p in ln.split("|")]
        if len(parts) == len(header):
            rows.append(parts)
    df = pd.DataFrame(rows, columns=header)
    # best-effort numeric conversion
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    return df

DEBUG_TICK = re.compile(r"===\s+DEBUG\s+([A-Z\.\-]+)\s+===")

def collect_failures_from_run(stdout: str):
    fails, current = {}, None
    for line in stdout.splitlines():
        m = DEBUG_TICK.search(line)
        if m:
            current = m.group(1)
            continue
        if current and "FAIL" in line and "❌" in line:
            tail = line.split("❌", 1)[-1].strip()
            reasons = [t.strip().replace("—","-").replace("–","-")
                       for t in re.split(r"[,\s]+", tail) if t.strip()]
            fails[current] = {"reasons": reasons}
            current = None
    return fails

_REASON_MAP = {
    "not_up_on_day": "Red day vs. prior close.",
    "relvol_below_threshold": "Relative volume below threshold.",
    "no_upside_to_resistance": "At/above resistance; no upside room.",
    "insufficient_atr_capacity_daily": "ATR capacity (daily) < req move.",
    "insufficient_atr_capacity_weekly": "ATR capacity (weekly) < req move.",
    "insufficient_atr_capacity_monthly": "ATR capacity (monthly) < req move.",
    "hist21d_insufficient": "No historical 21d move ≥ req% in last 12m.",
    "insufficient_data": "Insufficient price history.",
    "missing_series": "Price series missing or unusable.",
}

def humanize_reasons(codes):
    return "; ".join(_REASON_MAP.get(c, f"`{c}`") for c in codes) if codes else ""

_keyval = re.compile(r"(\w+)=([^\s]+)")

def parse_debug(stdout: str):
    header, verdict, reasons = {}, None, []
    for line in stdout.splitlines():
        for k, v in _keyval.findall(line):
            header[k] = v
        if "PASS" in line and "✅" in line:
            verdict = "PASS"
        if "FAIL" in line and "❌" in line:
            verdict = "FAIL"
            tail = line.split("❌", 1)[-1].strip()
            for token in re.split(r"[,\s]+", tail):
                t = token.strip().replace("—", "-").replace("–", "-")
                if t:
                    reasons.append(t)
    return header, verdict, reasons

def explain_ticker(tkr: str):
    rc, out, err = run_subprocess(["--explain", tkr.upper()])
    return parse_debug(out or ""), out, err

def near_misses(stdout: str, max_items=3) -> pd.DataFrame:
    fails = collect_failures_from_run(stdout)
    singles = [(t, v["reasons"][0]) for t, v in fails.items() if v.get("reasons") and len(v["reasons"]) == 1]
    close_to_green, others = [], []
    for tkr, reason in singles:
        if reason == "not_up_on_day":
            (hdr, verdict, reasons), _, _ = explain_ticker(tkr)
            try:
                entry = float(hdr.get("entry_used"))
                prev = float(hdr.get("prev_close_used"))
                pct = (entry - prev) / prev * 100.0
                gap = 0.0 if pct >= 0 else abs(pct)
                close_to_green.append({
                    "Ticker": tkr, "Reason": humanize_reasons([reason]),
                    "GapToGreen%": round(gap, 3),
                    "Entry": round(entry, 4), "PrevClose": round(prev, 4),
                    "EntryTimeET": hdr.get("EntryTimeET",""), "DataAgeMin": hdr.get("DataAgeMin",""),
                })
            except Exception:
                others.append({"Ticker": tkr, "Reason": humanize_reasons([reason])})
        else:
            others.append({"Ticker": tkr, "Reason": humanize_reasons([reason])})
    close_to_green.sort(key=lambda x: x.get("GapToGreen%", 9e9))
    ranked = close_to_green + others
    return pd.DataFrame(ranked[:max_items])

def build_pipe_text(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["|".join(map(str, cols))]
    for _, row in df.iterrows():
        parts = []
        for c in cols:
            val = row[c]
            if isinstance(val, float):
                parts.append(f"{val:.6g}")
            else:
                parts.append(str(val))
        lines.append("|".join(parts))
    return "\n".join(lines)

def _num(col_name, fmt=",.2f", help_txt=None, step=None):
    return st.column_config.NumberColumn(col_name, format=fmt, help=help_txt or "", step=step)

def show_tables(df: pd.DataFrame, title: str, key: str, compact_cols=None):
    """Compact table for mobile + full table in an expander. Keeps copy blocks."""
    if df.empty:
        st.warning(f"No rows for **{title}**.")
        return

    if compact_cols:
        present = [c for c in compact_cols if c in df.columns]
        st.markdown(f"### {title}")
        st.dataframe(
            df[present],
            use_container_width=True,
            hide_index=True,
            height=min(520, 80 + 32 * (len(df) + 1)),
        )
        pipe_text = build_pipe_text(df[present])
        st.caption("Copy compact table (pipe-delimited for Google Sheets)")
        st.code(pipe_text, language="text")
        st.download_button(
            "Copy / Download compact .txt",
            pipe_text.encode("utf-8"),
            file_name=f"{title.lower().replace(' ','_')}_compact.txt",
            mime="text/plain",
            key=f"copy_compact_{key}",
            use_container_width=True,
        )
    else:
        st.markdown(f"### {title}")

    with st.expander("Show all columns"):
        # numeric formatting
        col_cfg = {}
        percent_like = [c for c in df.columns if c.endswith("%") or "Pct" in c or "Percent" in c]
        money_like   = [c for c in df.columns if any(k in c for k in ["Price","TP","Risk$","ReqMove$","MaxProfit","Debit"])]
        int_like     = [c for c in df.columns if any(k in c for k in ["PassCount","Volume","Hist21d_PassCount"])]

        for c in percent_like: col_cfg[c] = _num(c, fmt="%.2f%%")
        for c in money_like:   col_cfg[c] = _num(c, fmt="$%,.2f")
        for c in int_like:     col_cfg[c] = _num(c, fmt="%,d", step=1)
        for c in df.select_dtypes("number").columns:
            col_cfg.setdefault(c, _num(c, fmt="%,.4f"))

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config=col_cfg,
            height=min(560, 100 + 32 * (len(df) + 1)),
        )

        pipe_text_full = build_pipe_text(df)
        st.caption("Copy full table (pipe-delimited for Google Sheets)")
        st.code(pipe_text_full, language="text")
        st.download_button(
            "Copy / Download full .txt",
            pipe_text_full.encode("utf-8"),
            file_name=f"{title.lower().replace(' ','_')}.txt",
            mime="text/plain",
            key=f"copy_full_{key}",
            use_container_width=True,
        )

# -------------------- UI --------------------
left, right = st.columns([2, 1])

with left:
    min_up_pct = st.number_input(
        "Min green % to qualify (applied at UI)",
        min_value=0.0, max_value=5.0, value=0.0, step=0.1, help="Require at least this % change vs prior close."
    )

    if st.button("Run Screener", use_container_width=True):
        st.info("Running screener… this may take a bit on first run.")
        rc, stdout, stderr = run_subprocess([])

        with st.expander("Console output"):
            st.code(stdout or "(no stdout)", language="bash")
            if stderr:
                st.error("stderr:")
                st.code(stderr, language="bash")

        # pass table (CSV preferred; fallback to stdout)
        df_pass = pd.DataFrame()
        if os.path.exists(CSV_PATH):
            try:
                df_pass = pd.read_csv(CSV_PATH)
            except Exception as e:
                st.error(f"Could not read {CSV_NAME}: {e}")
        if df_pass.empty:
            df_pass = parse_pipe_stdout_to_df(stdout)

        # apply UI-level min-up-pct requirement if column present
        if "Change%" in df_pass.columns and min_up_pct > 0:
            try:
                # Change% may be numeric or string like '0.04%' depending on backend
                if df_pass["Change%"].dtype == object:
                    df_pass["_chg_val"] = df_pass["Change%"].astype(str).str.replace("%","", regex=False)
                    df_pass["_chg_val"] = pd.to_numeric(df_pass["_chg_val"], errors="coerce")
                    mask = df_pass["_chg_val"] >= float(min_up_pct)
                else:
                    mask = df_pass["Change%"] >= float(min_up_pct)
                df_pass = df_pass[mask].drop(columns=[c for c in ["_chg_val"] if c in df_pass.columns])
            except Exception:
                pass

        if df_pass.empty:
            st.warning("No PASS tickers found (after UI filter) or CSV not produced.")
        else:
# show compact subset for quick view, but keep all columns in the expander
compact_cols = [
    "Ticker","EvalDate","Price","EntryTimeET","Change%","RelVol(TimeAdj63d)",
    "Resistance","TP","RR_to_Res","RR_to_TP","SupportType"
]

# If the options chain fields exist, tack them onto the compact view too:
for col in df_pass.columns:
    if col.startswith("OptExpiry") or col.startswith("RR_Spread") or col.startswith("MaxProfit"):
        compact_cols.append(col)

show_tables(df_pass, "PASS tickers", "pass", compact_cols=compact_cols)

        # Near-miss
        nm = near_misses(stdout, max_items=3)
        if not nm.empty:
            show_tables(nm, "🟡 Near-miss (Top 3)", "near_miss",
                        compact_cols=["Ticker","Reason","GapToGreen%","Entry","PrevClose","EntryTimeET"])
        else:
            st.caption("No single-reason near-misses detected this run.")

with right:
    st.markdown("### Explain a ticker")
    x_ticker = st.text_input("Ticker", placeholder="e.g., WMT, INTC, MOS")
    if st.button("Explain", use_container_width=True):
        if not x_ticker:
            st.warning("Type a ticker first.")
        else:
            st.info(f"Explaining {x_ticker.upper()}…")
            (hdr, verdict, reasons), out, err = explain_ticker(x_ticker)

            colA, colB, colC = st.columns(3)
            with colA:
                st.subheader(f"🔍 Debug: {x_ticker.upper()}")
                badge = "🟢 PASS" if verdict == "PASS" else ("🔴 FAIL" if verdict == "FAIL" else "⚪ Unknown")
                st.markdown(f"**Verdict:** {badge}")

            with colB:
                fields = []
                for k in ["session", "entry_src", "EntryTimeET", "DataAgeMin"]:
                    if k in hdr:
                        fields.append(f"**{k}**: `{hdr[k]}`")
                if fields:
                    st.markdown("**Context**  \n" + "  \n".join(fields))

            with colC:
                if "entry_used" in hdr and "prev_close_used" in hdr:
                    try:
                        entry = float(hdr["entry_used"])
                        prev = float(hdr["prev_close_used"])
                        pct = (entry - prev) / prev * 100.0
                        st.metric("Today vs Prior Close", f"{pct:+.2f}%")
                    except Exception:
                        pass

            if verdict == "FAIL":
                st.markdown("#### Why it failed")
                st.markdown(humanize_reasons(reasons) or "_See raw output below._")
            elif verdict == "PASS":
                st.success("All gates passed for this ticker.")

            with st.expander("Raw output"):
                st.code(out or "(no stdout)", language="bash")
                if err:
                    st.error("stderr:")
                    st.code(err, language="bash")

st.divider()
st.caption(
    "Use the **Min green %** input to require a real up-move (e.g., 0.5% or 1%). "
    "Compact tables show key columns; open *Show all columns* for the full dataset. "
    "Copy blocks under each table remain Google-Sheets ready (pipe-delimited). "
    "Prices are ~15-min delayed (Yahoo)."
)

