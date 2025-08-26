import os, sys, subprocess, time, re
import pandas as pd
import streamlit as st

st.set_page_config(page_title="S&P 500 Options Screener", layout="wide")

APP_TS = time.strftime("%Y-%m-%d %H:%M:%S ET", time.localtime())
st.title("📈 S&P 500 Options Screener")
st.caption(f"UI started: {APP_TS}")

CSV_PATH = "pass_tickers.csv"
SCRIPT = "swing_options_screener.py"

# --------------------------
# Utilities
# --------------------------
def run_subprocess(args):
    """Run the screener script and return (rc, stdout, stderr)."""
    try:
        proc = subprocess.run(
            [sys.executable, SCRIPT] + args,
            capture_output=True,
            text=True,
            timeout=600,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except Exception as e:
        return 1, "", f"Exception launching subprocess: {e}"

def run_screener():
    st.info("Running screener… this may take a bit on first run.")
    rc, out, err = run_subprocess([])
    with st.expander("Console output"):
        st.code(out or "(no stdout)", language="bash")
        if err:
            st.error("stderr:")
            st.code(err, language="bash")

    if os.path.exists(CSV_PATH):
        try:
            df = pd.read_csv(CSV_PATH)
        except Exception as e:
            st.error(f"Could not read {CSV_PATH}: {e}")
            return pd.DataFrame()
        return df
    else:
        return pd.DataFrame()

_REASON_MAP = {
    "not_up_on_day": "Red day vs. prior close.",
    "relvol_below_threshold": "Relative volume below threshold.",
    "no_upside_to_resistance": "Price already at/above resistance; no upside room.",
    "insufficient_atr_capacity_daily": "ATR capacity (daily) < required move.",
    "insufficient_atr_capacity_weekly": "ATR capacity (weekly) < required move.",
    "insufficient_atr_capacity_monthly": "ATR capacity (monthly) < required move.",
    "hist21d_insufficient": "No historical 21-day move ≥ required % in last 12 months.",
    "insufficient_data": "Insufficient price history.",
    "missing_series": "Price series missing or unusable.",
}

_keyval = re.compile(r"(\w+)=([^\s]+)")

def parse_debug(stdout: str):
    """
    Parse stdout from --explain into:
      - header dict (session/entry_src/entry_used/prev_close_used/EntryTimeET/DataAgeMin/…)
      - verdict ("PASS" or "FAIL")
      - reasons (list of codes)
    """
    header = {}
    verdict = None
    reasons = []
    for line in stdout.splitlines():
        # grab key=val pairs
        for k, v in _keyval.findall(line):
            header[k] = v
        # verdict lines
        if "PASS" in line and "✅" in line:
            verdict = "PASS"
        if "FAIL" in line and "❌" in line:
            verdict = "FAIL"
            # try to pull reason codes (words after FAIL icon)
            tail = line.split("❌", 1)[-1].strip()
            # split by spaces/commas
            for token in re.split(r"[,\s]+", tail):
                t = token.strip()
                if not t:
                    continue
                # normalize common tokens
                t = t.replace("—", "-").replace("–", "-")
                reasons.append(t)
    return header, verdict, reasons

def humanize_reasons(codes):
    out = []
    for c in codes:
        msg = _REASON_MAP.get(c, None)
        if msg:
            out.append(f"• **{msg}**  (`{c}`)")
    # if nothing matched, dump raw
    if not out and codes:
        out = [f"• `{c}`" for c in codes]
    return out

def explain_ticker(ticker: str):
    """Run --explain TICKER and show a human-readable summary + raw log."""
    if not ticker:
        st.warning("Type a ticker first.")
        return
    st.info(f"Explaining {ticker.upper()}…")
    rc, out, err = run_subprocess(["--explain", ticker.upper()])

    header, verdict, reasons = parse_debug(out or "")
    colA, colB, colC = st.columns(3)
    with colA:
        st.subheader(f"🔍 Debug: {ticker.upper()}")
        badge = "🟢 PASS" if verdict == "PASS" else ("🔴 FAIL" if verdict == "FAIL" else "⚪ Unknown")
        st.markdown(f"**Verdict:** {badge}")

    with colB:
        if header:
            # Show a few useful fields if present
            fields = []
            for k in ["session", "entry_src", "EntryTimeET", "DataAgeMin"]:
                if k in header:
                    fields.append(f"**{k}**: `{header[k]}`")
            if fields:
                st.markdown("**Context**  \n" + "  \n".join(fields))

    with colC:
        if "entry_used" in header and "prev_close_used" in header:
            try:
                entry = float(header["entry_used"])
                prev = float(header["prev_close_used"])
                pct = (entry - prev) / prev * 100.0
                st.metric("Today vs Prior Close", f"{pct:+.2f}%")
            except Exception:
                pass

    if verdict == "FAIL":
        bullets = humanize_reasons(reasons)
        if bullets:
            st.markdown("#### Why it failed")
            st.markdown("\n".join(bullets))
        else:
            st.markdown("#### Why it failed")
            st.markdown("_Couldn’t parse reasons; see raw output below._")
    elif verdict == "PASS":
        st.success("All gates passed for this ticker.")

    with st.expander("Raw output"):
        st.code(out or "(no stdout)", language="bash")
        if err:
            st.error("stderr:")
            st.code(err, language="bash")

# --------------------------
# Layout
# --------------------------
left, right = st.columns([2, 1])

with left:
    if st.button("Run Screener", use_container_width=True):
        df = run_screener()
        if df.empty:
            st.warning("No PASS tickers found (or CSV not produced).")
        else:
            st.success(f"Found {len(df)} PASS tickers")
            st.dataframe(df, use_container_width=True)
            st.download_button(
                "Download CSV",
                df.to_csv(index=False).encode("utf-8"),
                file_name="pass_tickers.csv",
                mime="text/csv",
                use_container_width=True,
            )

with right:
    st.markdown("### Explain a ticker")
    x_ticker = st.text_input("Ticker", placeholder="e.g., WMT, INTC, MOS")
    if st.button("Explain", use_container_width=True):
        explain_ticker(x_ticker)

st.divider()
st.caption(
    "Notes: Prices are ~15-min delayed from Yahoo. ‘Explain’ runs your CLI "
    "(`--explain TICKER`), parses the log, and shows human-readable reasons."
)

