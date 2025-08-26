# app.py
import io
import sys
import time
import contextlib
from datetime import datetime, timezone, timedelta

import pandas as pd
import streamlit as st

# ---- Utilities --------------------------------------------------------------

ET = timezone(timedelta(hours=-5), name="ET")  # crude ET; Streamlit Cloud uses UTC
def now_et() -> datetime:
    # Streamlit cloud runs in UTC; adjust to ET (no DST handling here on purpose)
    return datetime.utcnow().replace(tzinfo=timezone.utc).astimezone(ET)

def read_csv_if_exists(path: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
        if df.empty:
            return None
        return df
    except Exception:
        return None

def df_to_pipe(df: pd.DataFrame) -> str:
    # Produce Google-Sheets friendly pipe-delimited text with all columns
    cols = list(df.columns)
    lines = ["|".join(cols)]
    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r.get(c)
            if pd.isna(v):
                vals.append("")
            else:
                s = str(v)
                # keep pipes safe-ish by replacing with similar char
                vals.append(s.replace("|", "¦"))
        lines.append("|".join(vals))
    return "\n".join(lines)

def auto_compact_columns(df: pd.DataFrame) -> list[str]:
    """
    Base compact columns + append options columns if present.
    Hist21d/etc. will still be visible in the 'Show all columns' expander and copy block.
    """
    base = [
        "Ticker","EvalDate","Price","EntryTimeET","Change%","RelVol(TimeAdj63d)",
        "Resistance","TP","RR_to_Res","RR_to_TP","SupportType"
    ]
    cols = [c for c in base if c in df.columns]

    # Prefer to include options/synthetic columns in the compact table too
    prefer_prefixes = (
        "OptExpiry", "BuyK", "SellK", "Width", "DebitMid", "DebitCons",
        "MaxProfitMid", "MaxProfitCons", "RR_Spread_Mid", "RR_Spread_Cons",
        "BreakevenMid", "PricingNote"
    )
    for c in df.columns:
        if c.startswith(prefer_prefixes) and c not in cols:
            cols.append(c)
    return cols or list(df.columns)


@contextlib.contextmanager
def capture_stdout():
    buf = io.StringIO()
    old = sys.stdout
    try:
        sys.stdout = buf
        yield buf
    finally:
        sys.stdout = old


def invoke_run():
    """
    Calls swing_options_screener.run_scan(), captures console output,
    then loads pass/near CSVs if produced by your script.
    Works even if run_scan() prints instead of returning DataFrames.
    """
    try:
        from swing_options_screener import run_scan
    except Exception as e:
        return None, None, f"Import error: {e}"

    with capture_stdout() as out:
        try:
            result = run_scan()  # your existing function
        except Exception as e:
            print(f"[run_scan error] {e}", file=sys.stderr)
            result = None

    console_text = out.getvalue()

    # prefer DataFrame from return, else read CSV written by your script
    df_pass = None
    near_df = None
    if isinstance(result, dict):
        df_pass = result.get("pass_df")
        near_df = result.get("near_df")
    elif isinstance(result, pd.DataFrame):
        df_pass = result

    # fallback to files if present
    if df_pass is None:
        df_pass = read_csv_if_exists("pass_tickers.csv")
    if near_df is None:
        near_df = read_csv_if_exists("near_misses.csv")

    return df_pass, near_df, console_text


def explain_one(ticker: str) -> str:
    """
    Calls swing_options_screener.explain_ticker(t) and returns the printed explanation.
    """
    try:
        from swing_options_screener import explain_ticker
    except Exception as e:
        return f"Import error: {e}"

    with capture_stdout() as out:
        try:
            explain_ticker(ticker.strip().upper())
        except Exception as e:
            print(f"[explain_ticker error] {e}")
    return out.getvalue()


def render_table_block(df: pd.DataFrame, title: str, key_prefix: str):
    st.subheader(title)

    # Compact view first
    compact_cols = auto_compact_columns(df)
    st.dataframe(
        df[compact_cols],
        use_container_width=True,
        hide_index=True
    )

    # Full view (all columns)
    with st.expander("Show all columns", expanded=False):
        st.dataframe(df, use_container_width=True, hide_index=True)

    # Copy/download
    pipe_text = df_to_pipe(df)
    st.markdown("**Copy table (pipe-delimited for Google Sheets)**")
    st.code(pipe_text, language="text")
    st.download_button(
        "Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"{key_prefix}_tickers.csv",
        mime="text/csv",
        key=f"dl_{key_prefix}"
    )


# ---- UI ---------------------------------------------------------------------

st.set_page_config(page_title="S&P 500 Options Screener", layout="wide")
st.title("📈 S&P 500 Options Screener")

st.caption(f"UI started: {now_et().strftime('%Y-%m-%d %H:%M:%S ET')}")

col_left, col_right = st.columns([2, 1])

with col_left:
    if st.button("Run Screener", use_container_width=True):
        with st.status("Running screener… this may take a bit on first run.", expanded=True) as status:
            t0 = time.time()
            df_pass, near_df, console_text = invoke_run()
            status.update(state="complete")

        with st.expander("Console output", expanded=False):
            st.code(console_text or "(no console output)")

        if df_pass is None or df_pass.empty:
            st.warning("No PASS tickers found (or CSV not produced).")

            if isinstance(near_df, pd.DataFrame) and not near_df.empty:
                # Keep only the 3 nearest misses based on how your script scores them.
                # If your script already included a 'FailReason' or 'Score', we retain them.
                top3 = near_df.head(3)
                render_table_block(top3, "Closest 3 (failed) — reasons included", "near_misses")
        else:
            # Show PASS tickers with full column set (including Hist21d_* and options fields)
            render_table_block(df_pass, "PASS tickers", "pass")

with col_right:
    st.header("Explain a ticker")
    t = st.text_input("Ticker", placeholder="e.g., WMT, INTC, MOS")
    if st.button("Explain", use_container_width=True, type="primary", disabled=not t.strip()):
        with st.status(f"Explaining {t.strip().upper()}…"):
            txt = explain_one(t)
        st.subheader(f"🔍 Debug: {t.strip().upper()}")
        st.code(txt or "(no output)")

st.caption(
    "Notes: Yahoo prices are ~15-min delayed. "
    "'Explain' runs the same logic as your CLI and prints the exact gate that failed. "
    "All columns (Hist21d_*, ResLookbackDays, options pricing fields) are shown in the full view and copy block."
)

