#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=======================================================================
Bibliography (Section Index)
=======================================================================
1. Imports & CLI
2. Paths, Constants, Helpers
3. Robust IO (read/write CSV)
4. Outcomes Upsert (non-destructive append/update)
5. Screener Runner (invoke library, gather DataFrames)
6. Main (glue: run, save pass file, write logs, upsert outcomes)
=======================================================================
"""

# --------------------------------------------------------------------
# 1. Imports & CLI
# --------------------------------------------------------------------
import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
import inspect
import pandas as pd

# Screener module (must be importable from repo root)
try:
    import swing_options_screener as sos
except Exception as e:
    print(f"[FATAL] Could not import swing_options_screener: {e}", file=sys.stderr)
    sys.exit(1)

# Optional universe helper (only used if present)
try:
    import sp_universe as spuni  # optional
except Exception:
    spuni = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run swing scan and write history/outcomes."
    )
    p.add_argument(
        "--universe",
        default="sp500",
        help="Universe key (kept for logs/metadata; not passed to screener unless tickers param is supported).",
    )
    p.add_argument(
        "--with-options",
        action="store_true",
        help="Ask the screener to include options fields.",
    )
    p.add_argument(
        "--also-save-scan",
        action="store_true",
        help="If available, save raw scan DF alongside pass file.",
    )
    return p.parse_args()


# --------------------------------------------------------------------
# 2. Paths, Constants, Helpers
# --------------------------------------------------------------------
HISTORY_DIR = Path("data/history")
LOGS_DIR = Path("data/logs")
OUTCOMES_FILE = HISTORY_DIR / "outcomes.csv"

UTC_NOW = datetime.now(timezone.utc)
STAMP = UTC_NOW.strftime("%Y%m%d-%H%M")

PASS_PATH = HISTORY_DIR / f"pass_{STAMP}.csv"
SCAN_PATH = HISTORY_DIR / f"scan_{STAMP}.csv"
LOG_PATH = LOGS_DIR / f"scan_{STAMP}.txt"


def ensure_dirs() -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def pick(df: pd.DataFrame, col: str, default=None):
    """Safe column getter for a homogeneous value; returns default if missing."""
    try:
        if df is None or df.empty or col not in df.columns:
            return default
        vals = df[col].dropna().unique()
        if len(vals) == 0:
            return default
        return vals[0]
    except Exception:
        return default


def safe_str(x):
    return "" if x is None else str(x)


# --------------------------------------------------------------------
# 3. Robust IO (read/write CSV)
# --------------------------------------------------------------------
def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as e:
            print(f"[WARN] Failed reading {path}: {e}", file=sys.stderr)
    return pd.DataFrame()


def write_csv(path: Path, df: pd.DataFrame) -> None:
    try:
        df.to_csv(path, index=False)
    except Exception as e:
        print(f"[ERROR] Failed writing {path}: {e}", file=sys.stderr)


# --------------------------------------------------------------------
# 4. Outcomes Upsert (non-destructive append/update)
# --------------------------------------------------------------------
OUTCOME_COLS = [
    "Ticker",
    "EvalDate",
    "Price",
    "EntryTimeET",
    "Status",     # PENDING / SETTLED
    "HitDateET",  # filled when settled as HIT/MISS date
    "Expiry",     # options expiry (if available)
    "BuyK",       # option buy strike (if available)
    "SellK",      # option sell strike (if available)
    "TP",         # target price
    "Notes",
]


def _empty_outcomes_df() -> pd.DataFrame:
    return pd.DataFrame(columns=OUTCOME_COLS)


def _extract_value(row: pd.Series, *names, default=""):
    for n in names:
        if n in row and pd.notna(row[n]):
            return row[n]
    return default


def upsert_outcomes_from_pass(df_pass: pd.DataFrame, path: Path) -> None:
    """
    Add one 'PENDING' row per pass (Ticker, EvalDate) if not already present.
    Never overwrites existing non-empty rows.
    """
    base = read_csv_if_exists(path)
    if base.empty:
        base = _empty_outcomes_df()

    # Build new rows
    rows = []
    for _, r in df_pass.iterrows():
        tkr = _extract_value(r, "Ticker", "ticker", default="")
        if not tkr:
            continue

        eval_date = _extract_value(r, "EvalDate", "eval_date", default=UTC_NOW.date().isoformat())
        price = _extract_value(r, "Price", "price", default="")
        entry_time = _extract_value(r, "EntryTimeET", "EntryET", default="")
        expiry = _extract_value(r, "Expiry", "ExpDate", "OptionsExpiry", default="")
        buyk = _extract_value(r, "BuyK", "BuyStrike", "Buy", default="")
        sellk = _extract_value(r, "SellK", "SellStrike", "Sell", default="")
        tp = _extract_value(r, "TP", "Target", default="")

        rows.append(
            {
                "Ticker": tkr,
                "EvalDate": eval_date,
                "Price": price,
                "EntryTimeET": entry_time,
                "Status": "PENDING",
                "HitDateET": "",
                "Expiry": expiry,
                "BuyK": buyk,
                "SellK": sellk,
                "TP": tp,
                "Notes": "",
            }
        )

    add_df = pd.DataFrame(rows, columns=OUTCOME_COLS)
    if add_df.empty:
        return

    # Non-destructive upsert on key (Ticker, EvalDate).
    key_cols = ["Ticker", "EvalDate"]
    if not base.empty:
        merged = pd.concat([base, add_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=key_cols, keep="first")
    else:
        merged = add_df

    write_csv(path, merged)


# --------------------------------------------------------------------
# 5. Screener Runner (invoke library, gather DataFrames)
# --------------------------------------------------------------------
def _candidate_tickers_from_universe(universe_key: str):
    """Try to turn a universe key into a list of tickers (if helpers exist)."""
    if spuni is None:
        return None
    try:
        if hasattr(spuni, "get_universe"):
            return spuni.get_universe(universe_key)
        if hasattr(spuni, "UNIVERSES"):
            return spuni.UNIVERSES.get(universe_key)
    except Exception:
        pass
    return None


def _extract_df(out, *keys) -> pd.DataFrame:
    """If screener returns dict, try multiple keys; if DF, return it; else empty DF."""
    if isinstance(out, pd.DataFrame):
        return out
    if isinstance(out, dict):
        for k in keys:
            v = out.get(k)
            if isinstance(v, pd.DataFrame):
                return v
    return pd.DataFrame()


def run_screener(universe: str, with_options: bool, save_scan: bool):
    """
    Calls sos.run_scan defensively:
      - only passes kwargs that the function actually supports (introspection)
      - if it returns dict, tries common keys to pull pass/scan dataframes
    """
    fn = getattr(sos, "run_scan", None)
    if fn is None:
        raise RuntimeError("swing_options_screener.run_scan not found")

    sig = inspect.signature(fn)
    params = sig.parameters.keys()

    kwargs = {}
    if "with_options" in params:
        kwargs["with_options"] = bool(with_options)

    # only pass tickers if the function supports it
    if "tickers" in params:
        tickers = _candidate_tickers_from_universe(universe)
        if tickers:
            kwargs["tickers"] = tickers

    # Optional "return_scan_df" style flags (support several common names)
    for flag in ("return_scan_df", "return_full_scan", "return_scan"):
        if flag in params:
            kwargs[flag] = bool(save_scan)

    out = fn(**kwargs)

    # Try to extract dataframes
    df_pass = _extract_df(out, "pass", "passes", "df_pass", "pass_df", "passes_df")
    df_scan = _extract_df(out, "scan", "df_scan", "scan_df", "full_scan")

    return df_pass, df_scan


# --------------------------------------------------------------------
# 6. Main (glue: run, save pass file, write logs, upsert outcomes)
# --------------------------------------------------------------------
def main():
    args = parse_args()
    ensure_dirs()

    ts = STAMP
    log_lines = [
        f"UTC start: {UTC_NOW.isoformat()}",
        f"universe={args.universe}",
        f"with_options={args.with_options}",
        f"also_save_scan={args.also_save_scan}",
    ]

    try:
        df_pass, df_scan = run_screener(
            universe=args.universe,
            with_options=args.with_options,
            save_scan=args.also_save_scan,
        )
    except Exception as e:
        msg = f"[FATAL] Screener failed: {e}"
        print(msg, file=sys.stderr)
        log_lines.append(msg)
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "w") as f:
            f.write("\n".join(log_lines) + "\n")
        return

    # Save pass results
    if isinstance(df_pass, pd.DataFrame) and not df_pass.empty:
        write_csv(PASS_PATH, df_pass)
        upsert_outcomes_from_pass(df_pass, OUTCOMES_FILE)
        log_lines.append(f"pass_rows={len(df_pass)} -> {PASS_PATH.name}")
    else:
        log_lines.append("pass_rows=0 (nothing to save)")

    # Save scan results (optional)
    if args.also_save_scan and isinstance(df_scan, pd.DataFrame) and not df_scan.empty:
        write_csv(SCAN_PATH, df_scan)
        log_lines.append(f"scan_rows={len(df_scan)} -> {SCAN_PATH.name}")

    # Write log
    with open(LOG_PATH, "w") as f:
        f.write("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()
