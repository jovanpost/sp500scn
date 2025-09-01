import os
import glob
import pandas as pd


def latest_pass_file(pass_dir, hist_dir):
    """Return the newest pass_*.csv from either directory."""
    candidates = []
    for d in [pass_dir, hist_dir]:
        candidates.extend(glob.glob(os.path.join(d, "pass_*.csv")))
    return sorted(candidates)[-1] if candidates else None


def load_outcomes(out_file):
    """Load outcomes CSV if present."""
    if os.path.exists(out_file):
        return pd.read_csv(out_file)
    return pd.DataFrame()


def yf_fetch_daily(symbol: str):
    """Fetch ~6 months of daily bars via yfinance."""
    try:
        import yfinance as yf
        df = yf.download(symbol, period="6mo", interval="1d", auto_adjust=False, progress=False)
        if df is None or df.empty:
            return None
        cols = {c.lower(): c for c in df.columns}
        for need in ["Open", "High", "Low", "Close", "Volume"]:
            if need not in df.columns:
                for k, v in cols.items():
                    if k == need.lower():
                        break
                else:
                    return None
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        return df if not df.empty else None
    except Exception:
        return None
