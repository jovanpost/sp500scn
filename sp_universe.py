# sp_universe.py
import pandas as pd
import re

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

def _normalize_symbol(sym: str) -> str:
    if not isinstance(sym, str):
        return ""
    s = sym.strip().upper()
    # yfinance expects class shares with '-' not '.'
    s = s.replace(".", "-").replace(" ", "")
    # drop any stray footnote markers or non-ticker chars (keep A-Z, 0-9, and '-')
    s = re.sub(r"[^A-Z0-9\-]", "", s)
    return s

def get_sp500_tickers() -> list[str]:
    """
    Scrapes Wikipedia for the S&P 500 constituents and returns a cleaned list
    of tickers suitable for yfinance. Falls back to an empty list on any error.
    """
    try:
        tables = pd.read_html(WIKI_URL, attrs={"id": "constituents"})
        if not tables:
            # fallback: try any table that has a 'Symbol' column
            tables = pd.read_html(WIKI_URL)
            tables = [t for t in tables if "Symbol" in t.columns]
            if not tables:
                return []
        df = tables[0]
        syms = df["Symbol"].dropna().astype(str).tolist()
    except Exception:
        return []

    cleaned = [_normalize_symbol(s) for s in syms]
    # Remove empties and duplicates, preserve order
    out, seen = [], set()
    for t in cleaned:
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out
