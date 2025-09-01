# sp_universe.py
# Robust S&P 500 universe fetcher with caching and Wikipedia parsing
import json
import time
from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd
import requests
from utils.tickers import normalize_symbol

CACHE_PATH = "sp500_cache.json"
CACHE_MAX_AGE_SECONDS = 6 * 60 * 60  # 6 hours
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Expose last error for the UI to show a helpful message
LAST_ERROR: Optional[str] = None

def _load_cache() -> Optional[List[str]]:
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            obj = json.load(f)
        ts = obj.get("fetched_at_epoch", 0)
        if (time.time() - ts) <= CACHE_MAX_AGE_SECONDS:
            return obj.get("tickers", []) or None
    except Exception:
        pass
    return None

def _save_cache(tickers: List[str]) -> None:
    try:
        payload = {
            "fetched_at_epoch": int(time.time()),
            "fetched_at_iso": datetime.now(timezone.utc).isoformat(),
            "tickers": tickers,
        }
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        # Non-fatal if caching fails
        pass

def _fetch_wikipedia_html() -> str:
    # Use a real UA; some hosts rate-limit generic agents
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; sp500-screener/1.0; +https://example.com)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    resp = requests.get(WIKI_URL, headers=headers, timeout=20)
    resp.raise_for_status()
    return resp.text

def _parse_tickers_from_html(html: str) -> List[str]:
    # Prefer the table with id='constituents'; fall back to any table containing "Symbol"
    # Use bs4-backed parser to reduce need for lxml
    tables = pd.read_html(html, flavor="bs4")
    if not tables:
        raise RuntimeError("No HTML tables found on Wikipedia page.")

    # Look for a table with a 'Symbol' column (case-insensitive)
    candidates = []
    for df in tables:
        lower_cols = {c.lower(): c for c in df.columns if isinstance(c, str)}
        if any(k in lower_cols for k in ("symbol", "ticker symbol", "ticker")):
            candidates.append((df, lower_cols))
    if not candidates:
        raise RuntimeError("No table with a Symbol/Ticker column was found.")

    # Prefer the first candidate — Wikipedia typically has the constituents first
    df, lower_cols = candidates[0]
    sym_col_name = lower_cols.get("symbol") or lower_cols.get("ticker symbol") or lower_cols.get("ticker")
    if sym_col_name is None:
        raise RuntimeError("Could not identify the symbol column.")

    tickers = []
    for raw in df[sym_col_name].astype(str).tolist():
        sym = normalize_symbol(raw)
        if sym and sym.isascii():
            tickers.append(sym)

    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq

def get_sp500_tickers() -> List[str]:
    """
    Returns a list of current S&P 500 tickers (normalized for Yahoo Finance).
    Uses a 6-hour on-disk cache to avoid repeated scrapes.
    On failure, returns [] and sets LAST_ERROR.
    """
    global LAST_ERROR

    # 1) Cache
    cached = _load_cache()
    if cached:
        LAST_ERROR = None
        return cached

    # 2) Fresh pull
    try:
        html = _fetch_wikipedia_html()
        tickers = _parse_tickers_from_html(html)
        if not tickers:
            raise RuntimeError("Parsed zero tickers from Wikipedia.")
        _save_cache(tickers)
        LAST_ERROR = None
        return tickers
    except Exception as e:
        LAST_ERROR = f"{type(e).__name__}: {e}"
        return []

