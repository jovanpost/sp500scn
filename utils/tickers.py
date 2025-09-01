"""Ticker normalization helpers shared across modules."""
from __future__ import annotations

# Aliases for common company names -> tickers
ALIAS_MAP = {
    "NVIDIA": "NVDA", "NVIDIA CORPORATION": "NVDA",
    "TESLA": "TSLA", "TESLA INC": "TSLA",
    "APPLE": "AAPL", "APPLE INC": "AAPL",
    "MICROSOFT": "MSFT", "MICROSOFT CORPORATION": "MSFT",
    "ALPHABET": "GOOGL", "GOOGLE": "GOOGL",
    "META": "META", "META PLATFORMS": "META",
    "AMAZON": "AMZN", "AMAZONCOM": "AMZN", "AMAZON.COM": "AMZN",
    "NETFLIX": "NFLX", "WALMART": "WMT", "WALMART INC": "WMT",
    "JPMORGAN": "JPM", "JPMORGAN CHASE": "JPM",
    "BERKSHIRE": "BRK-B", "BERKSHIRE HATHAWAY": "BRK-B",
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
    """Handle Berkshire share classes in various formats."""
    s2 = (
        s.replace(" ", "")
        .replace("-", "")
        .replace("_", "")
        .replace(".", "")
        .upper()
    )
    if s2 == "BRKB":
        return "BRK-B"
    if s2 == "BRKA":
        return "BRK-A"
    return None


def normalize_symbol(inp: str) -> str | None:
    """Best-effort mapping: ticker-looking -> upper, aliases, and heuristics."""
    if not inp:
        return None

    s = str(inp)
    # Strip whitespace and odd spaces from HTML sources
    s = s.replace("\u200b", "").replace("\xa0", "").strip()
    if not s:
        return None

    # Company name path (aliases)
    key = s.upper()
    key = key.replace(",", "").replace(".", "")
    for kill in (" INC", " CORPORATION", " COMPANY", " HOLDINGS", " PLC", " LTD"):
        key = key.replace(kill, "")
    key = key.replace(" CLASS A", "").replace(" CLASS B", "")
    key = " ".join(key.split())

    if key in ALIAS_MAP:
        return ALIAS_MAP[key]

    # Looks like a ticker already?
    if 1 <= len(s) <= 6 and all(c.isalnum() or c in ".-_" for c in s):
        brk = _normalize_brk(s)
        if brk:
            return brk
        return s.upper().replace("_", "-").replace(".", "-")

    # Last chance Berkshire normalization
    brk = _normalize_brk(s)
    if brk:
        return brk

    return s.upper().replace("_", "-").replace(".", "-")

