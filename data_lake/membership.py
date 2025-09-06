"""Build historical S&P 500 membership information."""

from __future__ import annotations

import io
import re
import time
from datetime import date
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .schemas import MemberRow
from .storage import Storage

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
UA = {"User-Agent": "Mozilla/5.0"}


def _http_get(url: str) -> str:
    delays = [0.5, 1.0, 2.0]
    for i, d in enumerate(delays):
        try:
            resp = requests.get(url, headers=UA, timeout=10)
            resp.raise_for_status()
            return resp.text
        except Exception:
            if i == len(delays) - 1:
                raise
            time.sleep(d)
    raise RuntimeError("failed to fetch url")


def _scrape_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    html = _http_get(WIKI_URL)
    try:
        tables = pd.read_html(html)
    except ValueError:
        tables = []
    current = None
    changes = None
    for t in tables:
        # Flatten MultiIndex headers and normalize to simple strings for matching
        t.columns = [
            " ".join(
                [
                    str(x)
                    for x in (col if isinstance(col, tuple) else (col,))
                    if x is not None
                ]
            ).strip()
            for col in t.columns
        ]
        cols = {str(c).lower() for c in t.columns}
        if {"symbol", "security"} <= cols:
            current = t
        if {"date", "added", "removed"} <= cols:
            changes = t
    if current is not None and changes is not None:
        return current, changes
    soup = BeautifulSoup(html, "lxml")
    for table in soup.find_all("table"):
        headers = [str(th.get_text(strip=True)).lower() for th in table.find_all("th")]
        if current is None and "symbol" in headers and "security" in headers:
            current = pd.read_html(str(table))[0]
        elif changes is None and {"date", "added", "removed"} <= set(headers):
            changes = pd.read_html(str(table))[0]
    if current is None or changes is None:
        raise RuntimeError("could not locate membership tables")
    return current, changes


def _normalize_ticker(t: str) -> str:
    return t.upper().replace(".", "-").strip()


def _extract_ticker(text: str) -> str | None:
    if not isinstance(text, str):
        return None
    m = re.search(r"[A-Z]{1,5}(?:\.[A-Z])?", text.upper())
    if m:
        return _normalize_ticker(m.group(0))
    return None


def _extract_name(text: str) -> str:
    if not isinstance(text, str):
        return ""
    parts = re.split(r"\s+-\s+|\s+", text, maxsplit=1)
    return parts[1].strip() if len(parts) > 1 else text.strip()


def _load_overrides() -> pd.DataFrame:
    path = Path(__file__).with_name("manual_overrides.csv")
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(
        columns=["ticker", "replace_ticker", "start_date", "end_date", "notes"]
    )


def load_membership(storage: Storage | None = None) -> pd.DataFrame:
    if storage is None:
        storage = Storage()
    try:
        data = storage.read_bytes("membership/sp500_members.parquet")
        return pd.read_parquet(io.BytesIO(data))
    except Exception:
        preview_path = Path(__file__).with_name("sp500_members_preview.csv")
        if preview_path.exists():
            return pd.read_csv(preview_path)
        raise


def build_membership(storage: Storage) -> str:
    """
    Scrape, normalize, apply overrides, validate, write parquet + csv preview.
    Return human-readable summary (counts, date range).
    """

    current, changes = _scrape_tables()
    current["ticker"] = current["Symbol"].apply(_normalize_ticker)
    current["name"] = current["Security"].astype(str)

    # Ensure unique columns and locate date/added/removed headers by substring
    # Drop any duplicate headers and work on a copy so subsequent column
    # assignments are applied to the DataFrame we operate on.  Using ``copy``
    # avoids pandas' ``SettingWithCopyWarning`` which previously could leave
    # the expected "Date" column absent and lead to KeyError when dropping
    # missing values.
    changes = changes.loc[:, ~changes.columns.duplicated()].copy()
    norm = {c: str(c).strip().lower() for c in changes.columns}
    date_candidates = [c for c, s in norm.items() if "date" in s]
    if not date_candidates:
        raise RuntimeError("membership 'changes' table missing a Date column")
    cleaned = (
        changes[date_candidates]
        .apply(lambda s: s.astype(str).str.replace(r"\[.*?\]", "", regex=True).str.strip())
    )
    date_series = cleaned.bfill(axis=1).iloc[:, 0]
    changes = changes.drop(columns=date_candidates)
    changes["Date"] = pd.to_datetime(
        date_series, errors="coerce", infer_datetime_format=True
    )
    if changes["Date"].isna().all():
        # Fallback: separate year/month/day columns if present
        norm_cols = {c: re.sub(r"\s+", " ", str(c)).strip().lower() for c in changes.columns}
        y = next((c for c, s in norm_cols.items() if "year" in s), None)
        m = next((c for c, s in norm_cols.items() if "month" in s), None)
        d = next((c for c, s in norm_cols.items() if "day" in s), None)
        if y and m and d:
            changes["Date"] = pd.to_datetime(
                {"year": changes[y], "month": changes[m], "day": changes[d]},
                errors="coerce",
            )
    date_col = changes.get("Date")
    if date_col is None:
        raise RuntimeError("membership 'changes' table missing a parsable Date column")
    # Drop rows lacking a parsable date.  Using boolean indexing avoids the
    # ``KeyError`` previously seen when the "Date" column was unexpectedly
    # absent due to upstream parsing quirks.
    changes = changes[date_col.notna()]
    added_candidates = [c for c, s in norm.items() if "add" in s]
    removed_candidates = [c for c, s in norm.items() if "remov" in s]
    if not added_candidates or not removed_candidates:
        raise RuntimeError(
            "membership 'changes' table missing Added/Removed-like columns"
        )
    added_col, removed_col = added_candidates[0], removed_candidates[0]
    changes = changes.rename(columns={added_col: "Added", removed_col: "Removed"})[
        ["Date", "Added", "Removed"]
    ]
    records: List[dict] = []
    for _, row in changes.iterrows():
        d = row["Date"].date()
        add_t = _extract_ticker(row["Added"])
        rem_t = _extract_ticker(row["Removed"])
        if add_t:
            records.append(
                {
                    "action": "add",
                    "ticker": add_t,
                    "name": _extract_name(row["Added"]),
                    "date": d,
                }
            )
        if rem_t:
            records.append({"action": "remove", "ticker": rem_t, "date": d})
    records.sort(key=lambda r: r["date"])

    membership: Dict[str, List[Dict[str, date | None]]] = {}
    names: Dict[str, str] = {}
    for rec in records:
        t = rec["ticker"]
        if rec["action"] == "add":
            names[t] = rec.get("name", names.get(t, ""))
            membership.setdefault(t, []).append(
                {"start_date": rec["date"], "end_date": None}
            )
        else:
            intervals = membership.get(t)
            if intervals:
                intervals[-1]["end_date"] = rec["date"]

    today = date.today()
    for _, row in current.iterrows():
        t = row["ticker"]
        names[t] = row["name"]
        intervals = membership.get(t)
        if not intervals:
            membership[t] = [{"start_date": today, "end_date": None}]
        elif intervals[-1]["end_date"] is not None:
            membership[t].append({"start_date": today, "end_date": None})

    rows: List[MemberRow] = []
    for t, ivs in membership.items():
        for iv in ivs:
            row: MemberRow = {
                "ticker": t,
                "name": names.get(t),
                "start_date": str(iv["start_date"]),
                "end_date": str(iv["end_date"]) if iv["end_date"] else None,
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    overrides = _load_overrides()
    if not overrides.empty:
        overrides["ticker"] = overrides["ticker"].apply(_normalize_ticker)
        if "replace_ticker" in overrides.columns:
            overrides["replace_ticker"] = overrides["replace_ticker"].apply(
                lambda t: _normalize_ticker(t) if isinstance(t, str) else t
            )
    for ov in overrides.itertuples():
        mask = df["ticker"] == ov.ticker
        if pd.notna(getattr(ov, "replace_ticker", None)):
            df.loc[mask, "ticker"] = ov.replace_ticker
        if pd.notna(getattr(ov, "start_date", None)):
            df.loc[mask, "start_date"] = ov.start_date
        if pd.notna(getattr(ov, "end_date", None)):
            df.loc[mask, "end_date"] = ov.end_date

    df["ticker"] = df["ticker"].apply(_normalize_ticker)
    df = (
        df.sort_values(["ticker", "start_date", "end_date"])
        .drop_duplicates()
        .reset_index(drop=True)
    )

    cleaned: List[MemberRow] = []
    for t, grp in df.groupby("ticker"):
        grp = grp.sort_values("start_date")
        prev = None
        for row in grp.to_dict("records"):
            if row["end_date"] and row["end_date"] < row["start_date"]:
                row["end_date"] = row["start_date"]
            if (
                prev
                and prev["end_date"] == row["start_date"]
                and prev["name"] == row["name"]
            ):
                prev["end_date"] = row["end_date"]
            else:
                if prev:
                    cleaned.append(prev)
                prev = row
        if prev:
            cleaned.append(prev)
    df = pd.DataFrame(cleaned)

    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    storage.write_bytes("membership/sp500_members.parquet", buffer.getvalue())

    preview_path = Path(__file__).with_name("sp500_members_preview.csv")
    df.head(100).to_csv(preview_path, index=False)

    start = df["start_date"].min()
    end = (
        df["end_date"].dropna().max()
        if not df["end_date"].dropna().empty
        else "present"
    )
    summary = f"{len(df)} rows from {start} to {end}"
    return summary
