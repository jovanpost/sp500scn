"""Build historical S&P 500 membership information from GitHub dataset."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from .storage import Storage
import streamlit as st

URL = "https://raw.githubusercontent.com/fja05680/sp500/master/sp500_ticker_start_end.csv"


def _normalize_ticker(t: str) -> str:
    return t.upper().replace(".", "-").strip()


def _load_overrides() -> pd.DataFrame:
    path = Path(__file__).with_name("manual_overrides.csv")
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(
        columns=["ticker", "replace_ticker", "start_date", "end_date", "notes"]
    )


def _load_github() -> pd.DataFrame:
    df = pd.read_csv(URL)
    norm = {c.lower().replace(" ", ""): c for c in df.columns}
    tcol = norm.get("ticker") or norm.get("symbol")
    scol = norm.get("start") or norm.get("first") or norm.get("firstdate") or norm.get(
        "startdate"
    )
    ecol = norm.get("end") or norm.get("last") or norm.get("lastdate") or norm.get(
        "enddate"
    )
    df = df.rename(columns={tcol: "ticker", scol: "start_date", ecol: "end_date"})
    df["ticker"] = (
        df["ticker"].astype(str).str.upper().str.strip().str.replace(".", "-", regex=False)
    )
    df["start_date"] = (
        pd.to_datetime(df["start_date"], errors="coerce").dt.date.astype("string")
    )
    e = pd.to_datetime(df["end_date"], errors="coerce")
    df["end_date"] = e.dt.date.astype("string")
    df.loc[e.isna(), "end_date"] = None
    df["name"] = None
    df = (
        df[["ticker", "name", "start_date", "end_date"]]
        .drop_duplicates()
        .sort_values(["ticker", "start_date"])
        .reset_index(drop=True)
    )
    return df


@st.cache_data(show_spinner=False, hash_funcs={Storage: lambda _: 0})
def load_membership(
    storage: Storage | None = None, cache_salt: str = ""
) -> pd.DataFrame:
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


# --- New: small helper used by the UI tab ---
def historical_tickers(
    storage: Storage | None = None, limit: int | None = None
) -> list[str]:
    """
    Return normalized unique tickers from the membership table.
    Used by the UI to decide which symbols to ingest.
    """

    if storage is None:
        storage = Storage()
    df = load_membership(storage, cache_salt=storage.cache_salt())
    if df is None or df.empty:
        return []
    tickers = (
        df["ticker"]
        .astype(str)
        .str.upper()
        .str.strip()
        .dropna()
        .unique()
        .tolist()
    )
    return tickers[:limit] if limit is not None else tickers


def build_membership(storage: Storage) -> str:
    df = _load_github()

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

    df = df.drop_duplicates().sort_values(["ticker", "start_date"]).reset_index(drop=True)

    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    storage.write_bytes("membership/sp500_members.parquet", buffer.getvalue())

    preview_path = Path(__file__).with_name("sp500_members_preview.csv")
    df.head(100).to_csv(preview_path, index=False)

    summary = (
        f"{len(df)} rows, {df['end_date'].isna().sum()} current (source: github, 1996→present)"
    )
    return summary

