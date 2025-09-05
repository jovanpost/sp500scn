"""Lightweight type definitions for the data lake."""

from typing import TypedDict, Optional, Literal


class MemberRow(TypedDict):
    ticker: str
    name: Optional[str]
    start_date: str  # YYYY-MM-DD
    end_date: Optional[str]  # YYYY-MM-DD or None


class IngestJob(TypedDict):
    ticker: str
    start: str  # YYYY-MM-DD
    end: str  # YYYY-MM-DD


class IngestResult(TypedDict):
    ticker: str
    rows_written: int
    path: str
    error: Optional[str]


StorageMode = Literal["supabase", "local"]
