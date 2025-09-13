from __future__ import annotations

import uuid, json, io
import pandas as pd
from typing import Callable, Tuple

from data_lake.storage import Storage
from engine.signal_scan import scan_day, ScanParams

import pyarrow.parquet as pq
import pyarrow as pa


def trading_days(storage: Storage, start: str, end: str) -> pd.DatetimeIndex:
    """Derive trading days from AAPL parquet dates."""
    df = storage.read_parquet("prices/AAPL.parquet")
    if "date" not in df.columns:
        df["date"] = pd.to_datetime(df.get("index") or df.get("Date"))
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    mask = (df["date"] >= pd.to_datetime(start)) & (df["date"] <= pd.to_datetime(end))
    days = pd.DatetimeIndex(df.loc[mask, "date"].dropna().unique()).sort_values()
    return days


def run_range(
    storage: Storage,
    start: str,
    end: str,
    params: ScanParams,
    run_id: str | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
) -> Tuple[str, dict]:
    rid = run_id or pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S_") + str(uuid.uuid4())[:8]
    days = trading_days(storage, start, end)
    all_cands, all_out, totals = [], [], {"days": 0, "candidates": 0, "hits": 0, "fails": 0}
    total_days = len(days)
    for idx, D in enumerate(days, 1):
        cands, out, fails, _dbg = scan_day(storage, D, params)
        if not cands.empty:
            cands = cands.copy(); cands["D"] = D.normalize()
            all_cands.append(cands)
        if not out.empty:
            out = out.copy(); out["D"] = D.normalize()
            totals["hits"] += int(out["hit"].sum())
            all_out.append(out)
        totals["fails"] += int(fails); totals["candidates"] += int(len(cands)); totals["days"] += 1
        if progress_cb:
            progress_cb(idx, total_days)

    cand_df = pd.concat(all_cands, ignore_index=True) if all_cands else pd.DataFrame()
    out_df = pd.concat(all_out, ignore_index=True) if all_out else pd.DataFrame()
    summary = {
        "run_id": rid,
        "start": start,
        "end": end,
        **totals,
        "hit_rate": (float(totals["hits"]) / max(1, totals["candidates"]))
    }

    def _to_parquet_bytes(df: pd.DataFrame) -> bytes:
        buf = io.BytesIO()
        if not df.empty:
            pq.write_table(pa.Table.from_pandas(df), buf)
        return buf.getvalue()

    storage.write_bytes(f"runs/{rid}/candidates.parquet", _to_parquet_bytes(cand_df))
    storage.write_bytes(f"runs/{rid}/outcomes.parquet", _to_parquet_bytes(out_df))
    storage.write_bytes(f"runs/{rid}/summary.json", json.dumps(summary, indent=2).encode("utf-8"))
    return rid, summary
