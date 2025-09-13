from __future__ import annotations

import io
import logging
import datetime as dt
from typing import TypedDict, Tuple, List, Dict, Any

import numpy as np
import pandas as pd

from data_lake.storage import Storage
from .metrics import has_21d_runup_precedent
from .replay import replay_trade

log = logging.getLogger(__name__)


class ScanParams(TypedDict, total=False):
    min_close_up_pct: float
    min_vol_multiple: float
    min_gap_open_pct: float
    atr_window: int
    lookback_days: int
    horizon_days: int
    sr_min_ratio: float


def _load_members(storage: Storage) -> pd.DataFrame:
    raw = storage.read_bytes("membership/sp500_members.parquet")
    m = pd.read_parquet(io.BytesIO(raw))
    m["start_date"] = pd.to_datetime(m["start_date"], errors="coerce", utc=True).dt.tz_localize(None)
    m["end_date"] = pd.to_datetime(m["end_date"], errors="coerce", utc=True).dt.tz_localize(None)
    return m


def members_on_date(m: pd.DataFrame, date: dt.date) -> pd.DataFrame:
    D = pd.to_datetime(date)
    start = pd.to_datetime(m["start_date"])
    end = pd.to_datetime(m["end_date"])
    mask = (start <= D) & (end.isna() | (D <= end))
    return m.loc[mask]


def _load_prices(storage: Storage, ticker: str) -> pd.DataFrame | None:
    try:
        df = storage.read_parquet(f"prices/{ticker}.parquet")
        if "date" not in df.columns:
            df["date"] = pd.to_datetime(df.get("index") or df.get("Date"))
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        return df[["date", "open", "high", "low", "close", "volume"]].dropna().sort_values("date")
    except Exception as e:
        log.warning("price load failed for %s: %s", ticker, e)
        return None


def _compute_metrics(df: pd.DataFrame, D: dt.date, vol_lookback: int, atr_window: int) -> Dict[str, Any] | None:
    D = pd.to_datetime(D)
    if D not in df["date"].values:
        idx = df["date"].searchsorted(D)
        if idx == 0 or idx >= len(df):
            return None
        D = df["date"].iloc[idx]

    idx = df.index[df["date"] == D]
    if len(idx) == 0:
        return None
    i = idx[0]
    if i == 0:
        return None
    dm1 = i - 1

    close_up_pct = (
        (df.loc[dm1, "close"] / df.loc[dm1 - 1, "close"] - 1.0) * 100
        if dm1 > 0
        else np.nan
    )

    w0 = max(0, dm1 - vol_lookback + 1)
    lookback_vol = df.loc[w0:dm1, "volume"].mean()
    vol_multiple = (
        df.loc[dm1, "volume"] / lookback_vol
        if lookback_vol and not np.isnan(lookback_vol)
        else np.nan
    )

    gap_open_pct = (df.loc[i, "open"] / df.loc[dm1, "close"] - 1.0) * 100

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(atr_window, min_periods=1).mean()
    atr_val = float(atr.loc[dm1]) if dm1 in atr.index else np.nan
    atr_pct = atr_val / df.loc[dm1, "close"] * 100 if df.loc[dm1, "close"] else np.nan

    if dm1 >= atr_window:
        ret = (df.loc[dm1, "close"] / df.loc[dm1 - atr_window, "close"] - 1.0) * 100
    else:
        ret = np.nan

    lo_win = df.loc[max(0, dm1 - atr_window) : dm1, "low"].min()
    hi_win = df.loc[max(0, dm1 - atr_window) : dm1, "high"].max()

    entry = float(df.loc[i, "open"])
    support = float(lo_win)
    resistance = float(hi_win)
    sr_ratio = np.nan
    tp_halfway_pct = np.nan
    if support > 0 and resistance > entry:
        up = resistance - entry
        down = entry - support
        sr_ratio = up / down if down > 0 else np.nan
        tp_halfway_pct = (entry + up / 2) / entry - 1.0

    return {
        "close_up_pct": float(close_up_pct) if not np.isnan(close_up_pct) else np.nan,
        "vol_multiple": float(vol_multiple) if not np.isnan(vol_multiple) else np.nan,
        "gap_open_pct": float(gap_open_pct),
        "atr21": float(atr_val) if not np.isnan(atr_val) else np.nan,
        "atr21_pct": float(atr_pct) if not np.isnan(atr_pct) else np.nan,
        "ret21_pct": float(ret) if not np.isnan(ret) else np.nan,
        "support": support,
        "resistance": resistance,
        "sr_ratio": sr_ratio,
        "tp_halfway_pct": float(tp_halfway_pct) if not np.isnan(tp_halfway_pct) else np.nan,
        "entry_open": entry,
    }


def scan_day(storage: Storage, D: pd.Timestamp, params: ScanParams) -> Tuple[pd.DataFrame, pd.DataFrame, int, dict]:
    """
    Returns (candidates_df, outcomes_df, fail_count, debug_info).
    Must produce same columns as the UI page.
    """

    members = _load_members(storage)
    active = members_on_date(members, D.date())["ticker"].dropna().unique().tolist()

    vol_lookback = int(params.get("lookback_days", 63))
    atr_window = int(params.get("atr_window", 21))
    min_close = float(params.get("min_close_up_pct", 0.0))
    min_vol = float(params.get("min_vol_multiple", 0.0))
    min_gap = float(params.get("min_gap_open_pct", 0.0))
    horizon = int(params.get("horizon_days", 30))
    sr_min = float(params.get("sr_min_ratio", 2.0))

    cand_rows: List[Dict[str, Any]] = []
    out_rows: List[Dict[str, Any]] = []
    fail_count = 0

    stats = {"universe": len(active), "loaded": 0, "candidates": 0}

    for t in active:
        df = _load_prices(storage, t)
        if df is None or len(df) < max(vol_lookback, atr_window) + 2:
            fail_count += 1
            continue
        stats["loaded"] += 1

        m = _compute_metrics(df, D, vol_lookback, atr_window)
        if not m:
            fail_count += 1
            continue

        if (
            (not np.isnan(m["close_up_pct"]) and m["close_up_pct"] >= min_close)
            and (not np.isnan(m["vol_multiple"]) and m["vol_multiple"] >= min_vol)
            and m["gap_open_pct"] >= min_gap
            and (not np.isnan(m["sr_ratio"]) and m["sr_ratio"] >= sr_min)
        ):
            history_before_D = df[df["date"] < pd.to_datetime(D)]
            tp_halfway_pct = m.get("tp_halfway_pct")
            atr_pct = m.get("atr21_pct")
            precedent_ok = has_21d_runup_precedent(
                history_before_D, 252, atr_window, tp_halfway_pct
            ) if (tp_halfway_pct is not None and not np.isnan(tp_halfway_pct)) else False
            atr_ok = (
                (atr_pct / 100.0) * atr_window >= tp_halfway_pct
                if (atr_pct is not None and not np.isnan(atr_pct) and tp_halfway_pct and not np.isnan(tp_halfway_pct))
                else False
            )
            reasons: List[str] = []
            if not precedent_ok:
                reasons.append(
                    f"no {atr_window}d precedent ≥ {tp_halfway_pct:.2%} in 252d"
                )
            if not atr_ok:
                reasons.append("ATR×window < target")

            if precedent_ok and atr_ok:
                row = {
                    "ticker": t,
                    **m,
                    "precedent_ok": precedent_ok,
                    "atr_ok": atr_ok,
                    "reasons": "",
                }
                cand_rows.append(row)

                tp_price = row["entry_open"] * (1 + row["tp_halfway_pct"])
                stop_price = row["support"]
                out = replay_trade(
                    df[["date", "open", "high", "low", "close"]],
                    pd.to_datetime(D),
                    row["entry_open"],
                    tp_price,
                    stop_price,
                    horizon_days=horizon,
                )
                out_row = {**row, "tp_price": tp_price, **out}
                out_rows.append(out_row)
            else:
                # filtered by feasibility
                pass
        else:
            # did not meet basic filters
            pass

    cand_df = pd.DataFrame(cand_rows)
    out_df = pd.DataFrame(out_rows)
    stats["candidates"] = len(cand_df)

    return cand_df, out_df, fail_count, stats
