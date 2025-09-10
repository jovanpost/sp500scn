import io
import datetime as dt
import time
from dataclasses import dataclass, asdict

import pandas as pd
import streamlit as st
from data_lake.storage import Storage
from engine.universe import members_on_date


@dataclass
class ScanStats:
    universe: int = 0
    loaded: int = 0
    close_up: int = 0
    vol: int = 0
    gap: int = 0
    optional: int = 0
    sr: int = 0
    final: int = 0


@st.cache_resource
def _get_storage() -> Storage:
    return Storage()


@st.cache_data(show_spinner=False)
def _load_members(_storage) -> pd.DataFrame:
    raw = _storage.read_bytes("membership/sp500_members.parquet")
    df = pd.read_parquet(io.BytesIO(raw))
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    return df


def _run_signal_scan(
    active: pd.DataFrame,
    *,
    D,
    lookback,
    min_close_up,
    min_vol_mult,
    min_gap_next_open,
    limit=None,
    explain=False,
    use_sr=True,
    max_runtime=30,
) -> tuple[pd.DataFrame, ScanStats, dict[str, list[dict]], str | None]:
    """Run scan returning (results, stats, failures, timeout_msg)."""

    from data_lake.provider import get_daily_adjusted

    stats = ScanStats(universe=len(active))
    fails: dict[str, list[dict]] = {"close_up": [], "vol": [], "gap": [], "sr": []}

    tickers = sorted(active["ticker"].dropna().unique().tolist())
    if limit:
        tickers = tickers[:limit]
    if not tickers:
        return pd.DataFrame(), stats, fails, None

    D = pd.to_datetime(D)
    results: list[dict] = []
    start_time = time.time()
    timeout_msg = None

    for t in tickers:
        if time.time() - start_time > max_runtime:
            timeout_msg = f"Timed out while processing {t}"
            break

        back_days = max(lookback + 2, 70)
        start = (pd.Timestamp(D) - pd.Timedelta(days=back_days * 2)).date()
        hist = get_daily_adjusted(
            t,
            start=start,
            end=(pd.Timestamp(D) + pd.Timedelta(days=1)).date(),
        )
        if hist.empty or D not in hist.index:
            continue
        idx = hist.index.get_loc(D)
        d1 = hist.index[idx - 1] if idx > 0 else None
        if d1 is None:
            continue

        d1_row = hist.loc[d1]
        window = hist.loc[:d1].tail(lookback)
        if window.empty or window["volume"].mean() == 0:
            continue

        stats.loaded += 1

        close_up = (
            (d1_row["close"] - window.iloc[-2]["close"]) / window.iloc[-2]["close"] * 100.0
            if len(window) >= 2
            else 0.0
        )
        vol_mult = (
            (d1_row["volume"] / window["volume"].mean()) if window["volume"].mean() else 0.0
        )

        d_row = hist.loc[D] if D in hist.index else None
        if d_row is None:
            continue
        gap_pct = (d_row["open"] - d1_row["close"]) / d1_row["close"] * 100.0

        stage_info = {
            "ticker": t,
            "d1_close_up_pct": close_up,
            "d1_vol_mult": vol_mult,
            "gap_open_pct": gap_pct,
        }

        if close_up >= min_close_up:
            stats.close_up += 1
        else:
            if explain:
                fails["close_up"].append({"ticker": t, "value": close_up})
            continue

        if vol_mult >= min_vol_mult:
            stats.vol += 1
        else:
            if explain:
                fails["vol"].append({"ticker": t, "value": vol_mult})
            continue

        if gap_pct >= min_gap_next_open:
            stats.gap += 1
        else:
            if explain:
                fails["gap"].append({"ticker": t, "value": gap_pct})
            continue

        stats.optional += 1

        if use_sr:
            prior = hist.loc[:d1].tail(21)
            support = prior["low"].min()
            resistance = prior["high"].max()
            entry = d_row["open"]
            sr_ratio = (
                (resistance - entry) / (entry - support) if entry > support else 0.0
            )
            stage_info["sr_ratio"] = sr_ratio
            if sr_ratio >= 2.0:
                stats.sr += 1
            else:
                if explain:
                    fails["sr"].append({"ticker": t, "value": sr_ratio})
                continue
        else:
            stats.sr += 1

        results.append(stage_info)

    stats.final = len(results)
    res_df = pd.DataFrame(results).sort_values("gap_open_pct", ascending=False)
    return res_df, stats, fails, timeout_msg


def render_page():
    st.header("⚡ Yesterday Close+Volume → Buy Next Open")
    storage = _get_storage()

    _d = st.date_input("Entry day (D)", value=dt.date.today())
    if isinstance(_d, (list, tuple)):
        _d = _d[0]
    D = pd.Timestamp(_d)
    lookback = int(st.number_input("Lookback", value=63, min_value=1, step=1))
    min_close_up = float(
        st.number_input("Min close-up on D-1 (%)", value=3.0, step=0.5)
    )
    min_vol_mult = float(
        st.number_input("Min volume multiple", value=1.5, step=0.1)
    )
    min_gap = float(st.number_input("Min gap open (%)", value=0.0, step=0.1))
    dry_limit = int(st.number_input("Dry-run N tickers", value=30, min_value=1, step=1))
    run_full = st.checkbox("Run full scan", value=False)
    explain_zero = st.checkbox("Explain zero", value=False)
    use_sr = st.checkbox("Use S/R 2:1 filter", value=True)

    if st.button("Run scan"):
        with st.status("Scanning...", expanded=True) as status:
            status.update(label="Loading membership...")
            members = _load_members(storage)
            status.write(f"Loaded membership: {len(members)} rows")

            status.update(label="Resolving active tickers...")
            active = members_on_date(members, D)
            status.write(f"Universe size: {len(active)}")

            status.update(label="Storage check...")
            for t in ["AAPL", "MSFT", "XOM"]:
                try:
                    storage.read_bytes(f"prices/{t}.parquet")
                    status.write(f"{t} ✓")
                except Exception:
                    status.write(f"{t} ✗")

            if active.empty:
                status.update(label="No active members")
                st.warning("No active S&P members on selected date.")
                return

            limit = None if run_full else dry_limit
            status.update(label="Running filters...")
            results, stats, fails, timeout = _run_signal_scan(
                active,
                D=D,
                lookback=lookback,
                min_close_up=min_close_up,
                min_vol_mult=min_vol_mult,
                min_gap_next_open=min_gap,
                limit=limit,
                explain=explain_zero,
                use_sr=use_sr,
            )
            if timeout:
                status.write(timeout)
            status.update(label="Scan complete")

        counts_df = pd.DataFrame([asdict(stats)])[
            ["universe", "loaded", "close_up", "vol", "gap", "optional", "sr", "final"]
        ]
        st.table(counts_df)

        if results.empty:
            st.warning("No matches for the selected filters.")
            if explain_zero:
                for stage, items in fails.items():
                    if items:
                        st.write(stage)
                        st.dataframe(pd.DataFrame(items).head(5))
            return

        st.success(f"{len(results)} matches")
        st.dataframe(results, use_container_width=True)


def page():
    render_page()
