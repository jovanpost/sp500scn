import logging
import pandas as pd
import numpy as np
import datetime as dt
import streamlit as st
from typing import Dict, Any, Tuple, List

from engine.replay import replay_trade


def _emit(msg: str):
    logging.info(msg)
    st.write(msg)


@st.cache_data(show_spinner=False)
def _load_members(_bucket_key: str) -> pd.DataFrame:
    from data_lake.storage import Storage

    s = Storage()
    raw = s.read_bytes("membership/sp500_members.parquet")
    m = pd.read_parquet(pd.io.common.BytesIO(raw))
    m["start_date"] = pd.to_datetime(
        m["start_date"], errors="coerce", utc=True
    ).dt.tz_localize(None)
    m["end_date"] = pd.to_datetime(
        m["end_date"], errors="coerce", utc=True
    ).dt.tz_localize(None)
    return m


def members_on_date(m: pd.DataFrame, date: dt.date) -> pd.DataFrame:
    D = pd.to_datetime(date)
    start = pd.to_datetime(m["start_date"])
    end = pd.to_datetime(m["end_date"])
    mask = (start <= D) & (end.isna() | (D <= end))
    return m.loc[mask]


def _load_prices(storage, ticker: str) -> pd.DataFrame | None:
    try:
        path = f"prices/{ticker}.parquet"
        raw = storage.read_bytes(path)
        df = pd.read_parquet(pd.io.common.BytesIO(raw))
        df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
        return df.sort_values("date")
    except Exception as e:
        logging.warning("missing or unreadable price file for %s: %s", ticker, e)
        return None


@st.cache_data(show_spinner=False)
def _cached_prices(ticker: str) -> pd.DataFrame | None:
    from data_lake.storage import Storage

    s = Storage()
    try:
        df = s.read_parquet(f"prices/{ticker}.parquet")
        if "date" not in df.columns:
            df["date"] = pd.to_datetime(df.get("index") or df.get("Date"))
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        return df[["date", "open", "high", "low", "close"]].sort_values("date")
    except Exception as e:
        logging.warning("price load failed for %s: %s", ticker, e)
        return None


def _compute_metrics(
    df: pd.DataFrame, D: dt.date, vol_lookback: int
) -> Dict[str, Any] | None:
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
    atr = tr.rolling(21, min_periods=1).mean()
    atr21 = float(atr.loc[dm1]) if dm1 in atr.index else np.nan
    atr21_pct = atr21 / df.loc[dm1, "close"] * 100 if df.loc[dm1, "close"] else np.nan

    if dm1 >= 21:
        ret21 = (df.loc[dm1, "close"] / df.loc[dm1 - 21, "close"] - 1.0) * 100
    else:
        ret21 = np.nan

    lo_win = df.loc[max(0, dm1 - 21) : dm1, "low"].min()
    hi_win = df.loc[max(0, dm1 - 21) : dm1, "high"].max()

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
        "atr21": float(atr21) if not np.isnan(atr21) else np.nan,
        "atr21_pct": float(atr21_pct) if not np.isnan(atr21_pct) else np.nan,
        "ret21_pct": float(ret21) if not np.isnan(ret21) else np.nan,
        "support": support,
        "resistance": resistance,
        "sr_ratio": sr_ratio,
        "tp_halfway_pct": (
            float(tp_halfway_pct) if not np.isnan(tp_halfway_pct) else np.nan
        ),
        "entry_open": entry,
    }


def _run_signal_scan(
    storage,
    tickers: List[str],
    D: dt.date,
    min_close_up_pct: float,
    vol_lookback: int,
    min_vol_multiple: float,
    min_gap_open_pct: float,
    opts: Dict[str, Any],
    dry_run_n: int | None = None,
) -> Tuple[pd.DataFrame, Dict[str, int], List[str]]:

    from collections import OrderedDict

    stats = OrderedDict(
        universe=len(tickers),
        loaded=0,
        after_close_up=0,
        after_vol=0,
        after_gap=0,
        after_optional=0,
        after_sr=0,
        final=0,
    )
    dropped_examples: List[str] = []

    if dry_run_n:
        tickers = sorted(tickers)[:dry_run_n]

    results: List[Dict[str, Any]] = []

    for t in tickers:
        df = _load_prices(storage, t)
        if df is None or len(df) < 30:
            continue
        stats["loaded"] += 1

        m = _compute_metrics(df, D, vol_lookback)
        if not m:
            continue

        if not np.isnan(m["close_up_pct"]) and m["close_up_pct"] >= min_close_up_pct:
            pass
        else:
            dropped_examples.append(f"{t} close_up {m['close_up_pct']}")
            continue
        stats["after_close_up"] += 1

        if not np.isnan(m["vol_multiple"]) and m["vol_multiple"] >= min_vol_multiple:
            pass
        else:
            dropped_examples.append(f"{t} vol_mult {m['vol_multiple']}")
            continue
        stats["after_vol"] += 1

        if m["gap_open_pct"] >= min_gap_open_pct:
            pass
        else:
            dropped_examples.append(f"{t} gap {m['gap_open_pct']}")
            continue
        stats["after_gap"] += 1

        ok = True
        if v := opts.get("min_atr_dollars"):
            ok &= not np.isnan(m["atr21"]) and (m["atr21"] >= v)
        if v := opts.get("min_atr_pct"):
            ok &= not np.isnan(m["atr21_pct"]) and (m["atr21_pct"] >= v)
        if v := opts.get("min_close_dollars"):
            ok &= m["entry_open"] >= v
        if v := opts.get("min_ret21_pct"):
            ok &= not np.isnan(m["ret21_pct"]) and (m["ret21_pct"] >= v)
        if not ok:
            dropped_examples.append(f"{t} optional filters")
            continue
        stats["after_optional"] += 1

        if opts.get("require_sr", True):
            if np.isnan(m["sr_ratio"]) or m["sr_ratio"] < opts.get("sr_ratio_min", 2.0):
                dropped_examples.append(f"{t} sr {m['sr_ratio']}")
                continue
        stats["after_sr"] += 1

        row = {"ticker": t, **m}
        results.append(row)

    res_df = pd.DataFrame(results)
    if not res_df.empty and "gap_open_pct" in res_df.columns:
        res_df = res_df.sort_values("gap_open_pct", ascending=False)
    stats["final"] = len(res_df)

    return res_df, stats, dropped_examples[:10]


def render_page():
    st.header("⚡ Yesterday Close+Volume → Buy Next Open")

    _d = st.date_input("Entry day (D)", value=dt.date.today())
    if isinstance(_d, (list, tuple)):
        _d = _d[0]
    D = pd.to_datetime(_d).date()

    vol_lookback = int(
        st.number_input("Volume lookback", min_value=1, value=63, step=1)
    )
    min_close_up_pct = float(
        st.number_input("Min close-up on D-1 (%)", value=3.0, step=0.5)
    )
    min_vol_multiple = float(
        st.number_input("Min volume multiple", value=1.5, step=0.1)
    )
    min_gap_open_pct = float(st.number_input("Min gap open (%)", value=0.0, step=0.1))
    min_close_dollars = float(
        st.number_input("Min entry price ($)", value=0.0, step=1.0)
    )

    use_atr_dollars = st.checkbox("Use ATR $ filter", value=False)
    min_atr_dollars = (
        float(st.number_input("Min ATR ($)", value=1.0, step=0.1))
        if use_atr_dollars
        else 0.0
    )

    use_atr_pct = st.checkbox("Use ATR% filter", value=False)
    min_atr_pct = (
        float(st.number_input("Min ATR (%)", value=2.0, step=0.1))
        if use_atr_pct
        else 0.0
    )

    use_ret21 = st.checkbox("Use 21-day return filter", value=False)
    min_ret21_pct = (
        float(st.number_input("Min 21-day return (%)", value=0.0, step=0.1))
        if use_ret21
        else 0.0
    )

    dry_n = st.number_input(
        "Dry-run N tickers", min_value=0, max_value=3000, value=30, step=10
    )
    show_debug = st.checkbox("Show debug", value=True)

    if st.button("Run scan", type="primary"):
        with st.status("Running filters...", expanded=True):
            from data_lake.storage import Storage

            s = Storage()
            members = _load_members("anon")
            active = members_on_date(members, D)["ticker"].dropna().unique().tolist()
            st.write(f"Loaded membership: {len(members)} rows")
            st.write(f"Universe size: {len(active)}")

            res_df, stats, drops = _run_signal_scan(
                s,
                active,
                D,
                min_close_up_pct,
                vol_lookback,
                min_vol_multiple,
                min_gap_open_pct,
                opts={
                    "min_atr_dollars": min_atr_dollars if use_atr_dollars else None,
                    "min_atr_pct": min_atr_pct if use_atr_pct else None,
                    "min_close_dollars": min_close_dollars,
                    "min_ret21_pct": min_ret21_pct if use_ret21 else None,
                    "require_sr": True,
                    "sr_ratio_min": 2.0,
                },
                dry_run_n=(dry_n or None) if dry_n > 0 else None,
            )

        st.session_state["res_df"] = res_df
        st.session_state["stats"] = stats
        st.session_state["drops"] = drops

    res_df = st.session_state.get("res_df")
    stats = st.session_state.get("stats")
    drops = st.session_state.get("drops")

    if res_df is not None:
        st.dataframe(pd.DataFrame([stats]))
        if res_df.empty:
            st.warning("No matches for the selected filters.")
            if show_debug and drops:
                st.write("First few drops:", drops)
        else:
            st.success(f"{len(res_df)} matches")
            st.dataframe(res_df.head(100))

            use_stop = st.checkbox("Use stop at support", value=True)
            horizon = int(
                st.number_input("Horizon (days)", min_value=1, value=30, step=1)
            )
            save_outcomes = st.checkbox("Save outcomes to lake", value=False)

            run_df = res_df.copy()
            limit_n = 50
            if len(run_df) > limit_n:
                st.warning(
                    f"{len(run_df)} matches; limiting to top {limit_n} by sr_ratio unless confirmed"
                )
                if not st.checkbox("Process all matches", value=False):
                    run_df = run_df.sort_values("sr_ratio", ascending=False).head(
                        limit_n
                    )

            if st.button("Run outcomes for matches"):
                from data_lake.storage import Storage

                storage = Storage()
                rows: List[Dict[str, Any]] = []
                run_df["tp_price"] = run_df["entry_open"] * (
                    1 + run_df["tp_halfway_pct"]
                )
                for r in run_df.itertuples(index=False):
                    try:
                        df = storage.read_parquet(f"prices/{r.ticker}.parquet")

                        # Ensure we have a date column
                        if "date" not in df.columns:
                            idx_col = (
                                "index"
                                if "index" in df.columns
                                else ("Date" if "Date" in df.columns else None)
                            )
                            if idx_col is not None:
                                df["date"] = df[idx_col]
                            else:
                                raise ValueError(
                                    f"{r.ticker}: could not find a date/index column in parquet"
                                )

                        # Canonicalize to tz-naive midnight
                        d = pd.to_datetime(df["date"], errors="coerce")
                        if d.dt.tz is not None:
                            d = d.dt.tz_convert(None)
                        else:
                            d = d.dt.tz_localize(None)
                        df["date"] = d.dt.normalize()

                        # Keep only needed columns
                        df = (
                            df[["date", "open", "high", "low", "close"]]
                            .dropna()
                            .sort_values("date")
                        )

                        # Build normalized entry timestamp
                        entry_ts = pd.to_datetime(D)
                        try:
                            entry_ts = entry_ts.tz_localize(None)
                        except Exception:
                            pass
                        entry_ts = entry_ts.normalize()

                        # Diagnostics for missing windows
                        first_dt = df["date"].min()
                        last_dt = df["date"].max()
                        st.write(
                            f"ᴅɪᴀɢ: {r.ticker} bars {first_dt.date() if pd.notna(first_dt) else 'NA'} → {last_dt.date() if pd.notna(last_dt) else 'NA'}"
                        )

                        tp_price = float(r.entry_open * (1 + r.tp_halfway_pct))
                        stop_price = float(r.support) if use_stop else None
                        out = replay_trade(
                            df,
                            entry_ts,
                            float(r.entry_open),
                            tp_price,
                            stop_price,
                            horizon_days=horizon,
                        )
                        out["ticker"] = r.ticker
                        rows.append(out)
                    except Exception as e:
                        rows.append(
                            {
                                "ticker": r.ticker,
                                "hit": False,
                                "exit_reason": f"error:{e}",
                                "exit_price": float("nan"),
                                "days_to_exit": 0,
                                "mae_pct": float("nan"),
                                "mfe_pct": float("nan"),
                            }
                        )

                outcomes = pd.DataFrame(rows)
                res = run_df.merge(outcomes, on="ticker", how="left")

                hits = res["hit"].fillna(False)
                st.subheader("Outcomes summary")
                st.write(
                    {
                        "hit_rate": float(hits.mean()),
                        "median_days_to_exit": (
                            float(res.loc[hits, "days_to_exit"].median())
                            if hits.any()
                            else None
                        ),
                        "avg_MAE_pct": (
                            float(res["mae_pct"].dropna().mean())
                            if res["mae_pct"].notna().any()
                            else None
                        ),
                        "avg_MFE_pct": (
                            float(res["mfe_pct"].dropna().mean())
                            if res["mfe_pct"].notna().any()
                            else None
                        ),
                    }
                )
                st.dataframe(
                    res.sort_values(["hit", "days_to_exit"], ascending=[False, True]),
                    use_container_width=True,
                )

                if save_outcomes:
                    import io

                    buf = io.BytesIO()
                    res.to_parquet(buf, index=False)
                    storage.write_bytes(
                        f"runs/{pd.to_datetime(D).date().isoformat()}/outcomes.parquet",
                        buf.getvalue(),
                    )


def page():
    render_page()
