from __future__ import annotations

import io
import time
from datetime import date
from urllib.parse import urlparse

import pandas as pd
import requests  # used for Yahoo connectivity probe
import streamlit as st

try:  # pragma: no cover - optional dependency
    import yfinance as yf
except ModuleNotFoundError:  # pragma: no cover - handled in UI when invoked
    yf = None  # type: ignore[assignment]

from data_lake.storage import Storage, supabase_available
from data_lake.membership import build_membership, load_membership

# --- Robustly guard *all* ingest imports ---------------------------------------
# If data_lake.ingest has not shipped yet or has a load error, the page should still render.

try:
    from data_lake.ingest import ingest_batch as _ingest_batch  # type: ignore
except Exception as _e:
    _ingest_batch = None  # type: ignore
    _INGEST_IMPORT_ERROR = _e  # for optional diagnostics

try:
    from data_lake.ingest import lake_file_is_raw as _lake_file_is_raw  # type: ignore
except Exception:
    _lake_file_is_raw = None  # type: ignore

try:
    from data_lake.ingest import ingest_raw_yahoo_batch as _ingest_raw_yahoo_batch  # type: ignore
except Exception:
    _ingest_raw_yahoo_batch = None  # type: ignore
# -------------------------------------------------------------------------------

from data_lake.schemas import IngestJob
from ui.components.debug import debug_panel, _get_dbg


def _storage_has_file(storage: Storage, path: str) -> bool:
    """Return True if `path` exists on `storage`.

    Retains backward compatibility for environments where `Storage` may not
    implement `exists`; falls back to a `read_bytes` probe.
    """
    exists_fn = getattr(storage, "exists", None)
    if callable(exists_fn):
        try:
            return bool(exists_fn(path))
        except Exception:
            return False
    try:
        storage.read_bytes(path)
    except Exception:
        return False
    return True


def _ui_looks_raw_prices(df: pd.DataFrame) -> bool:
    """Local RAW detector used only if lake_file_is_raw is unavailable.

    RAW if, on any action day (div/split), Close != Adj Close.
    No action days => accept as RAW to avoid endless rebuilds.
    """
    needed = {"Close", "Adj Close", "Dividends", "Stock Splits"}
    if not needed.issubset(set(df.columns)):
        return False
    d = df.copy()
    d["Dividends"] = pd.to_numeric(d["Dividends"], errors="coerce").fillna(0)
    d["Stock Splits"] = pd.to_numeric(d["Stock Splits"], errors="coerce").fillna(0)
    actions = (d["Dividends"] != 0) | (d["Stock Splits"] != 0)
    if not actions.any():
        return True
    sub = d.loc[actions, ["Close", "Adj Close"]].dropna()
    if sub.empty:
        return True
    # avoid numpy dep: use pandas vector ops
    diff_ok = (sub["Close"] - sub["Adj Close"]).abs() <= 1e-6
    return not bool(diff_ok.all())


def _probe_is_raw(storage: Storage, ticker: str) -> bool:
    """Use ingest.lake_file_is_raw if available; else local fallback."""
    if callable(_lake_file_is_raw):
        try:
            return bool(_lake_file_is_raw(storage, ticker))
        except Exception:
            return False
    # Fallback: read a subset of columns and apply the local heuristic
    path = f"prices/{ticker.upper()}.parquet"
    if not _storage_has_file(storage, path):
        return False
    try:
        try:
            df = storage.read_parquet_df(
                path, columns=["date", "Close", "Adj Close", "Dividends", "Stock Splits"]
            )
        except TypeError:
            df = storage.read_parquet_df(path)
        return _ui_looks_raw_prices(df)
    except Exception:
        return False


def render_data_lake_tab() -> None:
    st.subheader("Data Lake (Phase 1)")
    dbg = _get_dbg("lake")

    storage = Storage()
    diag = storage.diagnostics()
    dbg.set_env(storage_mode=getattr(storage, "mode", "unknown"),
                bucket=getattr(storage, "bucket", None))
    st.caption(f"storage: mode={diag['mode']} bucket={diag['bucket']}")

    ok, reason = supabase_available()
    if storage.mode == "supabase":
        host = urlparse(storage.supabase_url or "").netloc
        st.success(f"✅ Supabase mode ({host}, bucket: {storage.bucket})")
    elif getattr(storage, "force_supabase", False):
        st.error(f"❌ Forced Supabase requested but unavailable: {reason}")
    else:
        st.info("Using local mode")

    if st.button("Clear data caches"):
        st.cache_data.clear()
        st.experimental_rerun()

    st.markdown("### Sanity check")
    sample_path = "prices/AAPL.parquet"
    if storage.exists(sample_path):
        try:
            df = storage.read_parquet_df(sample_path)
        except Exception as exc:  # pragma: no cover - UI feedback
            st.error(f"Failed to read {sample_path}: {exc}")
        else:
            if "date" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index().rename(columns={df.index.name or "index": "date"})
            st.dataframe(df.tail(5), use_container_width=True)
            date_col = "date" if "date" in df.columns else None
            if date_col is not None:
                date_series = pd.to_datetime(df[date_col], errors="coerce")
                try:
                    date_series = date_series.dt.tz_localize(None)
                except Exception:
                    pass
                df = df.assign(_date=date_series)
                df = df.dropna(subset=["_date"])
                if not df.empty:
                    df = df.set_index("_date")
                    value_col = None
                    for candidate in ["close", "Close", "adjclose", "Adj Close"]:
                        if candidate in df.columns:
                            value_col = candidate
                            break
                    if value_col:
                        st.line_chart(df[[value_col]].rename(columns={value_col: "Close"}))
    else:
        st.warning("AAPL.parquet not found in storage.")

    with st.expander("Diagnostics"):
        st.caption(storage.info())
        if st.button("Run Supabase self-test"):
            st.json(storage.selftest())

    if storage.key_info.get("kind") in {"publishable", "not_jwt", "invalid_jwt"}:
        st.error(
            "Supabase key is not a valid JWT (service_role/anon). Use Legacy API Keys. Skipping remote writes."
        )
        debug_panel("lake")
        # ---- RAW SUPABASE LIST DEBUG (kept available even if key invalid) ----
        with st.expander("🔎 Raw Supabase list() response (temporary debug)"):
            try:
                s = storage
                client = getattr(s, "supabase_client", None)
                bucket_name = getattr(s, "bucket", "lake")
                st.write({
                    "storage_mode": getattr(s, "mode", None),
                    "bucket_name": bucket_name,
                })

                if client:
                    api = client.storage.from_(bucket_name)
                    try:
                        resp = api.list("prices")
                    except TypeError:
                        resp = api.list(prefix="prices")
                    data = getattr(resp, "data", resp)
                    st.write({"resp_type": type(resp).__name__, "count": len(data or []) if isinstance(data, (list, tuple)) else None})
                    if isinstance(data, (list, tuple)):
                        head = list(data)[:10]
                        st.write("Head raw:", head)
                        st.write("Head names:", [
                            (d.get("name") if isinstance(d, dict) else getattr(d, "name", None))
                            for d in head
                        ])
                    elif isinstance(data, dict):
                        st.json(data)
                    else:
                        st.write(data)

                    helper_list = s.list_prefix("prices")
                    st.write({"list_prefix_len": len(helper_list), "list_prefix_sample": helper_list[:10]})
                else:
                    st.info("No Supabase client (likely local mode).")
            except Exception as e:
                st.error(repr(e))
        # ----------------------------------------------------------------------
        return

    if storage.mode == "local":
        st.caption("Using local .lake/ fallback")

    if st.button("Build membership parquet"):
        try:
            progress = st.progress(0)
            with st.spinner("Building membership..."):
                summary = build_membership(storage)
                progress.progress(100)
            st.success(summary)
            df_mem = load_membership(storage, cache_salt=storage.cache_salt())
            st.write(
                {
                    "rows": len(df_mem),
                    "tickers": df_mem["ticker"].nunique(),
                    "current": df_mem["end_date"].isna().sum(),
                    "source": "github",
                }
            )
            st.dataframe(df_mem.head(20), use_container_width=True)
        except Exception as e:  # pragma: no cover - UI
            st.exception(e)

    st.markdown("### Prices coverage")
    try:
        membership_df = load_membership(storage, cache_salt=storage.cache_salt())
        scope = st.radio("Scope", ["Historical (since 1996)", "Current only"], horizontal=True)
        if scope.startswith("Historical"):
            tickers = sorted(
                membership_df["ticker"].astype(str).str.upper().str.strip().unique().tolist()
            )
        else:
            current = (
                membership_df[
                    membership_df["end_date"].isna() | (membership_df["end_date"] == "")
                ]
                if "end_date" in membership_df
                else membership_df
            )
            tickers = sorted(
                current["ticker"].astype(str).str.upper().str.strip().unique().tolist()
            )

        target = st.radio(
            "Target dataset",
            options=["Files present", "RAW (unadjusted)"],
            index=1,
            horizontal=True,
        )

        present: list[tuple[str, bool]] = []
        needs_rebuild: list[tuple[str, bool]] = []

        with st.spinner("Scanning lake…"):
            for ticker in tickers:
                has_file = storage.exists(f"prices/{ticker.upper()}.parquet")
                if not has_file:
                    present.append((ticker, False))
                    needs_rebuild.append((ticker, False))
                    continue

                is_raw = _probe_is_raw(storage, ticker)
                present.append((ticker, True))
                needs_rebuild.append((ticker, not is_raw))

        if target == "Files present":
            covered = sum(1 for _, ok in present if ok)
            missing = [ticker for ticker, ok in present if not ok]
        else:
            covered = sum(
                1
                for (_, has_file), (_, rebuild) in zip(present, needs_rebuild)
                if has_file and not rebuild
            )
            missing = [
                ticker
                for (ticker, has_file), (_, rebuild) in zip(present, needs_rebuild)
                if (not has_file) or rebuild
            ]

        st.markdown(
            f"Coverage: **{covered} / {len(tickers)}** tickers with "
            f"{'files' if target == 'Files present' else 'RAW (unadjusted)'}"
        )

        with st.expander("Show first 25 missing tickers"):
            st.write(missing[:25])

        max_run = st.number_input("max tickers per run", 1, 1000, 50, key="cov_max_run")
        run_disabled = (
            (target == "Files present" and not callable(_ingest_batch)) or
            (target == "RAW (unadjusted)" and not (callable(_ingest_raw_yahoo_batch) or callable(_ingest_batch)))
        )

        if run_disabled:
            if target == "RAW (unadjusted)" and not callable(_ingest_raw_yahoo_batch):
                st.info("Yahoo RAW batch helper not available in this build; falling back to Supabase ingest if present.")
            if target == "Files present" and not callable(_ingest_batch):
                st.info("Supabase ingest is not available in this build.")

        if st.button("Run until target coverage", use_container_width=True, type="primary", disabled=run_disabled):
            jobs = [
                {"ticker": t, "start": "1990-01-01", "end": None}
                for t in missing[: int(max_run)]
            ]
            if not jobs:
                st.success("Already at target coverage.")
            else:
                progress_bar = st.progress(0.0)
                # Prefer Yahoo RAW when targeting RAW and helper exists
                if target == "RAW (unadjusted)" and callable(_ingest_raw_yahoo_batch):
                    summary = _ingest_raw_yahoo_batch(
                        storage,
                        jobs,
                        progress_cb=lambda d, t: progress_bar.progress(d / t),
                    )
                else:
                    # Fall back to Supabase path if available
                    if not callable(_ingest_batch):
                        st.error("No available ingester for this target in this build.")
                        st.stop()
                    summary = _ingest_batch(
                        storage,
                        jobs,
                        progress_cb=lambda d, t: progress_bar.progress(d / t),
                    )
                st.success(f"ok {summary['ok']}, failed {summary['failed']}")
                st.write(f"manifest: {summary['manifest_path']}")
                st.cache_data.clear()
                st.experimental_rerun()
    except Exception:
        st.caption("Membership parquet not available")

    # --- RAW OHLC (unadjusted) one-click writer ----------------------------------
    st.markdown("### RAW OHLC (unadjusted) — one-click writer")
    with st.expander(
        "🧽 Write RAW Yahoo OHLC to lake (unadjusted, with actions)",
        expanded=False,
    ):
        tickers_str = st.text_input(
            "Tickers (comma/space separated)",
            value="NVDA WMT ALB",
            help=(
                "Example: NVDA, WMT, ALB  · Dot tickers (BRK.B) are handled automatically."
            ),
        ).strip()

        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            start_date = st.date_input(
                "Start date", value=pd.Timestamp("1990-01-01").date(), key="raw_start"
            )
        with c2:
            use_end = st.checkbox(
                "Set explicit end date", value=False, key="raw_use_end"
            )
            end_date = (
                st.date_input("End date", value=pd.Timestamp.today().date(), key="raw_end")
                if use_end
                else None
            )
        with c3:
            pause_s = st.number_input(
                "Pause (sec) between tickers",
                min_value=0.0,
                max_value=5.0,
                value=0.25,
                step=0.05,
            )

        # Connectivity probe (Yahoo)
        def _yahoo_ok() -> bool:
            try:
                r = requests.get(
                    "https://query1.finance.yahoo.com/v7/finance/quote?symbols=AAPL",
                    timeout=6,
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                return r.status_code == 200 and "quoteResponse" in r.text
            except Exception:
                return False

        yahoo_reachable = _yahoo_ok()
        if not yahoo_reachable:
            st.warning(
                "Yahoo is not reachable from this runtime. Use **GitHub → Actions → "
                "migrate-and-verify-sample → Run workflow** to ingest RAW OHLC."
            )

        run_btn = st.button(
            "Write RAW to lake now (this runtime)",
            type="primary",
            use_container_width=True,
            disabled=not yahoo_reachable,
        )

        SCHEMA = [
            "date",
            "Ticker",
            "Open",
            "High",
            "Low",
            "Close",
            "Adj Close",
            "Volume",
            "Dividends",
            "Stock Splits",
        ]

        def _yahoo_symbol(tkr: str) -> str:
            """Return the Yahoo Finance symbol for ``tkr``."""
            return tkr.replace(".", "-").upper()

        def _download_raw_yahoo(tkr: str, start: str | None, end: str | None) -> pd.DataFrame:
            if yf is None:  # pragma: no cover - guarded before invocation
                raise RuntimeError("yfinance is required for RAW OHLC ingest")

            y = yf.download(
                _yahoo_symbol(tkr),
                start=start,
                end=end,
                auto_adjust=False,
                actions=True,
                progress=False,
                threads=False,
            )
            if y.empty:
                return pd.DataFrame(columns=SCHEMA)

            # Some yfinance versions return MultiIndex columns when actions=True
            if isinstance(y.columns, pd.MultiIndex):
                y = y.droplevel(1, axis=1)

            y = y.reset_index().rename(columns={"Date": "date"})
            out = pd.DataFrame(index=range(len(y)))
            out["date"] = pd.to_datetime(y["date"]).dt.tz_localize(None)
            out["Ticker"] = tkr.upper()
            out["Open"] = y.get("Open", pd.NA)
            out["High"] = y.get("High", pd.NA)
            out["Low"] = y.get("Low", pd.NA)
            out["Close"] = y.get("Close", pd.NA)
            out["Adj Close"] = y.get("Adj Close", pd.NA)
            out["Volume"] = y.get("Volume", pd.NA)
            out["Dividends"] = y.get("Dividends", 0.0 if "Dividends" in y else 0.0)
            out["Stock Splits"] = y.get("Stock Splits", 0.0 if "Stock Splits" in y else 0.0)
            return out[SCHEMA].sort_values("date")

        if run_btn:
            if yf is None:
                st.error("yfinance is required for RAW OHLC ingest. Please install yfinance.")
            else:
                try:
                    storage = Storage.from_env()  # reads SUPABASE creds from st.secrets
                except Exception as e:  # pragma: no cover - UI feedback
                    st.error(f"Storage not configured or unavailable: {e}")
                    st.stop()
                raw_tickers = [t for t in tickers_str.replace(",", " ").upper().split() if t]
                if not raw_tickers:
                    st.error("No tickers provided.")
                else:
                    start_s = str(pd.Timestamp(start_date).date()) if start_date else None
                    end_s = str(pd.Timestamp(end_date).date()) if end_date else None
                    ok, fail = 0, 0
                    try:
                        prog = st.progress(0.0, text="Starting…")
                        progress_supports_text = True
                    except TypeError:
                        prog = st.progress(0.0)
                        progress_supports_text = False
                    log = st.empty()

                    for i, tkr in enumerate(raw_tickers, start=1):
                        try:
                            df = _download_raw_yahoo(tkr, start_s, end_s)
                            if df.empty:
                                log.warning(f"{tkr}: Yahoo returned 0 rows for range [{start_s}…{end_s}]")
                                fail += 1
                            else:
                                buf = io.BytesIO()
                                dest = f"prices/{tkr}.parquet"
                                ts = pd.Timestamp.utcnow().strftime("%Y%m%d%H%M%S")
                                if _storage_has_file(storage, dest):
                                    try:
                                        prev = storage.read_bytes(dest)
                                        storage.write_bytes(f"backups/prices/{tkr}.{ts}.parquet", prev)
                                    except Exception as e:  # pragma: no cover
                                        log.warning(f"{tkr}: could not backup existing file: {e}")
                                df.to_parquet(buf, index=False, compression="snappy")
                                storage.write_bytes(dest, buf.getvalue())
                                log.info(
                                    f"{tkr}: wrote {len(df)} rows "
                                    f"[{df['date'].min().date()} → {df['date'].max().date()}]"
                                )
                                ok += 1
                        except Exception as e:  # pragma: no cover - UI feedback
                            log.error(f"{tkr}: ERROR {e}")
                            fail += 1
                        progress_value = i / max(len(raw_tickers), 1)
                        if progress_supports_text:
                            try:
                                prog.progress(progress_value, text=f"{i}/{len(raw_tickers)} processed…")
                            except TypeError:
                                progress_supports_text = False
                                prog.progress(progress_value)
                        else:
                            prog.progress(progress_value)
                        time.sleep(float(pause_s))

                    if fail == 0:
                        st.success(f"Done. ok={ok}, failed={fail}")
                    else:
                        st.warning(f"Done with issues. ok={ok}, failed={fail}. See log above.")
                    if ok > 0:
                        st.cache_data.clear()
                        st.experimental_rerun()
    # -------------------------------------------------------------------------------

    st.markdown("### Ingest prices")
    start = st.date_input("start date", date(1990, 1, 1), key="start_date")
    end = st.date_input("end date", date.today(), key="end_date")
    max_tickers = st.number_input("max tickers per run", 1, 1000, 25, key="max_tickers")
    dry_run = st.checkbox("dry run", value=False)
    ingest_disabled = not callable(_ingest_batch)
    if ingest_disabled:
        st.info("Supabase ingest is not available in this build.")

    if st.button("Ingest prices (batch)", disabled=ingest_disabled):
        try:
            membership_df = load_membership(storage, cache_salt=storage.cache_salt())
            tickers = list(membership_df["ticker"].unique())[: int(max_tickers)]
            jobs: list[IngestJob] = [
                {"ticker": t, "start": str(start), "end": str(end)} for t in tickers
            ]
            if dry_run:
                st.write(f"Would ingest {len(jobs)} tickers")
            else:
                progress_bar = st.progress(0)
                summary = _ingest_batch(  # type: ignore[operator]
                    storage, jobs, progress_cb=lambda d, t: progress_bar.progress(d / t)
                )
                st.success(f"ok {summary['ok']}, failed {summary['failed']}")
                st.write(f"manifest: {summary['manifest_path']}")
                for res in summary["results"][:2]:
                    st.write(res)
                    if not res["error"] and _storage_has_file(storage, res["path"]):
                        df = storage.read_parquet_df(res["path"])
                        st.dataframe(df.head(), use_container_width=True)
        except Exception as e:  # pragma: no cover - UI
            st.exception(e)

    st.markdown("### Sanity check")
    try:
        with dbg.step("lake_exists_probe"):
            exists = False
            if callable(getattr(storage, "exists", None)):
                exists = storage.exists("prices/AAPL.parquet")
            else:
                _ = storage.read_bytes("prices/AAPL.parquet")
                exists = True
            dbg.event("exists:AAPL", exists=exists)
        if exists:
            with dbg.step("load_sample_AAPL"):
                df2 = storage.read_parquet_df("prices/AAPL.parquet")
                dbg.event(
                    "sample_shape",
                    rows=len(df2),
                    cols=len(df2.columns),
                    min=str(df2["date"].min()),
                    max=str(df2["date"].max()),
                )
            st.dataframe(df2.tail(5), use_container_width=True)
            try:
                st.line_chart(df2.set_index("date")["close"])
            except Exception:
                if "Close" in df2.columns:
                    st.line_chart(df2.set_index("date")["Close"])
        else:
            st.warning("AAPL.parquet not found in storage.")
    except Exception as e:
        dbg.error("sanity_check", e)
        st.warning(f"Sanity check failed: {e}")

    debug_panel("lake")

    # --- TEMP DEBUG: raw Supabase list() response (inside the function) -------
    with st.expander("🔎 Raw Supabase list() response (temporary debug)"):
        try:
            s = storage
            client = getattr(s, "supabase_client", None)
            bucket_name = getattr(s, "bucket", "lake")
            st.write({"storage_mode": getattr(s, "mode", None),
                      "bucket_name": bucket_name})

            if client:
                api = client.storage.from_(bucket_name)
                # Different SDKs accept either list("prefix") or list(prefix="prefix")
                try:
                    resp = api.list("prices")
                except TypeError:
                    resp = api.list(prefix="prices")

                data = getattr(resp, "data", resp)
                st.write({"resp_type": type(resp).__name__})
                st.write("Has .data:", hasattr(resp, "data"))
                if isinstance(data, (list, tuple)):
                    st.write("Count:", len(data))
                    head = data[:10]
                    st.write("Head raw:", head)
                    st.write("Head names:", [
                        (d.get("name") if isinstance(d, dict) else getattr(d, "name", None))
                        for d in head
                    ])
                elif isinstance(data, dict):
                    st.json(data)
                else:
                    st.write(data)

                # Also try helper if present
                try:
                    if hasattr(s, "list_prefix"):
                        lp = s.list_prefix("prices")
                        st.write({"list_prefix_len": len(lp), "list_prefix_sample": lp[:10]})
                except Exception as e2:
                    st.error(f"list_prefix() raised: {repr(e2)}")
            else:
                st.info("No Supabase client (likely local mode).")
        except Exception as e:
            st.error(repr(e))
    # -------------------------------------------------------------------------


