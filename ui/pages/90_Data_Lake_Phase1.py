from __future__ import annotations

import io
from datetime import date
import time
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import streamlit as st

from data_lake.storage import Storage, supabase_available
from data_lake.membership import build_membership, load_membership
from data_lake.ingest import ingest_batch
from data_lake.schemas import IngestJob
from ui.components.debug import debug_panel, _get_dbg


def _storage_has_file(storage: Storage, path: str) -> bool:
    """Return ``True`` if ``path`` exists on ``storage``.

    Retains backward compatibility for environments where ``Storage`` may not
    implement ``exists``; falls back to a ``read_bytes`` probe.
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


def render_data_lake_tab() -> None:
    st.subheader("Data Lake (Phase 1)")
    dbg = _get_dbg("lake")
    if "auto_run" not in st.session_state:
        st.session_state.auto_run = False
    if "auto_meta" not in st.session_state:
        st.session_state.auto_meta = {}
    storage = Storage()
    diag = storage.diagnostics()
    dbg.set_env(storage_mode=getattr(storage, "mode", "unknown"), bucket=getattr(storage, "bucket", None))
    st.caption(f"storage: mode={diag['mode']} bucket={diag['bucket']}")

    ok, reason = supabase_available()
    if storage.mode == "supabase":
        host = urlparse(storage.supabase_url or "").netloc
        st.success(f"✅ Supabase mode ({host}, bucket: {storage.bucket})")
    elif storage.force_supabase:
        st.error(f"❌ Forced Supabase requested but unavailable: {reason}")
    else:
        st.info("Using local mode")

    if st.button("Clear data caches"):
        st.cache_data.clear()
        st.experimental_rerun()

    st.markdown("### Sanity check")
    if storage.exists("prices/AAPL.parquet"):
        df = pd.read_parquet(io.BytesIO(storage.read_bytes("prices/AAPL.parquet")))
        st.dataframe(df.tail(5))
        st.line_chart(df.set_index("date")["close"])
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
            df = load_membership(storage, cache_salt=storage.cache_salt())
            st.write(
                {
                    "rows": len(df),
                    "tickers": df["ticker"].nunique(),
                    "current": df["end_date"].isna().sum(),
                    "source": "github",
                }
            )
            st.dataframe(df.head(20))
        except Exception as e:  # pragma: no cover - UI
            st.exception(e)

    st.markdown("### Prices coverage")
    try:
        mdf = load_membership(storage, cache_salt=storage.cache_salt())
        scope = st.radio(
            "Scope", ["Historical (since 1996)", "Current only"], horizontal=True
        )
        if scope.startswith("Historical"):
            total = sorted(
                mdf["ticker"].astype(str).str.upper().str.strip().unique().tolist()
            )
        else:
            cur = mdf[mdf["end_date"].isna() | (mdf["end_date"] == "")] if "end_date" in mdf else mdf
            total = sorted(
                cur["ticker"].astype(str).str.upper().str.strip().unique().tolist()
            )

        files = storage.list_all("prices")
        present = {Path(p).stem.upper() for p in files if p.endswith(".parquet")}

        total_set = set(total)
        missing = sorted(total_set - present)

        st.write(
            f"Coverage: **{len(present)} / {len(total_set)}** tickers with price files"
        )
        st.caption(f"files discovered {len(files)} | coverage {len(present)} unique tickers")
        if missing:
            with st.expander("Show first 25 missing tickers"):
                st.code(", ".join(missing[:25]))

        max_run = st.number_input("max tickers per run", 1, 200, 50)

        if st.button("Ingest missing only"):
            jobs = [
                {"ticker": t, "start": "1990-01-01", "end": str(date.today())}
                for t in missing[: int(max_run)]
            ]
            progress_bar = st.progress(0)
            summary = ingest_batch(
                storage, jobs, progress_cb=lambda d, t: progress_bar.progress(d / t)
            )
            st.success(f"ok {summary['ok']}, failed {summary['failed']}")
            st.write(f"manifest: {summary['manifest_path']}")
            st.cache_data.clear()
            files = storage.list_all("prices")
            present = {Path(p).stem.upper() for p in files if p.endswith(".parquet")}
            missing = sorted(total_set - present)
            st.info(
                f"post-run coverage {len(present)} / {len(total_set)}; {len(missing)} remaining"
            )

        batch_size = st.number_input(
            "batch size", 1, 200, 50, key="auto_batch_size"
        )
        max_minutes = st.number_input(
            "max minutes", 1, 120, 15, key="auto_max_minutes"
        )
        max_iters = st.number_input(
            "max iters", 1, 1000, 50, key="auto_max_iters"
        )
        pause_seconds_between_batches = st.number_input(
            "pause seconds between batches", 0, 60, 5, key="auto_pause_seconds"
        )

        col_run, col_stop = st.columns(2)
        if col_run.button("Run until target coverage"):
            st.session_state.auto_run = True
            st.session_state.auto_meta = {"start_ts": time.time(), "iters": 0}
        if col_stop.button("Stop"):
            st.session_state.auto_run = False

        meta = st.session_state.get("auto_meta", {})
        elapsed = time.time() - meta.get("start_ts", time.time())
        elapsed_str = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
        last = meta.get("last")
        last_str = (
            f"ok {last['ok']}, failed {last['failed']}" if isinstance(last, dict) else "n/a"
        )
        st.info(
            f"present {len(present)} / {len(total_set)} | remaining {len(missing)} | "
            f"iters {meta.get('iters', 0)} | elapsed {elapsed_str} | last batch {last_str}"
        )

        if st.session_state.auto_run:
            if not missing or meta.get("iters", 0) >= int(max_iters) or elapsed >= int(max_minutes) * 60:
                st.session_state.auto_run = False
                if not missing:
                    st.success("Coverage complete")
                else:
                    st.info("Auto run stopped (limits reached)")
            else:
                chunk = missing[: int(batch_size)]
                jobs = [
                    {
                        "ticker": t,
                        "start": "1990-01-01",
                        "end": str(date.today()),
                    }
                    for t in chunk
                ]
                progress_placeholder = st.empty()
                progress_bar = progress_placeholder.progress(0)
                summary = ingest_batch(
                    storage,
                    jobs,
                    progress_cb=lambda d, t: progress_bar.progress(d / t),
                )
                progress_placeholder.empty()
                meta["iters"] = meta.get("iters", 0) + 1
                meta["last"] = summary
                st.session_state.auto_meta = meta
                st.cache_data.clear()
                files = storage.list_all("prices")
                present = {Path(p).stem.upper() for p in files if p.endswith(".parquet")}
                missing = sorted(total_set - present)
                st.info(
                    f"Batch {meta['iters']} ok {summary['ok']} failed {summary['failed']}; {len(missing)} remaining"
                )
                time.sleep(int(pause_seconds_between_batches))
                st.experimental_rerun()
    except Exception:
        st.caption("Membership parquet not available")

    st.markdown("### Ingest prices")
    start = st.date_input("start date", date(1990, 1, 1), key="start_date")
    end = st.date_input("end date", date.today(), key="end_date")
    max_tickers = st.number_input(
        "max tickers per run", 1, 1000, 25, key="max_tickers"
    )
    dry_run = st.checkbox("dry run", value=False)
    if st.button("Ingest prices (batch)"):
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
                summary = ingest_batch(
                    storage, jobs, progress_cb=lambda d, t: progress_bar.progress(d / t)
                )
                st.success(f"ok {summary['ok']}, failed {summary['failed']}")
                st.write(f"manifest: {summary['manifest_path']}")
                for res in summary["results"][:2]:
                    st.write(res)
                    if not res["error"] and storage.exists(res["path"]):
                        df = storage.read_parquet_df(res["path"])
                        st.dataframe(df.head())
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
                df = storage.read_parquet_df("prices/AAPL.parquet")
                dbg.event(
                    "sample_shape",
                    rows=len(df),
                    cols=len(df.columns),
                    min=str(df["date"].min()),
                    max=str(df["date"].max()),
                )
            st.dataframe(df.tail(5))
            st.line_chart(df.set_index("date")["close"])
        else:
            st.warning("AAPL.parquet not found in storage.")
    except Exception as e:
        dbg.error("sanity_check", e)
        st.warning(f"Sanity check failed: {e}")

    debug_panel("lake")

with st.expander("🔎 Raw Supabase list() response (temporary debug)"):
    try:
        from data_lake.storage import Storage
        s = Storage()
        api = getattr(s, "supabase_client", None).storage.from_(s.bucket)
        try:
            resp = api.list("prices")
        except TypeError:
            resp = api.list(prefix="prices")
        data = getattr(resp, "data", resp)
        st.write({"resp_type": type(resp).__name__})
        st.write("Has .data:", hasattr(resp, "data"))
        if isinstance(data, (list, tuple)):
            st.write("Count:", len(data))
            st.write("Head raw:", data[:3])
            st.write("Head names:", [ (d.get("name") if isinstance(d, dict) else getattr(d, "name", None)) for d in data[:10] ])
        elif isinstance(data, dict):
            st.json(data)
        else:
            st.write(data)
    except Exception as e:
        st.error(repr(e))

