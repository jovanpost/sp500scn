from __future__ import annotations

import io
from datetime import date
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from data_lake.storage import Storage
from data_lake.membership import build_membership, load_membership
from data_lake.ingest import ingest_batch
from data_lake.schemas import IngestJob


def _storage_has_file(storage: Storage, path: str) -> bool:
    """Return ``True`` if ``path`` exists on ``storage``.

    Some storage backends used in tests or development may not implement an
    ``exists`` method.  This helper mirrors the expected behaviour using
    ``read_bytes`` as a fallback so that the UI can safely check for files
    without raising an ``AttributeError``.
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
    if "auto_run" not in st.session_state:
        st.session_state.auto_run = False
    if "auto_meta" not in st.session_state:
        st.session_state.auto_meta = {}
    storage = Storage()
    with st.expander("Diagnostics"):
        st.caption(storage.info())
        if st.button("Run Supabase self-test"):
            st.json(storage.selftest())
    if storage.key_info.get("kind") in {"publishable", "not_jwt", "invalid_jwt"}:
        st.error(
            "Supabase key is not a valid JWT (service_role/anon). Use Legacy API Keys. Skipping remote writes."
        )
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
            df = load_membership(storage)
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
        mdf = load_membership(storage)
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
            membership_df = load_membership(storage)
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
                    if not res["error"] and _storage_has_file(storage, res["path"]):
                        df = pd.read_parquet(
                            io.BytesIO(storage.read_bytes(res["path"]))
                        )
                        st.dataframe(df.head())
        except Exception as e:  # pragma: no cover - UI
            st.exception(e)

    st.markdown("### Sanity check")
    try:
        has_aapl = getattr(storage, "exists", None) and storage.exists("prices/AAPL.parquet")
    except Exception as e:
        has_aapl = False
        st.warning(f"exists() check failed: {e}")

    cols = st.columns(2)
    with cols[0]:
        st.caption("🔎 First few objects under prices/")
        try:
            sample = storage.list_prefix("prices/")
            st.write(sample[:10] if sample else "— (none) —")
        except Exception as e:
            st.warning(f"list_prefix failed: {e}")
    with cols[1]:
        st.caption("📄 AAPL preview")
        if has_aapl:
            try:
                df = pd.read_parquet(io.BytesIO(storage.read_bytes("prices/AAPL.parquet")))
                st.dataframe(df.tail(5))
                st.line_chart(df.set_index("date")["close"])
            except Exception as e:
                st.error(f"Failed to read prices/AAPL.parquet: {e}")
        else:
            st.info("No AAPL.parquet found under prices/.")
