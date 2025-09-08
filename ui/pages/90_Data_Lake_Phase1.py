from __future__ import annotations

import io
from datetime import date

import pandas as pd
import streamlit as st

from data_lake.storage import Storage
from data_lake.membership import (
    build_membership,
    load_membership,
    historical_tickers,
)
from data_lake.ingest import ingest_batch
from data_lake.schemas import IngestJob
from data_lake.provider import ingest_batch as provider_ingest_batch


def render_data_lake_tab() -> None:
    st.subheader("Data Lake (Phase 1)")
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
        membership_df = load_membership(storage)
        scope = st.selectbox(
            "Scope", ["Historical (since 1996)", "Current only"]
        )
        if scope.startswith("Historical"):
            tickers = historical_tickers(storage)
        else:
            tickers = sorted(
                membership_df[membership_df["end_date"].isna()]["ticker"].unique()
            )
        present_files = storage.list_prefix("prices/")
        present = {
            f.split("/")[-1].split(".")[0].upper() for f in present_files
        }
        total = set(tickers)
        missing = sorted(total - present)
        st.write(f"{len(total) - len(missing)} / {len(total)} present")
        if missing:
            with st.expander("Missing (first 25)"):
                st.write(missing[:25])
            max_per_run = st.session_state.get("max_tickers", 25)
            start = st.session_state.get("start_date", date(1990, 1, 1))
            end = st.session_state.get("end_date", date.today())
            if st.button("Ingest missing only"):
                try:
                    progress_bar = st.progress(0)
                    ok, fail = provider_ingest_batch(
                        storage, missing, start, end, max_per_run
                    )
                    progress_bar.progress(100)
                    st.success(f"ok {len(ok)}, failed {len(fail)}")
                    if fail:
                        st.write("failed:", fail[:10])
                except Exception as e:
                    st.exception(e)
        else:
            st.caption("All tickers present")
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
                    if not res["error"] and storage.exists(res["path"]):
                        df = pd.read_parquet(
                            io.BytesIO(storage.read_bytes(res["path"]))
                        )
                        st.dataframe(df.head())
        except Exception as e:  # pragma: no cover - UI
            st.exception(e)

    st.markdown("### Sanity check")
    if storage.exists("prices/AAPL.parquet"):
        df = pd.read_parquet(io.BytesIO(storage.read_bytes("prices/AAPL.parquet")))
        st.dataframe(df.tail(5))
        st.line_chart(df.set_index("date")["close"])
    else:
        st.caption("No AAPL price file found")
