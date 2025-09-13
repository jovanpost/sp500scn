import datetime as dt
import streamlit as st

from data_lake.storage import Storage
from engine.signal_scan import ScanParams
from backtest.run_range import run_range, trading_days


def render_page() -> None:
    st.header("📈 Backtest Range")

    start = st.date_input("Start date", value=dt.date.today() - dt.timedelta(days=30))
    end = st.date_input("End date", value=dt.date.today())
    if isinstance(start, (list, tuple)):
        start = start[0]
    if isinstance(end, (list, tuple)):
        end = end[0]

    vol_lookback = int(st.number_input("Volume lookback", min_value=1, value=63, step=1))
    min_close_up_pct = float(st.number_input("Min close-up on D-1 (%)", value=3.0, step=0.5))
    min_vol_multiple = float(st.number_input("Min volume multiple", value=1.5, step=0.1))
    min_gap_open_pct = float(st.number_input("Min gap open (%)", value=0.0, step=0.1))
    atr_window = int(st.number_input("ATR window", min_value=1, value=21, step=1))
    horizon = int(st.number_input("Horizon (days)", min_value=1, value=30, step=1))
    sr_min_ratio = float(st.number_input("Min S:R ratio", value=2.0, step=0.1))

    if st.button("Run backtest", type="primary"):
        storage = Storage()
        params: ScanParams = {
            "min_close_up_pct": min_close_up_pct,
            "min_vol_multiple": min_vol_multiple,
            "min_gap_open_pct": min_gap_open_pct,
            "atr_window": atr_window,
            "lookback_days": vol_lookback,
            "horizon_days": horizon,
            "sr_min_ratio": sr_min_ratio,
        }
        days = trading_days(storage, str(start), str(end))
        progress = st.progress(0.0)

        def _cb(i: int, total: int) -> None:
            progress.progress(i / total if total else 0.0)

        rid, summary = run_range(storage, str(start), str(end), params, progress_cb=_cb)
        st.write(summary)

        cand_bytes = storage.read_bytes(f"runs/{rid}/candidates.parquet")
        out_bytes = storage.read_bytes(f"runs/{rid}/outcomes.parquet")
        sum_bytes = storage.read_bytes(f"runs/{rid}/summary.json")
        st.download_button("Candidates", cand_bytes, file_name="candidates.parquet")
        st.download_button("Outcomes", out_bytes, file_name="outcomes.parquet")
        st.download_button("Summary", sum_bytes, file_name="summary.json")


def page() -> None:
    render_page()
