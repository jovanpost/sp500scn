import sys
from pathlib import Path
import importlib
import pandas as pd
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_run_signal_scan_empty_active():
    mod = importlib.import_module("ui.pages.45_YdayVolSignal_Open")
    active = pd.DataFrame(columns=["ticker"])
    results, stats, fails, timeout = mod._run_signal_scan(
        active,
        D="2024-01-02",
        lookback=63,
        min_close_up=3.0,
        min_vol_mult=1.5,
        min_gap_next_open=0.0,
    )
    assert results.empty
    assert list(results.columns) == [
        "ticker",
        "d1_close_up_pct",
        "d1_vol_mult",
        "gap_open_pct",
        "sr_ratio",
    ]
    assert stats.universe == 0
    assert stats.final == 0
    assert fails == {"close_up": [], "vol": [], "gap": [], "sr": []}
    assert timeout is None


def test_run_signal_scan_no_results():
    mod = importlib.import_module("ui.pages.45_YdayVolSignal_Open")
    active = pd.DataFrame({"ticker": ["AAPL"]})

    fake_hist = pd.DataFrame(
        {
            "open": [100, 100],
            "close": [100, 100],
            "volume": [100, 100],
            "high": [100, 100],
            "low": [100, 100],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )

    def fake_get_daily_adjusted(t, start, end):
        return fake_hist

    with patch("data_lake.provider.get_daily_adjusted", fake_get_daily_adjusted):
        results, stats, fails, timeout = mod._run_signal_scan(
            active,
            D="2024-01-02",
            lookback=63,
            min_close_up=3.0,
            min_vol_mult=1.5,
            min_gap_next_open=0.0,
        )

    assert results.empty
    assert list(results.columns) == [
        "ticker",
        "d1_close_up_pct",
        "d1_vol_mult",
        "gap_open_pct",
        "sr_ratio",
    ]
    assert stats.universe == 1
    assert stats.loaded == 1
    assert stats.final == 0
    assert fails == {"close_up": [], "vol": [], "gap": [], "sr": []}
    assert timeout is None
