import sys
from pathlib import Path
import importlib
import pandas as pd

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
    assert stats.universe == 0
    assert stats.final == 0
    assert fails == {"close_up": [], "vol": [], "gap": [], "sr": []}
    assert timeout is None
