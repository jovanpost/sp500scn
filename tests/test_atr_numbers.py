import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

from engine.filters import _calc_atr
from engine.filters import atr_feasible


def test_calc_atr_14_sma_matches_manual():
    # Construct 20 bars with TR = 1.0 for last 14 bars
    n = 20
    highs = pd.Series(np.full(n, 2.0))
    lows = pd.Series(np.full(n, 1.0))
    close = pd.Series(np.full(n, 1.0))  # keep close flat so TR=1
    # Force TR==1 on each bar by aligning H/L to differ by 1 and close trailing
    # _calc_atr uses SMA over window
    atr14 = _calc_atr(highs, lows, close, window=14)
    # After the first 14 fully populated values, ATR should be exactly 1.0
    last = float(atr14.iloc[-1])
    assert abs(last - 1.0) < 1e-9


def test_atr_feasible_uses_passed_window_14():
    # Simple frame where ATR(D-1)=1.0 and entry_open=100, tp_frac=0.05 -> need $5
    # With window=14, budget = 14 * 1 = 14 >= 5 => feasible
    df = pd.DataFrame({
        "high": [10] * 30,
        "low": [9] * 30,
        "close": [9] * 30,
        "open": [100] * 30,
    })
    asof_idx = 28  # D-1
    assert atr_feasible(df, asof_idx, required_pct=0.05, atr_window=14) is True
    # Make it harder: require 0.20 => need $20 -> still 14 < 20 => not feasible
    assert atr_feasible(df, asof_idx, required_pct=0.20, atr_window=14) is False
