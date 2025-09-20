import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest

from engine.filters import _calc_atr, atr_feasible


def test_calc_atr_wilder_matches_manual_sequence():
    tr_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    base = 50.0
    highs = pd.Series(base + tr_values / 2.0)
    lows = pd.Series(base - tr_values / 2.0)
    close = pd.Series(np.full(len(tr_values), base))

    atr = _calc_atr(highs, lows, close, window=3, method="wilder")

    assert pd.isna(atr.iloc[0])
    assert pd.isna(atr.iloc[1])
    assert atr.iloc[2] == pytest.approx((1.0 + 2.0 + 3.0) / 3.0)
    assert atr.iloc[3] == pytest.approx((((1.0 + 2.0 + 3.0) / 3.0) * 2 + 4.0) / 3.0)
    assert atr.iloc[4] == pytest.approx((((((1.0 + 2.0 + 3.0) / 3.0) * 2 + 4.0) / 3.0) * 2 + 5.0) / 3.0)


def test_calc_atr_sma_matches_manual_constant():
    n = 20
    highs = pd.Series(np.full(n, 2.0))
    lows = pd.Series(np.full(n, 1.0))
    close = pd.Series(np.full(n, 1.5))
    atr14 = _calc_atr(highs, lows, close, window=14, method="sma")
    assert atr14.iloc[-1] == pytest.approx(1.0)


def test_atr_feasible_uses_passed_window_and_method():
    df = pd.DataFrame(
        {
            "high": [10] * 30,
            "low": [9] * 30,
            "close": [9] * 30,
            "open": [100] * 30,
        }
    )
    asof_idx = 28
    assert (
        atr_feasible(
            df,
            asof_idx,
            required_pct=0.05,
            atr_window=14,
            atr_method="wilder",
        )
        is True
    )
    assert (
        atr_feasible(
            df,
            asof_idx,
            required_pct=0.20,
            atr_window=14,
            atr_method="wilder",
        )
        is False
    )
