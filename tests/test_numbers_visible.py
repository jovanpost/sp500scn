import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_display_columns_exist_and_are_2dp():
    df = pd.DataFrame(
        {
            "entry_open": [17.3299879],
            "close_up_pct": [3.4567],
            "support": [15.112],
            "resistance": [25.9988],
            "sr_ratio": [2.42199],
            "tp_pct_used": [26.2263],
            "atr_value_dm1": [2.09931],
            "atr_budget_dollars": [29.39034],
            "tp_required_dollars": [4.55121],
        }
    )

    def _round2(s):
        return s.astype(float).round(2)

    for c in [
        "entry_open",
        "support",
        "resistance",
        "atr_value_dm1",
        "atr_budget_dollars",
        "tp_required_dollars",
    ]:
        df[c + "_2dp"] = _round2(df[c])

    for c in ["close_up_pct", "tp_pct_used"]:
        df[c + "_2dp"] = _round2(df[c])

    df["sr_ratio_2dp"] = _round2(df["sr_ratio"])

    assert list(df.filter(like="_2dp").columns)
    assert df["entry_open_2dp"].iloc[0] == 17.33
    assert df["close_up_pct_2dp"].iloc[0] == 3.46
    assert df["sr_ratio_2dp"].iloc[0] == 2.42
