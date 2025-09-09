from __future__ import annotations
import pandas as pd
from typing import Sequence, Dict, Any


def time_to_hit(
    df: pd.DataFrame,
    entry_ts: pd.Timestamp,
    entry_price: float,
    tps: Sequence[float],
    horizon: int = 30,
) -> Dict[str, Any]:
    """
    Evaluate when each TP (% gain) is first hit using next days' HIGHs
    starting from entry date (inclusive). Returns {tp_2_days: int|None, ...}
    """
    out: Dict[str, Any] = {}
    if entry_ts not in df.index:
        return {f"tp_{int(tp*100)}_days": None for tp in tps}
    highs = df['high'].loc[entry_ts:].iloc[:horizon]
    for tp in tps:
        target = entry_price * (1 + tp)
        hit = highs[highs >= target]
        out[f"tp_{int(tp*100)}_days"] = int((hit.index[0] - entry_ts).days) if len(hit) else None
    return out
