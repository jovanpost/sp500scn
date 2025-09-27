from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

EXPORT_ROOT = Path("data/exports")
EXPORT_ROOT.mkdir(parents=True, exist_ok=True)


def _resolve_name(base_name: str, suffix: str) -> Path:
    safe = base_name.replace("/", "_").replace("\\", "_")
    return EXPORT_ROOT / f"{safe}_{suffix}.csv"


def export_trades(trades_df: pd.DataFrame, base_name: str) -> str:
    if trades_df is None or trades_df.empty:
        path = _resolve_name(base_name, "trades")
        trades_df = pd.DataFrame(columns=["ticker", "entry_date", "exit_date", "pnl"])
    else:
        path = _resolve_name(base_name, "trades")
    trades_df.to_csv(path, index=False)
    return str(path)


def export_diagnostics(diag_df: pd.DataFrame, base_name: str) -> str:
    if diag_df is None:
        diag_df = pd.DataFrame()
    path = _resolve_name(base_name, "diagnostics")
    diag_df.to_csv(path, index=False)
    return str(path)


__all__ = ["export_trades", "export_diagnostics"]
