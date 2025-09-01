from __future__ import annotations

from pathlib import Path
from typing import Union
import pandas as pd
import csv

# Determine repository root based on this file's location
REPO_ROOT = Path(__file__).resolve().parents[1]

# Commonly used paths
HISTORY_DIR = REPO_ROOT / "data" / "history"
OUTCOMES_CSV = HISTORY_DIR / "outcomes.csv"


def read_csv(path: Union[str, Path]) -> pd.DataFrame:
    """Read a CSV file into a DataFrame; return empty DataFrame if missing/failed."""
    p = Path(path)
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception:
            pass
    return pd.DataFrame()


def write_csv(path: Union[str, Path], df: pd.DataFrame) -> None:
    """Write DataFrame to CSV with minimal quoting, creating directories as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, quoting=csv.QUOTE_MINIMAL)
