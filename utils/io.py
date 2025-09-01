from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

# Repository root and common data paths
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
HISTORY_DIR = DATA_DIR / "history"
OUTCOMES_CSV = HISTORY_DIR / "outcomes.csv"


def read_csv(path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Read a CSV file into a DataFrame.

    Returns an empty DataFrame on failure.
    """
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        return pd.DataFrame()


def write_csv(path: Union[str, Path], df: pd.DataFrame, **kwargs) -> None:
    """Write a DataFrame to a CSV file ensuring parent directory exists."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, **kwargs)
