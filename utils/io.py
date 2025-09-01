from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

# Repository paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
HISTORY_DIR = DATA_DIR / "history"
LOGS_DIR = DATA_DIR / "logs"
PASS_DIR = DATA_DIR / "pass_logs"
OUTCOMES_PATH = HISTORY_DIR / "outcomes.csv"

PathLike = Union[str, Path]

def read_csv(path: PathLike) -> pd.DataFrame:
    """Read a CSV file into a DataFrame, returning empty DF on failure."""
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def write_csv(path: PathLike, df: pd.DataFrame) -> None:
    """Write DataFrame to CSV, creating parent dirs as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
