import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ui.history as history
from utils import outcomes


def test_load_outcomes_wraps_read_outcomes(monkeypatch):
    sentinel = pd.DataFrame({"a": [1]})
    monkeypatch.setattr(history, "read_outcomes", lambda: sentinel)
    assert history.load_outcomes() is sentinel


def test_load_outcomes_missing_file(tmp_path, monkeypatch):
    dummy = tmp_path / "missing.csv"

    def fake_read_outcomes():
        return outcomes.read_outcomes(dummy)

    monkeypatch.setattr(history, "read_outcomes", fake_read_outcomes)
    df = history.load_outcomes()
    assert df.empty
    assert list(df.columns) == outcomes.OUTCOLS
