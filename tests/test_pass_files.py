import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import io
import ui.history as history
from datetime import date


def test_list_pass_files(tmp_path, monkeypatch):
    root = tmp_path
    data_dir = root / "data"
    pass_logs = data_dir / "pass_logs"
    hist_dir = data_dir / "history"
    legacy_dir = root / "history"
    pass_logs.mkdir(parents=True)
    hist_dir.mkdir(parents=True)
    legacy_dir.mkdir()

    f1 = pass_logs / "pass_20230102.csv"
    f1.touch()
    f2 = hist_dir / "pass_20230101.psv"
    f2.touch()
    f3 = legacy_dir / "pass_20230103.csv"
    f3.touch()

    monkeypatch.setattr(io, "REPO_ROOT", root)
    monkeypatch.setattr(io, "DATA_DIR", data_dir)
    monkeypatch.setattr(io, "HISTORY_DIR", hist_dir)
    monkeypatch.setattr(io, "PASS_DIR", pass_logs)

    paths = io.list_pass_files()
    assert paths == sorted(paths)
    assert {p.name for p in paths} == {"pass_20230102.csv", "pass_20230101.psv", "pass_20230103.csv"}


def test_load_pass_history(tmp_path, monkeypatch):
    f1 = tmp_path / "pass_20230101.csv"
    f1.write_text("Ticker,Price\nAAA,1\n")
    f2 = tmp_path / "pass_20230102.psv"
    f2.write_text("Ticker|Price\nBBB|2\n")

    monkeypatch.setattr(history, "list_pass_files", lambda: [f1, f2])

    df = history.load_pass_history()
    assert list(df["Ticker"]) == ["AAA", "BBB"]
    assert list(df["run_date"]) == [date(2023, 1, 1), date(2023, 1, 2)]
