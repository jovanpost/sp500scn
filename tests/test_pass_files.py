import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ui.history as history
from utils import io


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


def test_load_history_df_uses_list_pass_files(monkeypatch, tmp_path):
    psv = tmp_path / "pass_1.psv"
    psv.write_text("a|b\n1|2\n")
    monkeypatch.setattr(history, "list_pass_files", lambda: [psv])
    df = history.load_history_df()
    assert df["__source_file"].tolist() == ["pass_1.psv"]


def test_latest_pass_file_uses_list_pass_files(monkeypatch, tmp_path):
    csv1 = tmp_path / "pass_20230101.csv"
    csv1.touch()
    csv2 = tmp_path / "pass_20230102.csv"
    csv2.touch()
    monkeypatch.setattr(history, "list_pass_files", lambda: [csv1, csv2])
    assert history.latest_pass_file() == csv2
