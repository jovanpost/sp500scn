from __future__ import annotations

import re
import runpy
import sys
import tempfile
from pathlib import Path

import pytest


PAGE_PATH = Path("ui/pages/64_Spike_Precursor_Lab.py")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.mark.skipif(not PAGE_PATH.exists(), reason="Spike Precursor Lab page missing")
def test_precursor_lab_imports_clean() -> None:
    text = PAGE_PATH.read_text(encoding="utf-8")
    assert "engine.stocks_only_scanner import run_scan" not in text
    assert "ScanSummary" not in text


@pytest.mark.parametrize("path", [PAGE_PATH])
def test_no_double_write(path: Path) -> None:
    content = path.read_text(encoding="utf-8")
    pattern = r"st\.session_state\[[^\]]+\]\s*=\s*st\."
    assert not re.search(pattern, content)


def test_precursor_lab_import_handles_export_permission(monkeypatch: pytest.MonkeyPatch) -> None:
    import pathlib

    target = pathlib.Path("data/exports")
    original_mkdir = pathlib.Path.mkdir

    def fake_mkdir(self: pathlib.Path, *args, **kwargs):
        if pathlib.Path(self) == target:
            raise PermissionError("mocked permission failure")
        return original_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "mkdir", fake_mkdir)
    if hasattr(pathlib, "PosixPath"):
        monkeypatch.setattr(pathlib.PosixPath, "mkdir", fake_mkdir, raising=False)
    if hasattr(pathlib, "WindowsPath"):
        monkeypatch.setattr(pathlib.WindowsPath, "mkdir", fake_mkdir, raising=False)

    for module in ["utils.io_export"]:
        sys.modules.pop(module, None)

    module_globals = runpy.run_path(str(PAGE_PATH), run_name="__test__")
    assert module_globals

    io_export = sys.modules.get("utils.io_export")
    assert io_export is not None
    fallback_root = Path(tempfile.gettempdir()) / "sp500scn_exports"
    assert io_export.DEFAULT_EXPORT_PATH == Path("data/exports")
    assert io_export.EXPORT_ROOT == fallback_root
    assert io_export.EXPORT_ROOT.exists()
