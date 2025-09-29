from __future__ import annotations

import re
from pathlib import Path

import pytest


PAGE_PATH = Path("ui/pages/64_Spike_Precursor_Lab.py")


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
