from __future__ import annotations

import re
from pathlib import Path

import pytest

# Only include page modules that currently exist so the test suite remains robust
STREAMLIT_FILES = [
    p for p in (
        Path("ui/pages/65_Stock_Scanner_SharesOnly.py"),
        Path("ui/pages/64_Spike_Precursor_Lab.py"),
    ) if p.exists()
]

@pytest.mark.parametrize("path", STREAMLIT_FILES)
def test_no_session_state_double_write(path: Path) -> None:
    content = path.read_text(encoding="utf-8")
    pattern = re.compile(r"st\.session_state\[[^\]]+\]\s*=\s*st\.")
    assert not pattern.search(content), f"Found state double-write in {path}"
