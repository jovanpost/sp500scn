from __future__ import annotations

import re
from pathlib import Path

import pytest

STREAMLIT_FILES = [
    Path("ui/pages/65_Stock_Scanner_SharesOnly.py"),
    Path("ui/pages/66_Spike_Precursor_Lab.py"),
]


@pytest.mark.parametrize("path", STREAMLIT_FILES)
def test_no_session_state_double_write(path: Path) -> None:
    content = path.read_text()
    pattern = re.compile(r"st\.session_state\[[^\]]+\]\s*=\s*st\.")
    assert not pattern.search(content), f"Found state double-write in {path}"
