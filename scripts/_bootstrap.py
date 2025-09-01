from __future__ import annotations

"""Utilities for bootstrapping script imports.

This module exposes :func:`add_repo_root` which ensures that the
repository root directory is on ``sys.path``. It is safe to call multiple
times: if the path is already present, the function does nothing.
"""

from pathlib import Path
import sys


def add_repo_root() -> None:
    """Insert the repository root into ``sys.path`` if it's missing."""
    root = Path(__file__).resolve().parents[1]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
