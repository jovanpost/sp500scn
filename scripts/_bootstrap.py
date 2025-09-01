from __future__ import annotations

import sys
from pathlib import Path


def add_repo_root() -> None:
    """Insert the repository root into ``sys.path``.

    The path is added to the front of ``sys.path`` if it is not already present.
    This allows scripts to import project modules when executed directly.
    """
    root = Path(__file__).resolve().parents[1]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

