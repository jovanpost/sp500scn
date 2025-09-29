# ui/nav.py
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Callable, Optional, Sequence

RenderFn = Callable[[], None]

ROOT = Path(__file__).resolve().parent

def _load_render_fn(module_name: str, file_path: str, attrs: Sequence[str]) -> Optional[RenderFn]:
    """
    Safely load a render function from a module on disk.
    - Works with files whose stems start with digits (we choose our own module_name).
    - Tries attributes in order (e.g., page(), then main()).
    - Returns None on failure (caller can warn gracefully).
    """
    try:
        spec = importlib.util.spec_from_file_location(module_name, ROOT / file_path)
        if not spec or not spec.loader:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    except Exception:
        return None

    for attr in attrs:
        fn = getattr(mod, attr, None)
        if callable(fn):
            return fn
    return None

# Optional imports that may fail—keep app resilient
try:
    from ui.history import render_history_tab
except Exception:
    render_history_tab = None  # type: ignore

try:
    from ui.debugger import render_debugger_tab
except Exception:
    render_debugger_tab = None  # type: ignore

# Single source of truth for tabs: (label, route, render_fn)
TABS: list[tuple[str, str, Optional[RenderFn]]] = [
    (
        "⚡ Gap Scanner",
        "gap-scanner",
        _load_render_fn("ui.pages.yday_vol_signal_open", "pages/45_YdayVolSignal_Open.py", ("page",)),
    ),
    (
        "🚀 Spike Precursor Lab",
        "spike-precursor-lab",
        _load_render_fn("ui.pages.spike_precursor_lab", "pages/64_Spike_Precursor_Lab.py", ("page", "main")),
    ),
    (
        "📊 Stock Scanner (Shares Only)",
        "stock-scanner-shares-only",
        _load_render_fn("ui.pages.shares_only", "pages/65_Stock_Scanner_SharesOnly.py", ("page",)),
    ),
    (
        "📅 Backtest (range)",
        "backtest-range",
        _load_render_fn("ui.pages.backtest_range", "pages/55_Backtest_Range.py", ("page",)),
    ),
    (
        "📈 History & Outcomes",
        "history",
        render_history_tab,  # may be None
    ),
    (
        "💧 Data Lake (Phase 1)",
        "data-lake-phase1",
        _load_render_fn("ui.pages.data_lake_phase1", "pages/90_Data_Lake_Phase1.py", ("render_data_lake_tab", "page", "main")),
    ),
    (
        "🐞 Debugger",
        "debugger",
        render_debugger_tab,  # may be None
    ),
]

__all__ = ["TABS"]
