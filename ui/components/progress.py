from __future__ import annotations

import math
import streamlit as st


_BAR_WIDTH = 30  # characters


def _to_percent(v) -> int:
    """Normalize any numeric v to 0..100 int."""
    try:
        if isinstance(v, float):
            if 0.0 <= v <= 1.0:
                v = v * 100.0
        v = int(round(float(v)))
    except Exception:
        v = 0
    return max(0, min(100, v))


class _StatusLike:
    """Minimal shim compatible with st.status(...).update(label=..., state=...)."""

    def __init__(self, title_slot):
        self._title_slot = title_slot

    def update(self, label: str | None = None, state: str | None = None):
        if label:
            self._title_slot.markdown(f"**{label}**")
        # state is accepted but ignored for broad compatibility


class _ProgLike:
    """Simple progress renderer using Markdown; avoids Streamlit version differences."""

    def __init__(self, bar_slot):
        self._bar_slot = bar_slot
        self._last = -1

    def progress(self, v, *args, **kwargs):
        """Update progress, absorbing extra args/kwargs for API compatibility."""
        # Accept and ignore extra args/kwargs to mimic st.progress signature
        pct = _to_percent(v)
        if pct == self._last:
            return
        self._last = pct
        filled = math.floor(pct / 100 * _BAR_WIDTH)
        bar = "█" * filled + "░" * (_BAR_WIDTH - filled)
        # Render as plain markdown to keep it universal
        self._bar_slot.markdown(f"{pct}%\n\n`{bar}`")


def status_block(title: str, key_prefix: str = "prog"):
    """
    Version-proof progress block built from primitives only.
    Returns: (status_like, prog_widget, log_fn)
      - status_like.update(label=..., state=...)
      - prog_widget.progress(value) where value can be 0..1 (float) or 0..100 (int)
      - log_fn(text) appends to a code block
    """

    title_slot = st.empty()
    title_slot.markdown(f"**{title}**")

    bar_slot = st.empty()
    prog_widget = _ProgLike(bar_slot)

    log_slot = st.empty()
    _buf: list[str] = []

    def log_fn(msg: str):
        _buf.append(str(msg))
        # Keep last N lines to avoid unbounded growth
        log_slot.code("\n".join(_buf[-200:]), language="text")

    return _StatusLike(title_slot), prog_widget, log_fn

