from __future__ import annotations

import math
import streamlit as st


_BAR_WIDTH = 30  # characters


def _to_percent(value) -> int:
    """Normalize any numeric value to an integer 0..100. Swallow non-numerics."""
    try:
        x = float(value)
        if 0.0 <= x <= 1.0:
            x *= 100.0
        pct = int(round(x))
    except Exception:
        pct = 0
    return max(0, min(100, pct))


class _StatusLike:
    """Minimal shim compatible with st.status(...).update(label=..., state=...)."""

    def __init__(self, title_slot):
        self._title_slot = title_slot

    def update(self, label: str | None = None, state: str | None = None, **kwargs):
        # Accept and ignore extra kwargs for broader compatibility
        try:
            if label:
                self._title_slot.markdown(f"**{label}**")
        except Exception:
            pass
        # state is accepted but ignored for broad compatibility


class _ProgLike:
    """Progress renderer using only Markdown; tolerates extra kwargs like text=..."""

    def __init__(self, bar_slot):
        self._bar_slot = bar_slot
        self._last = -1

    def progress(self, value=None, **kwargs):
        pct = _to_percent(0 if value is None else value)
        if pct == self._last:
            return
        self._last = pct
        try:
            filled = math.floor(pct / 100 * _BAR_WIDTH)
            bar = "█" * filled + "░" * (_BAR_WIDTH - filled)
            # Render as plain markdown to keep it universal
            self._bar_slot.markdown(f"{pct}%\n\n`{bar}`")
        except Exception:
            # Never crash
            pass


def status_block(title: str, key_prefix: str = "prog"):
    """
    Returns: (status_like, prog_widget, log_fn)
      - status_like.update(label=..., state=..., **ignored)
      - prog_widget.progress(value[, text=...]) where value can be 0..1 or 0..100
      - log_fn(text) appends to a code block (keeps last ~200 lines)
    """

    title_slot = st.container(key=f"{key_prefix}_status").empty()
    try:
        title_slot.markdown(f"**{title}**")
    except Exception:
        pass

    bar_slot = st.container(key=f"{key_prefix}_prog").empty()
    prog_widget = _ProgLike(bar_slot)

    log_slot = st.container(key=f"{key_prefix}_log").empty()
    _buf: list[str] = []

    def log_fn(msg: str):
        try:
            _buf.append(str(msg))
            # Keep last N lines to avoid unbounded growth
            log_slot.code("\n".join(_buf[-200:]), language="text")
        except Exception:
            pass

    return _StatusLike(title_slot), prog_widget, log_fn

