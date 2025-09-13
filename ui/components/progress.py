from __future__ import annotations

import streamlit as st


class _StatusLike:
    """Drop-in for st.status(): supports .update(label=..., state=...)."""

    def __init__(self, title_slot):
        self._title_slot = title_slot

    def update(self, label: str | None = None, state: str | None = None):
        if label:
            self._title_slot.markdown(f"**{label}**")  # ignore state for compat


def status_block(title: str, key_prefix: str = "prog"):
    """
    Version-proof progress block.
    Returns: (status_like, prog_widget, log_fn)
      - status_like.update(label=..., state=...)
      - prog_widget.progress(int 0..100)
      - log_fn(text) appends to a code block
    """
    title_slot = st.empty()
    title_slot.markdown(f"**{title}**")
    prog_widget = st.progress(0, key=f"{key_prefix}_prog")
    log_slot = st.empty()
    _buf: list[str] = []

    def log_fn(msg: str):
        _buf.append(str(msg))
        log_slot.code("\n".join(_buf[-80:]), language="text")

    return _StatusLike(title_slot), prog_widget, log_fn
