from __future__ import annotations

import streamlit as st


class _DummyStatus:
    def update(self, **kwargs):
        pass


def status_block(title: str, key_prefix: str = "prog"):
    """
    Returns (status, prog, log) where:
      - status has .update(label=..., state=...) (noop in fallback)
      - prog is st.progress
      - log(msg: str) appends a message
    """
    root = st.container()

    try:
        status = st.status(title, expanded=True, key=f"{key_prefix}_status")
        prog = st.progress(0, text="", key=f"{key_prefix}_prog")

        def log(msg: str):
            status.write(msg)

        return status, prog, log
    except Exception:
        # Fallback for environments without st.status or when it errors
        root.markdown(f"**{title}**", key=f"{key_prefix}_title")
        prog = root.progress(0, text="", key=f"{key_prefix}_prog")
        log_area = root.empty(key=f"{key_prefix}_log")
        _buffer: list[str] = []

        def log(msg: str):
            _buffer.append(str(msg))
            log_area.code("\n".join(_buffer[-50:]), language="text")

        return _DummyStatus(), prog, log

