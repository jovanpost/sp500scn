from __future__ import annotations

import time
from typing import Iterable, Optional
from uuid import uuid4

import streamlit as st


def status_block(title: str, *, key_prefix: str = "stx"):
    """
    Convenience context that returns (status, progress, log_write).
    Use log_write(msg) to append lines safely.
    """
    key = f"{key_prefix}_{uuid4().hex[:8]}"
    container = st.container()
    with container:
        status = st.status(title, expanded=True, key=f"{key}_status")
        prog = st.progress(0, text="Starting…", key=f"{key}_prog")
        log_area = st.empty()
        lines: list[str] = []

        def log_write(msg: str):
            lines.append(msg)
            # Render last ~60 lines to avoid huge DOM
            tail = "\n".join(lines[-60:])
            log_area.code(tail, language="text")

    return status, prog, log_write
