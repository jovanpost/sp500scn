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
    Version-proof progress block built from primitives only.
    Returns: (status_like, prog_widget, log_fn)
      - status_like.update(label=..., state=...)
      - prog_widget.progress(value, *args, **kwargs) where value can be
        0..1 (float) or 0..100 (int)
      - log_fn(text) appends to a code block
    """
    title_slot = st.empty()
    title_slot.markdown(f"**{title}**")

    # Progress: avoid passing key= (not supported in some versions),
    # and normalize inputs so callers can pass float 0..1 or int 0..100.
    try:
        raw = st.progress(0)  # no key argument for max compatibility

        class _ProgLike:
            def __init__(self, raw_prog):
                self._raw = raw_prog

            def progress(self, v, *args, **kwargs):
                # Normalize to 0..100 int and forward any other args/kwargs
                try:
                    if isinstance(v, float):
                        if 0.0 <= v <= 1.0:
                            v = int(round(v * 100))
                        else:
                            v = int(round(v))
                    else:
                        v = int(v)
                except Exception:
                    v = 0
                v = max(0, min(100, v))
                self._raw.progress(v, *args, **kwargs)

        prog_widget = _ProgLike(raw)
    except Exception:
        class _NoopProg:
            def progress(self, v, *args, **kwargs):  # no-op fallback
                pass

        prog_widget = _NoopProg()

    log_slot = st.empty()
    _buf: list[str] = []

    def log_fn(msg: str):
        _buf.append(str(msg))
        # Keep last N lines to avoid growing forever
        log_slot.code("\n".join(_buf[-200:]), language="text")

    return _StatusLike(title_slot), prog_widget, log_fn
