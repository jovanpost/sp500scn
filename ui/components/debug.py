from __future__ import annotations
import time, json, traceback, sys, platform, datetime as dt
from contextlib import contextmanager
from typing import Any, Dict, Optional
import streamlit as st

REDACT_KEYS = ("key", "secret", "token", "password")


def _redact_secrets(d: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        if any(s in k.lower() for s in REDACT_KEYS):
            out[k] = "***REDACTED***"
        else:
            out[k] = v
    return out


class DebugLog:
    """Session-scoped debug collector."""

    def __init__(self, name: str = "dbg"):
        self.name = name
        self._data = {
            "meta": {
                "session_started": dt.datetime.utcnow().isoformat() + "Z",
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "streamlit": getattr(st, "__version__", "unknown"),
            },
            "params": {},
            "env": {},
            "events": [],  # list of {"t": "...", "name": "...", "data": {...}, "ms": N}
            "errors": [],  # list of {"t": "...", "where": "...", "exc": "..."}
        }

    def set_params(self, **kwargs):
        self._data["params"].update(kwargs)

    def set_env(self, **kwargs):
        self._data["env"].update(kwargs)

    def event(self, name: str, **data):
        self._data["events"].append({
            "t": dt.datetime.utcnow().isoformat() + "Z",
            "name": name,
            "data": data,
        })

    def error(self, where: str, exc: BaseException):
        self._data["errors"].append({
            "t": dt.datetime.utcnow().isoformat() + "Z",
            "where": where,
            "exc": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        })

    @contextmanager
    def step(self, name: str, **data):
        t0 = time.perf_counter()
        self.event(f"{name}:start", **data)
        try:
            yield
            ms = int((time.perf_counter() - t0) * 1000)
            self.event(f"{name}:done", ms=ms)
        except BaseException as e:
            ms = int((time.perf_counter() - t0) * 1000)
            self.event(f"{name}:fail", ms=ms)
            self.error(name, e)
            raise

    def export_text(self) -> str:
        # redact secrets in env/params
        data = dict(self._data)
        data["env"] = _redact_secrets(data.get("env", {}))
        data["params"] = _redact_secrets(data.get("params", {}))
        return json.dumps(data, indent=2, default=str)


def _get_dbg(name: str) -> DebugLog:
    key = f"__debuglog_{name}"
    if key not in st.session_state:
        st.session_state[key] = DebugLog(name)
    return st.session_state[key]


def debug_panel(name: str = "page", extra_info: Optional[dict] = None):
    """
    Returns (dbg) and renders a collapsible debug block at the bottom.
    Usage:
        dbg = debug_panel("backtest")
        dbg.set_params(**params_dict)
        with dbg.step("load_prices"):
            ...
    """
    dbg = _get_dbg(name)
    if extra_info:
        dbg.set_env(**extra_info)
    try:
        with st.expander("🐞 Debug panel", expanded=False):
            st.caption("Everything below is for diagnostics. Safe to share (secrets redacted).")
            extra = st.session_state.get(f"__debug_extra_{name}")
            if isinstance(extra, dict) and extra:
                pass_count = extra.get("precedent_pass_count")
                fail_count = extra.get("precedent_fail_count")
                median_hits_pass = extra.get("precedent_hits_median_pass")
                samples = extra.get("precedent_details_preview") or []
                st.markdown("**Precedent checks**")
                st.write(
                    {
                        "passes": int(pass_count) if pass_count is not None else None,
                        "fails": int(fail_count) if fail_count is not None else None,
                        "median_hits_pass": median_hits_pass,
                    }
                )
                if samples:
                    for idx, sample in enumerate(samples[:2], 1):
                        st.caption(
                            f"Sample {idx}: {sample.get('ticker')} on {sample.get('trade_date')} "
                            f"(hits={sample.get('precedent_hits')} ok={sample.get('precedent_ok')})"
                        )
                        st.write(sample.get("events", []))
            txt = dbg.export_text()
            st.text_area("Report (JSON)", value=txt, height=300, key=f"{name}_json", label_visibility="collapsed")
            st.download_button("Download JSON", data=txt.encode("utf-8"), file_name=f"{name}_debug.json", mime="application/json", key=f"{name}_dl")
    except Exception:
        pass
    return dbg
